"""The state machine: Idle -> Listen -> Parse -> Check -> Grasp -> Verify -> Handoff -> Return,
plus Safe Stop (freeze, servos powered) and Killed (everything disconnected).

Each state is one coroutine that returns the name of the next state. The UI can force any state
(go) or stand in for a model that is not trained yet (signal): "wake", "command", "found",
"grasped", "held", "grabbed". `detail` always says, in words, what the robot is doing and why;
`view` holds the numbers the video overlays draw.
"""

import asyncio
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field

from . import audio, config, infer, vision
from .arm import BODY, Arm, ArmError
from .collect import DemoRecorder, ImageCapture
from .parse import Command, parse_command

log = logging.getLogger("yogurt.fsm")

STATES = ["IDLE", "LISTEN", "PARSE", "CHECK", "GRASP", "VERIFY", "HANDOFF", "RETURN", "SAFE_STOP", "KILLED"]
YOGURT_LABELS = ["yogurt", "cup", "bowl"]  # "yogurt" from our fine-tuned detector, the others from COCO


@dataclass
class Person:
    id: int | None      # face-tracker id, None if only a direction is known
    bearing: float      # degrees from the base camera's axis, positive = right


@dataclass
class Request:
    """Everything remembered about the request in progress. Reset every time Idle is entered."""

    wake_at: float = 0.0
    audio: tuple[float, float] | None = None    # when the command was spoken
    heard: str | None = None                    # the command as text
    command: Command | None = None
    requester: Person | None = None
    others: list[Person] = field(default_factory=list)
    target: Person | None = None
    target_role: str = ""                       # "requester" or "other"
    yogurt_pan: float | None = None             # where the cup was found
    grasp_pose: dict | None = None              # arm pose when the gripper closed: where to put it back
    attempts: int = 0                           # grasp attempts so far
    centre_failures: int = 0


class Brain:
    def __init__(self, cfg, arm: Arm, mic: audio.Mic, wrist, base):
        self.cfg, self.arm, self.mic, self.wrist, self.base = cfg, arm, mic, wrist, base
        self.faces = vision.FaceTracker(cfg.cameras.base.hfov_deg)
        self.asr = audio.Transcriber(cfg)
        self.demos = DemoRecorder(cfg)
        self.capture = ImageCapture()
        self.demo_mode = False
        self.state = "SAFE_STOP"
        self.detail = "Started. The arm is holding still - press Idle to begin."
        self.entered = time.monotonic()
        self.events: deque = deque(maxlen=200)
        self.request = Request()
        self.view: dict = {}
        self.signals: dict = {}
        self.models: dict = {}          # name -> "ready" | "missing ..." | "error ..."
        self._task: asyncio.Future | None = None
        self._forced: str | None = None
        self._why = ""
        self._gripper_closed = False
        self.slm_status = "loading…"
        self.load_models()
        threading.Thread(target=self._warm_up_language_model, name="slm-warm-up", daemon=True).start()

    def _warm_up_language_model(self) -> None:
        """Ask the language model one question now, so the first real command does not wait for it to load."""
        try:
            parse_command("give me the yogurt", self.cfg.slm.url, self.cfg.slm.model, timeout_s=180)
            self.slm_status = f"ready ({self.cfg.slm.model})"
        except RuntimeError as e:
            self.slm_status = str(e)

    # ---- models -----------------------------------------------------------------------------

    def load_models(self) -> None:
        """(Re)load every model file that exists. A missing model just leaves its state to the operator."""
        paths = self.cfg.models

        def attempt(name: str, build):
            file = config.path(paths[name])
            if not file.exists():
                self.models[name] = f"missing ({paths[name]})"
                return None
            try:
                model = build(file)
                self.models[name] = "ready"
                return model
            except Exception as e:
                log.exception("loading %s", name)
                self.models[name] = f"error: {e}"
                return None

        def grasp_policy(folder):
            from .grasp import GraspPolicy
            return GraspPolicy(folder, self.cfg)

        self.wake = attempt("wake_word", lambda f: audio.WakeWord(f, self.cfg.wake.threshold))
        self.detector = attempt("yogurt_detector", lambda f: vision.YogurtDetector(infer.load(f), YOGURT_LABELS))
        self.verifier = attempt("grasp_verify", lambda f: vision.Classifier(infer.load(f)))
        self.grab_model = attempt("handoff_grab", lambda f: vision.Classifier(infer.load(f)))
        self.policy = attempt("act_grasp", grasp_policy)

    # ---- plumbing ---------------------------------------------------------------------------

    def say(self, message: str) -> None:
        """A decision worth keeping: shown as the current detail and added to the event log."""
        self.detail = message
        self.events.appendleft({"time": time.strftime("%H:%M:%S"), "state": self.state, "message": message})
        log.info("[%s] %s", self.state, message)

    def signal(self, name: str, value=True) -> None:
        self.signals[name] = value

    def take(self, name: str):
        return self.signals.pop(name, None)

    def go(self, state: str, why: str = "operator") -> None:
        """Force a state from the UI. Safe Stop and Kill act on the hardware at once."""
        if state not in STATES:
            raise ValueError(f"unknown state {state}")
        if state == "KILLED":
            self.disconnect(why if why != "operator" else "kill switch")
        elif state == "SAFE_STOP":
            self.arm.hold()
        elif not self.arm.connected:
            raise ArmError("the arm is disconnected - press Reconnect first")
        self._forced, self._why = state, why
        if self._task is not None:
            self._task.cancel()

    def start_demos(self) -> None:
        """Demo session: Check finds and centres the cup as usual, then the leader arm does the grasp."""
        if not self.cfg.leader.port:
            raise RuntimeError("no leader arm configured: set leader.port in the hardware config")
        self.demo_mode = True
        self.go("CHECK", "demonstration session")

    def connect(self) -> None:
        """Open cameras, microphone and arm. Each failure is reported, none stops the others."""
        self.wrist.open()  # camera problems show up as camera.error, which the UI displays
        self.base.open()
        for name, start in (("microphone", self.mic.open), ("arm", self.arm.connect)):
            try:
                start()
            except Exception as e:
                self.say(f"{name}: {e}")

    def disconnect(self, reason: str) -> None:
        """Kill: release the servos and let go of every device."""
        self.arm.kill(reason)
        self.demos.stop()
        self.demo_mode = False
        self.capture.stop()
        self.mic.close()
        self.wrist.close()
        self.base.close()

    async def run(self) -> None:
        watcher = asyncio.ensure_future(self._watch())
        try:
            while True:
                following, why = self.state, ""
                if self._forced is None:
                    self._task = asyncio.ensure_future(getattr(self, "_" + self.state.lower())())
                    try:
                        following = await self._task
                    except asyncio.CancelledError:
                        if self._forced is None:
                            raise  # the program is shutting down
                    except ArmError as e:
                        following, why = ("SAFE_STOP" if self.arm.connected else "KILLED"), f"arm problem: {e}"
                    except Exception as e:
                        log.exception("state %s failed", self.state)
                        following, why = "SAFE_STOP", f"{type(e).__name__}: {e}"
                        self.arm.hold()
                if self._forced is not None:  # a state forced by the operator always wins
                    following, why, self._forced = self._forced, self._why, None
                self.state, self.entered, self.view, self.signals = following, time.monotonic(), {}, {}
                self.say(f"entered {following}" + (f" ({why})" if why else ""))
        finally:
            watcher.cancel()
            if self._task is not None:
                self._task.cancel()

    async def _watch(self) -> None:
        """Background perception: follow faces on the base camera, feed the image recorder."""
        seen_base = seen_wrist = 0.0
        while True:
            await asyncio.sleep(0.03)
            if self.base.stamp != seen_base and (frame := self.base.read()) is not None:
                seen_base = self.base.stamp
                await asyncio.to_thread(self.faces.update, frame)
            if self.wrist.stamp != seen_wrist and (frame := self.wrist.read()) is not None:
                seen_wrist = self.wrist.stamp
                self.capture.offer(frame)
            if not self.arm.connected and self.state != "KILLED" and self._forced is None:
                # the arm dropped out by itself (overheated servo, pulled cable, never connected)
                self._forced, self._why = "KILLED", self.arm.fault or "arm disconnected"
                if self._task is not None:
                    self._task.cancel()

    def pose(self, name: str, pan: float | None = None, gripper: bool = True) -> dict[str, float]:
        pose = dict(self.cfg.poses[name])
        if pan is not None:
            pose["shoulder_pan"] = pan
        if not gripper:
            pose.pop("gripper")
        return pose

    def pan_for(self, bearing: float) -> float:
        """Shoulder angle that points the arm along a base-camera bearing."""
        return self.cfg.base_camera.yaw_deg + self.cfg.base_camera.pan_sign * bearing

    def _person_name(self, person: Person | None) -> str:
        if person is None:
            return "nobody"
        return (f"face #{person.id}" if person.id is not None else "voice") + f" at {person.bearing:+.0f}°"

    # ---- states -----------------------------------------------------------------------------

    async def _idle(self) -> str:
        self.request = Request()
        if self.demo_mode:
            self.demo_mode = False
            await asyncio.to_thread(self.demos.stop)
        await self.arm.goto(self.pose("idle"))
        entered, hits = time.monotonic(), 0
        while True:
            now = time.monotonic()
            typed = self.take("command")
            if self.take("wake") or typed:
                self.request.wake_at, self.request.heard = now, typed
                self.say("woken by the operator")
                return "LISTEN"
            if self.wake is None:
                self.detail = "Wake-word model not trained yet: press Wake (or type a command) to stand in for it."
            elif self.mic.error:
                self.detail = f"Microphone unavailable ({self.mic.error}): press Wake or type a command."
            elif now - entered > audio.WAKE_WINDOW_S:  # never judge audio from before Idle began
                p = self.wake.score(self.mic.audio(now - audio.WAKE_WINDOW_S))
                self.view["wake_p"] = p
                self.detail = f"Listening for “Yo, gurt” (score {p:.2f}, fires at {self.wake.threshold:.2f})."
                hits = hits + 1 if p >= self.wake.threshold else 0
                if hits >= self.cfg.wake.frames:
                    self.request.wake_at = now
                    self.say(f"wake word heard (score {p:.2f})")
                    return "LISTEN"
            await asyncio.sleep(0.1)

    async def _listen(self) -> str:
        cfg, request = self.cfg.listen, self.request
        voice_from = request.wake_at - audio.WAKE_WINDOW_S  # the wake phrase itself shows who is talking
        typed, speaker, voice = None, None, None
        while True:
            now = time.monotonic()
            voice = self.mic.bearing(voice_from)
            speaker = self.faces.speaker(voice, voice_from, cfg.audio_sigma_deg)
            bearing = speaker.bearing if speaker else voice
            self.view.update(voice=voice, speaker=speaker.id if speaker else None)
            if bearing is not None:  # turn towards whoever is talking, to show we are listening
                self.arm.set_target({"shoulder_pan": self.pan_for(bearing)}, cfg.follow_speed_dps)
            typed = request.heard or self.take("command")
            spoken = self.mic.speech_seconds(request.wake_at)
            quiet = now - max(self.mic.last_voice, request.wake_at)
            waited = now - request.wake_at
            if typed or (spoken >= 0.3 and quiet >= cfg.end_silence_s) or waited >= cfg.max_command_s:
                break
            if spoken < 0.3 and waited >= cfg.no_speech_timeout_s:
                self.say("no command followed the wake word")
                return "IDLE"
            facing = f"facing {self._person_name(Person(speaker.id if speaker else None, bearing))}" \
                if bearing is not None else "nobody located yet"
            self.detail = f"Listening to the command ({spoken:.1f} s of speech so far), {facing}."
            await asyncio.sleep(0.1)

        request.heard = typed
        request.audio = (request.wake_at - cfg.pre_roll_s, time.monotonic())
        faces = self.faces.visible()
        request.requester = Person(speaker.id, speaker.bearing) if speaker else Person(None, voice or 0.0)
        request.others = [Person(f.id, f.bearing) for f in faces if speaker is None or f.id != speaker.id]
        how = "mouth movement + voice direction" if speaker and voice is not None else \
            "mouth movement only" if speaker else "voice direction only" if voice is not None else \
            "no face and no voice direction, assuming straight ahead"
        self.say(f"requester = {self._person_name(request.requester)} ({how}); {len(request.others)} other(s) in view")
        return "PARSE"

    async def _parse(self) -> str:
        request = self.request
        if request.heard is None and request.audio is None:
            self.say("nothing to parse: no command has been recorded")
            return "IDLE"
        request.requester = request.requester or Person(None, 0.0)
        try:
            if request.heard is None:
                self.detail = "Transcribing the command…"
                request.heard = await asyncio.to_thread(self.asr.transcribe, self.mic.audio(*request.audio))
            self.say(f"heard: “{request.heard}”")
            self.detail = f"Asking the language model what “{request.heard}” means…"
            command = await asyncio.to_thread(parse_command, request.heard, self.cfg.slm.url, self.cfg.slm.model)
            self.slm_status = f"ready ({self.cfg.slm.model})"
        except RuntimeError as e:
            if "language model" in str(e):
                self.slm_status = str(e)
            self.say(str(e))
            return "IDLE"
        request.command = command
        if not command.give_yogurt:
            self.say("that was not a request for the yogurt")
            return "IDLE"
        if command.recipient == "someone_else" and request.others:
            # "him"/"her": the other person standing closest to the speaker
            request.target = min(request.others, key=lambda p: abs(p.bearing - request.requester.bearing))
            request.target_role = "other"
        else:
            request.target, request.target_role = request.requester, "requester"
        note = " (asked for someone else, but nobody else is in view)" \
            if command.recipient == "someone_else" and not request.others else ""
        self.say(f"yogurt goes to “{command.recipient_words or 'me'}” = the {request.target_role}, "
                 f"{self._person_name(request.target)}{note}")
        return "CHECK"

    async def _detect(self) -> list[vision.Detection]:
        """Yogurt cups in the newest wrist frame, best first."""
        frame = self.wrist.read()
        if frame is None or self.detector is None:
            await asyncio.sleep(0.05)
            return []
        found = await asyncio.to_thread(self.detector.detect, frame, self.cfg.check.confidence)
        self.view["detections"] = [(d.box, d.score, d.label) for d in found]
        self.view["width"] = frame.shape[1]
        return found

    async def _check(self) -> str:
        cfg, request = self.cfg.check, self.request
        self.detail = "Raising the wrist camera to look over the table."
        await self.arm.goto(self.pose("survey", pan=self.arm.pos["shoulder_pan"]))
        if self.detector is None:
            self.detail = ("Yogurt detector missing: jog the arm until the cup is centred in the wrist view, "
                           "then press Yogurt centred.")
            while not self.take("found"):
                await asyncio.sleep(0.1)
        else:
            # Look where the cup was last time first (a retry), then sweep the whole range left to right.
            found = False
            if request.yogurt_pan is not None:
                self.detail = "Looking where the cup was a moment ago."
                await self.arm.goto({"shoulder_pan": request.yogurt_pan})
                found = any([await self._detect(), await self._detect(), await self._detect()])
            if not found:
                self.detail = "Turning to the start of the sweep."
                await self.arm.goto({"shoulder_pan": cfg.pan_min_deg})
                self.arm.set_target({"shoulder_pan": cfg.pan_max_deg}, cfg.sweep_speed_dps)
                hits, end = 0, self.arm.clamp({"shoulder_pan": cfg.pan_max_deg})["shoulder_pan"]
                while hits < cfg.confirm_frames and not (by_hand := self.take("found")):
                    if self.arm.cmd["shoulder_pan"] >= end:
                        self.say("swept the whole table and saw no yogurt")
                        return "IDLE"
                    hits = hits + 1 if await self._detect() else 0
                    self.detail = (f"Sweeping for a yogurt cup: pan {self.arm.pos['shoulder_pan']:+.0f}° "
                                   f"of {cfg.pan_min_deg:+.0f}..{cfg.pan_max_deg:+.0f}°.")
                self.arm.hold()
                if by_hand:
                    self.say("operator says the cup is centred")
                    found = None
            if found is not None and not await self._centre():
                return "CHECK" if request.centre_failures < 3 else "IDLE"
        request.yogurt_pan = self.arm.pos["shoulder_pan"]
        self.say(f"yogurt centred at pan {request.yogurt_pan:+.0f}° - arm is in the pre-grasp pose")
        return "GRASP"

    async def _centre(self) -> bool:
        """Turn until the cup sits on the wrist camera's centre line. One look, one correction, repeat."""
        cfg = self.cfg.check
        deadline, misses = time.monotonic() + cfg.center_timeout_s, 0
        while time.monotonic() < deadline:
            await asyncio.sleep(0.2)  # let the picture settle after the last move
            found = await self._detect()
            if not found:
                misses += 1
                if misses >= 5:
                    break
                continue
            misses = 0
            width = self.view["width"]
            offset = (found[0].centre[0] - width / 2) / width  # fraction of the picture width, + = right
            self.view["offset"] = offset
            if abs(offset) <= cfg.center_tolerance:
                return True
            turn = cfg.pan_sign * offset * self.cfg.cameras.wrist.hfov_deg * 0.6
            self.detail = (f"Cup is {abs(offset) * 100:.0f}% of the picture to the {'right' if offset > 0 else 'left'} "
                           f"of centre: turning {turn:+.1f}°.")
            await self.arm.goto({"shoulder_pan": self.arm.pos["shoulder_pan"] + turn})
        self.request.centre_failures += 1
        self.say("lost the cup while centring on it")
        return False

    def _watch_gripper(self, gripper: float) -> None:
        """Remember the arm pose at the moment the gripper closes: that is where the cup stood."""
        closed = gripper < self.cfg.grasp.closed_below
        if closed and not self._gripper_closed:
            self.request.grasp_pose = dict(self.arm.pos)
        self._gripper_closed = closed

    async def _grasp(self) -> str:
        cfg = self.cfg.grasp
        self._gripper_closed = False
        self.request.attempts += 1
        if self.demo_mode:
            try:
                return await self._demonstrate()
            finally:
                if self.demos.dataset is not None:
                    self.demos.discard()  # never keep half an episode when the state is interrupted
        if self.policy is None:
            self.detail = ("No trained grasp policy (models/act_grasp). Record demonstrations and train one; "
                           "until then grasp by hand and press Grasp done.")
            while not self.take("grasped"):
                await asyncio.sleep(0.1)
            return "VERIFY"
        self.policy.reset()
        period, started = 1.0 / cfg.fps, time.monotonic()
        while (elapsed := time.monotonic() - started) < cfg.duration_s and not self.take("grasped"):
            tick = time.monotonic()
            frame = self.wrist.read()
            if frame is None:
                raise RuntimeError("wrist camera stopped during the grasp")
            action = await asyncio.to_thread(self.policy.act, frame, dict(self.arm.pos))
            self.arm.set_target(action, self.cfg.arm.max_speed_dps)
            self._watch_gripper(action["gripper"])
            self.detail = (f"ACT policy is grasping (attempt {self.request.attempts}, "
                           f"{elapsed:.0f}/{cfg.duration_s} s): gripper {action['gripper']:.0f}%.")
            await asyncio.sleep(max(0.0, period - (time.monotonic() - tick)))
        self.arm.hold()
        return "VERIFY"

    async def _demonstrate(self) -> str:
        """Demo mode's version of Grasp: the leader arm drives, and the motion is recorded."""
        recorder, period, fast = self.demos, 1.0 / self.cfg.grasp.fps, self.cfg.arm.max_speed_dps

        async def finish() -> str:
            self.arm.hold()
            await asyncio.to_thread(recorder.stop)
            self.demo_mode = False
            self.say("demonstration session finished")
            return "RETURN"

        if not recorder.open:
            self.detail = "Connecting the leader arm…"
            await asyncio.to_thread(recorder.start)
        while True:  # 1. wait until the leader is held where the follower already is: no jump at the start
            leader = await asyncio.to_thread(recorder.leader_pose)
            gap = {j: leader[j] - self.arm.pos[j] for j in BODY}
            worst = max(gap, key=lambda j: abs(gap[j]))
            if abs(gap[worst]) < 12:
                break
            if self.take("demo_finish"):
                return await finish()
            self.detail = (f"Demo: bring the leader arm to the follower's pose - {worst} is {gap[worst]:+.0f}° off. "
                           "Recording starts by itself when they match.")
            await asyncio.sleep(0.1)

        self.say("recording: grasp the cup, lift it a little, hold still, then press Save demo")
        started, frames = time.monotonic(), 0
        while True:  # 2. record
            tick = time.monotonic()
            action = self.arm.clamp(await asyncio.to_thread(recorder.leader_pose))
            frame = self.wrist.read()
            self.arm.set_target(action, fast)
            self._watch_gripper(action["gripper"])
            if frame is not None:
                await asyncio.to_thread(recorder.add, dict(self.arm.pos), action, frame)
                frames += 1
            if self.take("demo_save"):
                self.detail = "Saving the demonstration…"
                total = await asyncio.to_thread(recorder.save)
                self.say(f"demonstration saved ({frames} frames); the dataset now has {total}")
                break
            if self.take("demo_discard") or tick - started > 60:
                recorder.discard()
                self.say("demonstration discarded")
                break
            self.detail = f"RECORDING demonstration: {tick - started:.0f} s, {frames} frames."
            await asyncio.sleep(max(0.0, period - (time.monotonic() - tick)))

        self.detail = ("Not recording. The leader still drives the arm: put the cup somewhere new and let go, "
                       "then press Next demo or Finish demos.")
        while True:  # 3. the leader keeps driving so the operator can reset the scene
            self.arm.set_target(self.arm.clamp(await asyncio.to_thread(recorder.leader_pose)), fast)
            if self.take("demo_next"):
                self.arm.hold()
                return "CHECK"
            if self.take("demo_finish"):
                return await finish()
            await asyncio.sleep(period)

    async def _verify(self) -> str:
        cfg = self.cfg.verify
        self.detail = "Lifting the cup in front of the wrist camera to check the grasp."
        await self.arm.goto(self.pose("survey", pan=self.arm.pos["shoulder_pan"], gripper=False))
        if self.verifier is None:
            self.detail = ("Grasp-verification model not trained yet: is the yogurt in the gripper? "
                           "Press Held or Not held.")
            while (held := self.take("held")) is None:
                await asyncio.sleep(0.1)
            p = 1.0 if held else 0.0
        else:
            scores, started = [], time.monotonic()
            while time.monotonic() - started < cfg.window_s:
                if (frame := self.wrist.read()) is not None:
                    scores.append(await asyncio.to_thread(self.verifier.prob, frame))
                    self.view["held_p"] = sum(scores) / len(scores)
                await asyncio.sleep(0.05)
            p = sum(scores) / len(scores) if scores else 0.0
            held = p >= cfg.threshold
        if held:
            self.say(f"yogurt is in the gripper (P(held) = {p:.2f})")
            return "HANDOFF"
        self.say(f"grasp failed (P(held) = {p:.2f}, need {cfg.threshold:.2f}) - looking for the cup again")
        return "CHECK"

    def _locate(self, person: Person) -> float:
        """Where is this person now? Their tracked face; else the nearest face to where they were; else memory."""
        faces = self.faces.visible()
        taken = {p.id for p in [self.request.requester, *self.request.others] if p is not person}
        match = next((f for f in faces if f.id == person.id), None)
        if match is None:
            near = [f for f in faces if f.id not in taken
                    and abs(f.bearing - person.bearing) <= self.cfg.handoff.max_face_shift_deg]
            match = min(near, key=lambda f: abs(f.bearing - person.bearing), default=None)
        if match is not None:
            person.id, person.bearing = match.id, match.bearing
        self.view["target_seen"] = match is not None
        return person.bearing

    async def _handoff(self) -> str:
        cfg, request = self.cfg.handoff, self.request
        target = request.target or Person(None, 0.0)
        request.target = target
        pan = self.pan_for(self._locate(target))
        role = request.target_role or "person"
        self.detail = f"Turning to the {role} ({self._person_name(target)}), then reaching out."
        await self.arm.goto({"shoulder_pan": pan})
        await self.arm.goto(self.pose("handoff", pan=pan, gripper=False))
        deadline, above = time.monotonic() + cfg.timeout_s, None
        while (left := deadline - time.monotonic()) > 0:
            pan_now = self.pan_for(self._locate(target))
            if abs(pan_now - self.arm.target["shoulder_pan"]) > 8:  # they moved: follow, gently
                self.arm.set_target({"shoulder_pan": pan_now}, self.cfg.listen.follow_speed_dps / 2)
            taken = self.take("grabbed")
            if self.grab_model is None:
                waiting = "hand-off model not trained yet: press Taken when the person holds the cup"
            else:
                p = 0.0
                if (frame := self.wrist.read()) is not None:
                    p = await asyncio.to_thread(self.grab_model.prob, frame)
                self.view["grab_p"] = p
                above = (above or time.monotonic()) if p >= cfg.threshold else None
                taken = taken or (above is not None and time.monotonic() - above >= cfg.hold_s)
                waiting = f"P(grabbed) = {p:.2f}, lets go at {cfg.threshold:.2f}"
            self.view["countdown"] = left
            self.detail = (f"Holding the yogurt out to the {request.target_role or 'person'} "
                           f"({self._person_name(target)}); {waiting}. Giving up in {left:.0f} s.")
            if taken:
                self.say("the cup was taken - letting go")
                await self.arm.goto({"gripper": self.cfg.poses.survey.gripper})
                await asyncio.sleep(cfg.release_wait_s)
                return "RETURN"
            await asyncio.sleep(0.05)
        self.say(f"nobody took the yogurt in {cfg.timeout_s} s - putting it back")
        return await self._put_back()

    async def _put_back(self) -> str:
        request = self.request
        if request.grasp_pose is None:
            self.say("I do not know where the cup stood (the grasp was done by hand): holding it - "
                     "take the cup, then press Return")
            return "SAFE_STOP"
        pan = request.grasp_pose["shoulder_pan"]
        self.detail = "Carrying the yogurt back to where it was picked up."
        await self.arm.goto(self.pose("survey", pan=pan, gripper=False))
        await self.arm.goto({j: request.grasp_pose[j] for j in BODY})
        await self.arm.goto({"gripper": self.cfg.poses.survey.gripper})
        await self.arm.goto(self.pose("survey", pan=pan))
        self.say("yogurt is back where it was")
        return "RETURN"

    async def _return(self) -> str:
        self.detail = "Returning home."
        await self.arm.goto(self.pose("idle"))
        return "IDLE"

    async def _safe_stop(self) -> str:
        self.arm.hold()
        await asyncio.Event().wait()  # until the operator picks a state

    async def _killed(self) -> str:
        await asyncio.Event().wait()  # until the operator reconnects
