"""The fetch-and-handoff state machine. Pure logic: events in, bus messages out.

    IDLE --wake--> LISTEN --intent--> PARSE --fetch--> CHECK --cup seen--> GRASP --done--> VERIFY
    VERIFY --holding--> HANDOFF --presented, then taken--> RELEASE_HOME --done--> IDLE
    VERIFY --missed--> GRASP (retry) or RELEASE_HOME
    any state --stop, or arm latched--> SAFE_STOP --reset--> IDLE

Phase 1 config skips the optional steps: listen: false (wake goes straight to PARSE with
default_intent) and require_cup: false (CHECK passes without vision).

The policy process runs the arm motions; this machine only starts and stops policy tasks
("grasp", "handoff", "release_home") and reacts to their policy/status.
"""

import math
from dataclasses import dataclass
from enum import StrEnum

from yogurt.config import OrchestratorConfig
from yogurt.messages import (
    Message,
    OrchestratorState,
    PolicyCommand,
    Resume,
    Safety,
    Say,
    Stop,
)

CUP_FRESH_S = 1.0  # a cup detection this recent counts as "visible now"


class State(StrEnum):
    IDLE = "IDLE"
    LISTEN = "LISTEN"
    PARSE = "PARSE"
    CHECK = "CHECK"
    GRASP = "GRASP"
    VERIFY = "VERIFY"
    HANDOFF = "HANDOFF"
    RELEASE_HOME = "RELEASE_HOME"
    SAFE_STOP = "SAFE_STOP"


# Events ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Tick:
    pass


@dataclass(frozen=True)
class Wake:
    pass


@dataclass(frozen=True)
class Heard:
    intent: str
    text: str = ""


@dataclass(frozen=True)
class CupSeen:
    pass


@dataclass(frozen=True)
class Taken:
    """Someone is taking the cup (hand at the gripper, a pull on the FSR, or dev key 'h')."""


@dataclass(frozen=True)
class PolicyDone:
    task: str
    ok: bool
    detail: str = ""


@dataclass(frozen=True)
class ArmUpdate:
    gripper: float
    safety: str
    detail: str = ""


@dataclass(frozen=True)
class StopRequest:
    reason: str


@dataclass(frozen=True)
class Reset:
    pass


Event = Tick | Wake | Heard | CupSeen | Taken | PolicyDone | ArmUpdate | StopRequest | Reset
Action = tuple[str, Message]

LATCHED = (Safety.STOPPED, Safety.FAULT)


class Orchestrator:
    def __init__(self, cfg: OrchestratorConfig, now: float = 0.0):
        self.cfg = cfg
        self.state = State.IDLE
        self.now = now
        self.entered = now
        self.intent = ""
        self.attempts = 0
        self.gripper: float | None = None
        self.presenting = False
        self.policy_task = ""
        self._last_cup = -math.inf
        self._resuming = False
        self._last_stop_sent = -math.inf
        self._last_resume_sent = -math.inf
        self._out: list[Action] = []

    def handle(self, event: Event, now: float) -> list[Action]:
        self.now = now
        self._handle(event)
        out, self._out = self._out, []
        return out

    # Dispatch -------------------------------------------------------------------------------

    def _handle(self, ev: Event) -> None:
        S = State
        if isinstance(ev, StopRequest):
            self._safe_stop(ev.reason)
            return
        if isinstance(ev, ArmUpdate):
            self.gripper = ev.gripper
            latched = ev.safety in LATCHED
            if self._resuming:
                self._resuming = latched  # done once the driver reports it unlatched
            elif latched and self.state is not S.SAFE_STOP:
                self._safe_stop(f"arm {ev.safety}: {ev.detail}")
            return
        if isinstance(ev, CupSeen):
            self._last_cup = self.now
        if isinstance(ev, Heard) and ev.intent == "stop":
            self._safe_stop("voice: stop")
            return
        if isinstance(ev, Tick):
            self._tick()
            return

        s = self.state
        if s is S.IDLE and isinstance(ev, Wake):
            if self.cfg.listen:
                self._go(S.LISTEN, "wake")
            else:
                self.intent = self.cfg.default_intent
                self._go(S.PARSE, "wake")
        elif s is S.LISTEN and isinstance(ev, Heard):
            self.intent = ev.intent
            self._go(S.PARSE, f"heard {ev.intent}")
        elif s is S.CHECK and isinstance(ev, CupSeen):
            self._go(S.GRASP, "cup seen")
        elif isinstance(ev, PolicyDone) and ev.task == self.policy_task:
            self.policy_task = ""
            self._policy_done(ev)
        elif s is S.HANDOFF and isinstance(ev, Taken) and self.presenting:
            self._go(S.RELEASE_HOME, "taken")
        elif s is S.SAFE_STOP and isinstance(ev, Reset):
            self._resuming = True
            self._send_resume()
            self._go(S.IDLE, "reset")

    def _policy_done(self, ev: PolicyDone) -> None:
        S = State
        if self.state is S.GRASP:
            if ev.ok:
                self._go(S.VERIFY, "grasp done")
            else:
                self._say("I couldn't get it.")
                self._go(S.RELEASE_HOME, f"grasp failed: {ev.detail}")
        elif self.state is S.HANDOFF:
            if ev.ok:
                self.presenting = True
                self._say("Here you go.")
            else:
                self._safe_stop(f"handoff failed: {ev.detail}")
        elif self.state is S.RELEASE_HOME:
            if ev.ok:
                self._go(S.IDLE, "home")
            else:
                self._safe_stop(f"release_home failed: {ev.detail}")

    def _tick(self) -> None:
        S = State
        if self.state is S.SAFE_STOP and self.now - self._last_stop_sent >= self.cfg.stop_repeat_s:
            self._send_stop("safe stop (repeat)")
        if self._resuming and self.now - self._last_resume_sent >= self.cfg.stop_repeat_s:
            self._send_resume()

        limit = self.cfg.timeouts_s.get(self.state.value)
        if limit is None or self.now - self.entered < limit:
            return
        s = self.state
        if s is S.LISTEN:
            self._go(S.IDLE, "listen timeout")
        elif s is S.CHECK:
            self._say("I can't see a yogurt.")
            self._go(S.IDLE, "no cup")
        elif s is S.GRASP:
            self._say("I couldn't get it.")
            self._go(S.RELEASE_HOME, "grasp timeout")
        elif s is S.VERIFY:
            # VERIFY's timeout is the settle time before judging the grasp by the gripper.
            if self.gripper is not None and self.gripper >= self.cfg.holding_min_gripper:
                self._go(S.HANDOFF, f"holding (gripper {self.gripper:.0f})")
            elif self.attempts < self.cfg.max_grasp_attempts:
                self._say("Missed. Trying again.")
                self._go(S.GRASP, "missed")
            else:
                self._say("I couldn't get it.")
                self._go(S.RELEASE_HOME, "missed, out of attempts")
        elif s is S.HANDOFF:
            # Never drop the cup: keep holding it out and remind.
            self._say("Your yogurt is ready.")
            self.entered = self.now
        elif s is S.RELEASE_HOME:
            self._safe_stop("release_home timeout")

    # Transitions ----------------------------------------------------------------------------

    def _go(self, state: State, why: str) -> None:
        previous, self.state, self.entered = self.state, state, self.now
        self._emit(
            "orchestrator/state", OrchestratorState(state=state, previous=previous, event=why)
        )
        S = State
        if state is S.IDLE:
            self._stop_policy()
            self.attempts = 0
        elif state is S.LISTEN:
            self._say("Yes?")
        elif state is S.PARSE:
            if self.intent == "fetch":
                self._go(S.CHECK, "fetch")
            else:
                self._say("Sorry, I can only fetch yogurt.")
                self._go(S.IDLE, f"unknown intent {self.intent!r}")
        elif state is S.CHECK:
            self.attempts = 0
            if not self.cfg.require_cup or self.now - self._last_cup < CUP_FRESH_S:
                self._go(S.GRASP, "cup visible" if self.cfg.require_cup else "cup check off")
        elif state is S.GRASP:
            self.attempts += 1
            self._start_policy("grasp")
        elif state is S.HANDOFF:
            self.presenting = False
            self._start_policy("handoff")
        elif state is S.RELEASE_HOME:
            self._start_policy("release_home")

    def _safe_stop(self, reason: str) -> None:
        self._resuming = False
        self._stop_policy()
        self._send_stop(reason)
        if self.state is not State.SAFE_STOP:
            self._say("Stopping.", interrupt=True)
            self._go(State.SAFE_STOP, reason)

    # Outputs --------------------------------------------------------------------------------

    def _emit(self, topic: str, msg: Message) -> None:
        self._out.append((topic, msg))

    def _say(self, text: str, interrupt: bool = False) -> None:
        self._emit("tts/say", Say(text=text, interrupt=interrupt))

    def _start_policy(self, task: str) -> None:
        self._stop_policy()
        self.policy_task = task
        self._emit("policy/command", PolicyCommand(action="start", task=task))

    def _stop_policy(self) -> None:
        if self.policy_task:
            self._emit("policy/command", PolicyCommand(action="stop", task=self.policy_task))
            self.policy_task = ""

    def _send_stop(self, reason: str) -> None:
        self._last_stop_sent = self.now
        self._emit("system/stop", Stop(reason=reason))

    def _send_resume(self) -> None:
        self._last_resume_sent = self.now
        self._emit("system/resume", Resume(reason="operator reset"))
