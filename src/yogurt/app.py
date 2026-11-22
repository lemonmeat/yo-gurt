"""The web interface: `uv run yogurt laptop` or `uv run yogurt pi`, then open http://<host>:8080.

Everything is behind HTTP Basic Authentication (web.user / web.password in the hardware config).
"""

import argparse
import asyncio
import base64
import contextlib
import hmac
import json
import logging
import socket
import time
from pathlib import Path

import cv2
from aiohttp import web

from . import collect, config, lab, overlay
from .arm import JOINTS, Arm
from .audio import Mic
from .cameras import Camera
from .fsm import STATES, Brain

log = logging.getLogger("yogurt.app")
STATIC = Path(__file__).parent / "static"


def make_app(cfg, brain: Brain) -> web.Application:
    expected = "Basic " + base64.b64encode(f"{cfg.web.user}:{cfg.web.password}".encode()).decode()

    @web.middleware
    async def auth(request: web.Request, handler):
        if not hmac.compare_digest(request.headers.get("Authorization", ""), expected):
            return web.Response(status=401, text="Authentication required",
                                headers={"WWW-Authenticate": 'Basic realm="Yo-gurt"'})
        try:
            return await handler(request)
        except web.HTTPException:
            raise
        except Exception as e:  # show the operator what went wrong instead of a bare 500
            log.warning("%s %s: %s", request.method, request.path, e)
            return web.json_response({"error": str(e)}, status=400)

    routes = web.RouteTableDef()

    def teaching_only() -> None:
        if brain.state != "SAFE_STOP":
            raise RuntimeError("only possible in Safe Stop: press SAFE STOP first")

    # ---- pages and video --------------------------------------------------------------------

    @routes.get("/")
    async def index(request):
        return web.FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})

    @routes.get("/lab")
    async def lab_page(request):
        return web.FileResponse(STATIC / "lab.html", headers={"Cache-Control": "no-cache"})

    @routes.get("/video/{camera}")
    async def video(request):
        name = request.match_info["camera"]
        if name not in ("base", "wrist"):
            raise web.HTTPNotFound()
        response = web.StreamResponse(headers={"Content-Type": "multipart/x-mixed-replace; boundary=frame",
                                               "Cache-Control": "no-cache"})
        await response.prepare(request)

        def render() -> bytes:
            camera, draw = (brain.base, overlay.draw_base) if name == "base" else (brain.wrist, overlay.draw_wrist)
            frame = camera.read()
            img = draw(frame, brain) if frame is not None else overlay.no_signal(name, camera.error)
            return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()

        with contextlib.suppress(ConnectionResetError, asyncio.CancelledError):
            while True:
                started = time.monotonic()
                jpeg = await asyncio.to_thread(render)
                await response.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
                await asyncio.sleep(max(0.0, 1 / 12 - (time.monotonic() - started)))
        return response

    # ---- status -----------------------------------------------------------------------------

    def person(p) -> dict | None:
        if p is None:
            return None
        return {"id": p.id, "bearing": round(p.bearing, 1), "pan": round(brain.pan_for(p.bearing), 1)}

    @routes.get("/api/status")
    async def status(request):
        r = brain.request
        faces = [{"id": f.id, "bearing": round(f.bearing, 1), "pan": round(brain.pan_for(f.bearing), 1),
                  "mouth": round(f.mouth, 3)} for f in brain.faces.visible(1.0)]
        voice = brain.view.get("voice")
        mic = brain.mic
        mic_note = mic.error
        if mic.error is None and time.monotonic() - mic.stamp > 2:
            mic_note = "no audio is arriving"
        elif mic.error is None and mic.level == 0:
            mic_note = "pure silence - check the operating system's microphone permission"
        return web.json_response({
            "state": brain.state, "states": STATES, "detail": brain.detail,
            "seconds_in_state": round(time.monotonic() - brain.entered, 1),
            "events": list(brain.events)[:60],
            "arm": brain.arm.snapshot(),
            "poses": {name: dict(pose) for name, pose in cfg.poses.items()},
            "request": {
                "heard": r.heard, "target_role": r.target_role,
                "recipient_words": r.command.recipient_words if r.command else None,
                "requester": person(r.requester), "others": [person(p) for p in r.others],
                "target": person(r.target), "yogurt_pan": r.yogurt_pan, "attempts": r.attempts,
            },
            "faces": faces,
            "voice_pan": None if voice is None else round(brain.pan_for(voice), 1),
            "view": {k: v for k, v in brain.view.items() if isinstance(v, (int, float))},
            "models": brain.models,
            "hardware": {
                "wrist camera": brain.wrist.error or "ok", "base camera": brain.base.error or "ok",
                "microphone": mic_note or f"ok (level {mic.level:.3f}, room {mic.noise:.3f})",
                "speech-to-text": brain.asr.error or ("ready" if brain.asr.model else "loading…"),
                "language model": brain.slm_status,
                "leader arm": cfg.leader.port or "not configured",
            },
            "machine": cfg.machine, "session": brain.session_status(),
            "capture": capture_status(),
            "check": {"pan_min": cfg.check.pan_min_deg, "pan_max": cfg.check.pan_max_deg},
        })

    # ---- state machine ----------------------------------------------------------------------

    @routes.post("/api/state")
    async def set_state(request):
        brain.go((await request.json())["state"])
        return web.json_response({"ok": True})

    @routes.post("/api/signal")
    async def signal(request):
        body = await request.json()
        brain.signal(body["name"], body.get("value", True))
        return web.json_response({"ok": True})

    @routes.post("/api/reconnect")
    async def reconnect(request):
        await asyncio.to_thread(brain.connect)
        if brain.arm.connected:
            brain.go("SAFE_STOP", "reconnected - the arm is holding where it is")
        return web.json_response({"ok": brain.arm.connected, "error": brain.arm.fault})

    @routes.post("/api/models/reload")
    async def reload_models(request):
        teaching_only()
        await asyncio.to_thread(brain.load_models)
        return web.json_response(brain.models)

    # ---- arm: jog, poses, teaching (Safe Stop only) -----------------------------------------

    @routes.post("/api/arm/torque")
    async def torque(request):
        teaching_only()
        brain.arm.set_torque(bool((await request.json())["on"]))
        return web.json_response({"ok": True})

    @routes.post("/api/arm/jog")
    async def jog(request):
        teaching_only()
        body = await request.json()
        joint = body["joint"]
        brain.arm.set_target({joint: brain.arm.target[joint] + float(body["delta"])})
        return web.json_response({"ok": True})

    @routes.post("/api/pose/goto")
    async def pose_goto(request):
        teaching_only()
        brain.arm.set_target(cfg.poses[(await request.json())["name"]])
        return web.json_response({"ok": True})

    @routes.post("/api/pose/save")
    async def pose_save(request):
        teaching_only()
        name = (await request.json())["name"]
        if name not in cfg.poses:
            raise ValueError(f"unknown pose {name}")
        cfg.poses[name] = config.Cfg({j: round(brain.arm.pos[j], 1) for j in JOINTS})
        config.save_poses(cfg.poses)
        brain.say(f"pose '{name}' re-taught: {dict(cfg.poses[name])}")
        return web.json_response({"ok": True})

    # ---- data collection --------------------------------------------------------------------

    # ---- the Lab: wake-word clips ------------------------------------------------------------

    def served(file_or_bytes, content_type=None):
        if isinstance(file_or_bytes, bytes):
            return web.Response(body=file_or_bytes, content_type=content_type,
                                headers={"Cache-Control": "private, max-age=3600"})
        return web.FileResponse(file_or_bytes, headers={"Cache-Control": "no-cache"})

    async def find(lookup, *args):
        """A file the Lab asked for by name, or 404."""
        try:
            return await asyncio.to_thread(lookup, *args)
        except (LookupError, ValueError):
            raise web.HTTPNotFound() from None

    @routes.get("/api/lab/wake")
    async def lab_wake(request):
        clips = await asyncio.to_thread(collect.clips)
        totals = {label: {"clips": 0, "seconds": 0.0} for label in collect.AUDIO_LABELS}
        for c in clips:
            totals[c["label"]]["clips"] += 1
            totals[c["label"]]["seconds"] += c["seconds"] or 0.0
        return web.json_response({
            "clips": clips, "totals": totals, "prompts": json.loads(collect.PROMPTS.read_text()),
            "targets": {label: collect.TARGETS[f"wake_word/{label}"] for label in collect.AUDIO_LABELS},
            "train": collect.TRAIN["wake_word"], "model": brain.models.get("wake_word"),
        })

    @routes.get("/api/lab/wake/{label}/{name}")
    async def lab_wake_clip(request):
        return served(await find(collect.clip_file, request.match_info["label"], request.match_info["name"]))

    recording_clip = []

    @routes.post("/api/lab/wake/record")
    async def lab_wake_record(request):
        body = await request.json()
        seconds = min(max(float(body["seconds"]), 0.5), 120)
        if body.get("label") not in collect.AUDIO_LABELS:
            raise ValueError(f"label must be one of {collect.AUDIO_LABELS}")
        if brain.mic.error:
            raise RuntimeError(f"microphone: {brain.mic.error}")
        if recording_clip:
            raise RuntimeError("a clip is already being recorded")
        recording_clip.append(True)
        try:
            started = time.monotonic()
            await asyncio.sleep(seconds)
            clip = brain.mic.audio(started, started + seconds)
        finally:
            recording_clip.clear()
        file = await asyncio.to_thread(collect.save_audio_clip, body["label"], clip, body.get("phrase", ""))
        return web.json_response(await asyncio.to_thread(collect.clip_info, file, body["label"]))

    @routes.post("/api/lab/wake/delete")
    async def lab_wake_delete(request):
        clips = (await request.json())["clips"]
        return web.json_response({"deleted": await asyncio.to_thread(collect.delete_clips, clips)})

    @routes.post("/api/lab/wake/move")
    async def lab_wake_move(request):
        body = await request.json()
        return web.json_response({"moved": await asyncio.to_thread(collect.move_clips, body["clips"], body["to"])})

    # ---- the Lab: wrist images -------------------------------------------------------------

    def capture_status() -> dict:
        folder = brain.capture.folder
        if folder is None:
            return {"running": False, "saved": brain.capture.saved}
        return {"running": True, "saved": brain.capture.saved, "dataset": folder.parent.parent.name,
                "label": folder.parent.name, "burst": folder.name}

    def not_capturing(dataset: str, label: str, burst: str) -> None:
        c = capture_status()
        if c["running"] and (c["dataset"], c["label"], c["burst"]) == (dataset, label, burst):
            raise RuntimeError("pictures are still being saved into this burst: press Stop first")

    @routes.get("/api/lab/images")
    async def lab_images(request):
        return web.json_response({
            "sets": await asyncio.to_thread(collect.image_sets), "image_sets": collect.IMAGE_SETS,
            "shots": json.loads(collect.SHOTS.read_text()), "capture": capture_status(), "train": collect.TRAIN,
            "models": {d: brain.models.get(m) for d, m in collect.MODELS.items() if d != "wake_word"},
            "detector": brain.detector is not None,
        })

    @routes.get("/api/lab/images/next-unlabelled")
    async def lab_next_unlabelled(request):
        found = await asyncio.to_thread(collect.next_unlabelled)
        return web.json_response({"burst": found[0], "name": found[1]} if found else {"burst": None, "name": None})

    @routes.get("/api/lab/images/propose/{burst}/{name}")
    async def lab_propose(request):
        """The loaded detector's guess at the cup, drawn for the person to correct."""
        if brain.detector is None:
            return web.json_response({"box": None})
        file = await find(collect.image_file, "yogurt_detection", "images", request.match_info["burst"],
                          request.match_info["name"])
        frame = cv2.imread(str(file))
        found = await asyncio.to_thread(brain.detector.detect, frame, 0.1)
        if not found:
            return web.json_response({"box": None})
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = found[0].box
        return web.json_response({"box": [x1 / w, y1 / h, x2 / w, y2 / h]})

    @routes.get("/api/lab/images/{dataset}/{label}")
    async def lab_bursts(request):
        m = request.match_info
        return web.json_response({"bursts": await asyncio.to_thread(collect.bursts, m["dataset"], m["label"])})

    @routes.get("/api/lab/images/{dataset}/{label}/{burst}")
    async def lab_burst(request):
        m = request.match_info
        images = await asyncio.to_thread(collect.burst_images, m["dataset"], m["label"], m["burst"])
        return web.json_response({"images": images})

    @routes.get("/api/lab/images/{dataset}/{label}/{burst}/{name}")
    async def lab_image(request):
        m = request.match_info
        file = await find(collect.image_file, m["dataset"], m["label"], m["burst"], m["name"])
        if "thumb" in request.query:
            return served(await find(collect.thumbnail, file), "image/jpeg")
        return served(file)

    @routes.post("/api/lab/images/capture")
    async def lab_capture(request):
        body = await request.json()
        if body.get("stop"):
            brain.capture.stop()
        elif brain.capture.folder is not None:
            raise RuntimeError("pictures are already being saved: press Stop first")
        else:
            brain.capture.start(body["dataset"], body["label"], min(max(float(body.get("per_second", 2)), 0.2), 10))
        return web.json_response(capture_status())

    @routes.post("/api/lab/images/delete")
    async def lab_images_delete(request):
        """Delete some pictures of a burst (`names`), or the whole burst (no `names`)."""
        b = await request.json()
        not_capturing(b["dataset"], b["label"], b["burst"])
        if b.get("names") is None:
            gone = await asyncio.to_thread(collect.delete_burst, b["dataset"], b["label"], b["burst"])
            brain.say(f"Lab: deleted burst {b['dataset']}/{b['label']}/{b['burst']} ({gone} pictures)")
            return web.json_response({"deleted": gone, "left": 0})
        left = await asyncio.to_thread(collect.delete_images, b["dataset"], b["label"], b["burst"], b["names"])
        return web.json_response({"deleted": len(b["names"]), "left": left})

    @routes.post("/api/lab/images/move")
    async def lab_images_move(request):
        b = await request.json()
        not_capturing(b["dataset"], b["label"], b["burst"])
        name = await asyncio.to_thread(collect.move_burst, b["dataset"], b["label"], b["burst"], b["to"])
        brain.say(f"Lab: burst {b['burst']} relabelled {b['dataset']}/{b['label']} -> {b['to']}")
        return web.json_response({"burst": name})

    @routes.post("/api/lab/images/box")
    async def lab_box(request):
        b = await request.json()
        await asyncio.to_thread(collect.save_box, b["burst"], b["name"], b.get("box"))
        return web.json_response({"ok": True})

    # ---- the Lab: grasp demonstration datasets ---------------------------------------------

    def recording() -> str | None:
        """The dataset a session is writing into: not to be read or changed until the session ends."""
        return brain.session.settings.get("dataset") if brain.session.busy else None

    def not_recording(name: str) -> None:
        if name == recording():
            raise RuntimeError(f"{name} is being recorded into: stop the recording session first")

    @routes.get("/api/lab/datasets")
    async def lab_datasets(request):
        return web.json_response({
            "datasets": await asyncio.to_thread(lab.list_datasets, recording()),
            "defaults": {"task": cfg.grasp.task, "fps": cfg.grasp.fps,
                         "image": [cfg.grasp.image_width, cfg.grasp.image_height],
                         "dart_percent": round(cfg.dart.fraction * 100)},
            "name_rule": lab.NAME_RULE, "leader": cfg.leader.port, "detector": brain.detector is not None,
        })

    @routes.get("/api/lab/datasets/{name}")
    async def lab_dataset(request):
        name = request.match_info["name"]
        not_recording(name)
        return web.json_response(await asyncio.to_thread(lab.describe, name))

    @routes.get("/api/lab/datasets/{name}/episodes/{index:\\d+}/joints")
    async def lab_joints(request):
        name = request.match_info["name"]
        not_recording(name)
        return web.json_response(await asyncio.to_thread(lab.joints, name, int(request.match_info["index"])))

    @routes.get("/api/lab/datasets/{name}/videos/{camera}/{chunk:\\d+}/{file:\\d+}")
    async def lab_video(request):
        m = request.match_info
        not_recording(m["name"])
        try:
            video = await asyncio.to_thread(lab.video_file, m["name"], m["camera"], int(m["chunk"]), int(m["file"]))
        except (LookupError, ValueError):
            raise web.HTTPNotFound() from None
        # Deleting episodes rewrites a file under the same address: always check with the server.
        return web.FileResponse(video, headers={"Cache-Control": "no-cache"})

    async def rewrite(name: str, change, *args):
        """Run one dataset change at a time, never on the dataset a session is recording into."""
        lab.check_name(name)
        if brain.session.busy:
            raise RuntimeError("a recording session is running: stop it before changing or uploading a dataset")
        if not lab.LOCK.acquire(blocking=False):
            raise RuntimeError("another dataset change is still in progress")
        try:
            return await asyncio.to_thread(change, name, *args)
        finally:
            lab.LOCK.release()

    @routes.post("/api/lab/datasets/{name}/delete-episodes")
    async def lab_delete_episodes(request):
        name, episodes = request.match_info["name"], (await request.json()).get("episodes")
        if not isinstance(episodes, list):
            raise ValueError("episodes must be a list of episode numbers")
        left = await rewrite(name, lab.delete_episodes, episodes)
        brain.say(f"Lab: deleted episode(s) {sorted(set(episodes))} from {name}; {left} left")
        return web.json_response({"ok": True, "remaining": left})

    @routes.post("/api/lab/datasets/{name}/episodes/{index:\\d+}/cleaning")
    async def lab_cleaning(request):
        """Disable/enable an episode or set its trim. Stored beside the recording, never applied to it."""
        name, index, body = request.match_info["name"], int(request.match_info["index"]), await request.json()
        not_recording(name)
        args = {"disabled": body.get("disabled")}
        if "trim" in body:
            args["trim"] = body["trim"]
        if not lab.LOCK.acquire(blocking=False):  # never block the server behind a long export
            raise RuntimeError("a dataset change is in progress: try again when it has finished")
        try:
            edit = await asyncio.to_thread(lab.set_cleaning, name, index, **args)
        finally:
            lab.LOCK.release()
        return web.json_response({"ok": True, "episode": index, **edit})

    @routes.post("/api/lab/datasets/{name}/export-clean")
    async def lab_export_clean(request):
        name = request.match_info["name"]
        result = await rewrite(name, lab.export_clean)
        brain.say(f"Lab: wrote {result['name']} from {name} ({result['episodes']} episodes, {result['frames']} frames)")
        return web.json_response({"ok": True, **result})

    @routes.get("/api/lab/hub")
    async def lab_hub(request):
        return web.json_response(await asyncio.to_thread(lab.hub_status))

    @routes.post("/api/lab/hub/login")
    async def lab_hub_login(request):
        return web.json_response(await asyncio.to_thread(lab.hub_login, (await request.json()).get("token")))

    @routes.post("/api/lab/datasets/{name}/upload")
    async def lab_upload(request):
        name, body = request.match_info["name"], await request.json()
        result = await rewrite(name, lab.upload, body.get("repo"), body.get("private", True))
        brain.say(f"Lab: uploaded {name} to {result['url']}")
        return web.json_response({"ok": True, **result})

    @routes.post("/api/lab/datasets/{name}/delete")
    async def lab_delete_dataset(request):
        name = request.match_info["name"]
        if (await request.json()).get("confirm") != name:
            raise ValueError("type the dataset's name to confirm")
        await rewrite(name, lab.delete_dataset)
        brain.say(f"Lab: deleted dataset {name}")
        return web.json_response({"ok": True})

    @routes.post("/api/lab/record")
    async def lab_record(request):
        await brain.start_session(await request.json())
        return web.json_response({"ok": True})

    @routes.post("/api/lab/session")
    async def lab_session(request):
        """The recording screen's buttons. `phase` is the phase the operator saw: a press that arrives after
        the session has moved on (a double click) is refused instead of acting on the next phase."""
        body = await request.json()
        action = body.get("action")
        if action == "stop":
            await brain.stop_session()
            return web.json_response({"ok": True})
        status = brain.session_status()
        if not status["open"]:
            raise RuntimeError("no recording session is running")
        phase = status["phase"]
        if body.get("phase") != phase:
            raise RuntimeError(f"the session has moved on ({phase}): nothing was done")
        if action == "next" and phase in ("positioning", "recording", "resetting"):
            brain.signal({"positioning": "found", "recording": "end_recording", "resetting": "end_reset"}[phase])
        elif action == "rerecord" and phase == "recording":
            brain.signal("rerecord")
        elif action == "resume" and phase == "paused":
            brain.go("CHECK", "recording resumed")
        else:
            raise RuntimeError(f"'{action}' is not possible while {phase}")
        return web.json_response({"ok": True})

    # ---- lifecycle --------------------------------------------------------------------------

    tasks = []

    async def start(app):
        await asyncio.to_thread(brain.connect)
        tasks.append(asyncio.ensure_future(brain.run()))

    async def stop(app):
        for task in tasks:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        await asyncio.to_thread(brain.disconnect, "program stopped")

    app = web.Application(middlewares=[auth])
    app.add_routes(routes)
    app.on_startup.append(start)
    app.on_cleanup.append(stop)
    return app


def addresses(host: str, port: int) -> list[str]:
    """The addresses to type into a browser. 0.0.0.0 means "every network interface", which is
    not itself an address a browser can open, so list this machine's name and LAN address instead."""
    if host not in ("0.0.0.0", "::", ""):
        return [f"http://{host}:{port}"]
    urls = [f"http://127.0.0.1:{port}   (this machine)"]
    name = socket.gethostname()
    urls.append(f"http://{name if name.endswith('.local') else name + '.local'}:{port}   (other machines, by name)")
    with contextlib.suppress(OSError), socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("10.255.255.255", 1))  # UDP connect sends nothing; it only picks the outgoing interface
        urls.append(f"http://{s.getsockname()[0]}:{port}   (other machines, by IP)")
    return urls


def main() -> None:
    parser = argparse.ArgumentParser(description="Yo-gurt robot interface")
    parser.add_argument("machine", nargs="?", default="laptop", help="which config/hardware_<machine>.yaml to use")
    parser.add_argument("--sim", action="store_true", help="simulate the arm (no servos needed)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "huggingface_hub", "faster_whisper"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    cfg = config.load(args.machine)
    if args.sim:
        cfg.arm["port"] = "sim"
    brain = Brain(cfg, Arm(cfg), Mic(cfg), Camera("wrist", cfg.cameras.wrist), Camera("base", cfg.cameras.base))
    print(f"\n  Yo-gurt ({args.machine})   user {cfg.web.user} / password {cfg.web.password}")
    for url in addresses(cfg.web.host, cfg.web.port):
        print(f"    {url}")
    print()
    web.run_app(make_app(cfg, brain), host=cfg.web.host, port=cfg.web.port, print=None)


if __name__ == "__main__":
    main()
