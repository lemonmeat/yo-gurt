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
import time
from pathlib import Path

import cv2
from aiohttp import web

from . import collect, config, overlay
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

    @routes.get("/video/{camera}")
    async def video(request):
        name = request.match_info["camera"]
        camera, draw = {"base": (brain.base, overlay.draw_base), "wrist": (brain.wrist, overlay.draw_wrist)}[name]
        response = web.StreamResponse(headers={"Content-Type": "multipart/x-mixed-replace; boundary=frame",
                                               "Cache-Control": "no-cache"})
        await response.prepare(request)

        def render() -> bytes:
            frame = camera.read()
            img = draw(frame, brain) if frame is not None else overlay.no_signal(name, camera.error)
            return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tobytes()

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
            "machine": cfg.machine, "demo_mode": brain.demo_mode,
            "capture": {"running": brain.capture.folder is not None, "saved": brain.capture.saved},
            "check": {"pan_min": cfg.check.pan_min_deg, "pan_max": cfg.check.pan_max_deg},
        })

    @routes.get("/api/data")
    async def data_counts(request):
        prompts = json.loads((collect.DATA / "wake_word" / "prompts.json").read_text())
        shots = json.loads((collect.DATA / "shots.json").read_text())
        return web.json_response({"counts": await asyncio.to_thread(collect.counts), "prompts": prompts,
                                  "shots": shots, "image_sets": collect.IMAGE_SETS})

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

    @routes.post("/api/collect/audio")
    async def collect_audio(request):
        body = await request.json()
        seconds = min(max(float(body["seconds"]), 0.5), 120)
        if brain.mic.error:
            raise RuntimeError(f"microphone: {brain.mic.error}")
        started = time.monotonic()
        await asyncio.sleep(seconds)
        clip = brain.mic.audio(started, started + seconds)
        file = collect.save_audio_clip(body["label"], clip, body.get("phrase", ""))
        return web.json_response({"file": file.name, "peak": round(float(abs(clip).max()), 3) if len(clip) else 0})

    @routes.post("/api/collect/images")
    async def collect_images(request):
        body = await request.json()
        if body.get("stop"):
            return web.json_response({"saved": brain.capture.stop()})
        brain.capture.start(body["dataset"], body["label"], min(max(float(body.get("per_second", 2)), 0.2), 10))
        return web.json_response({"ok": True})

    @routes.get("/api/label/next")
    async def label_next(request):
        image = await asyncio.to_thread(collect.next_unlabelled)
        if image is None:
            return web.json_response({"image": None})
        proposal = None
        if brain.detector is not None:  # suggest a box; the person corrects it
            frame = cv2.imread(str(image))
            found = await asyncio.to_thread(brain.detector.detect, frame, 0.1)
            if found:
                h, w = frame.shape[:2]
                x1, y1, x2, y2 = found[0].box
                proposal = [x1 / w, y1 / h, x2 / w, y2 / h]
        name = str(image.relative_to(collect.DATA / "yogurt_detection" / "images"))
        return web.json_response({"image": name, "proposal": proposal})

    @routes.get("/api/label/image/{name:.*}")
    async def label_image(request):
        root = (collect.DATA / "yogurt_detection" / "images").resolve()
        file = (root / request.match_info["name"]).resolve()
        if not file.is_relative_to(root) or not file.is_file():
            raise web.HTTPNotFound()
        return web.FileResponse(file)

    @routes.post("/api/label")
    async def label_save(request):
        body = await request.json()
        collect.save_box(body["image"], body.get("box"))
        return web.json_response({"ok": True})

    @routes.post("/api/demos/start")
    async def demos_start(request):
        brain.start_demos()
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
    print(f"\n  Yo-gurt ({args.machine}): http://{cfg.web.host}:{cfg.web.port}"
          f"   user {cfg.web.user} / password {cfg.web.password}\n")
    web.run_app(make_app(cfg, brain), host=cfg.web.host, port=cfg.web.port, print=None)


if __name__ == "__main__":
    main()
