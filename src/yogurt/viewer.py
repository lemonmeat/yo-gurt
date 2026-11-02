"""viewer: live 3D view of the follower, for testing.

    uv run mjpython -m yogurt.viewer [--config PATH]    (macOS: MuJoCo's viewer needs mjpython)
    uv run python -m yogurt.viewer [--config PATH]      (Linux)

Read-only: subscribes arm/state and poses the SO-101 model at the measured joint positions (no
physics runs here). It also draws:
  - the config's keep-out boxes in red, with a faint shell for the margin (brighter while the
    safety filter is holding at that box),
  - a sphere at the commanded fingertip, colored by safety status: green ok, yellow watchdog,
    orange keep-out, red stopped or fault, grey when arm/state is stale.
It never publishes, so it works the same with --backend sim and real, and closing it changes
nothing.
"""

import logging
import time
from collections.abc import Sequence

import mujoco
import mujoco.viewer
import numpy as np

from yogurt import cli
from yogurt.arm_driver.sim import JointMap, load_model
from yogurt.bus import Bus
from yogurt.config import Box
from yogurt.kinematics import Chain
from yogurt.messages import GRIPPER, JointState, Safety

log = logging.getLogger("yogurt.viewer")

NAME = "viewer"
FPS = 60.0
STALE_S = 0.5
TIP_RADIUS_M = 0.012
STATUS_RGBA = {
    Safety.OK: (0.1, 0.85, 0.25, 0.9),
    Safety.WATCHDOG: (1.0, 0.85, 0.1, 0.9),
    Safety.KEEP_OUT: (1.0, 0.5, 0.0, 0.9),
    Safety.STOPPED: (0.95, 0.1, 0.1, 0.9),
    Safety.FAULT: (0.95, 0.1, 0.1, 0.9),
}
STALE_RGBA = (0.5, 0.5, 0.5, 0.6)

GeomSpec = tuple[int, Sequence[float], Sequence[float], Sequence[float]]  # type, size, pos, rgba


def overlay(
    boxes: list[Box],
    margin: float,
    tip: np.ndarray | None,
    status: str | None,
    active_zone: str = "",
) -> list[GeomSpec]:
    """Decorations for one frame: keep-out boxes, their margins, and the commanded-tip marker."""
    specs: list[GeomSpec] = []
    for b in boxes:
        lo, hi = np.array(b.min), np.array(b.max)
        center, half = (lo + hi) / 2, (hi - lo) / 2
        alpha = 0.5 if b.name == active_zone else 0.22
        specs.append((mujoco.mjtGeom.mjGEOM_BOX, half, center, (0.9, 0.1, 0.1, alpha)))
        specs.append((mujoco.mjtGeom.mjGEOM_BOX, half + margin, center, (0.9, 0.1, 0.1, 0.07)))
    if tip is not None:
        rgba = STATUS_RGBA.get(status, STALE_RGBA) if status else STALE_RGBA
        specs.append((mujoco.mjtGeom.mjGEOM_SPHERE, (TIP_RADIUS_M, 0, 0), tip, rgba))
    return specs


def draw(scene: mujoco.MjvScene, specs: list[GeomSpec]) -> None:
    scene.ngeom = 0
    for kind, size, pos, rgba in specs[: scene.maxgeom]:
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            kind,
            np.asarray(size, float),
            np.asarray(pos, float),
            np.eye(3).flatten(),
            np.asarray(rgba, np.float32),
        )
        scene.ngeom += 1


def main(argv: list[str] | None = None) -> None:
    _, cfg = cli.start(NAME, __doc__, argv)
    arm = cfg.follower
    try:
        model = load_model(cfg.path(arm.sim_model))
    except FileNotFoundError as e:
        log.error("%s", e)
        raise SystemExit(2) from e
    data = mujoco.MjData(model)
    joints = JointMap(model, arm.fk)
    chain = None
    if arm.fk is not None and cfg.path(arm.fk.urdf).exists():
        chain = Chain.from_config(arm.fk, cfg.path(arm.fk.urdf))
    boxes, margin = arm.safety.keep_out, arm.safety.keep_out_margin_m

    try:
        viewer = mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False)
    except RuntimeError as e:
        log.error("%s\nOn macOS, start it with: uv run mjpython -m yogurt.viewer", e)
        raise SystemExit(2) from e

    bus = Bus(cfg.bus, NAME)
    bus.subscribe("arm/state")
    state: JointState | None = None
    shown = None
    log.info("waiting for arm/state; keep-out boxes: %s", [b.name for b in boxes] or "none")
    try:
        with viewer:
            with viewer.lock():
                viewer.cam.lookat[:] = (0.15, 0.0, 0.12)
                viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 0.95, 145, -25
            while viewer.is_running():
                for _, msg in bus.drain():
                    state = msg
                fresh = state is not None and time.monotonic() - state.ts < STALE_S
                status = state.safety if fresh else None
                if status != shown:
                    log.info(
                        "arm %s %s", status or "no recent arm/state", state.detail if fresh else ""
                    )
                    shown = status
                with viewer.lock():
                    tip = None
                    if state is not None:
                        joints.set_qpos(data, state.positions | {GRIPPER: state.gripper})
                        mujoco.mj_forward(model, data)
                        if chain is not None and state.commanded:
                            tip = chain.tip(state.commanded)
                    zone = state.detail.split(":")[0] if fresh and status == Safety.KEEP_OUT else ""
                    draw(viewer.user_scn, overlay(boxes, margin, tip, status, zone))
                viewer.sync()
                time.sleep(1.0 / FPS)
    except KeyboardInterrupt:
        pass
    finally:
        bus.close()


if __name__ == "__main__":
    main()
