"""arm_driver with the MuJoCo backend: in-process loop tests and a subprocess smoke test."""

import os
import subprocess
import sys
import time

import pytest
import yaml
import zmq

from tests.helpers import free_bus_config, wait_for
from yogurt.arm_driver.driver import Driver, make_filter
from yogurt.bus import Bus
from yogurt.config import DEFAULT_CONFIG_DIR, load_config
from yogurt.devicelock import DeviceBusy, DeviceLock
from yogurt.messages import JointTarget, Safety, Stop

CFG = load_config()
pytestmark = [
    pytest.mark.sim,
    pytest.mark.skipif(
        not CFG.path(CFG.follower.sim_model).exists(),
        reason="run scripts/fetch_so_arm100.sh for the SO-101 model",
    ),
]


def test_sim_backend_tracks_commands():
    from yogurt.arm_driver.sim import SimBackend

    sim = SimBackend(CFG.path(CFG.follower.sim_model), rate_hz=50)
    sim.connect()
    start = sim.read()
    assert start.positions["elbow_flex"] == pytest.approx(0.0, abs=0.5)
    assert start.positions["gripper"] == pytest.approx(50.0, abs=1.0)
    cmd = start.positions | {"elbow_flex": 30.0, "gripper": 80.0}
    for _ in range(100):  # 2 s of sim time
        sim.write(cmd)
    end = sim.read()
    assert end.positions["elbow_flex"] == pytest.approx(30.0, abs=1.0)
    assert end.positions["gripper"] == pytest.approx(80.0, abs=2.0)
    assert set(end.loads) == set(cmd)


def _run_filter_on_sim(pose: dict, seconds: float):
    from yogurt.arm_driver.sim import SimBackend

    sim = SimBackend(CFG.path(CFG.follower.sim_model), rate_hz=50)
    sim.connect()
    safety, t, out, reading = make_filter(CFG.follower, CFG), 0.0, None, None
    for _ in range(int(seconds * 50)):
        t += 0.02
        safety.set_target(JointTarget(positions=pose, gripper=50.0, ts=t), t)
        reading = sim.read()
        out = safety.step(reading.positions, reading.loads, t)
        sim.write(out.command)
    return out, reading


def test_large_free_move_does_not_trip_overload():
    """Regression: sim loads once read 100% during any motion and faulted on big leader moves."""
    rest = {"shoulder_lift": -90.0, "elbow_flex": 90.0, "wrist_flex": 60.0}
    out, reading = _run_filter_on_sim(rest, 4.0)
    assert out.status == Safety.OK
    assert all(abs(reading.positions[j] - v) < 1.0 for j, v in rest.items())
    assert max(abs(v) for v in reading.loads.values()) < 50  # gravity only


def test_pushing_into_the_floor_still_faults():
    out, _ = _run_filter_on_sim(
        {"shoulder_lift": 60.0, "elbow_flex": 30.0, "wrist_flex": 80.0}, 4.0
    )
    assert out.status == Safety.FAULT
    assert "overload" in out.detail


def test_driver_loop_in_process(broker):
    from yogurt.arm_driver.sim import SimBackend

    ctx = zmq.Context()
    sim = SimBackend(CFG.path(CFG.follower.sim_model), rate_hz=50)
    sim.connect()
    driver = Driver(sim, make_filter(CFG.follower, CFG), Bus(broker, "arm_driver", ctx))
    with Bus(broker, "test", ctx) as client:
        client.subscribe("arm/state")
        # Until a target arrives the driver holds.
        wait_for(client, "arm/state", lambda: driver.tick(time.monotonic()))
        state = driver.tick(time.monotonic())
        assert state.safety == Safety.WATCHDOG

        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            client.publish("arm/target", JointTarget(positions={"elbow_flex": 20.0}))
            state = driver.tick(time.monotonic())
            time.sleep(0.02)
            if state and abs(state.positions["elbow_flex"] - 20.0) < 1.0:
                break
        assert state.safety == Safety.OK
        assert state.positions["elbow_flex"] == pytest.approx(20.0, abs=1.0)

        def stop_and_tick():
            client.publish("system/stop", Stop(reason="test"))
            time.sleep(0.01)
            return driver.tick(time.monotonic())

        state = None
        for _ in range(100):
            state = stop_and_tick()
            if state and state.safety == Safety.STOPPED:
                break
        assert state.safety == Safety.STOPPED
        assert "test" in state.detail
    driver.bus.close()
    ctx.term()


def test_device_lock_is_exclusive(tmp_path):
    with DeviceLock("/dev/tty.yogurt-test"), pytest.raises(DeviceBusy, match="pid"):
        DeviceLock("/dev/tty.yogurt-test").acquire()
    DeviceLock("/dev/tty.yogurt-test").acquire().release()


@pytest.fixture
def smoke_config(tmp_path):
    """The laptop config with free bus ports and absolute model paths."""
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    for name in ("joint_limits.yaml", "gripper_map.yaml"):
        (cfg_dir / name).write_text((DEFAULT_CONFIG_DIR / name).read_text())
    data = yaml.safe_load((DEFAULT_CONFIG_DIR / "hardware.laptop.yaml").read_text())
    bus = free_bus_config()
    data["bus"] = {"pub": bus.pub, "sub": bus.sub}
    data["follower"]["sim_model"] = str(CFG.path(CFG.follower.sim_model))
    data["follower"]["fk"]["urdf"] = str(CFG.path(CFG.follower.fk.urdf))
    path = cfg_dir / "hardware.smoke.yaml"
    path.write_text(yaml.safe_dump(data))
    return path, bus


def test_smoke_arm_driver_subprocess(smoke_config):
    """Boot broker + `arm_driver --backend sim`, send targets, and check arm/state."""
    path, bus_cfg = smoke_config
    env = os.environ | {"PYTHONUNBUFFERED": "1"}
    procs = [
        subprocess.Popen([sys.executable, "-m", module, "--config", str(path), *extra], env=env)
        for module, extra in [("yogurt.broker", []), ("yogurt.arm_driver", ["--backend", "sim"])]
    ]
    ctx = zmq.Context()
    try:
        with Bus(bus_cfg, "smoke", ctx) as client:
            client.subscribe("arm/state")
            first = wait_for(client, "arm/state", timeout=15.0)
            assert first.backend == "sim"
            assert first.source == "arm_driver"

            goal = {"shoulder_pan": 15.0, "elbow_flex": -20.0}
            state = wait_for(
                client,
                "arm/state",
                lambda: client.publish("arm/target", JointTarget(positions=goal, gripper=70.0)),
                timeout=5.0,
                pred=lambda s: (
                    all(abs(s.positions[j] - v) < 1.5 for j, v in goal.items())
                    and abs(s.gripper - 70.0) < 3.0
                ),
            )
            assert state.safety == Safety.OK

            # Stop sending: the watchdog holds the pose.
            held = wait_for(
                client, "arm/state", timeout=2.0, pred=lambda s: s.safety == Safety.WATCHDOG
            )
            assert held.positions["elbow_flex"] == pytest.approx(-20.0, abs=1.5)

            # system/stop latches.
            wait_for(
                client,
                "arm/state",
                lambda: client.publish("system/stop", Stop(reason="smoke")),
                pred=lambda s: s.safety == Safety.STOPPED,
            )
            for p in procs:
                assert p.poll() is None, "a process died"
    finally:
        for p in reversed(procs):
            p.terminate()
        for p in procs:
            assert p.wait(timeout=5) == 0
        ctx.term()
