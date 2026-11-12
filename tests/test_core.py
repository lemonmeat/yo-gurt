"""The few tests that guard what must not break: servo safety, direction of sound, the state flow, the login."""

import asyncio
import base64
from types import SimpleNamespace

import numpy as np
import pytest
from aiohttp.test_utils import TestClient, TestServer

from yogurt import config, fsm
from yogurt.app import make_app
from yogurt.arm import Arm, joint_limits
from yogurt.audio import Mic, delay_to_bearing, stereo_delay
from yogurt.parse import Command


class NoCamera:
    frame, stamp, error = None, 0.0, "no camera in tests"

    def open(self): ...
    def close(self): ...
    def read(self, max_age_s=1.0): ...


def make_brain(monkeypatch) -> fsm.Brain:
    cfg = config.load("laptop")
    cfg.arm["port"] = "sim"
    cfg.arm.update(speed_dps=2000, max_speed_dps=2000, gripper_speed=2000, max_lead_deg=400)  # a fast simulated arm
    # no trained models: the operator's buttons stand in for them
    cfg.models.update({name: "models/none" for name in cfg.models if name != "act_device"})
    monkeypatch.setattr(fsm, "parse_command", lambda text, url, model, **kw: Command(text, True, "speaker", "me"))
    monkeypatch.setattr(fsm.audio, "Transcriber", lambda cfg: SimpleNamespace(model=None, error=None))
    mic = Mic(cfg)
    monkeypatch.setattr(mic, "open", lambda: None)
    brain = fsm.Brain(cfg, Arm(cfg), mic, NoCamera(), NoCamera())
    brain.connect()
    return brain


async def reach(brain: fsm.Brain, state: str) -> None:
    for _ in range(400):
        if brain.state == state:
            return
        await asyncio.sleep(0.02)
    raise AssertionError(f"stuck in {brain.state} ({brain.detail}), expected {state}")


def test_joint_limits_come_from_the_calibration():
    calibration = {j: {"range_min": 916, "range_max": 3558} for j in ("shoulder_pan", "shoulder_lift", "elbow_flex",
                                                                    "wrist_flex", "wrist_roll")}
    low, high = joint_limits(calibration, margin_deg=3)["shoulder_pan"]
    assert high == pytest.approx(116.13 - 3, abs=0.01) and low == -high


def test_arm_never_jumps_overshoots_or_pushes_against_an_obstacle():
    cfg = config.load("laptop")
    cfg.arm["port"] = "sim"
    arm = Arm(cfg)
    arm.torque = True
    arm.set_target({"shoulder_pan": 500.0, "gripper": 150.0}, speed=9999)  # absurd request
    assert arm.target["shoulder_pan"] == arm.limits["shoulder_pan"][1] and arm.target["gripper"] == 100.0
    assert arm.speed == cfg.arm.max_speed_dps
    arm._step(1 / 30)
    assert arm.cmd["shoulder_pan"] == pytest.approx(cfg.arm.max_speed_dps / 30)  # one tick of travel, not a jump
    for _ in range(300):  # the arm is blocked: its measured position never changes
        arm._step(1 / 30)
    assert arm.cmd["shoulder_pan"] == pytest.approx(arm.pos["shoulder_pan"] + cfg.arm.max_lead_deg)
    # A joint found resting beyond the soft range (on its end stop) is held there, then only moves inwards.
    arm.pos["wrist_flex"] = arm.cmd["wrist_flex"] = arm.target["wrist_flex"] = 102.5
    arm._step(1 / 30)
    assert arm.cmd["wrist_flex"] == 102.5
    arm.set_target({"wrist_flex": 150.0})
    arm._step(1 / 30)
    assert arm.cmd["wrist_flex"] < 102.5


def test_voice_direction_from_the_two_microphones():
    sound = np.random.default_rng(0).standard_normal(4000)
    right, left_later = sound[100:2980], sound[97:2977]  # reaches the left microphone 3 samples later
    delay = stereo_delay(left_later, right, 48000, 0.06 / 343)
    assert delay * 48000 == pytest.approx(3, abs=0.2)
    assert delay_to_bearing(delay, 0.06) == pytest.approx(20.9, abs=1.5)        # so the voice is on the right
    assert delay_to_bearing(stereo_delay(right, left_later, 48000, 0.06 / 343), 0.06) < -15


def test_full_request_then_safe_stop_and_kill(monkeypatch):
    async def scenario():
        brain = make_brain(monkeypatch)
        runner = asyncio.ensure_future(brain.run())
        brain.go("IDLE")
        await reach(brain, "IDLE")
        for signal, value, state in [("command", "give me the yogurt", "CHECK"), ("found", True, "GRASP"),
                                     ("grasped", True, "VERIFY"), ("held", False, "CHECK"), ("found", True, "GRASP"),
                                     ("grasped", True, "VERIFY"), ("held", True, "HANDOFF")]:
            await asyncio.sleep(0.3)
            brain.signal(signal, value)
            await reach(brain, state)
        assert brain.request.target_role == "requester" and brain.request.attempts == 2
        await asyncio.sleep(0.5)
        brain.signal("grabbed")
        await reach(brain, "IDLE")

        brain.arm.set_target({"shoulder_pan": 60.0}, speed=5)   # a slow move in progress...
        brain.go("SAFE_STOP")                                    # ...is frozen at once, servos still on
        assert brain.arm.target == brain.arm.cmd and brain.arm.torque
        await reach(brain, "SAFE_STOP")
        brain.go("KILLED")
        assert not brain.arm.connected and not brain.arm.torque
        await reach(brain, "KILLED")
        with pytest.raises(Exception, match="Reconnect"):
            brain.go("IDLE")
        runner.cancel()

    asyncio.run(scenario())


def test_web_interface_requires_the_password(monkeypatch):
    async def scenario():
        brain = make_brain(monkeypatch)
        async with TestClient(TestServer(make_app(brain.cfg, brain))) as client:
            assert (await client.get("/api/status")).status == 401
            wrong = {"Authorization": "Basic " + base64.b64encode(b"gurt:nope").decode()}
            assert (await client.get("/api/status", headers=wrong)).status == 401
            right = {"Authorization": "Basic " + base64.b64encode(b"gurt:gurt").decode()}
            status = await client.get("/api/status", headers=right)
            assert status.status == 200 and (await status.json())["state"] == "SAFE_STOP"

    asyncio.run(scenario())
