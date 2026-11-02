import math

import pytest

from yogurt.config import JointLimit, SafetyConfig
from yogurt.messages import ALL_JOINTS, JointTarget, Safety
from yogurt.safety import SafetyFilter

LIMITS = {name: JointLimit(min=-90.0, max=90.0, max_velocity=100.0) for name in ALL_JOINTS}
LIMITS["gripper"] = JointLimit(min=0.0, max=100.0, max_velocity=100.0)
HOME = {name: 0.0 for name in ALL_JOINTS} | {"gripper": 50.0}
DT = 0.02


def make(keep_out=None, **safety) -> SafetyFilter:
    cfg = SafetyConfig(
        **({"max_tracking_error_deg": 1000.0, "max_gripper_tracking_error": 1000.0} | safety)
    )
    return SafetyFilter(LIMITS, cfg, keep_out)


def target(t: float, **positions) -> JointTarget:
    gripper = positions.pop("gripper", None)
    return JointTarget(positions=positions, gripper=gripper, ts=t)


def run(f: SafetyFilter, t0: float, ticks: int, measured=None, loads=None, retarget=None):
    """Step `ticks` times from t0, feeding back the command as the measured pose."""
    out, t = None, t0
    meas = dict(measured or HOME)
    for _ in range(ticks):
        t += DT
        if retarget is not None:
            assert f.set_target(retarget(t), t) is None
        out = f.step(meas if measured else (out.command if out else meas), loads or {}, t)
    return out, t


def test_first_tick_holds_measured_pose():
    f = make()
    out = f.step(HOME, {}, 0.0)
    assert out.command == HOME
    assert out.status == Safety.WATCHDOG


def test_target_is_clamped_to_joint_limits():
    f = make()
    f.step(HOME, {}, 0.0)
    out, _ = run(f, 0.0, 200, retarget=lambda t: target(t, elbow_flex=500.0, gripper=-20.0))
    assert out.command["elbow_flex"] == 90.0
    assert out.command["gripper"] == 0.0


def test_velocity_cap():
    f = make()
    f.step(HOME, {}, 0.0)
    out, _ = run(f, 0.0, 1, retarget=lambda t: target(t, shoulder_pan=80.0))
    assert out.command["shoulder_pan"] == pytest.approx(100.0 * DT)
    out, _ = run(f, DT, 9, retarget=lambda t: target(t, shoulder_pan=80.0))
    assert out.command["shoulder_pan"] == pytest.approx(100.0 * DT * 10)


def test_long_gap_between_ticks_does_not_jump():
    f = make()
    f.step(HOME, {}, 0.0)
    f.set_target(target(5.0, shoulder_pan=80.0), 5.0)
    out = f.step(HOME, {}, 5.0)
    assert out.command["shoulder_pan"] <= 100.0 * 0.1 + 1e-9


def test_partial_target_keeps_other_joints():
    f = make()
    f.step(HOME, {}, 0.0)
    out, _ = run(f, 0.0, 100, retarget=lambda t: target(t, wrist_flex=30.0))
    assert out.command["wrist_flex"] == pytest.approx(30.0)
    assert out.command["elbow_flex"] == 0.0
    assert out.command["gripper"] == 50.0


def test_watchdog_holds_last_command():
    f = make(watchdog_s=0.2)
    f.step(HOME, {}, 0.0)
    f.set_target(target(0.0, elbow_flex=80.0), 0.0)
    moving = f.step(HOME, {}, 0.1)
    assert moving.status == Safety.OK
    held = f.step(moving.command, {}, 0.35)
    assert held.status == Safety.WATCHDOG
    assert held.command == moving.command
    # A fresh target resumes motion.
    f.set_target(target(0.4, elbow_flex=80.0), 0.4)
    assert f.step(held.command, {}, 0.42).status == Safety.OK


def test_stale_and_bad_targets_are_rejected():
    f = make(watchdog_s=0.2)
    assert "stale" in f.set_target(target(0.0, elbow_flex=10.0), 1.0)
    assert "non-finite" in f.set_target(target(1.0, elbow_flex=math.nan), 1.0)
    assert "unknown" in f.set_target(JointTarget(positions={"elbo": 1.0}, ts=1.0), 1.0)


def test_stop_latches_until_resume():
    f = make()
    f.step(HOME, {}, 0.0)
    out, t = run(f, 0.0, 5, retarget=lambda t: target(t, elbow_flex=80.0))
    f.stop("voice")
    held = f.step(out.command, {}, t + DT)
    assert held.status == Safety.STOPPED
    assert held.command == out.command
    assert "latched" in f.set_target(target(t + DT, elbow_flex=80.0), t + DT)

    f.resume()
    # After resume the arm waits for a fresh target instead of finishing the old one.
    after = f.step(held.command, {}, t + 2 * DT)
    assert after.status == Safety.WATCHDOG
    assert after.command == held.command


def test_repeated_stop_keeps_first_reason_and_fault():
    f = make()
    f.step(HOME, {}, 0.0)
    f.stop("voice")
    f.stop("repeat")
    assert f.step(HOME, {}, DT).detail == "voice"
    f.resume()
    f.fault("overload")
    f.stop("repeat")
    out = f.step(HOME, {}, 2 * DT)
    assert (out.status, out.detail) == (Safety.FAULT, "overload")


def test_sustained_overload_faults_and_holds_measured_pose():
    f = make(max_load_pct={"elbow_flex": 80.0}, load_trip_s=0.3)
    f.step(HOME, {}, 0.0)
    pushed = HOME | {"elbow_flex": 12.0}
    out, t = run(f, 0.0, 10, retarget=lambda t: target(t, elbow_flex=40.0))
    assert f.step(pushed, {"elbow_flex": 95.0}, t + 0.1).status == Safety.OK  # brief spike
    out = f.step(pushed, {"elbow_flex": 95.0}, t + 0.5)
    assert out.status == Safety.FAULT
    assert "elbow_flex" in out.detail
    assert out.command == pushed


def test_short_overload_spike_is_ignored():
    f = make(max_load_pct={"elbow_flex": 80.0}, load_trip_s=0.3)
    f.step(HOME, {}, 0.0)
    f.step(HOME, {"elbow_flex": 95.0}, 0.1)
    f.step(HOME, {"elbow_flex": 10.0}, 0.2)
    assert f.step(HOME, {"elbow_flex": 95.0}, 0.45).status != Safety.FAULT


def test_tracking_cap_bounds_command_ahead_of_blocked_joint():
    f = make(max_tracking_error_deg=10.0, max_gripper_tracking_error=20.0)
    blocked = HOME | {"gripper": 40.0}  # gripper stopped by the cup at 40
    f.step(blocked, {}, 0.0)
    out, _ = run(
        f, 0.0, 100, measured=blocked, retarget=lambda t: target(t, elbow_flex=60.0, gripper=0.0)
    )
    assert out.command["elbow_flex"] == pytest.approx(10.0)
    assert out.command["gripper"] == pytest.approx(20.0)


class FakeZone:
    """Clearance shrinks as elbow_flex grows; the zone starts at elbow_flex = 30."""

    def clearance(self, positions):
        return (30.0 - positions["elbow_flex"]) / 100.0, "head"


def test_keep_out_blocks_entering():
    f = make(keep_out=FakeZone())
    f.step(HOME, {}, 0.0)
    out, _ = run(f, 0.0, 100, retarget=lambda t: target(t, elbow_flex=60.0))
    assert out.status == Safety.KEEP_OUT
    assert out.command["elbow_flex"] <= 30.0
    assert "head" in out.detail


def test_keep_out_allows_backing_out():
    f = make(keep_out=FakeZone())
    inside = HOME | {"elbow_flex": 50.0}
    f.step(inside, {}, 0.0)
    out, _ = run(f, 0.0, 100, retarget=lambda t: target(t, elbow_flex=0.0))
    assert out.status == Safety.OK
    assert out.command["elbow_flex"] == pytest.approx(0.0)


def test_missing_limits_rejected():
    with pytest.raises(ValueError):
        SafetyFilter({"elbow_flex": LIMITS["elbow_flex"]}, SafetyConfig())
