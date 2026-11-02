"""Leader -> arm/target -> arm_driver (sim), with a fake leader in place of the hardware."""

import json
import threading
import time

import pytest
import zmq

from yogurt.arm_driver.driver import Driver, make_filter
from yogurt.bus import Bus
from yogurt.config import load_config
from yogurt.leader import LeaderReader, stream, to_positions, to_target
from yogurt.messages import ALL_JOINTS, Safety

CFG = load_config()
POSE = {
    "shoulder_pan": 20.0,
    "shoulder_lift": -30.0,
    "elbow_flex": 25.0,
    "wrist_flex": 10.0,
    "wrist_roll": 0.0,
    "gripper": 70.0,
}


class FakeLeader:
    def read(self) -> dict[str, float]:
        return dict(POSE)


def test_to_positions_and_target():
    positions = to_positions({f"{j}.pos": v for j, v in POSE.items()})
    assert positions == POSE
    target = to_target(positions)
    assert target.gripper == 70.0
    assert "gripper" not in target.positions
    with pytest.raises(ValueError):
        to_positions({"elbow_flex.pos": 1.0})


def test_reader_refuses_wrapped_calibration(tmp_path):
    cfg = load_config()
    cfg.root = tmp_path
    (tmp_path / "calibration").mkdir()
    cal = {j: {"id": 1, "drive_mode": 0, "homing_offset": 0, "range_min": 1000, "range_max": 3000}
           for j in ALL_JOINTS} | {"elbow_flex": {"id": 3, "drive_mode": 0, "homing_offset": 0,
                                                  "range_min": 4, "range_max": 4083}}  # fmt: skip
    (tmp_path / "calibration" / f"{cfg.leader.id}.json").write_text(json.dumps(cal))
    with pytest.raises(RuntimeError, match="elbow_flex"):
        LeaderReader(cfg.leader, cfg)


@pytest.mark.sim
@pytest.mark.skipif(
    not CFG.path(CFG.follower.sim_model).exists(), reason="run scripts/fetch_so_arm100.sh"
)
def test_leader_drives_sim_follower(broker):
    from yogurt.arm_driver.sim import SimBackend

    ctx = zmq.Context()
    sim = SimBackend(CFG.path(CFG.follower.sim_model), rate_hz=50)
    sim.connect()
    driver = Driver(sim, make_filter(CFG.follower, CFG), Bus(broker, "arm_driver", ctx))
    leader_bus = Bus(broker, "leader", ctx)
    thread = threading.Thread(target=stream, args=(FakeLeader(), leader_bus, 30.0, 120))
    thread.start()
    state, deadline = None, time.monotonic() + 4.0
    while time.monotonic() < deadline:
        state = driver.tick(time.monotonic()) or state
        time.sleep(0.02)
        if state and all(abs(state.positions[j] - POSE[j]) < 1.5 for j in POSE if j != "gripper"):
            break
    thread.join()
    leader_bus.close()
    driver.bus.close()
    ctx.term()
    assert state.safety == Safety.OK
    for j in POSE:
        actual = state.gripper if j == "gripper" else state.positions[j]
        assert actual == pytest.approx(POSE[j], abs=2.0), j
