import pytest

from yogurt.config import load_config
from yogurt.messages import ALL_JOINTS, JOINTS
from yogurt.policy.replay import PoseSequence, Replayer, Waypoint, load_poses, save_poses

START = dict.fromkeys(ALL_JOINTS, 0.0) | {"gripper": 50.0}


def seq(*waypoints: Waypoint) -> PoseSequence:
    return PoseSequence(task="grasp", source="sim", waypoints=list(waypoints))


def test_interpolates_then_settles():
    r = Replayer(
        seq(Waypoint(duration_s=1.0, positions={"elbow_flex": 40.0}, gripper=100.0)), START, 0.0
    )
    mid = r.step(0.5, START)
    assert mid.positions["elbow_flex"] == pytest.approx(20.0)
    assert mid.gripper == pytest.approx(75.0)
    # Time is up but the arm has not arrived: keep holding the waypoint.
    hold = r.step(1.2, START)
    assert hold.positions["elbow_flex"] == 40.0
    assert r.status == "running"
    arrived = START | {"elbow_flex": 39.0}
    assert r.step(1.3, arrived) is None
    assert r.status == "done"


def test_partial_waypoints_keep_previous_values():
    r = Replayer(
        seq(
            Waypoint(duration_s=1.0, positions={"elbow_flex": 30.0}),
            Waypoint(duration_s=1.0, gripper=0.0),
        ),
        START,
        0.0,
    )
    r.step(1.0, START | {"elbow_flex": 30.0})  # reach waypoint 1
    t = r.step(1.5, START | {"elbow_flex": 30.0})
    assert t.positions["elbow_flex"] == 30.0
    assert t.gripper == pytest.approx(25.0)


def test_gripper_blocked_by_cup_does_not_stall():
    r = Replayer(seq(Waypoint(duration_s=0.5, gripper=0.0)), START, 0.0)
    assert r.step(0.6, START) is None  # gripper still at 50: body joints settled, so done
    assert r.status == "done"


def test_unreached_waypoint_fails_after_settle_timeout():
    r = Replayer(
        seq(Waypoint(duration_s=1.0, positions={"shoulder_lift": -40.0})), START, 0.0,
        settle_timeout_s=2.0,
    )  # fmt: skip
    r.step(2.5, START)
    assert r.status == "running"
    assert r.step(3.1, START) is None
    assert r.status == "failed"
    assert "shoulder_lift" in r.detail


def test_bad_waypoints_are_rejected():
    with pytest.raises(ValueError):
        Waypoint(duration_s=0.0)
    with pytest.raises(ValueError):
        Waypoint(duration_s=1.0, positions={"elbo": 1.0})


@pytest.mark.parametrize("task", ["grasp", "handoff", "release_home"])
def test_shipped_placeholder_poses_load_and_respect_limits(task):
    cfg = load_config()
    s = load_poses(cfg.path(cfg.policy.poses_dir), task)
    assert s.source == "sim"
    limits = cfg.follower.limits.joints
    for w in s.waypoints:
        for j, v in w.positions.items():
            assert limits[j].min <= v <= limits[j].max, (task, j, v)


def test_save_and_load_round_trip(tmp_path):
    s = PoseSequence(
        task="grasp",
        source="real",
        waypoints=[Waypoint(duration_s=2.0, positions=dict.fromkeys(JOINTS, 1.234), gripper=80.0)],
    )
    save_poses(tmp_path / "grasp.yaml", s, header="# taught\n")
    assert (tmp_path / "grasp.yaml").read_text().startswith("# taught\n")
    out = load_poses(tmp_path, "grasp")
    assert out.source == "real"
    assert out.waypoints[0].positions["elbow_flex"] == pytest.approx(1.23)
