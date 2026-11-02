"""The viewer's model posing and overlay, without opening a window."""

import mujoco
import numpy as np
import pytest

from yogurt.arm_driver.sim import JointMap, load_model
from yogurt.config import Box, load_config
from yogurt.kinematics import Chain
from yogurt.messages import ALL_JOINTS, Safety
from yogurt.viewer import STALE_RGBA, STATUS_RGBA, draw, overlay

CFG = load_config()
pytestmark = [
    pytest.mark.sim,
    pytest.mark.skipif(
        not CFG.path(CFG.follower.sim_model).exists(),
        reason="run scripts/fetch_so_arm100.sh for the SO-101 model",
    ),
]


@pytest.fixture(scope="module")
def model():
    return load_model(CFG.path(CFG.follower.sim_model))


def test_joint_map_round_trip(model):
    joints = JointMap(model, CFG.follower.fk)
    for name, value in zip(ALL_JOINTS, (10.0, -35.0, 50.0, -20.0, 90.0, 42.0), strict=True):
        assert joints.from_model(name, joints.to_model(name, value)) == pytest.approx(value)


def test_posed_model_matches_forward_kinematics(model):
    data = mujoco.MjData(model)
    joints = JointMap(model, CFG.follower.fk)
    chain = Chain.from_config(CFG.follower.fk, CFG.path(CFG.follower.fk.urdf))
    pose = {
        "shoulder_pan": 20.0,
        "shoulder_lift": -30.0,
        "elbow_flex": 40.0,
        "wrist_flex": 10.0,
        "wrist_roll": 0.0,
    }
    joints.set_qpos(data, pose | {"gripper": 50.0})
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.site("gripperframe").xpos, chain.tip(pose), atol=1e-4)


def test_overlay_boxes_and_tip_colors():
    boxes = [Box(name="wall", min=[0.3, -0.1, 0.0], max=[0.5, 0.1, 0.3])]
    tip = np.array([0.2, 0.0, 0.1])
    specs = overlay(boxes, 0.03, tip, Safety.OK)
    assert len(specs) == 3  # box, margin shell, tip
    kind, half, center, _ = specs[0]
    assert kind == mujoco.mjtGeom.mjGEOM_BOX
    np.testing.assert_allclose(center, [0.4, 0.0, 0.15])
    np.testing.assert_allclose(specs[1][1], np.asarray(half) + 0.03)
    assert specs[2][3] == STATUS_RGBA[Safety.OK]
    assert overlay(boxes, 0.03, tip, None)[2][3] == STALE_RGBA
    active = overlay(boxes, 0.03, tip, Safety.KEEP_OUT, active_zone="wall")
    assert active[0][3][3] > specs[0][3][3]  # the zone being held at is drawn brighter
    assert len(overlay([], 0.03, None, None)) == 0


def test_draw_into_scene(model):
    scene = mujoco.MjvScene(model, maxgeom=10)
    boxes = [Box(name="a", min=[0, 0, 0], max=[0.1, 0.1, 0.1])]
    draw(scene, overlay(boxes, 0.02, np.zeros(3), Safety.WATCHDOG))
    assert scene.ngeom == 3
    draw(scene, [])
    assert scene.ngeom == 0
