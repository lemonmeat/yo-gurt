import numpy as np
import pytest

from yogurt.config import Box, load_config
from yogurt.kinematics import Chain, KeepOut, box_signed_distance
from yogurt.messages import JOINTS

CFG = load_config()
URDF = CFG.path(CFG.follower.fk.urdf)
SCENE = CFG.path(CFG.follower.sim_model)
needs_model = pytest.mark.skipif(
    not URDF.exists(), reason="run scripts/fetch_so_arm100.sh for the SO-101 model"
)


def test_box_signed_distance():
    lo, hi = np.zeros(3), np.ones(3)
    pts = np.array([[0.5, 0.5, 0.5], [2.0, 0.5, 0.5], [0.5, 0.5, 0.9]])
    np.testing.assert_allclose(box_signed_distance(pts, lo, hi), [-0.5, 1.0, -0.1])


@needs_model
@pytest.mark.sim
def test_urdf_fk_matches_mujoco():
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    chain = Chain(URDF)
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = {name: float(rng.uniform(-80, 80)) for name in JOINTS}
        for name in JOINTS:
            data.qpos[model.joint(name).qposadr[0]] = np.radians(q[name])
        mujoco.mj_kinematics(model, data)
        np.testing.assert_allclose(chain.tip(q), data.site("gripperframe").xpos, atol=1e-4)
        np.testing.assert_allclose(chain.points(q)[2], data.body("wrist").xpos, atol=1e-4)


@needs_model
def test_tool_offset_and_joint_sign():
    base = Chain(URDF)
    q = dict.fromkeys(JOINTS, 0.0) | {"shoulder_pan": 30.0}
    flipped = Chain(URDF, joint_sign={"shoulder_pan": -1.0})
    tip, tip_flipped = base.tip(q), flipped.tip(q)
    assert tip[1] == pytest.approx(-tip_flipped[1], abs=1e-3)
    longer = Chain(URDF, tool_offset=[0.0, 0.0, 0.05])
    assert np.linalg.norm(longer.tip(q) - tip) == pytest.approx(0.05)


@needs_model
def test_keep_out_box_in_front_of_arm():
    chain = Chain(URDF)
    home = dict.fromkeys(JOINTS, 0.0)  # reaches ~0.39 m forward at ~0.23 m height
    wall = Box(name="wall", min=[0.35, -0.2, 0.0], max=[0.6, 0.2, 0.5])
    zone = KeepOut(chain, [wall], margin_m=0.02)
    clearance, name = zone.clearance(home)
    assert clearance < 0 and name == "wall"
    folded = home | {"shoulder_lift": -90.0, "elbow_flex": 90.0}
    assert zone.clearance(folded)[0] > 0


@needs_model
def test_shipped_head_zone_is_clear_at_home():
    chain = Chain.from_config(CFG.follower.fk, URDF)
    zone = KeepOut(chain, CFG.follower.safety.keep_out, CFG.follower.safety.keep_out_margin_m)
    assert zone.clearance(dict.fromkeys(JOINTS, 0.0))[0] > 0
