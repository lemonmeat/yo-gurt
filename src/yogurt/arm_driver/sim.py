"""MuJoCo backend: SO-ARM100's Simulation/SO101 model, stepped one control period per write().

The model's joint zero is the middle of each joint's range, like LeRobot's degrees, so the mapping
is the same FkConfig sign/offset the safety filter uses. The model's gripper is the stock SO-101
gripper, the same one on the real follower; percent maps linearly onto its actuator range
(0 = closed end). JointMap holds that mapping; the viewer uses it too.
"""

import math
from collections.abc import Mapping
from pathlib import Path

import mujoco
import numpy as np

from yogurt.arm_driver.backend import Reading
from yogurt.config import FkConfig
from yogurt.messages import ALL_JOINTS, GRIPPER, JOINTS

SIM_TEMPERATURE_C = 25.0


def load_model(path: Path) -> mujoco.MjModel:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run scripts/fetch_so_arm100.sh")
    return mujoco.MjModel.from_xml_path(str(path))


class JointMap:
    """Degrees / gripper percent <-> the model's joint positions."""

    def __init__(self, model: mujoco.MjModel, fk: FkConfig | None = None):
        self.qadr = {j: int(model.joint(j).qposadr[0]) for j in ALL_JOINTS}
        self._grip_lo, self._grip_hi = model.jnt_range[model.joint(GRIPPER).id]
        self._sign = {j: float((fk.joint_sign if fk else {}).get(j, 1.0)) for j in JOINTS}
        self._offset = {j: float((fk.joint_offset_deg if fk else {}).get(j, 0.0)) for j in JOINTS}

    def to_model(self, name: str, value: float) -> float:
        if name == GRIPPER:
            return self._grip_lo + value / 100.0 * (self._grip_hi - self._grip_lo)
        return math.radians(self._sign[name] * (value + self._offset[name]))

    def from_model(self, name: str, q: float) -> float:
        if name == GRIPPER:
            return 100.0 * (q - self._grip_lo) / (self._grip_hi - self._grip_lo)
        return math.degrees(q) / self._sign[name] - self._offset[name]

    def set_qpos(self, data: mujoco.MjData, positions: Mapping[str, float]) -> None:
        for name, value in positions.items():
            data.qpos[self.qadr[name]] = self.to_model(name, value)


class SimBackend:
    name = "sim"

    def __init__(
        self,
        model_path: Path,
        rate_hz: float,
        fk: FkConfig | None = None,
        initial: dict[str, float] | None = None,
    ):
        self.model = load_model(model_path)
        self.data = mujoco.MjData(self.model)
        self.joints = JointMap(self.model, fk)
        self._act = {j: self.model.actuator(j).id for j in ALL_JOINTS}
        self._force_max = {
            j: float(np.abs(self.model.actuator_forcerange[a]).max()) for j, a in self._act.items()
        }
        self._substeps = max(1, round(1.0 / (rate_hz * self.model.opt.timestep)))
        self._initial = dict.fromkeys(JOINTS, 0.0) | {GRIPPER: 50.0} | (initial or {})

    def connect(self) -> None:
        self.joints.set_qpos(self.data, self._initial)
        for name, value in self._initial.items():
            self.data.ctrl[self._act[name]] = self.joints.to_model(name, value)
        mujoco.mj_forward(self.model, self.data)

    def read(self) -> Reading:
        positions = {
            j: self.joints.from_model(j, float(self.data.qpos[a]))
            for j, a in self.joints.qadr.items()
        }
        loads = {
            j: 100.0 * float(self.data.actuator_force[a]) / self._force_max[j]
            for j, a in self._act.items()
        }
        temps = dict.fromkeys(ALL_JOINTS, SIM_TEMPERATURE_C)
        return Reading(positions, loads, temps)

    def write(self, command: dict[str, float]) -> None:
        for name, value in command.items():
            self.data.ctrl[self._act[name]] = self.joints.to_model(name, value)
        for _ in range(self._substeps):
            mujoco.mj_step(self.model, self.data)

    def close(self) -> None:
        pass
