"""MuJoCo backend: SO-ARM100's Simulation/SO101 model, stepped one control period per write().

The model's joint zero is the middle of each joint's range, like LeRobot's degrees, so the mapping
is the same FkConfig sign/offset the safety filter uses. The model has the stock gripper; percent
maps linearly onto its actuator range (0 = closed end).
"""

import math
from pathlib import Path

import mujoco
import numpy as np

from yogurt.arm_driver.backend import Reading
from yogurt.config import FkConfig
from yogurt.messages import ALL_JOINTS, GRIPPER, JOINTS

SIM_TEMPERATURE_C = 25.0


class SimBackend:
    name = "sim"

    def __init__(
        self,
        model_path: Path,
        rate_hz: float,
        fk: FkConfig | None = None,
        initial: dict[str, float] | None = None,
    ):
        if not model_path.exists():
            raise FileNotFoundError(f"{model_path} missing; run scripts/fetch_so_arm100.sh")
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self._qadr = {j: self.model.joint(j).qposadr[0] for j in ALL_JOINTS}
        self._act = {j: self.model.actuator(j).id for j in ALL_JOINTS}
        self._force_max = {
            j: float(np.abs(self.model.actuator_forcerange[a]).max()) for j, a in self._act.items()
        }
        self._grip_lo, self._grip_hi = self.model.actuator_ctrlrange[self._act[GRIPPER]]
        self._substeps = max(1, round(1.0 / (rate_hz * self.model.opt.timestep)))
        self._sign = {j: float((fk.joint_sign if fk else {}).get(j, 1.0)) for j in JOINTS}
        self._offset = {j: float((fk.joint_offset_deg if fk else {}).get(j, 0.0)) for j in JOINTS}
        self._initial = dict.fromkeys(JOINTS, 0.0) | {GRIPPER: 50.0} | (initial or {})

    def _to_sim(self, name: str, value: float) -> float:
        if name == GRIPPER:
            return self._grip_lo + value / 100.0 * (self._grip_hi - self._grip_lo)
        return math.radians(self._sign[name] * (value + self._offset[name]))

    def _from_sim(self, name: str, q: float) -> float:
        if name == GRIPPER:
            return 100.0 * (q - self._grip_lo) / (self._grip_hi - self._grip_lo)
        return math.degrees(q) / self._sign[name] - self._offset[name]

    def connect(self) -> None:
        for name, value in self._initial.items():
            q = self._to_sim(name, value)
            self.data.qpos[self._qadr[name]] = q
            self.data.ctrl[self._act[name]] = q
        mujoco.mj_forward(self.model, self.data)

    def read(self) -> Reading:
        positions = {j: self._from_sim(j, float(self.data.qpos[a])) for j, a in self._qadr.items()}
        loads = {
            j: 100.0 * float(self.data.actuator_force[a]) / self._force_max[j]
            for j, a in self._act.items()
        }
        temps = dict.fromkeys(ALL_JOINTS, SIM_TEMPERATURE_C)
        return Reading(positions, loads, temps)

    def write(self, command: dict[str, float]) -> None:
        for name, value in command.items():
            self.data.ctrl[self._act[name]] = self._to_sim(name, value)
        for _ in range(self._substeps):
            mujoco.mj_step(self.model, self.data)

    def close(self) -> None:
        pass
