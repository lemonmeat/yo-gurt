"""Real backend: an SO-101 on a Feetech bus through LeRobot (needs `uv sync --extra hw`).

UNTESTED ON HARDWARE. Read the TODO(hw) notes before the first run, with a hand on the power switch.

Body joints use LeRobot's calibrated degrees. The gripper goes through gripper_map.yaml in raw
counts. ACT policies trained on lerobot-record data speak LeRobot's gripper scale (range_min..
range_max of the calibration file), so connect() warns if the two disagree.
"""

import logging
import time

from yogurt.arm_driver.backend import Reading
from yogurt.calibration import RECALIBRATE_HINT, calibration_problems
from yogurt.config import ArmConfig, Config
from yogurt.devicelock import DeviceLock
from yogurt.messages import ALL_JOINTS, GRIPPER, JOINTS

log = logging.getLogger(__name__)

READ_RETRIES = 2
TEMPERATURE_PERIOD_S = 1.0
GRIPPER_MAP_TOLERANCE = 50  # counts


class RealBackend:
    name = "real"

    def __init__(self, arm: ArmConfig, cfg: Config):
        if not arm.limits.measured:
            raise RuntimeError(
                f"{arm.id}: joint_limits.yaml still has placeholder limits (measured: false). "
                "Measure them first (docs/TASKS.md T1)."
            )
        if not arm.gripper_map.measured:
            raise RuntimeError(
                f"{arm.id}: gripper_map.yaml is a placeholder (measured: false). "
                "Measure it first (docs/TASKS.md T2)."
            )
        try:
            from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        except ImportError as e:
            raise RuntimeError("real backend needs LeRobot: uv sync --extra hw") from e

        self.arm = arm
        self.gripper_map = arm.gripper_map
        self._lock = DeviceLock(arm.port)
        self.robot = SO101Follower(
            SO101FollowerConfig(
                port=arm.port,
                id=arm.id,
                calibration_dir=cfg.path(arm.calibration_dir),
                use_degrees=True,
                disable_torque_on_disconnect=arm.torque_off_on_exit,
                cameras={},  # camera_hub owns the cameras
            )
        )
        self._temps: dict[str, float] = {}
        self._temps_at = 0.0

    @property
    def bus(self):
        return self.robot.bus

    def connect(self) -> None:
        self._lock.acquire()
        if not self.robot.calibration:
            raise RuntimeError(
                f"no calibration for {self.arm.id} in {self.arm.calibration_dir}; "
                "run scripts/calibrate.py first"
            )
        problems = calibration_problems(self.robot.calibration_fpath)
        if problems:
            hint = RECALIBRATE_HINT.replace("<arm>", "follower")
            raise RuntimeError(f"{'; '.join(problems)}. {hint}")
        # calibrate=False: never start interactive calibration inside the driver. connect() also
        # writes PID gains and the gripper's torque/current caps (LeRobot's configure()).
        self.robot.connect(calibrate=False)
        if not self.robot.is_calibrated:
            raise RuntimeError(f"{self.arm.id}: motors disagree with the calibration file")
        self._check_gripper_map()

        # TODO(hw): confirm Torque_Limit (RAM, 0..1000) behaves the same on the STS3215 and ST3215
        # units in this arm, and that it stays at or below the gripper's Max_Torque_Limit (500).
        limit = round(10 * self.arm.safety.torque_limit_pct)
        for motor in self.bus.motors:
            cap = 500 if motor == GRIPPER else 1000
            self.bus.write("Torque_Limit", motor, min(limit, cap))
        log.info("%s connected on %s", self.arm.id, self.arm.port)

    def _check_gripper_map(self) -> None:
        cal = self.robot.calibration[GRIPPER]
        g = self.gripper_map
        if (
            abs(cal.range_min - g.closed_counts) > GRIPPER_MAP_TOLERANCE
            or abs(cal.range_max - g.open_counts) > GRIPPER_MAP_TOLERANCE
        ):
            log.warning(
                "gripper_map.yaml (closed %d, open %d) differs from the LeRobot calibration "
                "(range_min %d, range_max %d). Policies trained on lerobot-record data will "
                "command the gripper on the calibration's scale.",
                g.closed_counts,
                g.open_counts,
                cal.range_min,
                cal.range_max,
            )

    def read(self) -> Reading:
        positions = self.bus.sync_read("Present_Position", list(JOINTS), num_retry=READ_RETRIES)
        raw = self.bus.sync_read(
            "Present_Position", [GRIPPER], normalize=False, num_retry=READ_RETRIES
        )
        positions[GRIPPER] = self.gripper_map.to_percent(raw[GRIPPER])
        # Present_Load is signed, in 0.1 % of max torque.
        loads = {
            k: v / 10.0
            for k, v in self.bus.sync_read("Present_Load", num_retry=READ_RETRIES).items()
        }
        now = time.monotonic()
        if now - self._temps_at > TEMPERATURE_PERIOD_S:
            self._temps = {
                k: float(v)
                for k, v in self.bus.sync_read(
                    "Present_Temperature", num_retry=READ_RETRIES
                ).items()
            }
            self._temps_at = now
        return Reading({k: float(v) for k, v in positions.items()}, loads, dict(self._temps))

    def write(self, command: dict[str, float]) -> None:
        missing = set(ALL_JOINTS) - set(command)
        if missing:
            raise ValueError(f"command is missing {sorted(missing)}")
        self.bus.sync_write("Goal_Position", {j: command[j] for j in JOINTS})
        self.bus.write(
            "Goal_Position", GRIPPER, self.gripper_map.to_counts(command[GRIPPER]), normalize=False
        )

    def close(self) -> None:
        try:
            if self.robot.is_connected:
                self.robot.disconnect()  # disables torque if torque_off_on_exit
        finally:
            self._lock.release()
