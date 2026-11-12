"""The arm. Every motion command in the project goes through Arm, which keeps the servos safe:

- targets are clamped to the calibrated joint range (minus a margin),
- the commanded position moves at a bounded speed, never in one jump,
- the command never runs more than a few degrees ahead of where the arm really is, so a blocked
  arm is not pushed harder and harder,
- servo temperature is watched, and torque is released if one overheats.

hold() is Safe Stop (freeze, servos stay powered); kill() is Kill (servos released, port closed).
"""

import asyncio
import glob
import json
import logging
import threading
import time
from pathlib import Path

from . import config

log = logging.getLogger("yogurt.arm")

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
BODY = JOINTS[:5]
ENCODER_MAX = 4095  # STS3215 ticks per turn - 1, as LeRobot uses for its degree conversion


class ArmError(RuntimeError):
    pass


def joint_limits(calibration: dict, margin_deg: float) -> dict[str, tuple[float, float]]:
    """Usable range per joint. LeRobot degrees are measured from the middle of the calibrated range."""
    limits = {}
    for joint in BODY:
        half = (calibration[joint]["range_max"] - calibration[joint]["range_min"]) / 2 * 360 / ENCODER_MAX
        limits[joint] = (-half + margin_deg, half - margin_deg)
    limits["gripper"] = (0.0, 100.0)
    return limits


def find_port(exclude: str | None = None) -> str:
    patterns = ["/dev/tty.usbmodem*", "/dev/tty.usbserial*", "/dev/ttyACM*", "/dev/ttyUSB*"]
    ports = sorted({p for pattern in patterns for p in glob.glob(pattern)} - {exclude})
    if len(ports) == 1:
        return ports[0]
    if not ports:
        raise ArmError("no USB serial adapter found - is the arm plugged in and powered?")
    raise ArmError(f"several serial ports found ({', '.join(ports)}) - set the port in the hardware config")


def open_feetech(kind: str, port: str, calibration_file: Path):
    """Open a LeRobot SO-101 follower or leader without its interactive calibration prompt."""
    try:  # LeRobot >= 0.6
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
    except ImportError:  # older LeRobot, as in the project's Movement_mvp.py
        from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
        from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig

    if not calibration_file.exists():
        raise ArmError(f"calibration file {calibration_file} not found")
    make, make_cfg = (SO101Follower, SO101FollowerConfig) if kind == "follower" else (SO101Leader, SO101LeaderConfig)
    device = make(make_cfg(port=port, id=calibration_file.stem, calibration_dir=calibration_file.parent,
                           use_degrees=True))
    bus = device.bus
    bus.connect()
    try:
        bus.disable_torque()
        if not bus.is_calibrated:
            log.warning("%s: servo memory differs from %s - writing the file's calibration to the servos",
                        kind, calibration_file.name)
            bus.write_calibration(device.calibration)
        if kind == "follower":
            # configure() ends by switching torque on; make sure that cannot move anything.
            bus.sync_write("Goal_Position", bus.sync_read("Present_Position", num_retry=3))
        device.configure()
    except Exception:
        bus.disconnect(disable_torque=True)
        raise
    return device


class ServoBus:
    """The real follower arm."""

    def __init__(self, port: str, calibration_file: Path):
        # Keep the follower object itself: LeRobot disconnects the bus when that object is garbage-collected.
        self.follower = open_feetech("follower", port, calibration_file)
        self.bus = self.follower.bus

    def read_positions(self) -> dict[str, float]:
        return self.bus.sync_read("Present_Position", num_retry=3)

    def write_goals(self, goals: dict[str, float]) -> None:
        self.bus.sync_write("Goal_Position", goals)

    def set_torque(self, on: bool) -> None:
        if on:
            self.bus.sync_write("Goal_Position", self.read_positions())
            self.bus.enable_torque(num_retry=3)
        else:
            self.bus.disable_torque(num_retry=3)

    def read_health(self) -> dict[str, dict]:
        temps = self.bus.sync_read("Present_Temperature", normalize=False, num_retry=3)
        loads = self.bus.sync_read("Present_Load", normalize=False, num_retry=3)
        return {j: {"temp_c": int(temps[j]), "load_pct": loads[j] / 10} for j in JOINTS}

    def close(self) -> None:
        self.bus.disconnect(disable_torque=True)


class SimBus:
    """Stands in for the servos when arm.port is 'sim': joints glide toward their goals."""

    def __init__(self, start: dict[str, float]):
        self.pos, self.goal, self.torque, self.t = dict(start), dict(start), False, time.monotonic()

    def read_positions(self) -> dict[str, float]:
        now = time.monotonic()
        blend = min(1.0, (now - self.t) * 15)
        self.t = now
        if self.torque:
            for joint in JOINTS:
                self.pos[joint] += (self.goal[joint] - self.pos[joint]) * blend
        return dict(self.pos)

    def write_goals(self, goals: dict[str, float]) -> None:
        self.goal.update(goals)

    def set_torque(self, on: bool) -> None:
        self.goal, self.torque = dict(self.pos), on

    def read_health(self) -> dict[str, dict]:
        return {j: {"temp_c": 30, "load_pct": 0.0} for j in JOINTS}

    def close(self) -> None:
        self.torque = False


class Arm:
    def __init__(self, cfg):
        self.cfg, self.settings = cfg, cfg.arm
        self.calibration_file = config.path(cfg.arm.calibration)
        self.limits = joint_limits(json.loads(self.calibration_file.read_text()), cfg.arm.limit_margin_deg)
        self.sim = cfg.arm.port == "sim"
        self.bus = None
        self.connected = False
        self.torque = False
        self.fault: str | None = None      # why the arm is down, shown in the UI
        self.pos = dict.fromkeys(JOINTS, 0.0)       # measured
        self.cmd = dict(self.pos)                    # what the servos are being told right now
        self.target = dict(self.pos)                 # where cmd is heading
        self.speed = cfg.arm.speed_dps
        self.health: dict[str, dict] = {}
        self.port: str | None = None
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._running = False

    # ---- connection -------------------------------------------------------------------------

    def connect(self) -> None:
        """Open the bus and hold the arm exactly where it is."""
        self.kill("reconnecting")
        try:
            if self.sim:
                self.port, self.bus = "sim", SimBus(self.clamp(self.cfg.poses.idle))
            else:
                port = self.cfg.arm.port
                self.port = find_port(exclude=self.cfg.leader.port) if port == "auto" else port
                self.bus = ServoBus(self.port, self.calibration_file)
            with self._lock:
                self.bus.set_torque(True)
                self.pos = self.bus.read_positions()
                self.cmd, self.target = dict(self.pos), dict(self.pos)
                self.torque, self.connected, self.fault = True, True, None
        except Exception as e:
            self.kill(f"could not connect to the arm: {e}".strip())
            raise ArmError(self.fault) from e
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="arm", daemon=True)
        self._thread.start()
        log.info("arm connected on %s", self.port)

    def kill(self, reason: str = "kill switch") -> None:
        """Release every servo and close the port."""
        self._running = False
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=2)
        with self._lock:
            if self.bus is not None:
                try:
                    self.bus.close()
                except Exception as e:  # the port may already be gone
                    log.warning("closing the arm: %s", e)
                self.bus = None
                log.info("arm released (%s)", reason)
            self.connected = self.torque = False
            self.fault = reason

    # ---- commands ---------------------------------------------------------------------------

    def clamp(self, pose: dict[str, float]) -> dict[str, float]:
        return {j: min(max(float(v), self.limits[j][0]), self.limits[j][1]) for j, v in pose.items()}

    def set_target(self, pose: dict[str, float], speed: float | None = None) -> None:
        """Head for pose (any subset of joints) at speed deg/s. Returns immediately."""
        with self._lock:
            self.target.update(self.clamp(pose))
            self.speed = min(speed or self.settings.speed_dps, self.settings.max_speed_dps)

    async def goto(self, pose: dict[str, float], speed: float | None = None) -> None:
        """Move to pose and wait until the arm is there. Raises ArmError if it cannot get there."""
        if not (self.connected and self.torque):
            raise ArmError(self.fault or "arm servos are released")
        self.set_target(pose, speed)
        goal = self.clamp(pose)
        travel = max((abs(goal[j] - self.pos[j]) for j in goal), default=0.0)
        deadline = time.monotonic() + travel / self.speed * 2 + 4
        while True:
            await asyncio.sleep(0.05)
            if not (self.connected and self.torque):
                raise ArmError(self.fault or "arm servos were released during a move")
            if any(self.target[j] != goal[j] for j in goal):
                raise ArmError("move interrupted by another command")
            body = [j for j in goal if j != "gripper"]
            if all(self.cmd[j] == goal[j] for j in goal) and all(
                abs(self.pos[j] - goal[j]) < self.settings.tolerance_deg for j in body
            ):
                return
            if time.monotonic() > deadline:
                self.hold()
                off = {j: round(self.pos[j] - goal[j], 1) for j in body}
                raise ArmError(f"arm did not reach its target (blocked?) - error per joint {off}")

    def hold(self) -> None:
        """Safe Stop: freeze the commanded position, so the servos stop and keep holding there."""
        with self._lock:
            self.target = dict(self.cmd)

    def set_torque(self, on: bool) -> None:
        """Release (False) or re-engage (True) the servos while staying connected, for teaching poses."""
        with self._lock:
            if self.bus is None:
                raise ArmError(self.fault or "arm is not connected")
            self.bus.set_torque(on)
            self.pos = self.bus.read_positions()
            self.cmd, self.target, self.torque = dict(self.pos), dict(self.pos), on

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "connected": self.connected, "torque": self.torque, "sim": self.sim, "port": self.port,
                "fault": self.fault, "limits": self.limits, "health": self.health,
                "pos": {j: round(v, 1) for j, v in self.pos.items()},
                "target": {j: round(v, 1) for j, v in self.target.items()},
            }

    # ---- control loop -----------------------------------------------------------------------

    def _step(self, dt: float) -> None:
        """Advance the commanded position one tick toward the target, inside every safety bound."""
        for joint in JOINTS:
            max_step = (self.settings.gripper_speed if joint == "gripper" else self.speed) * dt
            delta = self.target[joint] - self.cmd[joint]
            cmd = self.cmd[joint] + min(max(delta, -max_step), max_step)
            if joint != "gripper":  # the gripper is allowed to squeeze; LeRobot caps its torque
                lead = self.settings.max_lead_deg
                cmd = min(max(cmd, self.pos[joint] - lead), self.pos[joint] + lead)
            # Inside the soft range always; a joint that starts outside it (resting on an end stop)
            # may stay where it is or come inwards, never go further out.
            low, high = self.limits[joint]
            self.cmd[joint] = min(max(cmd, min(low, self.cmd[joint])), max(high, self.cmd[joint]))

    def _loop(self) -> None:
        period = 1.0 / self.settings.rate_hz
        failures, next_health = 0, 0.0
        while self._running:
            started = time.monotonic()
            try:
                with self._lock:
                    self.pos = self.bus.read_positions()
                    if self.torque:
                        self._step(period)
                        self.bus.write_goals(self.cmd)
                    if started >= next_health:
                        self.health = self.bus.read_health()
                        next_health = started + 1.0
                failures = 0
            except Exception as e:
                failures += 1
                log.warning("arm bus error (%d in a row): %s", failures, e)
                if failures >= 10:
                    threading.Thread(target=self.kill, args=(f"lost contact with the arm: {e}",)).start()
                    return
            hot = [f"{j} {h['temp_c']}°C" for j, h in self.health.items() if h["temp_c"] >= self.settings.max_temp_c]
            if hot:
                threading.Thread(target=self.kill, args=(f"servo overheating: {', '.join(hot)}",)).start()
                return
            time.sleep(max(0.0, period - (time.monotonic() - started)))
