"""leader: owns the SO-101 leader arm and streams its pose as arm/target.

    python -m yogurt.leader [--rate 30] [--config PATH]          (needs `uv sync --extra hw`)

Reads the leader through LeRobot with its motors unpowered and publishes every reading as
arm/target, so arm_driver moves the follower through the full safety filter: the MuJoCo arm with
--backend sim, or the real one with --backend real. Teleop in sim, with the live viewer:

    uv run honcho -f Procfile.sim-teleop start

Don't run the policy at the same time; both would publish arm/target. If the leader stops (Ctrl-C,
unplugged), arm_driver's watchdog holds the last pose.
"""

import argparse
import logging
from collections.abc import Mapping
from typing import Protocol

from yogurt import cli
from yogurt.bus import Bus
from yogurt.calibration import RECALIBRATE_HINT, calibration_problems
from yogurt.config import Config, LeaderConfig
from yogurt.devicelock import DeviceLock
from yogurt.messages import ALL_JOINTS, GRIPPER, JOINTS, JointTarget
from yogurt.rate import Rate

log = logging.getLogger("yogurt.leader")

NAME = "leader"
MAX_READ_FAILURES = 10


class PoseSource(Protocol):
    def read(self) -> dict[str, float]: ...


class LeaderReader:
    """The leader arm as a pose source: degrees for body joints, 0..100 for the gripper."""

    def __init__(self, leader: LeaderConfig, cfg: Config):
        path = cfg.path(leader.calibration_dir) / f"{leader.id}.json"
        if not path.exists():
            raise RuntimeError(f"no calibration at {path}; run scripts/calibrate.py --arm leader")
        problems = calibration_problems(path)
        if problems:
            raise RuntimeError(
                f"{'; '.join(problems)}. {RECALIBRATE_HINT.replace('<arm>', 'leader')}"
            )
        try:
            from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
        except ImportError as e:
            raise RuntimeError("the leader needs LeRobot: uv sync --extra hw") from e
        self.leader = leader
        self._lock = DeviceLock(leader.port)
        self.device = SO101Leader(
            SO101LeaderConfig(
                port=leader.port,
                id=leader.id,
                calibration_dir=cfg.path(leader.calibration_dir),
                use_degrees=True,
            )
        )

    def connect(self) -> None:
        self._lock.acquire()
        self.device.connect(calibrate=False)  # configure() leaves the leader's torque off
        if not self.device.is_calibrated:
            raise RuntimeError(f"{self.leader.id}: motors disagree with the calibration file")

    def read(self) -> dict[str, float]:
        return to_positions(self.device.get_action())

    def close(self) -> None:
        try:
            if self.device.is_connected:
                self.device.disconnect()
        finally:
            self._lock.release()


def to_positions(action: Mapping[str, float]) -> dict[str, float]:
    """LeRobot's {"elbow_flex.pos": deg, ...} -> {"elbow_flex": deg, ...}."""
    positions = {k.removesuffix(".pos"): float(v) for k, v in action.items()}
    missing = set(ALL_JOINTS) - set(positions)
    if missing:
        raise ValueError(f"leader reading is missing {sorted(missing)}")
    return positions


def to_target(positions: Mapping[str, float]) -> JointTarget:
    return JointTarget(positions={j: positions[j] for j in JOINTS}, gripper=positions[GRIPPER])


def stream(source: PoseSource, bus: Bus, rate_hz: float, max_ticks: int | None = None) -> None:
    """Publish one arm/target per tick. Stops after MAX_READ_FAILURES failed reads in a row."""
    rate, failures, ticks = Rate(rate_hz), 0, 0
    while max_ticks is None or ticks < max_ticks:
        ticks += 1
        try:
            positions = source.read()
            failures = 0
        except Exception as e:  # a flaky bus read; the follower holds via its watchdog
            failures += 1
            log.warning("leader read failed (%d in a row): %s", failures, e)
            if failures >= MAX_READ_FAILURES:
                raise RuntimeError("leader stopped responding") from e
            rate.sleep()
            continue
        bus.publish("arm/target", to_target(positions))
        rate.sleep()


def main(argv: list[str] | None = None) -> None:
    def add_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--rate", type=float, default=30.0, help="publish rate in Hz (default 30)")

    args, cfg = cli.start(NAME, __doc__, argv, add_args)
    try:
        reader = LeaderReader(cfg.leader, cfg)
        reader.connect()
    except (RuntimeError, OSError, ConnectionError) as e:
        log.error("cannot start: %s", e)
        raise SystemExit(2) from e
    bus = Bus(cfg.bus, NAME)
    log.info(
        "streaming %s on %s -> arm/target at %.0f Hz", cfg.leader.id, cfg.leader.port, args.rate
    )
    try:
        stream(reader, bus, args.rate)
    except KeyboardInterrupt:
        pass
    except RuntimeError as e:
        log.error("%s", e)
        raise SystemExit(1) from e
    finally:
        reader.close()
        bus.close()


if __name__ == "__main__":
    main()
