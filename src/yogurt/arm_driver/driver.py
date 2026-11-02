"""arm_driver's control loop: bus in, safety filter, servos, arm/state out."""

import argparse
import logging
import time

from yogurt import cli
from yogurt.arm_driver import __doc__ as doc
from yogurt.arm_driver.backend import Backend
from yogurt.bus import Bus
from yogurt.config import ArmConfig, Config
from yogurt.kinematics import Chain, KeepOut
from yogurt.messages import GRIPPER, JOINTS, JointState, JointTarget, Safety, Stop
from yogurt.rate import Rate
from yogurt.safety import SafetyFilter

log = logging.getLogger(__name__)

NAME = "arm_driver"
MAX_READ_FAILURES = 5
REJECT_LOG_PERIOD_S = 2.0


def make_filter(arm: ArmConfig, cfg: Config) -> SafetyFilter:
    keep_out = None
    if arm.safety.keep_out:
        if arm.fk is None:
            raise ValueError(f"{arm.id}: keep_out zones need an fk section")
        urdf = cfg.path(arm.fk.urdf)
        if not urdf.exists():
            raise FileNotFoundError(f"{urdf} missing; run scripts/fetch_so_arm100.sh")
        chain = Chain.from_config(arm.fk, urdf)
        keep_out = KeepOut(chain, arm.safety.keep_out, arm.safety.keep_out_margin_m)
    return SafetyFilter(arm.limits.joints, arm.safety, keep_out)


def make_backend(kind: str, arm: ArmConfig, cfg: Config) -> Backend:
    if kind == "sim":
        from yogurt.arm_driver.sim import SimBackend

        return SimBackend(cfg.path(arm.sim_model), arm.rate_hz, arm.fk)
    if kind == "real":
        from yogurt.arm_driver.real import RealBackend

        return RealBackend(arm, cfg)
    raise ValueError(f"unknown backend {kind!r}")


class Driver:
    def __init__(self, backend: Backend, safety: SafetyFilter, bus: Bus):
        self.backend = backend
        self.safety = safety
        self.bus = bus
        bus.subscribe("arm/target", "system/stop", "system/resume")
        self._status: tuple[Safety, str] | None = None
        self._read_failures = 0
        self._last_reject_log = 0.0

    def tick(self, now: float) -> JointState | None:
        for topic, msg in self.bus.drain():
            if topic == "arm/target":
                assert isinstance(msg, JointTarget)
                reason = self.safety.set_target(msg, now)
                if reason and now - self._last_reject_log > REJECT_LOG_PERIOD_S:
                    log.warning("rejected target from %s: %s", msg.source, reason)
                    self._last_reject_log = now
            elif topic == "system/stop":
                assert isinstance(msg, Stop)
                self.safety.stop(f"{msg.source}: {msg.reason}" if msg.reason else msg.source)
            elif topic == "system/resume":
                self.safety.resume()

        try:
            reading = self.backend.read()
            self._read_failures = 0
        except Exception as e:  # a flaky servo bus must not kill the loop
            self._read_failures += 1
            log.warning("read failed (%d in a row): %s", self._read_failures, e)
            if self._read_failures >= MAX_READ_FAILURES:
                self.safety.fault(f"{self._read_failures} consecutive read failures")
            return None  # servos keep their last goal position

        out = self.safety.step(reading.positions, reading.loads, now)
        self.backend.write(out.command)

        # Log transitions only. The watchdog detail counts up every tick, so ignore it.
        status = (out.status, "" if out.status == Safety.WATCHDOG else out.detail)
        if status != self._status:
            level = logging.INFO if out.status in (Safety.OK, Safety.WATCHDOG) else logging.WARNING
            log.log(level, "safety %s %s", out.status.value, out.detail)
            self._status = status

        state = JointState(
            positions={j: reading.positions[j] for j in JOINTS},
            gripper=reading.positions[GRIPPER],
            commanded=out.command,
            loads=reading.loads,
            temperatures=reading.temperatures,
            safety=out.status,
            detail=out.detail,
            backend=self.backend.name,
        )
        self.bus.publish("arm/state", state)
        return state

    def run(self, rate_hz: float) -> None:
        rate = Rate(rate_hz)
        try:
            while True:
                self.tick(time.monotonic())
                rate.sleep()
        except KeyboardInterrupt:
            log.info("shutting down")
        finally:
            if rate.overruns:
                log.info("%d loop overruns at %.0f Hz", rate.overruns, rate_hz)
            self.backend.close()
            self.bus.close()


def main(argv: list[str] | None = None) -> None:
    def add_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--backend", choices=["sim", "real"], help="override the config's backend")

    args, cfg = cli.start(NAME, doc, argv, add_args)
    arm = cfg.follower
    kind = args.backend or arm.backend
    try:
        safety = make_filter(arm, cfg)
        backend = make_backend(kind, arm, cfg)
        backend.connect()
    except (RuntimeError, OSError, ValueError) as e:
        log.error("cannot start %s backend: %s", kind, e)
        raise SystemExit(2) from e
    log.info(
        "%s backend=%s rate=%.0f Hz keep-out=%s",
        arm.id,
        kind,
        arm.rate_hz,
        [b.name for b in arm.safety.keep_out] or "none",
    )
    Driver(backend, safety, Bus(cfg.bus, NAME)).run(arm.rate_hz)
