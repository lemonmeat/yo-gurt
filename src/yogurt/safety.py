"""The safety filter between every target and the servos. Pure logic; the drivers own the I/O.

Each control tick, step() turns the latest target into the next command:

1. Latches: a system/stop (STOPPED) or a driver fault such as sustained overload (FAULT) holds
   the last command until resume(). On overload the hold pose becomes the measured pose, so the
   arm stops pushing.
2. Watchdog: no fresh target within `watchdog_s` holds the last command.
3. Joint limits: targets are clamped to joint_limits.yaml.
4. Velocity cap: each joint moves at most max_velocity * dt per tick.
5. Tracking cap: the command stays within max_tracking_error of the measured position.
6. Keep-out: a step that brings the arm closer to a keep-out box than its margin is rejected
   (hold), unless it increases clearance, so an arm that starts inside can still back out.

Positions are degrees for body joints and 0..100 for the gripper, keyed by joint name.
"""

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from yogurt.config import JointLimit, SafetyConfig
from yogurt.messages import ALL_JOINTS, GRIPPER, JointTarget, Safety

log = logging.getLogger(__name__)

MAX_DT = 0.1  # a stalled loop must not turn into one big step


class Clearance(Protocol):
    def clearance(self, positions_deg: Mapping[str, float]) -> tuple[float, str]: ...


@dataclass
class FilterOutput:
    command: dict[str, float]
    status: Safety
    detail: str = ""


class SafetyFilter:
    def __init__(
        self,
        limits: Mapping[str, JointLimit],
        cfg: SafetyConfig,
        keep_out: Clearance | None = None,
    ):
        missing = set(ALL_JOINTS) - set(limits)
        if missing:
            raise ValueError(f"no limits for {sorted(missing)}")
        self.limits = dict(limits)
        self.cfg = cfg
        self.keep_out = keep_out
        self._cmd: dict[str, float] | None = None
        self._target: dict[str, float] = {}
        self._target_time: float | None = None
        self._last_step: float | None = None
        self._latch: Safety | None = None
        self._latch_reason = ""
        self._overload_since: dict[str, float] = {}

    # Inputs -------------------------------------------------------------------------------

    def set_target(self, target: JointTarget, now: float) -> str | None:
        """Accept a target. Returns why it was rejected, or None."""
        values = dict(target.positions)
        if target.gripper is not None:
            values[GRIPPER] = target.gripper
        unknown = set(values) - set(ALL_JOINTS)
        if unknown:
            return f"unknown joints {sorted(unknown)}"
        if not all(math.isfinite(v) for v in values.values()):
            return "non-finite value"
        if now - target.ts > self.cfg.watchdog_s:
            return f"stale by {now - target.ts:.3f} s"
        if self._latch is not None:
            return f"latched {self._latch.value}"
        for name, value in values.items():
            lim = self.limits[name]
            self._target[name] = min(lim.max, max(lim.min, value))
        self._target_time = now
        return None

    def stop(self, reason: str) -> None:
        # Keep the first reason (and never downgrade a FAULT): repeats carry less information.
        if self._latch is None:
            log.warning("STOP: %s", reason)
            self._latch, self._latch_reason = Safety.STOPPED, reason

    def fault(self, reason: str) -> None:
        if self._latch is not Safety.FAULT:
            log.error("FAULT: %s", reason)
        self._latch, self._latch_reason = Safety.FAULT, reason

    def resume(self) -> None:
        """Clear a latch. The arm then holds until a fresh target arrives."""
        if self._latch is not None:
            log.warning("resume after %s: %s", self._latch.value, self._latch_reason)
        self._latch, self._latch_reason = None, ""
        self._target_time = None
        if self._cmd is not None:
            self._target = dict(self._cmd)

    @property
    def latched(self) -> Safety | None:
        return self._latch

    # Control tick -------------------------------------------------------------------------

    def step(
        self, measured: Mapping[str, float], loads: Mapping[str, float], now: float
    ) -> FilterOutput:
        if self._cmd is None:
            # First tick: hold wherever the arm is.
            self._cmd = {name: float(measured[name]) for name in ALL_JOINTS}
            self._target = dict(self._cmd)
        dt = 0.0 if self._last_step is None else min(MAX_DT, max(0.0, now - self._last_step))
        self._last_step = now

        self._check_overload(measured, loads, now)
        if self._latch is not None:
            return FilterOutput(dict(self._cmd), self._latch, self._latch_reason)

        if self._target_time is None:
            return FilterOutput(dict(self._cmd), Safety.WATCHDOG, "no target yet")
        age = now - self._target_time
        if age > self.cfg.watchdog_s:
            return FilterOutput(dict(self._cmd), Safety.WATCHDOG, f"no target for {age:.2f} s")

        nxt = {}
        for name in ALL_JOINTS:
            cur, goal = self._cmd[name], self._target[name]
            step = self.limits[name].max_velocity * dt
            value = cur + min(step, max(-step, goal - cur))
            track = (
                self.cfg.max_gripper_tracking_error
                if name == GRIPPER
                else self.cfg.max_tracking_error_deg
            )
            lo, hi = measured[name] - track, measured[name] + track
            # Never pull the command further from the measured position than it already is.
            value = min(max(value, min(lo, cur)), max(hi, cur))
            nxt[name] = value

        if self.keep_out is not None:
            after, zone = self.keep_out.clearance(nxt)
            if after < 0:
                before, _ = self.keep_out.clearance(self._cmd)
                if after < before:
                    detail = f"{zone}: clearance {after * 100:.1f} cm"
                    return FilterOutput(dict(self._cmd), Safety.KEEP_OUT, detail)

        self._cmd = nxt
        return FilterOutput(dict(nxt), Safety.OK)

    def _check_overload(self, measured, loads: Mapping[str, float], now: float) -> None:
        for name, limit in self.cfg.max_load_pct.items():
            load = loads.get(name)
            if load is None or abs(load) <= limit:
                self._overload_since.pop(name, None)
                continue
            since = self._overload_since.setdefault(name, now)
            if now - since >= self.cfg.load_trip_s and self._latch is not Safety.FAULT:
                self.fault(f"overload: {name} at {load:.0f}% for {now - since:.2f} s")
                # Stop pushing: hold where the arm actually is.
                self._cmd = {n: float(measured[n]) for n in ALL_JOINTS}
                self._target = dict(self._cmd)
