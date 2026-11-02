"""Taught-pose replay: the Phase 1 grasp, and a fallback for every task after that.

A pose file (data/poses/<task>.yaml) is a list of waypoints taught with scripts/teach_poses.py.
Each waypoint gives any subset of joints (the rest keep their previous value) and a duration.
The replayer interpolates linearly from the arm's pose at start, then holds each waypoint until
the body joints are within `settle_tol_deg` (the gripper is not waited on, since a cup may stop
it) or `settle_timeout_s` passes, which fails the task.

Targets still go through arm_driver's safety filter like any other.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

from yogurt.messages import ALL_JOINTS, GRIPPER, JOINTS, JointTarget
from yogurt.typed import from_dict


@dataclass
class Waypoint:
    duration_s: float
    positions: dict[str, float] = field(default_factory=dict)  # degrees; subset of JOINTS
    gripper: float | None = None  # 0 closed .. 100 open

    def __post_init__(self):
        if self.duration_s <= 0:
            raise ValueError("duration_s must be positive")
        unknown = set(self.positions) - set(JOINTS)
        if unknown:
            raise ValueError(f"unknown joints {sorted(unknown)}")


@dataclass
class PoseSequence:
    task: str
    # "real" if taught on the real arm; "sim" placeholders are refused on real hardware.
    source: Literal["sim", "real"]
    waypoints: list[Waypoint]

    def __post_init__(self):
        if not self.waypoints:
            raise ValueError("no waypoints")


def load_poses(poses_dir: Path, task: str) -> PoseSequence:
    path = poses_dir / f"{task}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"no poses for {task!r}: {path} (teach with scripts/teach_poses.py)"
        )
    seq = from_dict(PoseSequence, yaml.safe_load(path.read_text()), path=path.name)
    if seq.task != task:
        raise ValueError(f"{path} is for task {seq.task!r}")
    return seq


def save_poses(path: Path, seq: PoseSequence, header: str = "") -> None:
    data = {
        "task": seq.task,
        "source": seq.source,
        "waypoints": [
            {
                "duration_s": w.duration_s,
                "positions": {k: round(v, 2) for k, v in w.positions.items()},
                **({"gripper": round(w.gripper, 1)} if w.gripper is not None else {}),
            }
            for w in seq.waypoints
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + yaml.safe_dump(data, sort_keys=False))


class Replayer:
    def __init__(
        self,
        seq: PoseSequence,
        start: dict[str, float],
        now: float,
        settle_tol_deg: float = 3.0,
        settle_timeout_s: float = 2.0,
    ):
        self.task = seq.task
        self.tol = settle_tol_deg
        self.settle_timeout = settle_timeout_s
        # Resolve every waypoint to a full pose.
        pose = {j: start[j] for j in ALL_JOINTS}
        self._poses = [pose]
        self._durations = []
        for w in seq.waypoints:
            pose = pose | w.positions | ({GRIPPER: w.gripper} if w.gripper is not None else {})
            self._poses.append(pose)
            self._durations.append(w.duration_s)
        self._index = 0  # current segment: poses[i] -> poses[i + 1]
        self._segment_start = now
        self.status = "running"
        self.detail = ""

    @property
    def waypoint(self) -> int:
        return self._index

    def step(self, now: float, measured: dict[str, float]) -> JointTarget | None:
        """The next target, or None once done or failed (see .status and .detail)."""
        if self.status != "running":
            return None
        a, b = self._poses[self._index], self._poses[self._index + 1]
        elapsed = now - self._segment_start
        duration = self._durations[self._index]
        if elapsed < duration:
            s = elapsed / duration
            pose = {j: a[j] + (b[j] - a[j]) * s for j in ALL_JOINTS}
        else:
            pose = b
            if all(abs(measured[j] - b[j]) <= self.tol for j in JOINTS):
                self._index += 1
                self._segment_start = now
                if self._index == len(self._durations):
                    self.status = "done"
                    return None
            elif elapsed > duration + self.settle_timeout:
                worst = max(JOINTS, key=lambda j: abs(measured[j] - b[j]))
                self.status = "failed"
                self.detail = (
                    f"waypoint {self._index + 1}: {worst} at {measured[worst]:.1f}, "
                    f"wanted {b[worst]:.1f}"
                )
                return None
        return JointTarget(positions={j: pose[j] for j in JOINTS}, gripper=pose[GRIPPER])
