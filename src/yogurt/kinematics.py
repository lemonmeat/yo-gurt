"""Forward kinematics and keep-out zones for the SO-101, from its URDF.

Pure numpy, so the safety filter needs neither MuJoCo nor LeRobot. tests/test_kinematics.py checks
the result against MuJoCo's model of the same arm.

Frames: the follower base frame of the SO-ARM100 model: +x forward, +y left, +z up, origin under
the shoulder at the bottom of the base, in metres.
"""

import math
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from yogurt.config import Box, FkConfig
from yogurt.messages import JOINTS

TIP_LINK = "gripper_frame_link"


def _rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = axis
    c, s, t = math.cos(angle), math.sin(angle), 1 - math.cos(angle)
    return np.array(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ]
    )


def _floats(text: str | None, default: str) -> list[float]:
    return [float(v) for v in (text or default).split()]


@dataclass
class _Joint:
    name: str
    kind: str  # "revolute" | "fixed"
    origin: np.ndarray  # 4x4, parent link -> joint frame at zero angle
    axis: np.ndarray


class Chain:
    """Serial chain from the URDF root to TIP_LINK, plus a fixed tool offset."""

    def __init__(
        self,
        urdf: str | Path,
        tool_offset: list[float] | None = None,
        joint_sign: Mapping[str, float] | None = None,
        joint_offset_deg: Mapping[str, float] | None = None,
    ):
        root = ET.parse(urdf).getroot()
        by_child = {j.find("child").get("link"): j for j in root.findall("joint")}
        joints = []
        link = TIP_LINK
        while link in by_child:
            j = by_child[link]
            origin, axis_el = j.find("origin"), j.find("axis")
            tf = np.eye(4)
            tf[:3, :3] = _rpy(*_floats(origin.get("rpy") if origin is not None else None, "0 0 0"))
            tf[:3, 3] = _floats(origin.get("xyz") if origin is not None else None, "0 0 0")
            axis = np.array(_floats(axis_el.get("xyz") if axis_el is not None else None, "1 0 0"))
            if j.get("type") != "fixed":
                axis = axis / np.linalg.norm(axis)
            joints.append(_Joint(j.get("name"), j.get("type"), tf, axis))
            link = j.find("parent").get("link")
        if not joints:
            raise ValueError(f"{urdf}: no chain ends at {TIP_LINK}")
        self.joints = joints[::-1]
        moving = [j.name for j in self.joints if j.kind != "fixed"]
        if moving != list(JOINTS):
            raise ValueError(f"{urdf}: expected joints {list(JOINTS)}, chain has {moving}")
        self.tool_offset = np.array(tool_offset or [0.0, 0.0, 0.0], dtype=float)
        self.sign = {name: float((joint_sign or {}).get(name, 1.0)) for name in JOINTS}
        self.offset = {name: float((joint_offset_deg or {}).get(name, 0.0)) for name in JOINTS}

    @classmethod
    def from_config(cls, fk: FkConfig, urdf_path: Path) -> "Chain":
        return cls(urdf_path, fk.tool_offset, fk.joint_sign, fk.joint_offset_deg)

    def points(self, positions_deg: Mapping[str, float]) -> np.ndarray:
        """Base-frame positions (N x 3) of every moving joint, the gripper frame, and the tool tip.

        The first joint's origin is skipped: it sits on the pan axis and cannot move.
        """
        tf = np.eye(4)
        out = []
        for i, j in enumerate(self.joints):
            tf = tf @ j.origin
            if i > 0:
                out.append(tf[:3, 3].copy())
            if j.kind != "fixed":
                angle = math.radians(
                    self.sign[j.name] * (positions_deg[j.name] + self.offset[j.name])
                )
                rot = np.eye(4)
                rot[:3, :3] = _axis_angle(j.axis, angle)
                tf = tf @ rot
        out.append((tf @ np.append(self.tool_offset, 1.0))[:3])
        return np.array(out)

    def tip(self, positions_deg: Mapping[str, float]) -> np.ndarray:
        return self.points(positions_deg)[-1]


def box_signed_distance(points: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Signed distance from each point to an axis-aligned box (negative inside)."""
    center, half = (lo + hi) / 2, (hi - lo) / 2
    q = np.abs(points - center) - half
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
    inside = np.minimum(q.max(axis=-1), 0.0)
    return outside + inside


class KeepOut:
    """Clearance between the arm (as a sampled polyline through its joints) and keep-out boxes."""

    def __init__(self, chain: Chain, boxes: list[Box], margin_m: float, samples_per_link: int = 4):
        self.chain = chain
        self.boxes = [(b.name, np.array(b.min, float), np.array(b.max, float)) for b in boxes]
        self.margin = margin_m
        t = np.linspace(0.0, 1.0, samples_per_link, endpoint=False)
        self._t = t[None, :, None]

    def _samples(self, positions_deg: Mapping[str, float]) -> np.ndarray:
        p = self.chain.points(positions_deg)
        a, b = p[:-1, None, :], p[1:, None, :]
        return np.concatenate([(a + (b - a) * self._t).reshape(-1, 3), p[-1:]])

    def clearance(self, positions_deg: Mapping[str, float]) -> tuple[float, str]:
        """(metres to the nearest margin-inflated box, its name). Negative means inside."""
        if not self.boxes:
            return math.inf, ""
        pts = self._samples(positions_deg)
        best, name = math.inf, ""
        for box_name, lo, hi in self.boxes:
            d = float(box_signed_distance(pts, lo, hi).min()) - self.margin
            if d < best:
                best, name = d, box_name
        return best, name
