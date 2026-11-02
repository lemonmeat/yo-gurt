"""Sanity checks for LeRobot calibration files (calibration/<id>.json).

LeRobot calibration sets each joint to mid-scale (2047 counts) at the pose you hold when you press
Enter, then records min and max while you sweep each joint. SO-101 joints travel about 200 degrees
(about 2300 counts), so if that pose is near one end of a joint's travel instead of the middle (the
folded rest pose, say), the sweep crosses the encoder's 0/4095 wrap and the recorded range covers
almost a full turn. Degrees then jump by 360 partway through the joint's motion, which makes
teleop and policies command violent reversals.
"""

import json
from pathlib import Path

FULL_TURN_JOINTS = {"wrist_roll"}  # LeRobot sets these to the full 0..4095 on purpose
WRAPPED_SPAN = 3800  # counts; a sane SO-101 joint travels about 1000-2600 counts


def calibration_problems(path: Path) -> list[str]:
    data = json.loads(path.read_text())
    problems = []
    for joint, cal in data.items():
        if joint in FULL_TURN_JOINTS:
            continue
        lo, hi = cal["range_min"], cal["range_max"]
        if hi - lo >= WRAPPED_SPAN:
            problems.append(
                f"{joint}: recorded range {lo}..{hi} covers almost a full turn, so the encoder "
                "wrapped during calibration"
            )
    return problems


MIDDLE_POSE = (
    "every joint about halfway between its two hard stops (the 'middle position' in LeRobot's "
    "SO-101 guide), not the folded rest pose"
)
RECALIBRATE_HINT = (
    f"Recalibrate with {MIDDLE_POSE} when you press Enter: "
    "uv run python scripts/calibrate.py --arm <arm>, then type c at the prompt."
)
