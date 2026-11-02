# Calibration files

`scripts/calibrate.py` writes LeRobot calibration JSONs here as `<id>.json`
(`yogurt_follower.json`, `yogurt_leader.json`, `yogurt_head.json`). They are committed so the Pi
and the laptop use the same calibration: recording, training, and the real backend all depend on
the same zero pose and gripper range.

Recalibrating changes what every joint angle means. Afterwards, re-check joint_limits.yaml and
gripper_map.yaml, and expect datasets recorded with the old calibration to stop matching.
