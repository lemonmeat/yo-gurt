"""Calibrate an arm with lerobot-calibrate, using the port and id from the config.

    uv run python scripts/calibrate.py --arm follower|leader [--dry-run]

Torque is off during calibration. LeRobot asks you to put the arm in the middle of its range
(that pose becomes 0 degrees), then to move every joint except wrist_roll through its full range.
Open and close the gripper all the way so the calibration's gripper range matches
gripper_map.yaml. The file lands in calibration/<id>.json; commit it so the Pi uses
the same calibration.
"""

from _lerobot import (
    calibration_path,
    check_port,
    follower_args,
    leader_args,
    lerobot_cli,
    parse,
    parser,
    run,
)

from yogurt.calibration import MIDDLE_POSE, RECALIBRATE_HINT, calibration_problems


def main() -> None:
    p = parser(__doc__)
    p.add_argument("--arm", required=True, choices=["follower", "leader"])
    args, extra, cfg = parse(p)

    if args.arm == "leader":
        device, device_args = cfg.leader, leader_args(cfg.leader, cfg)
    else:
        device, device_args = cfg.follower, follower_args(cfg.follower, cfg)
    if not args.dry_run:
        check_port(device.port, args.arm, cfg)

    print(f"Calibrating {args.arm} ({device.id} on {device.port}). Torque stays off.")
    print(f"When LeRobot asks for the middle of the range, put {MIDDLE_POSE}.")
    code = run([lerobot_cli("calibrate"), *device_args, *extra], [device.port], args.dry_run)
    path = calibration_path(device, cfg)
    if code == 0 and not args.dry_run and path.exists():
        problems = calibration_problems(path)
        if problems:
            print(f"\nWARNING: {path.name} looks wrong:\n  " + "\n  ".join(problems))
            print(RECALIBRATE_HINT.replace("<arm>", args.arm))
            code = 1
        else:
            print(f"{path.name} looks sane. Commit it so the Pi uses the same calibration.")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
