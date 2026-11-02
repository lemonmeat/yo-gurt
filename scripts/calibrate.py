"""Calibrate an arm with lerobot-calibrate, using the port and id from the config.

    uv run python scripts/calibrate.py --arm follower|leader|head [--dry-run]

Torque is off during calibration. LeRobot asks you to put the arm in the middle of its range
(that pose becomes 0 degrees), then to move every joint except wrist_roll through its full range.
For the follower, open and close the Robonine gripper all the way so the calibration's gripper
range matches gripper_map.yaml. The file lands in calibration/<id>.json; commit it so the Pi uses
the same calibration.
"""

from _lerobot import (
    check_port,
    follower_args,
    leader_args,
    lerobot_cli,
    parse,
    parser,
    run,
)


def main() -> None:
    p = parser(__doc__)
    p.add_argument("--arm", required=True, choices=["follower", "leader", "head"])
    args, extra, cfg = parse(p)

    if args.arm == "leader":
        device, device_args = cfg.leader, leader_args(cfg.leader, cfg)
    else:
        arm = getattr(cfg, args.arm)
        device, device_args = arm, follower_args(arm, cfg)
    if not args.dry_run:
        check_port(device.port, args.arm, cfg)

    print(f"Calibrating {args.arm} ({device.id} on {device.port}). Torque stays off.")
    run([lerobot_cli("calibrate"), *device_args, *extra], [device.port], args.dry_run)


if __name__ == "__main__":
    main()
