"""Teleoperate the follower with the leader arm, through lerobot-teleoperate.

    uv run python scripts/teleoperate.py [--fps 30] [--cameras wrist,scene] [--display]
                                         [--dry-run] [-- extra lerobot args]

SUPERVISED EXCEPTION to the arm_driver rule: LeRobot drives the follower directly, so there are
no joint limits, keep-out zone, or watchdog, only LeRobot's max_relative_target clamp (from the
config) on each step. A person holds the leader the whole time. Autonomous motion always goes
through arm_driver.

Before starting:
  - Follower on its 12 V supply, leader on its 5 V supply. Never swap them.
  - Both arms calibrated (scripts/calibrate.py).
  - Put the leader in roughly the follower's current pose: the follower jumps to the leader's
    pose as fast as max_relative_target allows.
  - On exit the follower's torque turns off (torque_off_on_exit): support it or rest it first.
"""

from _lerobot import (
    cameras_arg,
    check_calibrated,
    check_port,
    fail,
    follower_args,
    leader_args,
    lerobot_cli,
    parse,
    parser,
    run,
)


def main() -> None:
    p = parser(__doc__)
    p.add_argument("--fps", type=int, default=30, help="control rate (default 30)")
    p.add_argument("--cameras", default="", help="comma-separated camera names to show")
    p.add_argument("--display", action="store_true", help="show data in rerun")
    args, extra, cfg = parse(p)
    follower, leader = cfg.follower, cfg.leader
    if follower.max_relative_target_deg is None:
        fail("set follower.max_relative_target_deg in the config; teleop needs a step clamp")
    if follower.port == leader.port:
        fail("follower and leader have the same port in the config")

    if not args.dry_run:
        check_port(follower.port, "follower", cfg)
        check_port(leader.port, "leader", cfg)
        check_calibrated(follower, cfg, "follower")
        check_calibrated(leader, cfg, "leader")

    cameras = [c for c in args.cameras.split(",") if c]
    cmd = [
        lerobot_cli("teleoperate"),
        *follower_args(follower, cfg),
        f"--robot.max_relative_target={follower.max_relative_target_deg}",
        f"--robot.disable_torque_on_disconnect={str(follower.torque_off_on_exit).lower()}",
        *leader_args(leader, cfg),
        f"--fps={args.fps}",
    ]
    if cameras:
        cmd.append(cameras_arg(cfg, cameras))
    if args.display:
        cmd.append("--display_data=true")
    cmd += extra

    step = follower.max_relative_target_deg
    print(
        "SUPERVISED TELEOP: no joint limits, keep-out, or watchdog. "
        f"Max {step:g} deg per step at {args.fps} Hz ({step * args.fps:g} deg/s).\n"
        "Leader on 5 V, follower on 12 V. Match the leader to the follower's pose first.\n"
        "Ctrl-C stops; the follower then goes limp, so support it."
    )
    devices = [follower.port, leader.port] + [f"camera-{cfg.cameras[c].device}" for c in cameras]
    raise SystemExit(run(cmd, devices, args.dry_run))


if __name__ == "__main__":
    main()
