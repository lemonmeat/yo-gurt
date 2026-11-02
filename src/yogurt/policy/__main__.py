import argparse
import logging
import time

from yogurt import cli
from yogurt.bus import Bus
from yogurt.messages import GRIPPER, JointState, PolicyCommand, PolicyStatus
from yogurt.policy import __doc__ as doc
from yogurt.policy.replay import Replayer, load_poses
from yogurt.rate import Rate
from yogurt.typed import SchemaError

log = logging.getLogger("yogurt.policy")

NAME = "policy"
ARM_STATE_MAX_AGE_S = 0.5


def main(argv: list[str] | None = None) -> None:
    def add_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--backend", choices=["replay", "act"], help="override the config")

    args, cfg = cli.start(NAME, doc, argv, add_args)
    backend = args.backend or cfg.policy.backend
    poses_dir = cfg.path(cfg.policy.poses_dir)
    bus = Bus(cfg.bus, NAME)
    bus.subscribe("policy/command", "arm/state")
    log.info("backend=%s poses=%s", backend, poses_dir)

    arm: JointState | None = None
    active: Replayer | None = None

    def status(task: str, state: str, detail: str = "") -> None:
        log.log(
            logging.WARNING if state == "failed" else logging.INFO, "%s %s %s", task, state, detail
        )
        bus.publish("policy/status", PolicyStatus(task=task, status=state, detail=detail))

    def start(task: str) -> Replayer | None:
        if backend == "act":
            # TODO(act): see yogurt/policy/act.py.
            status(task, "failed", "ACT backend not implemented yet")
            return None
        if arm is None or time.monotonic() - arm.ts > ARM_STATE_MAX_AGE_S:
            status(task, "failed", "no recent arm/state; is arm_driver running?")
            return None
        try:
            seq = load_poses(poses_dir, task)
        except (OSError, SchemaError, ValueError) as e:
            status(task, "failed", str(e))
            return None
        if seq.source == "sim" and arm.backend == "real":
            status(task, "failed", f"{task}.yaml has sim placeholder poses; teach them first")
            return None
        status(task, "running", f"{len(seq.waypoints)} waypoints")
        return Replayer(seq, arm.positions | {GRIPPER: arm.gripper}, time.monotonic())

    rate = Rate(cfg.policy.rate_hz)
    try:
        while True:
            for topic, msg in bus.drain():
                if topic == "arm/state":
                    arm = msg
                elif topic == "policy/command":
                    assert isinstance(msg, PolicyCommand)
                    if active is not None:
                        status(active.task, "stopped", f"{msg.action} {msg.task} from {msg.source}")
                        active = None
                    if msg.action == "start":
                        active = start(msg.task)
            if active is not None and arm is not None:
                target = active.step(time.monotonic(), arm.positions)
                if target is not None:
                    bus.publish("arm/target", target)
                else:
                    status(active.task, active.status, active.detail)
                    active = None
            rate.sleep()
    except KeyboardInterrupt:
        pass
    finally:
        bus.close()


if __name__ == "__main__":
    main()
