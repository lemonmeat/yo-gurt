"""ACT policy backend (Phase 2). Not implemented yet.

TODO(act): implement once the first checkpoint exists (docs/TASKS.md T7):
  - Load: lerobot.policies.act.modeling_act.ACTPolicy.from_pretrained(cfg.policy.act.checkpoint),
    with the dataset's normalization stats (they ship inside the checkpoint's pretrained_model/).
  - Observation per tick: arm/state positions as "observation.state" (same joint order and units
    as the dataset: LeRobot degrees, gripper on the calibration's 0..100) and the latest wrist and
    scene frames from camera_hub (yogurt.camera_hub.ring.FrameReader), RGB, at the dataset's
    resolution, as "observation.images.wrist" / "observation.images.scene".
  - select_action() returns one action per call from its chunk; publish it as arm/target.
  - Remote GPU: if cfg.policy.act.server is set, use LeRobot's async inference instead
    (lerobot.async_inference.robot_client / policy_server) and adapt its action stream here.
  - Never talk to the follower directly: arm/target is the only output.
"""


class ActPolicy:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("ACT backend not implemented yet (docs/TASKS.md T7)")
