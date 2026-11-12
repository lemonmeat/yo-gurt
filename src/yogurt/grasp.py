"""The learned grasp: an ACT policy trained with LeRobot on leader-arm demonstrations.

Input is exactly what the demonstrations recorded (the wrist picture and the six joint positions);
output is the next joint target. The arm's own safety limits still apply to whatever comes out.
"""

from pathlib import Path

import cv2
import numpy as np

from .arm import JOINTS


def wrist_image(frame: np.ndarray, cfg) -> np.ndarray:
    """The wrist frame as demonstrations and the policy see it: RGB, resized to the configured size."""
    small = cv2.resize(frame, (cfg.grasp.image_width, cfg.grasp.image_height), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2RGB)


class GraspPolicy:
    def __init__(self, folder: Path, cfg):
        import torch
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.factory import make_pre_post_processors

        self.cfg = cfg
        self.device = torch.device(cfg.models.act_device)
        self.policy = ACTPolicy.from_pretrained(str(folder))
        self.policy.to(self.device).eval()
        self.preprocess, self.postprocess = make_pre_post_processors(
            self.policy.config, pretrained_path=str(folder),
            preprocessor_overrides={"device_processor": {"device": self.device.type}},
        )

    def reset(self) -> None:
        """Call before each grasp attempt: forget the previous attempt's planned actions."""
        self.policy.reset()

    def act(self, frame: np.ndarray, joints: dict[str, float]) -> dict[str, float]:
        """One control step: wrist frame (BGR, as the camera gives it) + joint positions -> joint targets."""
        from lerobot.common.control_utils import predict_action

        observation = {
            "observation.images.wrist": wrist_image(frame, self.cfg),
            "observation.state": np.array([joints[j] for j in JOINTS], dtype=np.float32),
        }
        action = predict_action(observation, self.policy, self.device, self.preprocess, self.postprocess,
                                use_amp=False, task=self.cfg.grasp.task, robot_type="so101_follower")
        values = np.asarray(action.detach().cpu(), dtype=np.float64).reshape(-1)
        return dict(zip(JOINTS, values.tolist(), strict=True))
