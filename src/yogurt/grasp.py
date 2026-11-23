"""The learned grasp: an ACT policy trained with LeRobot on leader-arm demonstrations.

Input is exactly what the demonstrations recorded (the wrist picture and the six joint positions);
output is the next joint target. The arm's own safety limits still apply to whatever comes out.
The camera name and picture size come from the trained policy, so a policy trained on a dataset
recorded outside the Lab (e.g. with `lerobot-record`, camera "Wrist" at 640x480) runs too.
"""

from pathlib import Path

import cv2
import numpy as np

from .arm import JOINTS


def wrist_image(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """The wrist frame as demonstrations and the policy see it: RGB at width x height. A frame of another
    shape is centre-cropped to it first (a 16:9 frame for a 4:3 policy loses its sides, not its proportions),
    as a camera asked for that shape would do."""
    h, w = frame.shape[:2]
    if w * height > h * width:      # too wide: trim the sides
        keep = round(h * width / height)
        frame = frame[:, (w - keep) // 2:(w - keep) // 2 + keep]
    elif w * height < h * width:    # too tall: trim top and bottom
        keep = round(w * height / width)
        frame = frame[(h - keep) // 2:(h - keep) // 2 + keep]
    small = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2RGB)


class GraspPolicy:
    def __init__(self, folder: Path, cfg):
        import torch
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.factory import make_pre_post_processors

        self.cfg = cfg
        self.device = torch.device(cfg.models.act_device)
        self.policy = ACTPolicy.from_pretrained(str(folder))
        # Re-plan from a fresh picture every n_action_steps instead of the trained default (often the
        # whole chunk): the rest of each predicted chunk is thrown away.
        steps, chunk = int(cfg.grasp.n_action_steps), self.policy.config.chunk_size
        if not 1 <= steps <= chunk:
            raise ValueError(f"grasp.n_action_steps is {steps}: it must be between 1 and the policy's chunk_size {chunk}")
        self.policy.config.n_action_steps = steps
        self.policy.reset()   # the action queue's length is set from n_action_steps here
        self.policy.to(self.device).eval()
        self.preprocess, self.postprocess = make_pre_post_processors(
            self.policy.config, pretrained_path=str(folder),
            preprocessor_overrides={"device_processor": {"device": self.device.type}},
        )
        cameras = self.policy.config.image_features
        if len(cameras) != 1:
            raise ValueError(f"the grasp policy must take one camera (the wrist), not {list(cameras) or 'none'}")
        (self.camera, feature), = cameras.items()   # e.g. "observation.images.wrist"
        _, self.image_height, self.image_width = feature.shape   # channels first

    def reset(self) -> None:
        """Call before each grasp attempt: forget the previous attempt's planned actions."""
        self.policy.reset()

    def act(self, frame: np.ndarray, joints: dict[str, float]) -> dict[str, float]:
        """One control step: wrist frame (BGR, as the camera gives it) + joint positions -> joint targets."""
        from lerobot.common.control_utils import predict_action

        observation = {
            self.camera: wrist_image(frame, self.image_width, self.image_height),
            "observation.state": np.array([joints[j] for j in JOINTS], dtype=np.float32),
        }
        action = predict_action(observation, self.policy, self.device, self.preprocess, self.postprocess,
                                use_amp=False, task=self.cfg.grasp.task, robot_type="so101_follower")
        values = np.asarray(action.detach().cpu(), dtype=np.float64).reshape(-1)
        return dict(zip(JOINTS, values.tolist(), strict=True))
