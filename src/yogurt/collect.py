"""Data collection, all driven from the UI: audio clips, labelled wrist images, grasp demonstrations.

Everything lands under data/, in the layout the trainers in yogurt.train read back.
"""

import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np

from . import audio, config
from .arm import JOINTS, open_feetech
from .grasp import wrist_image

log = logging.getLogger("yogurt.collect")
DATA = config.ROOT / "data"

# dataset -> the labels it accepts. Folder layout: data/<dataset>/<label>/<session>/<n>.jpg
IMAGE_SETS = {
    "grasp_verify": ["held", "empty"],
    "handoff_grab": ["grabbed", "not_grabbed"],
    "yogurt_detection": ["images"],
}
AUDIO_LABELS = ["positive", "negative", "background"]


def _stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def save_audio_clip(label: str, clip: np.ndarray, phrase: str = "") -> Path:
    """One wake-word clip: data/wake_word/<label>/<time>[_phrase].wav"""
    if label not in AUDIO_LABELS:
        raise ValueError(f"label must be one of {AUDIO_LABELS}")
    slug = "".join(c if c.isalnum() else "-" for c in phrase.lower()).strip("-")[:40]
    name = f"{_stamp()}-{int(time.time() * 1000) % 1000:03d}{'_' + slug if slug else ''}.wav"
    file = DATA / "wake_word" / label / name
    audio.write_wav(file, clip)
    return file


class ImageCapture:
    """Saves wrist-camera frames at a steady rate while it is running. One instance, one burst at a time."""

    def __init__(self):
        self.folder: Path | None = None
        self.saved = 0
        self._next = 0.0
        self._period = 0.5

    def start(self, dataset: str, label: str, per_second: float) -> None:
        if label not in IMAGE_SETS.get(dataset, []):
            raise ValueError(f"unknown dataset/label {dataset}/{label}")
        self.folder = DATA / dataset / label / _stamp()
        self.folder.mkdir(parents=True, exist_ok=True)
        self.saved, self._next, self._period = 0, 0.0, 1.0 / per_second

    def stop(self) -> int:
        self.folder = None
        return self.saved

    def offer(self, frame: np.ndarray) -> None:
        """Called with every new wrist frame; keeps the ones that are due."""
        now = time.monotonic()
        if self.folder is not None and now >= self._next:
            cv2.imwrite(str(self.folder / f"{self.saved:05d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            self.saved, self._next = self.saved + 1, now + self._period


def counts() -> dict:
    """How much of each kind of data exists, for the UI."""
    result = {f"wake_word/{label}": len(list((DATA / "wake_word" / label).glob("*.wav"))) for label in AUDIO_LABELS}
    for dataset, labels in IMAGE_SETS.items():
        for label in labels:
            result[f"{dataset}/{label}"] = len(list((DATA / dataset / label).rglob("*.jpg")))
    result["yogurt_detection/labelled"] = len(list((DATA / "yogurt_detection" / "labels").rglob("*.txt")))
    info = DATA / "grasp_act" / "dataset" / "meta" / "info.json"
    result["grasp_act/episodes"] = json.loads(info.read_text())["total_episodes"] if info.exists() else 0
    return result


# ---- box labels for the yogurt detector -----------------------------------------------------


def _label_file(image: Path) -> Path:
    images = DATA / "yogurt_detection" / "images"
    return (DATA / "yogurt_detection" / "labels" / image.relative_to(images)).with_suffix(".txt")


def next_unlabelled() -> Path | None:
    for image in sorted((DATA / "yogurt_detection" / "images").rglob("*.jpg")):
        if not _label_file(image).exists():
            return image
    return None


def save_box(image_name: str, box: list[float] | None) -> None:
    """box = [x1, y1, x2, y2] as fractions of the picture, or None for 'no yogurt in this picture'.

    Written in YOLO format (class cx cy w h); an empty file marks a picture with no yogurt.
    """
    images = DATA / "yogurt_detection" / "images"
    image = (images / image_name).resolve()
    if not image.is_relative_to(images.resolve()) or not image.exists():
        raise ValueError("no such image")
    label = _label_file(image)
    label.parent.mkdir(parents=True, exist_ok=True)
    if box is None:
        label.write_text("")
    else:
        x1, y1, x2, y2 = (min(max(float(v), 0.0), 1.0) for v in box)
        label.write_text(f"0 {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {abs(x2 - x1):.6f} {abs(y2 - y1):.6f}\n")


# ---- grasp demonstrations -------------------------------------------------------------------


class DemoRecorder:
    """Leader-arm demonstrations, written as a LeRobot dataset that `lerobot-train` reads directly."""

    REPO_ID = "local/yogurt_grasp"

    def __init__(self, cfg):
        self.cfg = cfg
        self.root = DATA / "grasp_act" / "dataset"
        self.leader = None
        self.dataset = None

    @property
    def open(self) -> bool:
        return self.leader is not None

    def start(self) -> None:
        """Connect the leader arm and open (or continue) the dataset."""
        from lerobot.datasets import LeRobotDataset
        from lerobot.utils.feature_utils import hw_to_dataset_features

        if not self.cfg.leader.port:
            raise RuntimeError("no leader arm configured: set leader.port in the hardware config")
        self.leader = open_feetech("leader", self.cfg.leader.port, config.path(self.cfg.leader.calibration))
        try:
            if (self.root / "meta" / "info.json").exists():
                self.dataset = LeRobotDataset.resume(self.REPO_ID, root=self.root)
            else:
                joints = {f"{j}.pos": float for j in JOINTS}
                size = (self.cfg.grasp.image_height, self.cfg.grasp.image_width, 3)
                features = {**hw_to_dataset_features(joints, "action"),
                            **hw_to_dataset_features({**joints, "wrist": size}, "observation")}
                self.dataset = LeRobotDataset.create(self.REPO_ID, self.cfg.grasp.fps, features=features,
                                                     root=self.root, robot_type="so101_follower", use_videos=True)
        except Exception:
            self.stop()
            raise

    def leader_pose(self) -> dict[str, float]:
        return self.leader.bus.sync_read("Present_Position", num_retry=3)

    def add(self, joints: dict[str, float], action: dict[str, float], frame: np.ndarray) -> None:
        self.dataset.add_frame({
            "observation.state": np.array([joints[j] for j in JOINTS], dtype=np.float32),
            "action": np.array([action[j] for j in JOINTS], dtype=np.float32),
            "observation.images.wrist": wrist_image(frame, self.cfg),
            "task": self.cfg.grasp.task,
        })

    def save(self) -> int:
        self.dataset.save_episode()
        return self.dataset.meta.total_episodes

    def discard(self) -> None:
        if self.dataset.has_pending_frames():
            self.dataset.clear_episode_buffer()

    def stop(self) -> None:
        """Close the dataset properly (it is unreadable until finalised) and let go of the leader."""
        if self.dataset is not None:
            self.discard()
            self.dataset.finalize()
            self.dataset = None
        if self.leader is not None:
            try:
                self.leader.bus.disconnect(disable_torque=True)
            except Exception as e:
                log.warning("closing the leader arm: %s", e)
            self.leader = None
