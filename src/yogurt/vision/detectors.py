"""Detector backends behind one interface: YOLO on the laptop, Hailo on the Pi."""

from typing import Protocol

import numpy as np

from yogurt.config import Config
from yogurt.messages import Detection


class Detector(Protocol):
    def detect(self, rgb: np.ndarray) -> list[Detection]:
        """Detections with boxes normalized to 0..1. Labels: cup, hand, face, person."""
        ...


class NullDetector:
    """Sees nothing. Lets the rest of the pipeline run before a model is set up."""

    def detect(self, rgb: np.ndarray) -> list[Detection]:
        return []


class YoloDetector:
    # TODO(vision): ultralytics is not a dependency yet; ask before adding it (likely a `vision`
    # extra). Then: model = ultralytics.YOLO(cfg.path(cfg.vision.model)); map COCO "cup" (41) and
    # "person" (0); hands and faces need a second model (e.g. a hand/face YOLO fine-tune).
    def __init__(self, cfg: Config):
        raise NotImplementedError("YOLO backend: add ultralytics first (see detectors.py)")

    def detect(self, rgb: np.ndarray) -> list[Detection]:
        raise NotImplementedError


class HailoDetector:
    # TODO(vision, Pi): run a compiled .hef on the AI HAT+ through the Hailo Python API once the
    # hardware arrives; same labels and normalized boxes as YoloDetector.
    def __init__(self, cfg: Config):
        raise NotImplementedError("Hailo backend: needs the AI HAT+ (see detectors.py)")

    def detect(self, rgb: np.ndarray) -> list[Detection]:
        raise NotImplementedError


def make_detector(cfg: Config) -> Detector:
    backend = cfg.vision.backend
    if backend == "none":
        return NullDetector()
    if backend == "yolo":
        return YoloDetector(cfg)
    if backend == "hailo":
        return HailoDetector(cfg)
    raise ValueError(f"unknown vision backend {backend!r}")
