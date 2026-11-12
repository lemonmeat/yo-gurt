"""Cameras: one background thread per camera that always holds the newest frame."""

import glob
import json
import logging
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("yogurt.cameras")
ROTATIONS = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def _key(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def find_device(device) -> int | str:
    """Turn a camera name into what OpenCV needs. Names beat indices: indices move when USB is replugged."""
    if isinstance(device, int) or str(device).startswith("/dev/"):
        return device
    if sys.platform == "darwin":
        listing = subprocess.run(["system_profiler", "-json", "SPCameraDataType"], capture_output=True, text=True)
        cameras = json.loads(listing.stdout).get("SPCameraDataType", [])
        cameras.sort(key=lambda c: c.get("spcamera_unique-id", ""))  # the order OpenCV numbers them in
        names = [c["_name"] for c in cameras]
        matches = [i for i, name in enumerate(names) if _key(device) in _key(name)]
    else:
        names = sorted(glob.glob("/dev/v4l/by-id/*-video-index0"))
        matches = [name for name in names if _key(device) in _key(Path(name).name)]
    if len(matches) != 1:
        problem = "not found" if not matches else "matches more than one camera"
        raise RuntimeError(f"camera '{device}' {problem}; cameras present: {names or 'none'}")
    return matches[0]


class Camera:
    def __init__(self, name: str, cfg):
        self.name, self.cfg = name, cfg
        self.frame: np.ndarray | None = None
        self.stamp = 0.0                 # time.monotonic() of the newest frame
        self.error: str | None = "not opened yet"
        self._running = False
        self._thread: threading.Thread | None = None

    def open(self) -> None:
        self.close()
        self._running = True
        self._thread = threading.Thread(target=self._loop, name=f"camera-{self.name}", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        self._thread, self.frame, self.error = None, None, "closed"

    def read(self, max_age_s: float = 1.0) -> np.ndarray | None:
        """Newest frame (BGR), or None if the camera has gone quiet."""
        if self.frame is None or time.monotonic() - self.stamp > max_age_s:
            return None
        return self.frame

    def _connect(self) -> cv2.VideoCapture:
        device = find_device(self.cfg.device)
        cap = cv2.VideoCapture(device, cv2.CAP_V4L2) if sys.platform == "linux" else cv2.VideoCapture(device)
        if sys.platform == "linux":  # USB cameras only reach 30 fps at 720p when sending MJPEG
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
        cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            raise RuntimeError(f"could not open camera '{self.cfg.device}'")
        return cap

    def _loop(self) -> None:
        cap = None
        while self._running:
            try:
                if cap is None:
                    cap = self._connect()
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("camera stopped sending frames")
                if self.cfg.rotate:
                    frame = cv2.rotate(frame, ROTATIONS[self.cfg.rotate])
                self.frame, self.stamp, self.error = frame, time.monotonic(), None
            except Exception as e:
                if self.error != str(e):
                    log.warning("%s camera: %s", self.name, e)
                self.error = str(e)
                if cap is not None:
                    cap.release()
                    cap = None
                for _ in range(20):  # retry in 2 s, but stop promptly when closed
                    if self._running:
                        time.sleep(0.1)
        if cap is not None:
            cap.release()
