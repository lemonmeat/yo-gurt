import argparse
import logging
import threading
import time

import numpy as np

from yogurt import cli
from yogurt.bus import Bus
from yogurt.camera_hub import __doc__ as doc
from yogurt.camera_hub.ring import FrameWriter, shm_name
from yogurt.config import CameraConfig
from yogurt.devicelock import DeviceLock
from yogurt.messages import FrameInfo
from yogurt.rate import Rate

log = logging.getLogger("yogurt.camera_hub")

NAME = "camera_hub"


class SyntheticSource:
    def __init__(self, cam: CameraConfig):
        self.cam = cam
        self.rate = Rate(cam.fps)
        x = np.linspace(0, 255, cam.width, dtype=np.float32)
        self._base = np.broadcast_to(x[None, :, None], (cam.height, cam.width, 3)).astype(np.uint8)
        self.n = 0

    def read(self) -> np.ndarray:
        self.rate.sleep()
        self.n += 1
        frame = np.roll(self._base, self.n * 4, axis=1).copy()
        row = (self.n * 3) % self.cam.height
        frame[row : row + 10] = 255
        return frame

    def close(self) -> None:
        pass


class OpenCvSource:
    """UNTESTED on the DFRobot FIT0892 / FIT0729. Needs OpenCV (`uv sync --extra hw`)."""

    def __init__(self, cam: CameraConfig):
        try:
            import cv2
        except ImportError as e:
            raise RuntimeError("opencv backend needs OpenCV: uv sync --extra hw") from e
        self.cv2 = cv2
        api = cv2.CAP_V4L2 if isinstance(cam.device, str) else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(cam.device, api)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open camera {cam.device!r}")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cam.fourcc))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam.height)
        self.cap.set(cv2.CAP_PROP_FPS, cam.fps)
        self.cam = cam

    def read(self) -> np.ndarray:
        ok, bgr = self.cap.read()
        if not ok:
            raise RuntimeError(f"camera {self.cam.device!r} returned no frame")
        if bgr.shape[:2] != (self.cam.height, self.cam.width):
            # TODO(hw): if a camera refuses 800x600, pick a mode it supports and update the config.
            bgr = self.cv2.resize(bgr, (self.cam.width, self.cam.height))
        return self.cv2.cvtColor(bgr, self.cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        self.cap.release()


def capture(name: str, source, ring: FrameWriter, bus: Bus, lock: threading.Lock, stop):
    failures = 0
    while not stop.is_set():
        try:
            frame = source.read()
            failures = 0
        except RuntimeError as e:
            failures += 1
            log.warning("%s: %s", name, e)
            time.sleep(min(1.0, 0.1 * failures))
            continue
        ts = time.monotonic()
        seq = ring.write(frame, ts)
        info = FrameInfo(
            camera=name, seq=seq, frame_ts=ts, width=ring.width, height=ring.height,
            shm_name=shm_name(name),
        )  # fmt: skip
        with lock:  # ZeroMQ sockets are not thread-safe
            bus.publish("camera/frame", info)


def main(argv: list[str] | None = None) -> None:
    def add_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--backend", choices=["opencv", "synthetic"], help="override the config")

    args, cfg = cli.start(NAME, doc, argv, add_args)
    backend = args.backend or cfg.camera_hub.backend
    bus = Bus(cfg.bus, NAME)
    bus_lock, stop = threading.Lock(), threading.Event()
    opened, threads = [], []
    try:
        for name, cam in cfg.cameras.items():
            device_lock = None
            if backend == "opencv":
                device_lock = DeviceLock(f"camera-{cam.device}").acquire()
                source = OpenCvSource(cam)
            else:
                source = SyntheticSource(cam)
            ring = FrameWriter(name, cam.width, cam.height, cfg.camera_hub.ring_slots)
            opened.append((source, ring, device_lock))
            t = threading.Thread(
                target=capture, args=(name, source, ring, bus, bus_lock, stop), daemon=True
            )
            t.start()
            threads.append(t)
            log.info(
                "%s: %s %s %dx%d@%d", name, backend, cam.device, cam.width, cam.height, cam.fps
            )
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    except RuntimeError as e:
        log.error("%s", e)
        raise SystemExit(2) from e
    finally:
        stop.set()
        for t in threads:
            t.join(timeout=2)
        for source, ring, device_lock in opened:
            source.close()
            ring.close()
            if device_lock:
                device_lock.release()
        bus.close()


if __name__ == "__main__":
    main()
