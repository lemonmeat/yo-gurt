"""Shared-memory frame rings: camera_hub writes, any process reads, nobody opens a camera twice.

Layout of segment "yogurt_cam_<camera>": a header [latest seq: u64], then `slots` slots of
[seq: u64][capture ts: f64][H x W x 3 uint8 RGB]. The writer marks a slot invalid while it copies,
and a reader checks the slot's seq before and after its own copy, so a torn read returns None
instead of a mixed frame. Readers see the newest frame only; they never block the writer.
"""

import contextlib
from multiprocessing import resource_tracker, shared_memory

import numpy as np

HEADER = 8
SLOT_HEADER = 16
WRITING = np.uint64(2**64 - 1)


def shm_name(camera: str) -> str:
    return f"yogurt_cam_{camera}"


def _size(width: int, height: int, slots: int) -> int:
    return HEADER + slots * (SLOT_HEADER + width * height * 3)


class _Ring:
    def __init__(self, shm: shared_memory.SharedMemory, width: int, height: int, slots: int):
        self.shm, self.width, self.height, self.slots = shm, width, height, slots
        frame = width * height * 3
        buf = shm.buf
        self._latest = np.ndarray((1,), np.uint64, buf, 0)
        self._seq, self._ts, self._frames = [], [], []
        for i in range(slots):
            off = HEADER + i * (SLOT_HEADER + frame)
            self._seq.append(np.ndarray((1,), np.uint64, buf, off))
            self._ts.append(np.ndarray((1,), np.float64, buf, off + 8))
            self._frames.append(np.ndarray((height, width, 3), np.uint8, buf, off + SLOT_HEADER))

    def _release_views(self) -> None:
        self._latest = None
        self._seq, self._ts, self._frames = [], [], []


class FrameWriter(_Ring):
    def __init__(self, camera: str, width: int, height: int, slots: int = 4):
        name = shm_name(camera)
        size = _size(width, height, slots)
        try:
            shm = shared_memory.SharedMemory(name, create=True, size=size)
        except FileExistsError:  # left behind by a crashed camera_hub
            stale = shared_memory.SharedMemory(name)
            stale.close()
            stale.unlink()
            shm = shared_memory.SharedMemory(name, create=True, size=size)
        super().__init__(shm, width, height, slots)
        self._latest[0] = 0
        for s in self._seq:
            s[0] = WRITING
        self.seq = 0

    def write(self, frame: np.ndarray, ts: float) -> int:
        if frame.shape != (self.height, self.width, 3):
            raise ValueError(f"frame {frame.shape}, ring expects {(self.height, self.width, 3)}")
        self.seq += 1
        i = self.seq % self.slots
        self._seq[i][0] = WRITING
        np.copyto(self._frames[i], frame)
        self._ts[i][0] = ts
        self._seq[i][0] = self.seq
        self._latest[0] = self.seq
        return self.seq

    def close(self) -> None:
        self._release_views()
        self.shm.close()
        with contextlib.suppress(FileNotFoundError):
            self.shm.unlink()


class FrameReader(_Ring):
    def __init__(self, camera: str, width: int, height: int, slots: int = 4):
        shm = shared_memory.SharedMemory(shm_name(camera))
        # Python 3.12's resource tracker would unlink the writer's segment when we exit.
        resource_tracker.unregister(shm._name, "shared_memory")
        if shm.size < _size(width, height, slots):
            shm.close()
            raise ValueError(f"{shm_name(camera)} is smaller than {width}x{height} x {slots}")
        super().__init__(shm, width, height, slots)

    def latest(self) -> tuple[int, float, np.ndarray] | None:
        """(seq, capture ts, RGB frame copy) of the newest frame, or None if none or torn."""
        seq = int(self._latest[0])
        if seq == 0:
            return None
        i = seq % self.slots
        if int(self._seq[i][0]) != seq:
            return None
        frame = self._frames[i].copy()
        ts = float(self._ts[i][0])
        if int(self._seq[i][0]) != seq:
            return None
        return seq, ts, frame

    def close(self) -> None:
        self._release_views()
        self.shm.close()
