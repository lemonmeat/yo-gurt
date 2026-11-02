"""Advisory per-device locks: one process owns each serial port or camera.

arm_driver's real backend, camera_hub, and every scripts/ wrapper that opens hardware take the lock
for the device they use, so e.g. lerobot-teleoperate cannot start while arm_driver drives the
follower. Locks are flock()s, released by the kernel when the holder exits, even on a crash.
"""

import fcntl
import os
import re
import sys
import tempfile
from pathlib import Path


class DeviceBusy(RuntimeError):
    pass


class DeviceLock:
    def __init__(self, device: str | int):
        device = str(device)
        if device.startswith("/dev/"):
            device = os.path.realpath(device)  # /dev/serial/by-id/... and /dev/ttyACM0 share a lock
        self.device = device
        name = re.sub(r"[^A-Za-z0-9_.-]", "_", device).strip("_")
        self.path = Path(tempfile.gettempdir()) / f"yogurt-{name}.lock"
        self._file = None

    def acquire(self) -> "DeviceLock":
        f = open(self.path, "a+")  # noqa: SIM115 - held open for the lock's lifetime
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            f.seek(0)
            owner = f.read().strip() or "another process"
            f.close()
            raise DeviceBusy(f"{self.device} is in use by {owner}") from None
        f.truncate(0)
        f.write(f"pid {os.getpid()} ({' '.join(sys.argv[:2])})")
        f.flush()
        self._file = f
        return self

    def release(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> "DeviceLock":
        return self.acquire()

    def __exit__(self, *exc) -> None:
        self.release()
