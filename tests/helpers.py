"""Test helpers shared across test modules."""

import socket
import time

from yogurt.bus import Bus
from yogurt.config import BusConfig


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def free_bus_config() -> BusConfig:
    return BusConfig(pub=f"tcp://127.0.0.1:{free_port()}", sub=f"tcp://127.0.0.1:{free_port()}")


def wait_for(bus: Bus, topic: str, publish=None, timeout: float = 3.0, pred=None):
    """Receive the first `topic` message matching `pred`, calling `publish()` while waiting.

    Re-publishing covers PUB/SUB's slow-joiner window after connect.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if publish:
            publish()
        item = bus.recv(timeout=0.05)
        if item and item[0] == topic and (pred is None or pred(item[1])):
            return item[1]
    raise TimeoutError(f"no {topic} message within {timeout} s")
