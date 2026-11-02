import threading

import pytest
import zmq

from tests.helpers import free_bus_config
from yogurt.bus import run_broker
from yogurt.config import load_config


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def broker():
    """A broker on free ports in a background thread; yields its BusConfig."""
    bus_cfg = free_bus_config()
    ctx = zmq.Context()
    thread = threading.Thread(target=run_broker, args=(bus_cfg, ctx), daemon=True)
    thread.start()
    yield bus_cfg
    ctx.term()
    thread.join(timeout=2)
