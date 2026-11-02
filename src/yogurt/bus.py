"""ZeroMQ pub/sub helpers.

All traffic goes through a tiny XSUB/XPUB broker (`python -m yogurt.broker`): every process
connects its PUB socket to `bus.pub` and its SUB socket to `bus.sub`, so any process (or a test, or
a debugging script) can publish without knowing who listens. If the broker dies, drivers stop
receiving targets and their watchdogs hold the pose.

PUB/SUB drops messages sent before a subscription has propagated (~tens of ms after connect), so
anything safety-relevant is published repeatedly (targets stream, SAFE_STOP re-sends system/stop).
"""

import dataclasses
import logging
import time

import msgpack
import zmq

from yogurt.config import BusConfig
from yogurt.messages import Message, check_topic, decode, encode
from yogurt.typed import SchemaError

log = logging.getLogger(__name__)


class Bus:
    def __init__(self, cfg: BusConfig, source: str, ctx: zmq.Context | None = None):
        self.source = source
        self._ctx = ctx or zmq.Context.instance()
        self._pub = self._ctx.socket(zmq.PUB)
        self._pub.setsockopt(zmq.LINGER, 0)
        self._pub.connect(cfg.pub)
        self._sub = self._ctx.socket(zmq.SUB)
        self._sub.setsockopt(zmq.LINGER, 0)
        self._sub.connect(cfg.sub)
        self._topics: set[str] = set()
        self._all = False

    def subscribe(self, *topics: str) -> None:
        for topic in topics:
            check_topic(topic)
            self._sub.setsockopt(zmq.SUBSCRIBE, topic.encode())
            self._topics.add(topic)

    def subscribe_all(self) -> None:
        self._sub.setsockopt(zmq.SUBSCRIBE, b"")
        self._all = True

    def publish(self, topic: str, msg: Message) -> None:
        if not msg.source:
            msg = dataclasses.replace(msg, source=self.source)
        self._pub.send_multipart([topic.encode(), encode(topic, msg)])

    def recv_raw(self, timeout: float | None = 0.0) -> tuple[str, bytes] | None:
        """Next (topic, payload), waiting up to `timeout` seconds (None waits forever)."""
        ms = None if timeout is None else int(timeout * 1000)
        if not self._sub.poll(ms, zmq.POLLIN):
            return None
        frames = self._sub.recv_multipart()
        if len(frames) != 2:
            log.warning("dropping malformed message with %d frames", len(frames))
            return None
        return frames[0].decode(errors="replace"), frames[1]

    def recv(self, timeout: float | None = 0.0) -> tuple[str, Message] | None:
        """Next decoded (topic, message). Undecodable messages are logged and skipped."""
        raw = self.recv_raw(timeout)
        if raw is None:
            return None
        topic, payload = raw
        # SUBSCRIBE filters by prefix; keep only exact topic matches.
        if not self._all and topic not in self._topics:
            return None
        try:
            return topic, decode(topic, payload)
        except (KeyError, SchemaError, ValueError, msgpack.UnpackException) as e:
            log.warning("dropping bad %s message: %s", topic, e)
            return None

    def drain(self, max_messages: int = 1000) -> list[tuple[str, Message]]:
        """Every message already queued, without waiting."""
        out = []
        for _ in range(max_messages):
            if not self._sub.poll(0, zmq.POLLIN):
                break
            item = self.recv(0)
            if item is not None:
                out.append(item)
        return out

    def settle(self, seconds: float = 0.3) -> None:
        """Wait for fresh connections before a one-shot publish (see the module docstring)."""
        time.sleep(seconds)

    def close(self) -> None:
        self._pub.close(0)
        self._sub.close(0)

    def __enter__(self) -> "Bus":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def run_broker(cfg: BusConfig, ctx: zmq.Context | None = None) -> None:
    """Forward everything from bus.pub to bus.sub until the context is terminated."""
    ctx = ctx or zmq.Context.instance()
    frontend = ctx.socket(zmq.XSUB)
    backend = ctx.socket(zmq.XPUB)
    try:
        frontend.bind(cfg.pub)
        backend.bind(cfg.sub)
        zmq.proxy(frontend, backend)
    except zmq.ContextTerminated:
        pass
    finally:
        frontend.close(0)
        backend.close(0)
