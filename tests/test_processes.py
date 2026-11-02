"""camera_hub's frame ring, the recorder, and asr's intent matching."""

import os
import threading
import time

import numpy as np
import pytest
import zmq

from tests.helpers import wait_for
from yogurt.asr import match_intent
from yogurt.bus import Bus
from yogurt.camera_hub.ring import WRITING, FrameReader, FrameWriter
from yogurt.messages import Stop
from yogurt.recorder import read_log, record


@pytest.fixture
def camera():
    name = f"t{os.getpid() % 100000}"
    writer = FrameWriter(name, width=8, height=6, slots=3)
    yield name, writer
    writer.close()


def test_frame_ring_round_trip(camera):
    name, writer = camera
    reader = FrameReader(name, width=8, height=6, slots=3)
    assert reader.latest() is None
    for i in range(1, 6):
        frame = np.full((6, 8, 3), i, np.uint8)
        writer.write(frame, ts=100.0 + i)
    seq, ts, frame = reader.latest()
    assert (seq, ts) == (5, 105.0)
    assert (frame == 5).all()
    reader.close()


def test_frame_ring_detects_torn_slot(camera):
    name, writer = camera
    reader = FrameReader(name, width=8, height=6, slots=3)
    writer.write(np.zeros((6, 8, 3), np.uint8), ts=1.0)
    writer._seq[1][0] = WRITING  # simulate the writer mid-copy
    assert reader.latest() is None
    reader.close()


def test_frame_ring_rejects_wrong_shape(camera):
    _, writer = camera
    with pytest.raises(ValueError):
        writer.write(np.zeros((600, 800, 3), np.uint8), ts=0.0)


def test_recorder_writes_readable_log(broker, tmp_path):
    ctx = zmq.Context()
    path = tmp_path / "log.msgpack"
    stop = threading.Event()
    rec_bus = Bus(broker, "recorder", ctx)
    rec_bus.subscribe_all()
    thread = threading.Thread(target=record, args=(rec_bus, path, {"audio/pcm"}, stop))
    with Bus(broker, "tester", ctx) as pub, Bus(broker, "probe", ctx) as probe:
        probe.subscribe("system/stop")
        thread.start()
        wait_for(probe, "system/stop", lambda: pub.publish("system/stop", Stop(reason="warmup")))
        time.sleep(0.2)
        pub.publish("system/stop", Stop(reason="logged"))
        time.sleep(0.2)
    stop.set()
    thread.join(timeout=2)
    rec_bus.close()
    ctx.term()
    records = list(read_log(path))
    assert any(topic == "system/stop" and msg.reason == "logged" for _, topic, msg in records)
    assert all(ts > 0 for ts, _, _ in records)


def test_match_intent():
    intents = {"get me a yogurt": "fetch", "stop": "stop"}
    assert match_intent("Get me a  yogurt", intents) == "fetch"
    assert match_intent("stop", intents) == "stop"
    assert match_intent("[unk]", intents) == "unknown"
