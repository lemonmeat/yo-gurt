import pytest
import zmq

from tests.helpers import wait_for
from yogurt.bus import Bus
from yogurt.messages import JointTarget, Stop


def test_publish_subscribe_through_broker(broker):
    ctx = zmq.Context()
    with Bus(broker, "tester", ctx) as pub, Bus(broker, "listener", ctx) as sub:
        sub.subscribe("system/stop")
        msg = wait_for(sub, "system/stop", lambda: pub.publish("system/stop", Stop(reason="t")))
        assert msg.reason == "t"
        assert msg.source == "tester"  # filled in by publish
    ctx.term()


def test_subscriber_only_sees_its_topics(broker):
    ctx = zmq.Context()
    with Bus(broker, "tester", ctx) as pub, Bus(broker, "listener", ctx) as sub:
        sub.subscribe("system/stop")

        def send_both():
            pub.publish("arm/target", JointTarget(positions={"elbow_flex": 1.0}))
            pub.publish("system/stop", Stop())

        wait_for(sub, "system/stop", send_both)
        assert all(topic == "system/stop" for topic, _ in sub.drain())
    ctx.term()


def test_subscribe_rejects_unknown_topic(broker):
    with Bus(broker, "tester") as bus, pytest.raises(KeyError):
        bus.subscribe("arm/targets")


def test_garbage_payload_is_skipped(broker):
    ctx = zmq.Context()
    raw = ctx.socket(zmq.PUB)
    raw.connect(broker.pub)
    with Bus(broker, "listener", ctx) as sub:
        sub.subscribe("system/stop")

        def send():
            raw.send_multipart([b"system/stop", b"\xc1not msgpack"])
            raw.send_multipart([b"system/stop", b"\x80"])  # empty map: a valid Stop

        msg = wait_for(sub, "system/stop", send)
        assert msg.reason == ""
    raw.close(0)
    ctx.term()
