import pytest

from yogurt import messages as m
from yogurt.typed import SchemaError

SAMPLES: dict[str, m.Message] = {
    "arm/state": m.JointState(
        positions={j: 1.5 for j in m.JOINTS},
        gripper=42.0,
        commanded={j: 1.0 for j in m.ALL_JOINTS},
        loads={"gripper": -12.5},
        temperatures={"elbow_flex": 31.0},
        safety=m.Safety.WATCHDOG,
        detail="no target for 0.3 s",
        backend="sim",
    ),
    "arm/target": m.JointTarget(positions={"elbow_flex": -20.0}, gripper=None),
    "head/state": m.JointState(positions={}, gripper=0.0),
    "head/target": m.JointTarget(positions={"shoulder_pan": 10.0}, gripper=50.0),
    "system/stop": m.Stop(reason="voice"),
    "system/resume": m.Resume(reason="operator"),
    "orchestrator/state": m.OrchestratorState(state="GRASP", previous="CHECK", event="cup_seen"),
    "camera/frame": m.FrameInfo(
        camera="wrist", seq=7, frame_ts=12.5, width=800, height=600, shm_name="yogurt_cam_wrist"
    ),
    "vision/detections": m.Detections(
        camera="scene",
        frame_ts=1.0,
        items=[m.Detection(label="cup", confidence=0.9, box=[0.1, 0.2, 0.3, 0.4])],
    ),
    "audio/wake": m.Wake(keyword="hey_jarvis", score=0.8),
    "audio/stop": m.StopHeard(score=0.7),
    "audio/bearing": m.Bearing(azimuth_deg=-30.0, kind="voice"),
    "audio/pcm": m.AudioChunk(samples=b"\x00\x01" * 160, sample_rate=16000),
    "audio/intent": m.Intent(text="get me a yogurt", intent="fetch", slots={"item": "yogurt"}),
    "tts/say": m.Say(text="Here you go", interrupt=True),
    "sensors/fsr": m.Fsr(volts=1.2),
    "sensors/battery": m.Battery(volts=11.7, percent=64.0),
    "policy/command": m.PolicyCommand(action="start", task="grasp"),
    "policy/status": m.PolicyStatus(task="grasp", status="done"),
    "dev/key": m.DevKey(key="w"),
}


def test_every_topic_has_a_sample():
    assert set(SAMPLES) == set(m.TOPICS)


@pytest.mark.parametrize("topic", sorted(SAMPLES))
def test_round_trip(topic):
    msg = SAMPLES[topic]
    out = m.decode(topic, m.encode(topic, msg))
    assert out == msg
    assert type(out) is type(msg)


def test_header_fields_survive():
    msg = m.Stop(reason="x", ts=123.25, source="audio")
    out = m.decode("system/stop", m.encode("system/stop", msg))
    assert (out.ts, out.source) == (123.25, "audio")


def test_ints_decode_as_floats():
    import msgpack

    payload = msgpack.packb({"positions": {"elbow_flex": 3}, "gripper": 50})
    out = m.decode("arm/target", payload)
    assert out.positions == {"elbow_flex": 3.0}
    assert isinstance(out.gripper, float)


def test_wrong_message_type_for_topic():
    with pytest.raises(TypeError):
        m.encode("arm/target", m.Stop())


def test_unknown_topic():
    with pytest.raises(KeyError):
        m.encode("arm/teleport", m.Stop())


def test_bad_payload_is_rejected():
    import msgpack

    with pytest.raises(SchemaError):
        m.decode("arm/target", msgpack.packb({"positions": {"elbow_flex": "fast"}}))
    with pytest.raises(SchemaError):
        m.decode("arm/target", msgpack.packb({"positions": {}, "speed": 3}))


def test_no_topic_is_a_prefix_of_another():
    # ZeroMQ subscriptions match by prefix; Bus also filters exactly, but keep topics unambiguous.
    topics = sorted(m.TOPICS)
    for a in topics:
        for b in topics:
            assert a == b or not b.startswith(a)
