"""Bus message schemas. Each topic's payload is defined once here, in TOPICS.

Every message carries `ts` (time.monotonic() when it was created, in seconds; comparable across
processes on the same host) and `source` (the publishing process; Bus.publish fills it in).

Units: body joint positions are degrees in LeRobot's calibrated frame (use_degrees=True, 0 = middle
of the recorded range). The gripper is 0 (fully closed) to 100 (fully open). Loads are signed
percent of max torque.

Wire format: a two-frame ZeroMQ message [topic, msgpack(dataclasses.asdict(msg))].
"""

import dataclasses
import time
from dataclasses import dataclass, field
from enum import StrEnum

import msgpack

from yogurt.typed import from_dict

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
GRIPPER = "gripper"
ALL_JOINTS = (*JOINTS, GRIPPER)


@dataclass(kw_only=True)
class Message:
    ts: float = field(default_factory=time.monotonic)
    source: str = ""


class Safety(StrEnum):
    """JointState.safety: what the driver's safety filter did on its last tick."""

    OK = "ok"
    WATCHDOG = "watchdog"  # no fresh target within the watchdog window; holding
    KEEP_OUT = "keep_out"  # last step would have entered a keep-out zone; holding
    STOPPED = "stopped"  # latched by system/stop; holding until system/resume
    FAULT = "fault"  # latched by the driver itself (overload, bus error); needs system/resume


@dataclass
class JointState(Message):
    """arm/state: published by arm_driver every control tick."""

    positions: dict[str, float]  # body joints, degrees
    gripper: float  # 0 closed .. 100 open
    commanded: dict[str, float] = field(default_factory=dict)  # what the filter sent, incl. gripper
    loads: dict[str, float] = field(default_factory=dict)  # incl. gripper
    temperatures: dict[str, float] = field(default_factory=dict)  # deg C, incl. gripper
    safety: str = Safety.OK
    detail: str = ""  # why the filter is holding, if it is
    backend: str = ""  # "sim" | "real"


@dataclass
class JointTarget(Message):
    """arm/target. Joints left out keep their previous target."""

    positions: dict[str, float] = field(default_factory=dict)  # body joints, degrees
    gripper: float | None = None  # 0 closed .. 100 open; None keeps the previous target


@dataclass
class Stop(Message):
    """system/stop: arm_driver latches into a hold. Any process may publish it."""

    reason: str = ""


@dataclass
class Resume(Message):
    """system/resume: clears a latched stop or fault. Only the orchestrator publishes it."""

    reason: str = ""


@dataclass
class OrchestratorState(Message):
    """orchestrator/state: published on every transition."""

    state: str
    previous: str
    event: str


@dataclass
class FrameInfo(Message):
    """camera/frame: camera_hub wrote a new frame to shared memory."""

    camera: str
    seq: int
    frame_ts: float  # monotonic capture time
    width: int
    height: int
    shm_name: str


@dataclass
class Detection:
    label: str  # "cup", "hand", "face", "person"
    confidence: float
    box: list[float]  # x1, y1, x2, y2, normalized 0..1


@dataclass
class Detections(Message):
    """vision/detections: one message per processed frame."""

    camera: str
    frame_ts: float
    items: list[Detection] = field(default_factory=list)


@dataclass
class Wake(Message):
    """audio/wake: the wake word was heard."""

    keyword: str
    score: float


@dataclass
class StopHeard(Message):
    """audio/stop: the always-on "stop" keyword was heard (audio also publishes system/stop)."""

    score: float


@dataclass
class Bearing(Message):
    """audio/bearing: direction of a voice or a loud noise, in the follower base frame."""

    azimuth_deg: float
    kind: str  # "voice" | "loud"
    confidence: float = 1.0


@dataclass
class AudioChunk(Message):
    """audio/pcm: raw microphone audio for asr. Mono int16 little-endian."""

    samples: bytes
    sample_rate: int


@dataclass
class Intent(Message):
    """audio/intent: what asr understood."""

    text: str
    intent: str  # e.g. "fetch", "stop", "unknown"
    slots: dict[str, str] = field(default_factory=dict)
    confidence: float = 1.0


@dataclass
class Say(Message):
    """tts/say."""

    text: str
    interrupt: bool = False


@dataclass
class Fsr(Message):
    """sensors/fsr: force-sensing resistor voltage from the ADS1115."""

    volts: float


@dataclass
class Battery(Message):
    """sensors/battery."""

    volts: float
    percent: float


@dataclass
class PolicyCommand(Message):
    """policy/command: start or stop a task ("grasp", "handoff", "release_home")."""

    action: str  # "start" | "stop"
    task: str = ""


@dataclass
class PolicyStatus(Message):
    """policy/status."""

    task: str
    status: str  # "running" | "done" | "failed" | "stopped"
    detail: str = ""


@dataclass
class DevKey(Message):
    """dev/key: a keypress from `python -m yogurt.devkeys`, for dev-mode triggers."""

    key: str


TOPICS: dict[str, type[Message]] = {
    "arm/state": JointState,
    "arm/target": JointTarget,
    "system/stop": Stop,
    "system/resume": Resume,
    "orchestrator/state": OrchestratorState,
    "camera/frame": FrameInfo,
    "vision/detections": Detections,
    "audio/wake": Wake,
    "audio/stop": StopHeard,
    "audio/bearing": Bearing,
    "audio/pcm": AudioChunk,
    "audio/intent": Intent,
    "tts/say": Say,
    "sensors/fsr": Fsr,
    "sensors/battery": Battery,
    "policy/command": PolicyCommand,
    "policy/status": PolicyStatus,
    "dev/key": DevKey,
}


def check_topic(topic: str) -> type[Message]:
    try:
        return TOPICS[topic]
    except KeyError:
        raise KeyError(f"unknown topic {topic!r}; add it to yogurt.messages.TOPICS") from None


def encode(topic: str, msg: Message) -> bytes:
    cls = check_topic(topic)
    if not isinstance(msg, cls):
        raise TypeError(f"{topic} carries {cls.__name__}, got {type(msg).__name__}")
    return msgpack.packb(dataclasses.asdict(msg), use_bin_type=True)


def decode(topic: str, payload: bytes) -> Message:
    cls = check_topic(topic)
    return from_dict(cls, msgpack.unpackb(payload, raw=False), path=topic)
