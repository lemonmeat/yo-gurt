"""Bus messages and dev keys -> state machine events."""

from yogurt.config import Config
from yogurt.messages import Detections, DevKey, Intent, JointState, Message, PolicyStatus, Stop
from yogurt.orchestrator.fsm import (
    ArmUpdate,
    CupSeen,
    Event,
    Heard,
    Orchestrator,
    PolicyDone,
    Reset,
    StopRequest,
    Taken,
    Wake,
)

NAME = "orchestrator"
DEV_KEYS = {
    "w": ("wake word", Wake()),
    "f": ('heard "fetch"', Heard("fetch", "dev key")),
    "c": ("cup seen", CupSeen()),
    "h": ("cup taken", Taken()),
    "d": ("policy task done", None),
    "x": ("policy task failed", None),
    "s": ("stop", StopRequest("dev key")),
    "r": ("reset from SAFE_STOP", Reset()),
}


def to_events(topic: str, msg: Message, fsm: Orchestrator, cfg: Config, dev: bool) -> list[Event]:
    if topic == "audio/wake":
        return [Wake()]
    if topic == "audio/intent":
        assert isinstance(msg, Intent)
        return [Heard(msg.intent, msg.text)]
    if topic == "system/stop":
        assert isinstance(msg, Stop)
        # Our own repeats would otherwise re-trigger SAFE_STOP right after a reset.
        return [] if msg.source == NAME else [StopRequest(f"{msg.source}: {msg.reason}")]
    if topic == "arm/state":
        assert isinstance(msg, JointState)
        return [ArmUpdate(msg.gripper, msg.safety, msg.detail)]
    if topic == "policy/status":
        assert isinstance(msg, PolicyStatus)
        if msg.status in ("done", "failed"):
            return [PolicyDone(msg.task, msg.status == "done", msg.detail)]
        return []
    if topic == "vision/detections":
        assert isinstance(msg, Detections)
        # TODO(vision): emit Taken when a hand reaches the gripper during HANDOFF.
        min_conf = cfg.vision.min_confidence
        if any(d.label == "cup" and d.confidence >= min_conf for d in msg.items):
            return [CupSeen()]
        return []
    if topic == "dev/key" and dev:
        assert isinstance(msg, DevKey)
        if msg.key in ("d", "x"):
            return [PolicyDone(fsm.policy_task, msg.key == "d", "dev key")]
        if msg.key in DEV_KEYS:
            return [DEV_KEYS[msg.key][1]]
    return []
