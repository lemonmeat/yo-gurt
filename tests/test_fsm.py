import pytest

from yogurt.config import OrchestratorConfig
from yogurt.messages import Detection, Detections, DevKey, JointState, Safety, Stop
from yogurt.orchestrator.fsm import (
    ArmUpdate,
    CupSeen,
    Heard,
    Orchestrator,
    PolicyDone,
    Reset,
    State,
    StopRequest,
    Taken,
    Tick,
    Wake,
)
from yogurt.orchestrator.inputs import to_events

TIMEOUTS = {"LISTEN": 6, "CHECK": 3, "GRASP": 20, "VERIFY": 1, "HANDOFF": 30, "RELEASE_HOME": 15}


def make(**overrides) -> Orchestrator:
    cfg = OrchestratorConfig(timeouts_s=TIMEOUTS, **overrides)
    return Orchestrator(cfg, now=0.0)


class Driver:
    """Feed events with a clock; collect what the machine publishes."""

    def __init__(self, fsm: Orchestrator):
        self.fsm, self.t, self.out = fsm, 0.0, []

    def __call__(self, event, dt: float = 0.1):
        self.t += dt
        actions = self.fsm.handle(event, self.t)
        self.out.extend(actions)
        return actions

    def wait(self, seconds: float):
        end = self.t + seconds
        while self.t < end:
            self(Tick(), 0.05)

    def topics(self, actions=None):
        return [topic for topic, _ in (self.out if actions is None else actions)]

    def policy_commands(self):
        return [(m.action, m.task) for t, m in self.out if t == "policy/command"]

    def said(self):
        return [m.text for t, m in self.out if t == "tts/say"]


def to_handoff(d: Driver):
    d(Wake())
    d(Heard("fetch"))
    d(CupSeen())
    d(PolicyDone("grasp", ok=True))
    d(ArmUpdate(gripper=35.0, safety=Safety.OK))
    d.wait(1.1)
    assert d.fsm.state is State.HANDOFF


def test_happy_path_full_config():
    d = Driver(make(listen=True, require_cup=True))
    d(Wake())
    assert d.fsm.state is State.LISTEN
    assert "Yes?" in d.said()
    d(Heard("fetch"))
    assert d.fsm.state is State.CHECK  # passed through PARSE, waiting for the cup
    d(CupSeen())
    assert d.fsm.state is State.GRASP
    assert d.policy_commands() == [("start", "grasp")]

    d(PolicyDone("grasp", ok=True))
    assert d.fsm.state is State.VERIFY
    d(ArmUpdate(gripper=35.0, safety=Safety.OK))
    d.wait(1.1)
    assert d.fsm.state is State.HANDOFF
    assert d.policy_commands()[-1] == ("start", "handoff")

    d(Taken())
    assert d.fsm.state is State.HANDOFF, "must not release before the cup is presented"
    d(PolicyDone("handoff", ok=True))
    d(Taken())
    assert d.fsm.state is State.RELEASE_HOME
    assert d.policy_commands()[-1] == ("start", "release_home")
    d(PolicyDone("release_home", ok=True))
    assert d.fsm.state is State.IDLE

    states = [m.state for t, m in d.out if t == "orchestrator/state"]
    assert states == [
        "LISTEN", "PARSE", "CHECK", "GRASP", "VERIFY", "HANDOFF", "RELEASE_HOME", "IDLE"
    ]  # fmt: skip


def test_phase1_config_goes_from_wake_to_grasp():
    d = Driver(make(listen=False, require_cup=False))
    d(Wake())
    assert d.fsm.state is State.GRASP
    assert d.policy_commands() == [("start", "grasp")]


def test_listen_timeout_returns_to_idle():
    d = Driver(make(listen=True))
    d(Wake())
    d.wait(6.1)
    assert d.fsm.state is State.IDLE


def test_unknown_intent_apologizes():
    d = Driver(make(listen=True))
    d(Wake())
    d(Heard("dance"))
    assert d.fsm.state is State.IDLE
    assert any("only fetch" in s for s in d.said())


def test_check_uses_recent_cup_and_times_out():
    d = Driver(make(listen=True, require_cup=True))
    d(CupSeen())
    d(Wake())
    d(Heard("fetch"))
    assert d.fsm.state is State.GRASP  # cup seen 0.2 s ago counts

    d = Driver(make(listen=True, require_cup=True))
    d(Wake())
    d(Heard("fetch"))
    d.wait(3.1)
    assert d.fsm.state is State.IDLE
    assert "I can't see a yogurt." in d.said()


def test_missed_grasp_retries_then_gives_up():
    d = Driver(make(listen=False, require_cup=False, max_grasp_attempts=2))
    d(Wake())
    d(PolicyDone("grasp", ok=True))
    d(ArmUpdate(gripper=2.0, safety=Safety.OK))  # closed on nothing
    d.wait(1.1)
    assert d.fsm.state is State.GRASP
    assert d.fsm.attempts == 2
    d(PolicyDone("grasp", ok=True))
    d.wait(1.1)
    assert d.fsm.state is State.RELEASE_HOME


def test_grasp_failure_and_timeout_go_home():
    d = Driver(make(listen=False, require_cup=False))
    d(Wake())
    d(PolicyDone("grasp", ok=False, detail="no poses"))
    assert d.fsm.state is State.RELEASE_HOME

    d = Driver(make(listen=False, require_cup=False))
    d(Wake())
    d.wait(20.1)
    assert d.fsm.state is State.RELEASE_HOME
    assert ("stop", "grasp") in d.policy_commands()


def test_stale_policy_status_is_ignored():
    d = Driver(make(listen=False, require_cup=False))
    d(Wake())
    d(PolicyDone("release_home", ok=True))  # leftover from an earlier run
    assert d.fsm.state is State.GRASP


def test_handoff_never_drops_the_cup_on_timeout():
    d = Driver(make(listen=True, require_cup=True))
    to_handoff(d)
    d(PolicyDone("handoff", ok=True))
    d.wait(31)
    assert d.fsm.state is State.HANDOFF
    assert "Your yogurt is ready." in d.said()


def test_release_home_timeout_is_a_safe_stop():
    d = Driver(make(listen=False, require_cup=False))
    d(Wake())
    d(PolicyDone("grasp", ok=False))
    d.wait(15.1)
    assert d.fsm.state is State.SAFE_STOP


@pytest.mark.parametrize(
    "state_setup",
    [
        [],
        [Wake()],
        [Wake(), Heard("fetch")],
        [Wake(), Heard("fetch"), CupSeen()],
        [Wake(), Heard("fetch"), CupSeen(), PolicyDone("grasp", True)],
    ],
    ids=["IDLE", "LISTEN", "CHECK", "GRASP", "VERIFY"],
)
def test_safe_stop_reachable_from_every_state(state_setup):
    d = Driver(make(listen=True, require_cup=True))
    for ev in state_setup:
        d(ev)
    running = d.fsm.policy_task
    actions = d(StopRequest("test"))
    assert d.fsm.state is State.SAFE_STOP
    assert "system/stop" in d.topics(actions)
    if running:
        assert ("stop", running) in d.policy_commands()


def test_safe_stop_from_handoff_and_release_home():
    for extra in ([], [PolicyDone("handoff", True), Taken()]):
        d = Driver(make(listen=True, require_cup=True))
        to_handoff(d)
        for ev in extra:
            d(ev)
        d(StopRequest("test"))
        assert d.fsm.state is State.SAFE_STOP


def test_voice_stop_intent_and_latched_arm_trigger_safe_stop():
    d = Driver(make(listen=True))
    d(Wake())
    d(Heard("stop"))
    assert d.fsm.state is State.SAFE_STOP

    d = Driver(make(listen=False, require_cup=False))
    d(Wake())
    d(ArmUpdate(gripper=50.0, safety=Safety.FAULT, detail="overload"))
    assert d.fsm.state is State.SAFE_STOP


def test_safe_stop_repeats_stop_and_ignores_everything_but_reset():
    d = Driver(make(stop_repeat_s=0.5))
    d(StopRequest("test"))
    d.out.clear()
    d.wait(1.6)
    assert d.topics().count("system/stop") >= 3
    d(Wake())
    d(Heard("fetch"))
    assert d.fsm.state is State.SAFE_STOP


def test_reset_resumes_until_the_arm_unlatches():
    d = Driver(make(stop_repeat_s=0.5))
    d(StopRequest("test"))
    actions = d(Reset())
    assert d.fsm.state is State.IDLE
    assert "system/resume" in d.topics(actions)

    # Queued arm/state from before the resume still says STOPPED: don't bounce back.
    d(ArmUpdate(gripper=50.0, safety=Safety.STOPPED))
    assert d.fsm.state is State.IDLE
    d.out.clear()
    d.wait(1.1)
    assert "system/resume" in d.topics()  # keeps asking

    d(ArmUpdate(gripper=50.0, safety=Safety.WATCHDOG))
    d.out.clear()
    d.wait(1.1)
    assert "system/resume" not in d.topics()
    d(ArmUpdate(gripper=50.0, safety=Safety.STOPPED))  # a new latch now counts
    assert d.fsm.state is State.SAFE_STOP


# Bus -> events ---------------------------------------------------------------------------------


def test_bus_messages_become_events(cfg):
    fsm = make()
    det = Detections(camera="scene", frame_ts=0.0, items=[Detection("cup", 0.9, [0, 0, 1, 1])])
    weak = Detections(camera="scene", frame_ts=0.0, items=[Detection("cup", 0.1, [0, 0, 1, 1])])
    assert to_events("vision/detections", det, fsm, cfg, dev=False) == [CupSeen()]
    assert to_events("vision/detections", weak, fsm, cfg, dev=False) == []
    state = JointState(positions={}, gripper=12.0, safety=Safety.OK)
    assert to_events("arm/state", state, fsm, cfg, dev=False) == [ArmUpdate(12.0, "ok", "")]
    own = Stop(reason="repeat", source="orchestrator")
    assert to_events("system/stop", own, fsm, cfg, dev=False) == []
    other = Stop(reason="voice", source="audio")
    assert to_events("system/stop", other, fsm, cfg, dev=False) == [StopRequest("audio: voice")]


def test_dev_keys_only_in_dev_mode(cfg):
    fsm = make()
    assert to_events("dev/key", DevKey(key="w"), fsm, cfg, dev=False) == []
    assert to_events("dev/key", DevKey(key="w"), fsm, cfg, dev=True) == [Wake()]
    fsm.policy_task = "grasp"
    assert to_events("dev/key", DevKey(key="d"), fsm, cfg, dev=True) == [
        PolicyDone("grasp", True, "dev key")
    ]
