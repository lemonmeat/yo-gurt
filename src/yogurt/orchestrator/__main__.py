import argparse
import logging
import time

from yogurt import cli
from yogurt.bus import Bus
from yogurt.messages import Message, OrchestratorState, Say
from yogurt.orchestrator import __doc__ as doc
from yogurt.orchestrator.fsm import Orchestrator, Tick
from yogurt.orchestrator.inputs import DEV_KEYS, NAME, to_events
from yogurt.rate import Rate

log = logging.getLogger("yogurt.orchestrator")

TICK_HZ = 20.0


def publish(bus: Bus, actions: list[tuple[str, Message]]) -> None:
    for topic, msg in actions:
        if isinstance(msg, OrchestratorState):
            log.info("%s -> %s (%s)", msg.previous, msg.state, msg.event)
        elif isinstance(msg, Say):
            log.info("say: %s", msg.text)
        bus.publish(topic, msg)


def main(argv: list[str] | None = None) -> None:
    def add_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--dev", action="store_true", help="accept dev/key triggers")

    args, cfg = cli.start(NAME, doc, argv, add_args)
    dev = args.dev or cfg.orchestrator.dev_mode
    bus = Bus(cfg.bus, NAME)
    bus.subscribe(
        "audio/wake",
        "audio/intent",
        "system/stop",
        "arm/state",
        "policy/status",
        "vision/detections",
    )
    if dev:
        bus.subscribe("dev/key")
        keys = ", ".join(f"{k}={what}" for k, (what, _) in DEV_KEYS.items())
        log.info("dev mode: run `python -m yogurt.devkeys` (%s)", keys)

    fsm = Orchestrator(cfg.orchestrator, time.monotonic())
    log.info(
        "state %s (listen=%s require_cup=%s)",
        fsm.state,
        cfg.orchestrator.listen,
        cfg.orchestrator.require_cup,
    )
    rate = Rate(TICK_HZ)
    try:
        while True:
            for topic, msg in bus.drain():
                for event in to_events(topic, msg, fsm, cfg, dev):
                    publish(bus, fsm.handle(event, time.monotonic()))
            publish(bus, fsm.handle(Tick(), time.monotonic()))
            rate.sleep()
    except KeyboardInterrupt:
        pass
    finally:
        bus.close()


if __name__ == "__main__":
    main()
