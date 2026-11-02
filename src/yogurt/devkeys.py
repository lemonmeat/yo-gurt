"""devkeys: single-key dev triggers for the orchestrator, published on dev/key.

    python -m yogurt.devkeys          (run in its own terminal next to `honcho start`)

's' also publishes system/stop directly, so stopping never depends on the orchestrator.
Keys are listed in yogurt/orchestrator/inputs.py (DEV_KEYS). 'q' quits.
"""

import contextlib
import sys
import termios
import tty

from yogurt import cli
from yogurt.bus import Bus
from yogurt.messages import DevKey, Stop
from yogurt.orchestrator.inputs import DEV_KEYS


@contextlib.contextmanager
def cbreak():
    """Read single keypresses without Enter, when stdin is a terminal."""
    if not sys.stdin.isatty():
        yield
        return
    old = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin)
        yield
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)


def main(argv: list[str] | None = None) -> None:
    _, cfg = cli.start("devkeys", __doc__, argv)
    bus = Bus(cfg.bus, "devkeys")
    print("keys: " + "  ".join(f"[{k}] {what}" for k, (what, _) in DEV_KEYS.items()) + "  [q] quit")
    try:
        with cbreak():
            while True:
                key = sys.stdin.read(1)
                if key in ("", "q"):
                    break
                if key.isspace():
                    continue
                if key == "s":
                    bus.publish("system/stop", Stop(reason="dev key"))
                if key in DEV_KEYS:
                    bus.publish("dev/key", DevKey(key=key))
                    print(f"-> {DEV_KEYS[key][0]}", flush=True)
                else:
                    print(f"unknown key {key!r}", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        bus.close()


if __name__ == "__main__":
    main()
