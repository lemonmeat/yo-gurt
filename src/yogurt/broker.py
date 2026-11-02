"""Bus broker: forwards every published message to every subscriber.

    python -m yogurt.broker

Start it first; everything else connects to it. See yogurt.bus.
"""

import contextlib
import logging

from yogurt import cli
from yogurt.bus import run_broker

log = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    _, cfg = cli.start("broker", __doc__, argv)
    log.info("forwarding %s -> %s", cfg.bus.pub, cfg.bus.sub)
    with contextlib.suppress(KeyboardInterrupt):
        run_broker(cfg.bus)


if __name__ == "__main__":
    main()
