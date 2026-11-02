"""Shared command-line boilerplate for every `python -m yogurt.<module>` process."""

import argparse
import logging
import os
import signal
from collections.abc import Callable
from pathlib import Path

from yogurt.config import ENV_VAR, Config, load_config


def _raise_interrupt(signum, frame):
    raise KeyboardInterrupt


def start(
    name: str,
    description: str,
    argv: list[str] | None = None,
    add_args: Callable[[argparse.ArgumentParser], None] | None = None,
) -> tuple[argparse.Namespace, Config]:
    """Parse args, set up logging, load config, and turn SIGTERM into KeyboardInterrupt.

    Every process cleans up in `except KeyboardInterrupt` / `finally`, so honcho's and systemd's
    SIGTERM get the same orderly shutdown as Ctrl-C.
    """
    parser = argparse.ArgumentParser(prog=f"python -m yogurt.{name}", description=description)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=f"hardware YAML (default: config/hardware.${ENV_VAR}.yaml, ${ENV_VAR}=laptop)",
    )
    parser.add_argument(
        "--log-level", default=os.environ.get("YOGURT_LOG", "INFO"), help="DEBUG, INFO, ..."
    )
    if add_args:
        add_args(parser)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format=f"%(asctime)s.%(msecs)03d {name} %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    signal.signal(signal.SIGTERM, _raise_interrupt)
    cfg = load_config(args.config)
    logging.getLogger(name).info("config %s (%s)", cfg.source_path, cfg.name)
    return args, cfg
