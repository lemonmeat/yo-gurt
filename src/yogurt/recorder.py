"""recorder: logs every bus message with its receive time, for debugging.

    python -m yogurt.recorder [--config PATH]          record to recorder.dir/<timestamp>.msgpack
    python -m yogurt.recorder --dump FILE [--topic T]   print a log

Each record is a msgpack array [receive monotonic ts, topic, payload bytes]; the payload decodes
with yogurt.messages.decode. Topics in recorder.exclude (raw audio by default) are skipped.
"""

import argparse
import logging
import threading
import time
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import msgpack

from yogurt import cli
from yogurt.bus import Bus
from yogurt.messages import Message, decode

log = logging.getLogger("yogurt.recorder")

FLUSH_PERIOD_S = 1.0


def read_log(path: Path) -> Iterator[tuple[float, str, Message]]:
    with open(path, "rb") as f:
        for ts, topic, payload in msgpack.Unpacker(f, raw=False):
            yield ts, topic, decode(topic, payload)


def record(bus: Bus, path: Path, exclude: set[str], stop: threading.Event | None = None) -> int:
    """Append messages to `path` until Ctrl-C/SIGTERM, or until `stop` is set."""
    count, last_flush = 0, time.monotonic()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "ab") as f:
        try:
            while stop is None or not stop.is_set():
                raw = bus.recv_raw(timeout=0.2)
                now = time.monotonic()
                if raw is not None and raw[0] not in exclude:
                    f.write(msgpack.packb([now, raw[0], raw[1]], use_bin_type=True))
                    count += 1
                if now - last_flush > FLUSH_PERIOD_S:
                    f.flush()
                    last_flush = now
        except KeyboardInterrupt:
            pass
    return count


def main(argv: list[str] | None = None) -> None:
    def add_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--dump", type=Path, help="print this log instead of recording")
        p.add_argument("--topic", help="with --dump: only this topic")

    args, cfg = cli.start("recorder", __doc__, argv, add_args)
    if args.dump:
        for ts, topic, msg in read_log(args.dump):
            if args.topic is None or topic == args.topic:
                print(f"{ts:.3f} {topic} {msg}")
        return

    path = cfg.path(cfg.recorder.dir) / f"{datetime.now():%Y%m%d-%H%M%S}.msgpack"
    bus = Bus(cfg.bus, "recorder")
    bus.subscribe_all()
    log.info("recording to %s (excluding %s)", path, cfg.recorder.exclude)
    try:
        count = record(bus, path, set(cfg.recorder.exclude))
    finally:
        bus.close()
    log.info("wrote %d messages", count)


if __name__ == "__main__":
    main()
