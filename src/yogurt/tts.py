"""tts: Piper text to speech on the USB speaker.

    python -m yogurt.tts [--config PATH]

Subscribes tts/say. STUB: logs the text instead of speaking it.

TODO(tts): piper-tts is listed for the project but not added yet; add it as an `audio` extra.
Load cfg.path(cfg.tts.voice) once, synthesize each Say, play on cfg.tts.output_device, and cut the
current utterance short when Say.interrupt is set. Stretch: a custom Piper voice.
"""

import logging
import time

from yogurt import cli
from yogurt.bus import Bus
from yogurt.messages import Say

log = logging.getLogger("yogurt.tts")


def main(argv: list[str] | None = None) -> None:
    _, cfg = cli.start("tts", __doc__, argv)
    bus = Bus(cfg.bus, "tts")
    bus.subscribe("tts/say")
    log.warning("stub: logging speech instead of playing it")
    try:
        while True:
            for _, msg in bus.drain():
                assert isinstance(msg, Say)
                log.info("say%s: %s", " (interrupt)" if msg.interrupt else "", msg.text)
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        bus.close()


if __name__ == "__main__":
    main()
