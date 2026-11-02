"""asr: Vosk speech recognition restricted to a fixed grammar.

    python -m yogurt.asr [--config PATH]

Subscribes audio/pcm (audio owns the mic); publishes audio/intent. STUB: logs what it receives.

TODO(asr): vosk.KaldiRecognizer(vosk.Model(cfg.path(cfg.asr.model)), 16000,
json.dumps(cfg.asr.grammar)); feed AudioChunk.samples; on a final result publish
Intent(text, match_intent(text, cfg.asr.intents)). Only listen between audio/wake and the
orchestrator leaving LISTEN, so TTS output is not transcribed.
"""

import logging
import time

from yogurt import cli
from yogurt.bus import Bus

log = logging.getLogger("yogurt.asr")


def match_intent(text: str, intents: dict[str, str]) -> str:
    """Map a recognized phrase to an intent name; anything else is "unknown"."""
    phrase = " ".join(text.lower().split())
    return intents.get(phrase, "unknown")


def main(argv: list[str] | None = None) -> None:
    _, cfg = cli.start("asr", __doc__, argv)
    bus = Bus(cfg.bus, "asr")
    bus.subscribe("audio/pcm")
    log.warning("stub: no recognizer yet; grammar has %d phrases", len(cfg.asr.grammar))
    chunks = 0
    try:
        while True:
            chunks += len(bus.drain())
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        log.info("received %d audio chunks", chunks)
        bus.close()


if __name__ == "__main__":
    main()
