"""audio: owns the microphones. Wake word, the always-on "stop" keyword, and sound direction.

    python -m yogurt.audio [--config PATH]

Publishes audio/wake, audio/stop (plus system/stop, so stopping never waits on the orchestrator),
audio/bearing, and audio/pcm for asr. STUB: publishes nothing yet; use devkeys meanwhile.

TODO(audio), in order:
  1. Capture: a mic library is not a dependency yet; ask before adding one (e.g. sounddevice).
     16 kHz mono int16 from cfg.audio.input_device (laptop mic now, SPH0645 I2S pair on the Pi).
  2. Publish 80 ms audio/pcm chunks for asr.
  3. openWakeWord on the same chunks: score >= wake_threshold -> audio/wake.
  4. Stop keyword (stop_model) -> audio/stop and system/stop, debounced but never rate-limited.
  5. Pi, with i2s_mics: GCC-PHAT on the stereo pair -> audio/bearing (voice or loud noise).
"""

import logging
import time

from yogurt import cli
from yogurt.bus import Bus

log = logging.getLogger("yogurt.audio")


def main(argv: list[str] | None = None) -> None:
    _, cfg = cli.start("audio", __doc__, argv)
    bus = Bus(cfg.bus, "audio")
    log.warning("stub: no microphone capture yet; use `python -m yogurt.devkeys` for wake/stop")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        bus.close()


if __name__ == "__main__":
    main()
