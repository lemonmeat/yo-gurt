"""sensors: owns the ADS1115 ADC on the Pi (round FSR in the gripper, battery voltage).

    python -m yogurt.sensors [--config PATH]

Publishes sensors/fsr and sensors/battery at sensors.rate_hz. With sensors.enabled: false (the
laptop) it publishes nothing and idles, so the same Procfile works everywhere.

TODO(sensors, Pi): an ADS1115 library is not a dependency yet; ask before adding one (e.g.
adafruit-circuitpython-ads1x15, or raw smbus2 reads). Read fsr_channel and battery_channel
single-ended, scale the battery by battery_divider, and estimate percent from a 3S Li-ion curve.
"""

import logging
import time

from yogurt import cli
from yogurt.bus import Bus

log = logging.getLogger("yogurt.sensors")


def main(argv: list[str] | None = None) -> None:
    _, cfg = cli.start("sensors", __doc__, argv)
    bus = Bus(cfg.bus, "sensors")
    if cfg.sensors.enabled:
        log.warning("stub: ADS1115 reading not implemented yet; publishing nothing")
    else:
        log.info("disabled in config; publishing nothing")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        bus.close()


if __name__ == "__main__":
    main()
