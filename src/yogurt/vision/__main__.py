import logging
import time

from yogurt import cli
from yogurt.bus import Bus
from yogurt.camera_hub.ring import FrameReader
from yogurt.messages import Detections
from yogurt.rate import Rate
from yogurt.vision import __doc__ as doc
from yogurt.vision.detectors import make_detector

log = logging.getLogger("yogurt.vision")

NAME = "vision"


def main(argv: list[str] | None = None) -> None:
    _, cfg = cli.start(NAME, doc, argv)
    camera = cfg.vision.camera
    cam = cfg.cameras[camera]
    try:
        detector = make_detector(cfg)
    except NotImplementedError as e:
        log.error("%s", e)
        raise SystemExit(2) from e
    bus = Bus(cfg.bus, NAME)
    reader = None
    last_seq = 0
    rate = Rate(cfg.vision.rate_hz)
    log.info("detector=%s camera=%s", cfg.vision.backend, camera)
    try:
        while True:
            rate.sleep()
            if reader is None:
                try:
                    reader = FrameReader(camera, cam.width, cam.height, cfg.camera_hub.ring_slots)
                    log.info("attached to %s frames", camera)
                except FileNotFoundError:
                    time.sleep(1.0)  # camera_hub not up yet
                    continue
            frame = reader.latest()
            if frame is None or frame[0] == last_seq:
                continue
            last_seq, frame_ts, rgb = frame
            items = [d for d in detector.detect(rgb) if d.confidence >= cfg.vision.min_confidence]
            bus.publish(
                "vision/detections", Detections(camera=camera, frame_ts=frame_ts, items=items)
            )
    except KeyboardInterrupt:
        pass
    finally:
        if reader is not None:
            reader.close()
        bus.close()


if __name__ == "__main__":
    main()
