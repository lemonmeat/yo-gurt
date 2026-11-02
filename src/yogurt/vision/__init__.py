"""vision: cup, hand and face detection on camera_hub frames.

    python -m yogurt.vision [--config PATH]

Reads the newest frame of `vision.camera` from shared memory at `vision.rate_hz`, runs the
configured detector (detectors.py), and publishes vision/detections.
"""
