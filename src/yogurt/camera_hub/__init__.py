"""camera_hub: owns both USB cameras and shares frames through shared memory.

    python -m yogurt.camera_hub [--backend opencv|synthetic] [--config PATH]

Each camera in config `cameras` gets a capture thread writing RGB frames into a FrameWriter ring
(ring.py) and publishing camera/frame. Other processes read with FrameReader and never open a
camera themselves. The synthetic backend draws a moving test pattern, for work without cameras.
"""
