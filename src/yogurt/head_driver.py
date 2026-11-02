"""head_driver: the only process that touches the head arm's servo bus.

    python -m yogurt.head_driver [--backend sim|real] [--config PATH]

Same loop and safety filter as arm_driver, with the head's own limits. Subscribes head/target,
system/stop, system/resume; publishes head/state. The head carries the scene camera (later the
mics) and turns toward speakers and loud noises; it never carries the cup.
"""

from yogurt.arm_driver.driver import HEAD, main

if __name__ == "__main__":
    main(HEAD, __doc__)
