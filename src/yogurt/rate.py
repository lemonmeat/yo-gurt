"""Fixed-rate loop timing on the monotonic clock."""

import time


class Rate:
    """Call sleep() once per iteration to run at `hz`. Overruns reset the schedule, not burst."""

    def __init__(self, hz: float):
        self.period = 1.0 / hz
        self._next = time.monotonic() + self.period
        self.overruns = 0

    def sleep(self) -> None:
        now = time.monotonic()
        if now < self._next:
            time.sleep(self._next - now)
            self._next += self.period
        else:
            self.overruns += 1
            self._next = now + self.period
