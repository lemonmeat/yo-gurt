"""What the driver loop needs from a servo backend."""

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Reading:
    positions: dict[str, float]  # every joint: body in degrees, gripper 0..100
    loads: dict[str, float] = field(default_factory=dict)  # signed percent of max torque
    temperatures: dict[str, float] = field(default_factory=dict)  # deg C


class Backend(Protocol):
    name: str

    def connect(self) -> None: ...

    def read(self) -> Reading: ...

    def write(self, command: dict[str, float]) -> None:
        """Send one command. Only the driver calls this, with SafetyFilter output."""
        ...

    def close(self) -> None: ...
