"""Typed configuration, loaded from YAML under config/.

Which hardware file is active:
    --config PATH           explicit file (every process takes this flag)
    YOGURT_HW=laptop|pi     config/hardware.<name>.yaml (default: laptop)

joint_limits.yaml and gripper_map.yaml are read from the same directory as the hardware file and
merged into the `follower` and `head` sections. Relative paths inside the config are relative to the
repository root, i.e. the config directory's parent; use Config.path() to resolve them.

Laptop and Pi differ only in these files. Code must not branch on the host.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

from yogurt.messages import ALL_JOINTS, GRIPPER
from yogurt.typed import SchemaError, from_dict

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = REPO_ROOT / "config"
ENV_VAR = "YOGURT_HW"


@dataclass
class BusConfig:
    pub: str  # publishers connect here (the broker's XSUB)
    sub: str  # subscribers connect here (the broker's XPUB)


@dataclass
class JointLimit:
    min: float
    max: float
    max_velocity: float  # deg/s (gripper: percent/s)

    def __post_init__(self):
        if not self.min < self.max:
            raise ValueError(f"min {self.min} must be below max {self.max}")
        if self.max_velocity <= 0:
            raise ValueError("max_velocity must be positive")


@dataclass
class ArmLimits:
    measured: bool  # False while the values are placeholders; the real backend refuses to run
    joints: dict[str, JointLimit]


@dataclass
class GripperMap:
    """Raw encoder counts at the gripper's mechanical ends, mapped linearly to 0..100."""

    closed_counts: int
    open_counts: int
    measured: bool = False  # False while the values are placeholders

    def __post_init__(self):
        if self.closed_counts == self.open_counts:
            raise ValueError("closed_counts and open_counts must differ")

    def to_percent(self, counts: float) -> float:
        pct = 100.0 * (counts - self.closed_counts) / (self.open_counts - self.closed_counts)
        return min(100.0, max(0.0, pct))

    def to_counts(self, percent: float) -> int:
        percent = min(100.0, max(0.0, percent))
        return round(self.closed_counts + percent / 100.0 * (self.open_counts - self.closed_counts))


@dataclass
class Box:
    """Axis-aligned keep-out box in the follower base frame (metres; +x forward, +z up)."""

    name: str
    min: list[float]
    max: list[float]

    def __post_init__(self):
        if len(self.min) != 3 or len(self.max) != 3:
            raise ValueError("box corners need 3 coordinates")
        if not all(lo < hi for lo, hi in zip(self.min, self.max, strict=True)):
            raise ValueError(f"box {self.name}: min must be below max on every axis")


@dataclass
class FkConfig:
    urdf: str
    # Fingertip offset from the URDF gripper_frame_link, in that link's frame (metres).
    tool_offset: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    # URDF angle = sign * (lerobot_deg + offset_deg). Defaults 1 and 0; verify against the real arm.
    joint_sign: dict[str, float] = field(default_factory=dict)
    joint_offset_deg: dict[str, float] = field(default_factory=dict)


@dataclass
class SafetyConfig:
    watchdog_s: float = 0.2
    # |Present_Load| percent that latches a FAULT if exceeded for load_trip_s.
    max_load_pct: dict[str, float] = field(default_factory=dict)
    load_trip_s: float = 0.5
    # Written to each servo's Torque_Limit register at connect (real backend).
    torque_limit_pct: float = 80.0
    # The command never runs further than this ahead of the measured position, so a blocked joint
    # doesn't wind up and then jump when released. For the gripper this also bounds the squeeze.
    max_tracking_error_deg: float = 20.0
    max_gripper_tracking_error: float = 30.0
    keep_out: list[Box] = field(default_factory=list)
    keep_out_margin_m: float = 0.03


@dataclass
class ArmConfig:
    port: str
    id: str
    limits: ArmLimits  # from joint_limits.yaml
    gripper_map: GripperMap  # from gripper_map.yaml
    backend: Literal["sim", "real"] = "sim"
    calibration_dir: str = "calibration"
    rate_hz: float = 50.0
    sim_model: str = "third_party/SO-ARM100/Simulation/SO101/scene.xml"
    fk: FkConfig | None = None
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    torque_off_on_exit: bool = True
    # LeRobot's own per-command clamp in the real backend; a second line of defence.
    max_relative_target_deg: float | None = 10.0


@dataclass
class LeaderConfig:
    port: str
    id: str
    calibration_dir: str = "calibration"


@dataclass
class CameraConfig:
    device: int | str  # OpenCV index on macOS, /dev/v4l/by-id/... path on the Pi
    width: int = 800
    height: int = 600
    fps: int = 30
    fourcc: str = "MJPG"


@dataclass
class CameraHubConfig:
    backend: Literal["opencv", "synthetic"] = "opencv"
    ring_slots: int = 4


@dataclass
class AudioConfig:
    input_device: int | str | None = None  # sounddevice name or index; None = default mic
    sample_rate: int = 16000
    wake_model: str = "hey_jarvis"  # openWakeWord model name or .onnx path
    wake_threshold: float = 0.5
    stop_model: str = ""  # TODO: train or pick a "stop" keyword model
    stop_threshold: float = 0.5
    i2s_mics: bool = False  # Pi: SPH0645 pair for audio/bearing


@dataclass
class VisionConfig:
    backend: Literal["none", "yolo", "hailo"] = "none"
    model: str = ""
    camera: str = "scene"
    rate_hz: float = 10.0
    min_confidence: float = 0.4


@dataclass
class AsrConfig:
    model: str = "models/vosk-model-small-en-us-0.15"
    # Vosk restricts recognition to these phrases; "[unk]" catches everything else.
    grammar: list[str] = field(default_factory=list)
    # Phrase -> intent name.
    intents: dict[str, str] = field(default_factory=dict)


@dataclass
class TtsConfig:
    voice: str = "models/piper/en_US-lessac-medium.onnx"
    output_device: int | str | None = None


@dataclass
class SensorsConfig:
    enabled: bool = False
    i2c_bus: int = 1
    ads1115_address: int = 0x48
    fsr_channel: int = 0
    battery_channel: int = 1
    battery_divider: float = 1.0  # volts at the pack per volt at the ADC pin
    rate_hz: float = 20.0


@dataclass
class ActConfig:
    checkpoint: str = ""  # outputs/train/<run>/checkpoints/last/pretrained_model
    device: str = "cpu"  # "mps" on the laptop, "cuda" on a GPU box
    server: str | None = None  # host:port of a LeRobot policy server, instead of running locally


@dataclass
class PolicyConfig:
    backend: Literal["replay", "act"] = "replay"
    rate_hz: float = 30.0
    poses_dir: str = "data/poses"
    act: ActConfig = field(default_factory=ActConfig)


@dataclass
class OrchestratorConfig:
    dev_mode: bool = False  # accept dev/key triggers
    listen: bool = True  # False: wake word goes straight to default_intent (Phase 1)
    require_cup: bool = True  # False: CHECK passes without vision (Phase 1)
    default_intent: str = "fetch"
    # Gripper percent above which a closed grasp counts as holding something.
    holding_min_gripper: float = 8.0
    max_grasp_attempts: int = 2
    timeouts_s: dict[str, float] = field(default_factory=dict)
    stop_repeat_s: float = 0.5  # SAFE_STOP re-publishes system/stop at this period


@dataclass
class RecorderConfig:
    dir: str = "outputs/logs"
    exclude: list[str] = field(default_factory=lambda: ["audio/pcm"])


@dataclass
class Config:
    name: str
    bus: BusConfig
    follower: ArmConfig
    head: ArmConfig
    leader: LeaderConfig
    cameras: dict[str, CameraConfig]
    camera_hub: CameraHubConfig = field(default_factory=CameraHubConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    asr: AsrConfig = field(default_factory=AsrConfig)
    tts: TtsConfig = field(default_factory=TtsConfig)
    sensors: SensorsConfig = field(default_factory=SensorsConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    orchestrator: OrchestratorConfig = field(default_factory=OrchestratorConfig)
    recorder: RecorderConfig = field(default_factory=RecorderConfig)
    root: Path = REPO_ROOT
    source_path: Path | None = None

    def path(self, p: str | Path) -> Path:
        """Resolve a config path relative to the repository root."""
        p = Path(p).expanduser()
        return p if p.is_absolute() else self.root / p


def default_config_path() -> Path:
    name = os.environ.get(ENV_VAR, "laptop")
    return DEFAULT_CONFIG_DIR / f"hardware.{name}.yaml"


def _read_yaml(path: Path) -> dict:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise SchemaError(f"{path}: top level must be a mapping")
    return data


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else default_config_path()
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path} (set {ENV_VAR} or --config)")
    path = path.resolve()
    data = _read_yaml(path)
    limits = _read_yaml(path.parent / "joint_limits.yaml")
    grippers = _read_yaml(path.parent / "gripper_map.yaml")

    for arm in ("follower", "head"):
        section = data.get(arm)
        if not isinstance(section, dict):
            raise SchemaError(f"{path}: missing '{arm}' section")
        for key, source, name in (
            ("limits", limits, "joint_limits.yaml"),
            ("gripper_map", grippers, "gripper_map.yaml"),
        ):
            if key in section:
                raise SchemaError(f"{path}: {arm}.{key} belongs in {name}")
            if arm not in source:
                raise SchemaError(f"{name}: missing '{arm}' section")
            section[key] = source[arm]

    for key in ("root", "source_path"):
        if key in data:
            raise SchemaError(f"{path}: '{key}' is set by the loader")
    data["root"] = str(path.parent.parent)
    data["source_path"] = str(path)

    cfg = from_dict(Config, data, path=path.name)
    for arm_name in ("follower", "head"):
        arm = getattr(cfg, arm_name)
        missing = set(ALL_JOINTS) - set(arm.limits.joints)
        extra = set(arm.limits.joints) - set(ALL_JOINTS)
        if missing or extra:
            raise SchemaError(
                f"joint_limits.yaml {arm_name}: missing {sorted(missing)}, unknown {sorted(extra)}"
            )
        g = arm.limits.joints[GRIPPER]
        if g.min < 0 or g.max > 100:
            raise SchemaError(f"joint_limits.yaml {arm_name}.gripper must stay within 0..100")
    return cfg
