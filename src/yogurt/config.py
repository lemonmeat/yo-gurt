"""Configuration: config/robot.yaml (shared) + config/hardware_<machine>.yaml + config/poses.yaml."""

import os
from pathlib import Path

import yaml

ROOT = Path(os.environ.get("YOGURT_ROOT", Path(__file__).resolve().parents[2]))
POSES_FILE = ROOT / "config" / "poses.yaml"
POSES_HEADER = (
    "# Taught arm poses, in LeRobot degrees (gripper: 0 closed .. 100 open).\n"
    "# Re-teach any of them from the Arm panel of the UI; that rewrites this file.\n"
    "# handoff: its gripper value is ignored (the gripper keeps holding the cup).\n"
)


class Cfg(dict):
    """A dict whose keys can also be read as attributes: cfg.arm.port."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"missing config key '{key}'") from None


def _wrap(value):
    if isinstance(value, dict):
        return Cfg({k: _wrap(v) for k, v in value.items()})
    return value


def _merge(base: dict, extra: dict) -> dict:
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def load(machine: str) -> Cfg:
    """machine is the suffix of a config/hardware_<machine>.yaml file, e.g. 'laptop' or 'pi'."""
    hardware = ROOT / "config" / f"hardware_{machine}.yaml"
    if not hardware.exists():
        known = sorted(p.stem.removeprefix("hardware_") for p in hardware.parent.glob("hardware_*.yaml"))
        raise SystemExit(f"No {hardware}. Known machines: {', '.join(known)}")
    cfg = yaml.safe_load((ROOT / "config" / "robot.yaml").read_text())
    _merge(cfg, yaml.safe_load(hardware.read_text()))
    cfg["machine"] = machine
    cfg["poses"] = yaml.safe_load(POSES_FILE.read_text())
    return _wrap(cfg)


def save_poses(poses: dict) -> None:
    plain = {name: {j: round(float(v), 1) for j, v in pose.items()} for name, pose in poses.items()}
    POSES_FILE.write_text(POSES_HEADER + yaml.safe_dump(plain, sort_keys=False))


def path(relative: str) -> Path:
    """Resolve a path from the config files against the repository root."""
    return (ROOT / relative).resolve()
