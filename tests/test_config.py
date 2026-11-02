import shutil
from pathlib import Path

import pytest
import yaml

from yogurt.config import DEFAULT_CONFIG_DIR, ENV_VAR, GripperMap, load_config
from yogurt.messages import ALL_JOINTS
from yogurt.typed import SchemaError


@pytest.mark.parametrize("name", ["laptop", "pi"])
def test_shipped_configs_load(name):
    cfg = load_config(DEFAULT_CONFIG_DIR / f"hardware.{name}.yaml")
    assert cfg.name == name
    assert set(cfg.follower.limits.joints) == set(ALL_JOINTS)
    assert cfg.root == DEFAULT_CONFIG_DIR.parent
    assert {"wrist", "scene"} <= set(cfg.cameras)


def test_env_var_selects_config(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "pi")
    assert load_config().name == "pi"
    monkeypatch.setenv(ENV_VAR, "laptop")
    assert load_config().name == "laptop"


def test_missing_config_names_the_env_var(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "toaster")
    with pytest.raises(FileNotFoundError, match=ENV_VAR):
        load_config()


@pytest.fixture
def config_dir(tmp_path) -> Path:
    d = tmp_path / "config"
    shutil.copytree(DEFAULT_CONFIG_DIR, d)
    return d


def _edit(path: Path, fn):
    data = yaml.safe_load(path.read_text())
    fn(data)
    path.write_text(yaml.safe_dump(data))


def test_paths_resolve_against_repo_root(config_dir):
    cfg = load_config(config_dir / "hardware.laptop.yaml")
    assert cfg.root == config_dir.parent
    assert cfg.path(cfg.follower.sim_model) == config_dir.parent / cfg.follower.sim_model
    assert cfg.path("/abs/model.xml") == Path("/abs/model.xml")


def test_unknown_key_is_rejected(config_dir):
    _edit(config_dir / "hardware.laptop.yaml", lambda d: d["follower"].update(rate_hx=50))
    with pytest.raises(SchemaError, match="rate_hx"):
        load_config(config_dir / "hardware.laptop.yaml")


def test_missing_joint_limit_is_rejected(config_dir):
    _edit(config_dir / "joint_limits.yaml", lambda d: d["follower"]["joints"].pop("wrist_roll"))
    with pytest.raises(SchemaError, match="wrist_roll"):
        load_config(config_dir / "hardware.laptop.yaml")


def test_inverted_limit_is_rejected(config_dir):
    _edit(
        config_dir / "joint_limits.yaml",
        lambda d: d["follower"]["joints"]["elbow_flex"].update(min=10, max=-10),
    )
    with pytest.raises(SchemaError, match="elbow_flex"):
        load_config(config_dir / "hardware.laptop.yaml")


def test_limits_cannot_live_in_hardware_file(config_dir):
    _edit(config_dir / "hardware.laptop.yaml", lambda d: d["follower"].update(limits={}))
    with pytest.raises(SchemaError, match="joint_limits.yaml"):
        load_config(config_dir / "hardware.laptop.yaml")


def test_gripper_map():
    g = GripperMap(closed_counts=2000, open_counts=3000)
    assert g.to_percent(2000) == 0.0
    assert g.to_percent(3000) == 100.0
    assert g.to_percent(2500) == 50.0
    assert g.to_percent(1500) == 0.0  # clamped
    assert g.to_counts(25.0) == 2250
    assert g.to_counts(150.0) == 3000


def test_gripper_map_handles_reversed_direction():
    g = GripperMap(closed_counts=3000, open_counts=2000)
    assert g.to_percent(2900) == pytest.approx(10.0)
    assert g.to_counts(10.0) == 2900


def _calibration(tmp_path, **ranges):
    import json

    data = {
        name: {"id": i + 1, "drive_mode": 0, "homing_offset": 0, "range_min": lo, "range_max": hi}
        for i, (name, (lo, hi)) in enumerate(ranges.items())
    }
    path = tmp_path / "arm.json"
    path.write_text(json.dumps(data))
    return path


def test_calibration_check_flags_wrapped_joints(tmp_path):
    from yogurt.calibration import calibration_problems

    sane = _calibration(tmp_path, shoulder_lift=(820, 3290), wrist_roll=(0, 4095))
    assert calibration_problems(sane) == []
    wrapped = _calibration(tmp_path, shoulder_lift=(3, 4094), gripper=(1967, 3395))
    problems = calibration_problems(wrapped)
    assert len(problems) == 1 and "shoulder_lift" in problems[0]
