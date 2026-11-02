"""Shared helpers for the scripts/ wrappers around LeRobot's CLIs.

Every wrapper reads ports, ids, and calibration paths from the active config (YOGURT_HW or
--config), holds the device locks for whatever hardware LeRobot will open (so arm_driver and
camera_hub cannot run at the same time), prints the exact command, and runs it.
"""

import argparse
import json
import shlex
import shutil
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path

from yogurt.calibration import RECALIBRATE_HINT, calibration_problems
from yogurt.config import ENV_VAR, ArmConfig, Config, LeaderConfig, load_config
from yogurt.devicelock import DeviceBusy, DeviceLock


def parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", type=Path, help=f"hardware YAML (default: ${ENV_VAR}, laptop)")
    p.add_argument("--dry-run", action="store_true", help="print the command, run nothing")
    return p


def parse(p: argparse.ArgumentParser) -> tuple[argparse.Namespace, list[str], Config]:
    """Parse our flags; anything after `--` goes to LeRobot unchanged."""
    argv = sys.argv[1:]
    extra = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1 :]
    args = p.parse_args(argv)
    return args, extra, load_config(args.config)


def fail(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(2)


def lerobot_cli(name: str) -> str:
    exe = shutil.which(f"lerobot-{name}", path=str(Path(sys.executable).parent))
    if exe is None:
        fail("LeRobot is not installed in this environment: uv sync --extra hw")
    return exe


def check_port(port: str, what: str, cfg: Config) -> None:
    if not Path(port).exists():
        fail(
            f"{what} port {port} does not exist. Plug the {what} in, find its port with "
            f"`uv run lerobot-find-port`, and set it in {cfg.source_path}."
        )


def calibration_path(device: ArmConfig | LeaderConfig, cfg: Config) -> Path:
    return cfg.path(device.calibration_dir) / f"{device.id}.json"


def check_calibrated(device: ArmConfig | LeaderConfig, cfg: Config, arm: str) -> None:
    path = calibration_path(device, cfg)
    if not path.exists():
        fail(f"no calibration at {path}. Run: uv run python scripts/calibrate.py --arm {arm}")
    problems = calibration_problems(path)
    if problems:
        fail(
            f"{path.name} looks wrong:\n  "
            + "\n  ".join(problems)
            + "\n"
            + RECALIBRATE_HINT.replace("<arm>", arm)
        )


def follower_args(arm: ArmConfig, cfg: Config, prefix: str = "robot") -> list[str]:
    return [
        f"--{prefix}.type=so101_follower",
        f"--{prefix}.port={arm.port}",
        f"--{prefix}.id={arm.id}",
        f"--{prefix}.calibration_dir={cfg.path(arm.calibration_dir)}",
    ]


def leader_args(leader: LeaderConfig, cfg: Config) -> list[str]:
    return [
        "--teleop.type=so101_leader",
        f"--teleop.port={leader.port}",
        f"--teleop.id={leader.id}",
        f"--teleop.calibration_dir={cfg.path(leader.calibration_dir)}",
    ]


def cameras_arg(cfg: Config, names: list[str]) -> str:
    cams = {}
    for name in names:
        if name not in cfg.cameras:
            fail(f"unknown camera {name!r}; config has {sorted(cfg.cameras)}")
        c = cfg.cameras[name]
        cams[name] = {
            "type": "opencv",
            "index_or_path": c.device,
            "width": c.width,
            "height": c.height,
            "fps": c.fps,
            "fourcc": c.fourcc,
        }
    return f"--robot.cameras={json.dumps(cams)}"


def run(cmd: list[str], devices: list[str | int], dry_run: bool) -> int:
    """Print and run `cmd` while holding a DeviceLock on each device. Returns its exit code."""
    print("$ " + shlex.join(cmd), flush=True)
    if dry_run:
        return 0
    try:
        with ExitStack() as stack:
            for device in devices:
                stack.enter_context(DeviceLock(device))
            code = subprocess.run(cmd).returncode
    except DeviceBusy as e:
        fail(f"{e}. Stop it first (honcho, arm_driver, camera_hub, or another script).")
    except KeyboardInterrupt:
        code = 130
    return code
