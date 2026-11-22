"""The Lab: grasp demonstrations as LeRobot datasets, recorded and managed the way LeLab does it.

Every dataset is a standard LeRobot v3 dataset in data/grasp_act/<name>/ (repo id local/<name>), so
`lerobot-train` reads it directly. A recording session (`Session`) writes N episodes into one dataset;
the state machine drives its phases (fsm.Brain._demonstrate). The functions below list the datasets,
read an episode back for the browser (video window and joint trace) and delete episodes or datasets.
"""

import json
import logging
import re
import shutil
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

from . import collect, config
from .arm import JOINTS, open_feetech
from .grasp import wrist_image

log = logging.getLogger("yogurt.lab")

NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
NAME_RULE = "1-64 letters, digits, '.', '_' or '-', starting with a letter or digit"
CAMERA = "observation.images."  # LeRobot's prefix for camera features
ROBOT_TYPE = "so101_follower"
LOCK = threading.Lock()  # held while a dataset is being rewritten or deleted


def root() -> Path:
    return collect.DATA / "grasp_act"


def repo_id(name: str) -> str:
    return f"local/{name}"


def check_name(name) -> str:
    if not isinstance(name, str) or not NAME.fullmatch(name):
        raise ValueError(f"a dataset name is {NAME_RULE}")
    return name


def dataset_dir(name: str) -> Path:
    path = root() / check_name(name)
    if not (path / "meta" / "info.json").is_file():
        raise LookupError(f"there is no dataset called {name}")
    return path


def _info(path: Path) -> dict:
    return json.loads((path / "meta" / "info.json").read_text())


def _cameras(info: dict) -> list[str]:
    return sorted(k.removeprefix(CAMERA) for k, f in info["features"].items()
                  if k.startswith(CAMERA) and f.get("dtype") == "video")


def _episodes(path: Path, info: dict) -> list[dict]:
    """Per-episode rows from meta/episodes, sorted by index. Raises if the index cannot be read."""
    import pyarrow.parquet as pq

    fps, cameras, rows = info["fps"], _cameras(info), []
    for file in sorted((path / "meta" / "episodes").glob("chunk-*/file-*.parquet")):
        wanted = ["episode_index", "length", "tasks", "data/chunk_index", "data/file_index"]
        for cam in cameras:
            wanted += [f"videos/{CAMERA}{cam}/{k}" for k in ("chunk_index", "file_index", "from_timestamp",
                                                             "to_timestamp")]
        for r in pq.read_table(file, columns=wanted).to_pylist():
            rows.append({
                "index": int(r["episode_index"]), "length": int(r["length"]),
                "duration_s": round(int(r["length"]) / fps, 2), "tasks": list(r["tasks"] or []),
                "data": [int(r["data/chunk_index"]), int(r["data/file_index"])],
                "videos": {cam: {"chunk": int(r[f"videos/{CAMERA}{cam}/chunk_index"]),
                                 "file": int(r[f"videos/{CAMERA}{cam}/file_index"]),
                                 "from": float(r[f"videos/{CAMERA}{cam}/from_timestamp"]),
                                 "to": float(r[f"videos/{CAMERA}{cam}/to_timestamp"])} for cam in cameras},
            })
    rows.sort(key=lambda r: r["index"])
    if [r["index"] for r in rows] != list(range(info["total_episodes"])):
        raise RuntimeError(f"its episode index lists {len(rows)} of {info['total_episodes']} episodes")
    return rows


def _problem(path: Path, info: dict) -> str | None:
    """Why LeRobot cannot use this dataset, or None if it can."""
    try:
        _episodes(path, info)
        return None
    except Exception as e:
        return (f"Unreadable: {e}. Most likely its recording was interrupted (power cut, crash) before the "
                "dataset was finalised. It cannot be repaired; delete it.")


DART_FILE = "dart.json"  # {"perturbed": [episode indices]}: which takes had DART noise (see dart.py)


def perturbed(path: Path) -> set[int]:
    try:
        return set(json.loads((path / DART_FILE).read_text())["perturbed"])
    except (OSError, ValueError, KeyError, TypeError):
        return set()


def _write_perturbed(path: Path, indices) -> None:
    (path / DART_FILE).write_text(json.dumps({"perturbed": sorted(indices)}))


# ---- cleaning: trims and disabled episodes ----------------------------------------------------
# Kept beside the recording in <dataset>/cleaning.json and never applied to it, so every edit can be
# undone: {"episodes": {"<index>": {"disabled": true, "trim": [first frame, end frame (exclusive)]}}}.
# lerobot-train cannot read it, so export_clean() writes a copy with the edits applied, to train on.
CLEAN_FILE = "cleaning.json"
CLEAN_SUFFIX = "-clean"


def cleaning(path: Path) -> dict[int, dict]:
    try:
        raw = json.loads((path / CLEAN_FILE).read_text())["episodes"]
        return {int(k): v for k, v in raw.items() if isinstance(v, dict)}
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return {}


def _write_cleaning(path: Path, edits: dict[int, dict]) -> None:
    edits = {i: e for i, e in edits.items() if e.get("disabled") or e.get("trim")}
    file = path / CLEAN_FILE
    if not edits and not file.exists():
        return
    tmp = file.with_suffix(".tmp")
    tmp.write_text(json.dumps({"episodes": {str(i): edits[i] for i in sorted(edits)}}, indent=1))
    tmp.replace(file)  # never a half-written file, even on a power cut


def _kept_range(edit: dict | None, length: int) -> tuple[int, int]:
    start, end = (edit or {}).get("trim") or (0, length)
    return max(0, min(int(start), length)), max(0, min(int(end), length))


def clean_name(name: str) -> str:
    return name[:64 - len(CLEAN_SUFFIX)] + CLEAN_SUFFIX


def set_cleaning(name: str, index: int, disabled=None, trim="keep") -> dict:
    """Disable/enable one episode and/or set its trim ([start, end) in frames, or None for the whole take)."""
    path = dataset_dir(name)
    info = _info(path)
    if problem := _problem(path, info):
        raise RuntimeError(problem)
    episode = next((e for e in _episodes(path, info) if e["index"] == index), None)
    if episode is None:
        raise LookupError(f"{name} has no episode {index}")
    edits = cleaning(path)
    edit = dict(edits.get(index, {}))
    if disabled is not None:
        if not isinstance(disabled, bool):
            raise ValueError("disabled must be true or false")
        edit["disabled"] = disabled
    if trim != "keep":
        if trim is None:
            edit.pop("trim", None)
        else:
            if (not isinstance(trim, list) or len(trim) != 2
                    or not all(isinstance(v, int) and not isinstance(v, bool) for v in trim)):
                raise ValueError("trim must be [first frame, end frame]")
            start, end = trim
            if not 0 <= start < end <= episode["length"]:
                raise ValueError(f"trim must satisfy 0 <= start < end <= {episode['length']}")
            if end - start < info["fps"] // 2:
                raise ValueError("a trimmed episode must keep at least half a second")
            edit["trim"] = None if (start, end) == (0, episode["length"]) else [start, end]
    edits[index] = {k: v for k, v in edit.items() if v}
    _write_cleaning(path, edits)
    return edits[index]


def export_clean(name: str) -> dict:
    """Write <name>-clean: the enabled episodes, each cut to its trim, as a normal LeRobot dataset.
    Built in a staging folder and swapped in at the end, so an earlier copy survives a failure."""
    import pyarrow.parquet as pq
    from lerobot.datasets import LeRobotDataset
    from torchcodec.decoders import VideoDecoder

    path = dataset_dir(name)
    info = _info(path)
    if problem := _problem(path, info):
        raise RuntimeError(problem)
    if name.endswith(CLEAN_SUFFIX):
        raise ValueError("this is already a cleaned copy: clean the original dataset instead")
    edits, marks = cleaning(path), perturbed(path)
    episodes = [e for e in _episodes(path, info) if not edits.get(e["index"], {}).get("disabled")]
    if not episodes:
        raise ValueError("every episode is disabled: nothing to export")
    target_name = clean_name(name)
    target = root() / target_name
    cams = _cameras(info)
    builtin = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
    feats = {k: v for k, v in info["features"].items() if k not in builtin}
    staging = Path(tempfile.mkdtemp(prefix=f".{target_name}.export-", dir=root()))
    out_dir = staging / "new"
    try:
        out = LeRobotDataset.create(repo_id(target_name), info["fps"], features=feats, root=out_dir,
                                    robot_type=info.get("robot_type"), use_videos=True, image_writer_threads=4)
        tables, decoders, new_marks, frames = {}, {}, [], 0
        for e in episodes:
            key = tuple(e["data"])
            if key not in tables:
                file = path / info["data_path"].format(chunk_index=key[0], file_index=key[1])
                tables[key] = pq.read_table(file).to_pandas()
            rows = tables[key][tables[key]["episode_index"] == e["index"]].sort_values("frame_index")
            start, end = _kept_range(edits.get(e["index"]), e["length"])
            pictures = {}
            for cam in cams:
                v = e["videos"][cam]
                vkey = (cam, v["chunk"], v["file"])
                if vkey not in decoders:
                    decoders[vkey] = VideoDecoder(str(video_file(name, cam, v["chunk"], v["file"])))
                first = round(v["from"] * info["fps"])
                batch = decoders[vkey].get_frames_at(list(range(first + start, first + end)))
                pictures[cam] = batch.data.permute(0, 2, 3, 1).numpy()  # N,H,W,C uint8
            task = (e["tasks"] or [""])[0]
            for k, (_, row) in enumerate(rows.iloc[start:end].iterrows()):
                frame = {key: np.asarray(row[key], dtype=np.float32) for key in feats
                         if not key.startswith(CAMERA)}
                frame.update({CAMERA + cam: pictures[cam][k] for cam in cams})
                out.add_frame({**frame, "task": task})
                frames += 1
            out.save_episode()
            if e["index"] in marks:
                new_marks.append(out.meta.total_episodes - 1)
        out.finalize()
        if new_marks:
            _write_perturbed(out_dir, new_marks)
        (out_dir / "SOURCE.json").write_text(json.dumps(
            {"source": name, "episodes": [e["index"] for e in episodes],
             "trims": {str(i): edits[i]["trim"] for i in edits if edits[i].get("trim")}}, indent=1))
        if target.exists():
            target.rename(staging / "old")
        out_dir.rename(target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    log.info("exported %s: %d episodes, %d frames", target_name, len(episodes), frames)
    return {"name": target_name, "episodes": len(episodes), "frames": frames,
            "train_command": train_command(target_name)}


def _size_mb(path: Path) -> float:
    return round(sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6, 1)


def list_datasets(recording: str | None = None) -> list[dict]:
    """Every dataset, newest first. The one a session is writing into is not read (its files are open)."""
    out = []
    if not root().is_dir():
        return out
    for path in root().iterdir():
        if not (NAME.fullmatch(path.name) and (path / "meta" / "info.json").is_file()):
            continue
        try:
            info = _info(path)
        except Exception as e:
            out.append({"name": path.name, "episodes": 0, "frames": 0, "problem": f"Unreadable meta/info.json: {e}",
                        "modified": path.stat().st_mtime})
            continue
        out.append({"name": path.name, "episodes": info.get("total_episodes", 0),
                    "frames": info.get("total_frames", 0), "recording": path.name == recording,
                    "problem": None if path.name == recording else _problem(path, info),
                    "modified": (path / "meta" / "info.json").stat().st_mtime})
    out.sort(key=lambda d: d["modified"], reverse=True)
    return out


def total_episodes() -> int:
    return sum(d["episodes"] for d in list_datasets() if not d["problem"])


def train_command(name: str) -> str:
    return (f"lerobot-train --dataset.repo_id={repo_id(name)} --dataset.root=data/grasp_act/{name} "
            f"--policy.type=act --policy.device=cuda --policy.push_to_hub=false "
            f"--output_dir=outputs/act_{name} --steps=60000 --batch_size=8 --wandb.enable=false")


def describe(name: str) -> dict:
    """One dataset with its episodes, for the browser."""
    path = dataset_dir(name)
    info = _info(path)
    base = {"name": name, "repo_id": repo_id(name), "path": f"data/grasp_act/{name}", "fps": info["fps"],
            "robot_type": info.get("robot_type"), "total_episodes": info.get("total_episodes", 0),
            "total_frames": info.get("total_frames", 0), "cameras": _cameras(info), "size_mb": _size_mb(path),
            # changes whenever the dataset is rewritten: the browser adds it to video addresses
            "version": (path / "meta" / "info.json").stat().st_mtime_ns,
            "train_command": train_command(name)}
    problem = _problem(path, info)
    if problem:
        return {**base, "problem": problem, "episodes": []}
    marked, edits = perturbed(path), cleaning(path)
    episodes = []
    for e in _episodes(path, info):
        edit = edits.get(e["index"], {})
        start, end = _kept_range(edit, e["length"])
        episodes.append({**e, "perturbed": e["index"] in marked, "disabled": bool(edit.get("disabled")),
                         "trim": edit.get("trim"), "kept_frames": end - start})
    usable = [e for e in episodes if not e["disabled"]]
    copy_info = root() / clean_name(name) / "meta" / "info.json"
    changed = [p for p in (path / CLEAN_FILE, path / "meta" / "info.json") if p.is_file()]
    return {**base, "problem": None, "episodes": episodes, "perturbed_episodes": len(marked),
            "cleaning": {"edited": bool(edits), "usable_episodes": len(usable),
                         "usable_frames": sum(e["kept_frames"] for e in usable),
                         "is_copy": name.endswith(CLEAN_SUFFIX), "copy": clean_name(name),
                         # the copy is stale when the recording or its edits changed after it was made
                         "copy_time": copy_info.stat().st_mtime if copy_info.is_file() else None,
                         "edit_time": max((p.stat().st_mtime for p in changed), default=0)},
            "image_sizes": {cam: info["features"][CAMERA + cam]["shape"][:2] for cam in base["cameras"]}}


def joints(name: str, index: int) -> dict:
    """One episode's joint trace: what the leader commanded (action) and where the follower was (state)."""
    import pyarrow.parquet as pq

    path = dataset_dir(name)
    info = _info(path)
    episode = next((r for r in _episodes(path, info) if r["index"] == index), None)
    if episode is None:
        raise LookupError(f"{name} has no episode {index}")
    chunk, file = episode["data"]
    table = pq.read_table(path / info["data_path"].format(chunk_index=chunk, file_index=file),
                          columns=["episode_index", "frame_index", "action", "observation.state"],
                          filters=[("episode_index", "=", index)]).to_pydict()
    order = np.argsort(table["frame_index"])
    action = np.round(np.asarray(table["action"], dtype=float)[order], 2)
    state = np.round(np.asarray(table["observation.state"], dtype=float)[order], 2)
    names = [n.removesuffix(".pos") for n in info["features"]["action"]["names"]]
    return {"index": index, "fps": info["fps"], "names": names, "action": action.tolist(), "state": state.tolist()}


def video_file(name: str, camera: str, chunk: int, file: int) -> Path:
    path = dataset_dir(name)
    info = _info(path)
    if camera not in _cameras(info) or chunk < 0 or file < 0:
        raise LookupError("no such video")
    video = (path / info["video_path"].format(video_key=CAMERA + camera, chunk_index=chunk, file_index=file))
    if not video.resolve().is_relative_to(path.resolve()) or not video.is_file():
        raise LookupError("no such video")
    return video


def delete_episodes(name: str, indices: list[int]) -> int:
    """Remove episodes; the ones after them are renumbered. Returns how many episodes are left.

    LeRobot writes the edited dataset into a staging folder (re-encoding any video file shared with a
    deleted episode); only when that succeeded is it swapped in, so a failure leaves the dataset as it was.
    """
    from lerobot.datasets import LeRobotDataset
    from lerobot.datasets.dataset_tools import delete_episodes as lerobot_delete

    path = dataset_dir(name)
    info = _info(path)
    if problem := _problem(path, info):
        raise RuntimeError(problem)
    total = info["total_episodes"]
    if not indices or not all(isinstance(i, int) and not isinstance(i, bool) for i in indices):
        raise ValueError("choose at least one episode")
    chosen = sorted(set(indices))
    if chosen[0] < 0 or chosen[-1] >= total:
        raise ValueError(f"{name} has episodes 0 to {total - 1}")
    if len(chosen) == total:
        raise ValueError("those are all of its episodes: delete the whole dataset instead")
    staging = Path(tempfile.mkdtemp(prefix=f".{name}.delete-", dir=root()))
    try:
        lerobot_delete(LeRobotDataset(repo_id(name), root=path), chosen, output_dir=staging / "kept",
                       repo_id=repo_id(name))
        # the episodes after a deleted one move down: renumber the DART marks the same way
        kept = [i for i in range(total) if i not in set(chosen)]
        _write_perturbed(staging / "kept", [new for new, old in enumerate(kept) if old in perturbed(path)])
        old_edits = cleaning(path)
        _write_cleaning(staging / "kept", {new: old_edits[old] for new, old in enumerate(kept) if old in old_edits})
        path.rename(staging / "old")
        try:
            (staging / "kept").rename(path)
        except OSError:
            (staging / "old").rename(path)
            raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    left = _info(path)["total_episodes"]
    log.info("deleted episodes %s from %s; %d left", chosen, name, left)
    return left


def delete_dataset(name: str) -> None:
    path = dataset_dir(name)
    shutil.rmtree(path)
    log.info("deleted dataset %s", path)


# ---- recording ------------------------------------------------------------------------------


def settings(raw: dict, cfg) -> dict:
    """The recording form, checked. Mirrors LeLab's: dataset, task, episodes, episode time, reset time."""
    def number_value(label, value, low, high):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value != int(value):
            raise ValueError(f"{label} must be a whole number")
        if not low <= value <= high:
            raise ValueError(f"{label} must be between {low} and {high}")
        return int(value)

    def number(key, low, high):
        return number_value(key.replace("_", " "), raw.get(key), low, high)

    name = check_name(raw.get("dataset"))
    task = str(raw.get("task") or "").strip() or cfg.grasp.task
    new = bool(raw.get("new"))
    exists = (root() / name).exists()
    if new and exists:
        raise ValueError(f"a dataset called {name} already exists: choose another name, or Record more into it")
    if not new:
        path = dataset_dir(name)
        info = _info(path)
        if problem := _problem(path, info):
            raise RuntimeError(problem)
        _check_compatible(info, cfg)
    dart = raw.get("dart_percent", round(cfg.dart.fraction * 100))
    return {"dataset": name, "new": new, "task": task[:200],
            "dart_fraction": number_value("dart percent", dart, 0, 100) / 100,
            "num_episodes": number("num_episodes", 1, 200),
            "episode_time_s": number("episode_time_s", 3, 600),
            "reset_time_s": number("reset_time_s", 0, 600)}  # 0 = wait for "Start Next Episode"


def features(cfg) -> dict:
    from lerobot.utils.feature_utils import hw_to_dataset_features

    names = {f"{j}.pos": float for j in JOINTS}
    size = (cfg.grasp.image_height, cfg.grasp.image_width, 3)
    return {**hw_to_dataset_features(names, "action"),
            **hw_to_dataset_features({**names, "wrist": size}, "observation")}


def _check_compatible(info: dict, cfg) -> None:
    """Appending only works with the same frame rate, joints and picture size the dataset was made with."""
    want, have = features(cfg), info["features"]
    differences = []
    if info["fps"] != cfg.grasp.fps:
        differences.append(f"{info['fps']} fps (config: {cfg.grasp.fps})")
    for key in want:
        if key not in have or list(have[key]["shape"]) != list(want[key]["shape"]) \
                or have[key].get("names") != want[key].get("names"):
            differences.append(f"{key} {have.get(key, {}).get('shape')} (config: {list(want[key]['shape'])})")
    if info.get("robot_type") != ROBOT_TYPE:
        differences.append(f"robot {info.get('robot_type')}")
    if differences:
        raise ValueError("this dataset was recorded differently, so new episodes cannot be added to it: "
                         + "; ".join(differences) + ". Record into a new dataset instead.")


class Session:
    """One recording session: the leader arm, the dataset being written and the counters the Lab shows.

    Every dataset operation takes `lock`, because Kill can close the session from the web server while
    the state machine is adding a frame in a worker thread.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.lock = threading.Lock()
        self.leader = None
        self.dataset = None
        self.starting = False
        self.settings: dict = {}
        self.phase = ""
        self.phase_id = 0  # counts phase changes, so the Lab plays each sound cue once
        self.phase_started = self.started = 0.0
        self.saved = 0
        self.created = False
        self.last: dict | None = None  # what the previous session achieved, for the Lab's summary

    @property
    def open(self) -> bool:
        return self.dataset is not None

    @property
    def busy(self) -> bool:
        return self.starting or self.open

    def set_phase(self, phase: str) -> None:
        self.phase, self.phase_started, self.phase_id = phase, time.monotonic(), self.phase_id + 1

    def start(self, s: dict) -> None:
        """Connect the leader arm and create (or reopen) the dataset. Blocking."""
        from lerobot.datasets import LeRobotDataset

        path = root() / s["dataset"]
        leader = open_feetech("leader", self.cfg.leader.port, config.path(self.cfg.leader.calibration))
        try:
            if s["new"]:
                if path.exists():
                    raise ValueError(f"a dataset called {s['dataset']} already exists")
                root().mkdir(parents=True, exist_ok=True)
                try:
                    dataset = LeRobotDataset.create(repo_id(s["dataset"]), self.cfg.grasp.fps,
                                                    features=features(self.cfg), root=path, robot_type=ROBOT_TYPE,
                                                    use_videos=True, image_writer_threads=4)
                except Exception:
                    shutil.rmtree(path, ignore_errors=True)
                    raise
            else:
                _check_compatible(_info(dataset_dir(s["dataset"])), self.cfg)
                dataset = LeRobotDataset.resume(repo_id(s["dataset"]), root=path, image_writer_threads=4)
        except Exception:
            leader.bus.disconnect(disable_torque=True)
            raise
        with self.lock:
            self.leader, self.dataset, self.settings, self.created = leader, dataset, s, s["new"]
            self.saved, self.last, self.started = 0, None, time.monotonic()
            self.set_phase("preparing")

    def leader_pose(self) -> dict[str, float]:
        with self.lock:
            if self.leader is None:
                raise RuntimeError("the recording session has ended")
            return self.leader.bus.sync_read("Present_Position", num_retry=3)

    def add(self, joints: dict[str, float], action: dict[str, float], frame: np.ndarray) -> None:
        with self.lock:
            if self.dataset is None:
                return
            self.dataset.add_frame({
                "observation.state": np.array([joints[j] for j in JOINTS], dtype=np.float32),
                "action": np.array([action[j] for j in JOINTS], dtype=np.float32),
                "observation.images.wrist": wrist_image(frame, self.cfg.grasp.image_width, self.cfg.grasp.image_height),
                "task": self.settings["task"],
            })

    def save(self, perturbed_take: bool = False) -> int:
        """Keep the episode just recorded. Returns how many episodes the dataset now has."""
        with self.lock:
            self.dataset.save_episode()
            self.saved += 1
            total = self.dataset.meta.total_episodes
            if perturbed_take:
                path = root() / self.settings["dataset"]
                _write_perturbed(path, perturbed(path) | {total - 1})
            return total

    def discard(self) -> None:
        with self.lock:
            self._discard()

    def _discard(self) -> None:
        if self.dataset is None or not self.dataset.has_pending_frames():
            return
        # LeRobot deletes the take's frame folder without waiting for its image writer, and the next take
        # reuses that folder: frames still in the queue would end up in the next episode's video.
        writer = self.dataset.writer.image_writer
        if writer is not None:
            writer.wait_until_done()
        self.dataset.clear_episode_buffer()

    def close(self, reason: str) -> None:
        """End the session: drop any unsaved take, finalise the dataset (it is unreadable until then) and
        let go of the leader arm. A dataset this session created and saved nothing into is removed."""
        with self.lock:
            if self.dataset is None:
                return
            name, dataset, leader = self.settings["dataset"], self.dataset, self.leader
            self.dataset = self.leader = None
            note = None
            try:
                if dataset.has_pending_frames():
                    writer = dataset.writer.image_writer
                    if writer is not None:
                        writer.wait_until_done()
                    dataset.clear_episode_buffer()
                dataset.finalize()
                total = dataset.meta.total_episodes
                if self.created and total == 0:
                    shutil.rmtree(root() / name, ignore_errors=True)
                    note = f"nothing was saved, so {name} was not kept"
            except Exception as e:
                log.exception("closing dataset %s", name)
                total, note = None, f"the dataset could not be finalised: {e}"
            finally:
                try:
                    leader.bus.disconnect(disable_torque=True)
                except Exception as e:
                    log.warning("closing the leader arm: %s", e)
            self.last = {"dataset": name, "saved": self.saved, "total": total, "reason": reason, "note": note,
                         "kept": not (self.created and total == 0) and total is not None}
            log.info("recording session ended (%s): %s", reason, self.last)

    def status(self, paused: bool, switching: bool, paused_at: float = 0.0) -> dict:
        """`switching`: the robot is about to change state, which wipes button presses: report "preparing".
        `paused_at`: when Safe Stop was entered; a paused session's clock counts from then."""
        dataset = self.dataset  # read once: Kill may close the session at any moment
        if dataset is None:
            return {"open": False, "starting": self.starting, "last": self.last}
        s, now = self.settings, time.monotonic()
        phase = "preparing" if switching else "paused" if paused else self.phase
        limit = {"recording": s["episode_time_s"], "resetting": s["reset_time_s"] or None}.get(phase)
        return {"open": True, "starting": False, "last": None, **s, "phase": phase,
                "phase_id": 0 if switching else -self.phase_id if paused else self.phase_id,
                "episode": min(self.saved + 1, s["num_episodes"]), "saved": self.saved,
                "done": self.saved >= s["num_episodes"], "dataset_episodes": dataset.meta.total_episodes,
                "phase_elapsed_s": round(now - (paused_at if phase == "paused" else self.phase_started), 1),
                "phase_limit_s": limit,
                "session_elapsed_s": round(now - self.started, 1)}
