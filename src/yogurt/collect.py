"""Wake-word clips and labelled wrist images: recorded, browsed and tidied up from the Lab.

Everything lands under data/, in the layout the trainers in yogurt.train read back:
    data/wake_word/<label>/<time>[_phrase].wav
    data/<dataset>/<label>/<burst>/<n>.jpg            (one folder per Start/Stop burst)
    data/yogurt_detection/labels/<burst>/<n>.txt      (the detector's boxes, mirroring images/)
The trainers hold out whole bursts for validation, so bursts are moved and deleted as a whole or
image by image, never renamed or merged. Grasp demonstrations are LeRobot datasets: see lab.py.
"""

import logging
import re
import shutil
import time
import wave
from pathlib import Path

import cv2
import numpy as np

from . import audio, config

log = logging.getLogger("yogurt.collect")
DATA = config.ROOT / "data"

# dataset -> the labels it accepts. Folder layout: data/<dataset>/<label>/<session>/<n>.jpg
IMAGE_SETS = {
    "grasp_verify": ["held", "empty"],
    "handoff_grab": ["grabbed", "not_grabbed"],
    "yogurt_detection": ["images"],
}
AUDIO_LABELS = ["positive", "negative", "background"]
# how much data the guides in data/ ask for (background: seconds of audio, the rest: clips or images)
TARGETS = {"wake_word/positive": 150, "wake_word/negative": 300, "wake_word/background": 600,
           "grasp_verify/held": 300, "grasp_verify/empty": 300, "handoff_grab/grabbed": 300,
           "handoff_grab/not_grabbed": 300, "yogurt_detection/images": 300}
TRAIN = {"wake_word": "uv run python -m yogurt.train.wake_word",
         "grasp_verify": "uv run python -m yogurt.train.classifier grasp_verify",
         "handoff_grab": "uv run python -m yogurt.train.classifier handoff_grab",
         "yogurt_detection": "uv run python -m yogurt.train.detector train"}
MODELS = {"wake_word": "wake_word", "grasp_verify": "grasp_verify", "handoff_grab": "handoff_grab",
          "yogurt_detection": "yogurt_detector"}
PROMPTS = config.ROOT / "data" / "wake_word" / "prompts.json"  # what the Lab asks people to say
SHOTS = config.ROOT / "data" / "shots.json"                    # what it asks them to photograph
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")  # file and folder names the Lab may touch


def _stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _unique(path: Path) -> Path:
    """path, or path-2, path-3 ... if it is taken (two bursts in the same second must not share a folder)."""
    candidate, n = path, 2
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}-{n}{path.suffix}")
        n += 1
    return candidate


def _name(value) -> str:
    if not isinstance(value, str) or not NAME.fullmatch(value):
        raise LookupError(f"not a valid name: {value!r}")
    return value


# ---- wake-word clips ------------------------------------------------------------------------


def _clip_dir(label: str) -> Path:
    if label not in AUDIO_LABELS:
        raise ValueError(f"label must be one of {AUDIO_LABELS}")
    return DATA / "wake_word" / label


def clip_file(label: str, name: str) -> Path:
    file = _clip_dir(label) / _name(name)
    if file.suffix != ".wav" or not file.is_file():
        raise LookupError(f"no clip {label}/{name}")
    return file


def save_audio_clip(label: str, clip: np.ndarray, phrase: str = "") -> Path:
    """One wake-word clip: data/wake_word/<label>/<time>[_phrase].wav"""
    slug = re.sub("-+", "-", "".join(c if c.isalnum() else "-" for c in phrase.lower())).strip("-")[:40]
    name = f"{_stamp()}-{int(time.time() * 1000) % 1000:03d}{'_' + slug if slug else ''}.wav"
    file = _unique(_clip_dir(label) / name)
    audio.write_wav(file, clip)
    return file


_clip_cache: dict = {}  # (path, mtime, size) -> (seconds, peak): reading every clip on each listing is slow


def clip_info(file: Path, label: str) -> dict:
    stat = file.stat()
    key = (str(file), stat.st_mtime_ns, stat.st_size)
    if key not in _clip_cache:
        try:
            with wave.open(str(file), "rb") as w:
                seconds = w.getnframes() / w.getframerate()
                samples = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.int32)
            peak = float(np.abs(samples).max()) / 32767 if len(samples) else 0.0
            _clip_cache[key] = (round(seconds, 2), round(peak, 3))
        except (wave.Error, EOFError, ValueError, ZeroDivisionError):
            _clip_cache[key] = (None, None)
    seconds, peak = _clip_cache[key]
    stem = file.stem
    return {"label": label, "name": file.name, "phrase": stem.split("_", 1)[1].replace("-", " ") if "_" in stem else "",
            "seconds": seconds, "peak": peak, "time": stat.st_mtime}


def clips() -> list[dict]:
    """Every clip, newest first."""
    out = [clip_info(f, label) for label in AUDIO_LABELS for f in _clip_dir(label).glob("*.wav")
           if NAME.fullmatch(f.name)]
    return sorted(out, key=lambda c: c["time"], reverse=True)


def _clip_list(items) -> list[Path]:
    if not isinstance(items, list) or not items:
        raise ValueError("choose at least one clip")
    return [clip_file(i.get("label"), i.get("name")) for i in items]


def delete_clips(items) -> int:
    files = _clip_list(items)
    for f in files:
        f.unlink()
    return len(files)


def move_clips(items, to: str) -> int:
    """Relabel clips, e.g. a 'positive' in which the phrase was fumbled becomes a 'negative'."""
    target, files = _clip_dir(to), _clip_list(items)
    target.mkdir(parents=True, exist_ok=True)
    for f in files:
        if f.parent != target:
            f.rename(_unique(target / f.name))
    return len(files)


class ImageCapture:
    """Saves wrist-camera frames at a steady rate while it is running. One instance, one burst at a time."""

    def __init__(self):
        self.folder: Path | None = None
        self.saved = 0
        self._next = 0.0
        self._period = 0.5

    def start(self, dataset: str, label: str, per_second: float) -> None:
        if label not in IMAGE_SETS.get(dataset, []):
            raise ValueError(f"unknown dataset/label {dataset}/{label}")
        folder = _unique(DATA / dataset / label / _stamp())
        folder.mkdir(parents=True)
        self.folder, self.saved, self._next, self._period = folder, 0, 0.0, 1.0 / per_second

    def stop(self) -> int:
        self.folder = None
        return self.saved

    def offer(self, frame: np.ndarray) -> None:
        """Called with every new wrist frame; keeps the ones that are due."""
        now = time.monotonic()
        if self.folder is not None and now >= self._next:
            cv2.imwrite(str(self.folder / f"{self.saved:05d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            self.saved, self._next = self.saved + 1, now + self._period


def counts() -> dict:
    """How much of each kind of data exists."""
    result = {f"wake_word/{label}": len(list(_clip_dir(label).glob("*.wav"))) for label in AUDIO_LABELS}
    for dataset, labels in IMAGE_SETS.items():
        for label in labels:
            result[f"{dataset}/{label}"] = len(list((DATA / dataset / label).rglob("*.jpg")))
    result["yogurt_detection/labelled"] = len(list((DATA / "yogurt_detection" / "labels").rglob("*.txt")))
    return result


# ---- wrist images ---------------------------------------------------------------------------


def _set_dir(dataset: str, label: str) -> Path:
    if label not in IMAGE_SETS.get(dataset, []):
        raise ValueError(f"unknown image set {dataset}/{label}")
    return DATA / dataset / label


def burst_dir(dataset: str, label: str, burst: str) -> Path:
    folder = _set_dir(dataset, label) / _name(burst)
    if not folder.is_dir():
        raise LookupError(f"no burst {dataset}/{label}/{burst}")
    return folder


def image_file(dataset: str, label: str, burst: str, name: str) -> Path:
    file = burst_dir(dataset, label, burst) / _name(name)
    if file.suffix != ".jpg" or not file.is_file():
        raise LookupError(f"no image {burst}/{name}")
    return file


def _label_file(image: Path) -> Path:
    images = DATA / "yogurt_detection" / "images"
    return (DATA / "yogurt_detection" / "labels" / image.relative_to(images)).with_suffix(".txt")


def _box(image: Path):
    """The detector label of a picture: [x1, y1, x2, y2] (fractions), "none" (no yogurt) or None (not labelled)."""
    label = _label_file(image)
    if not label.exists():
        return None
    parts = label.read_text().split()
    if len(parts) < 5:
        return "none"
    cx, cy, w, h = (float(v) for v in parts[1:5])
    return [round(cx - w / 2, 4), round(cy - h / 2, 4), round(cx + w / 2, 4), round(cy + h / 2, 4)]


def image_sets() -> dict:
    """Images, bursts (and for the detector, labelled pictures) per dataset/label."""
    out = {}
    for dataset, labels in IMAGE_SETS.items():
        for label in labels:
            folder = DATA / dataset / label
            bursts = [b for b in folder.iterdir() if b.is_dir() and NAME.fullmatch(b.name)] if folder.is_dir() else []
            entry = {"images": sum(len(list(b.glob("*.jpg"))) for b in bursts), "bursts": len(bursts),
                     "target": TARGETS[f"{dataset}/{label}"]}
            if dataset == "yogurt_detection":
                boxes = [_box(i) for b in bursts for i in b.glob("*.jpg")]
                entry["labelled"] = sum(b is not None for b in boxes)
                entry["no_yogurt"] = sum(b == "none" for b in boxes)
            out[f"{dataset}/{label}"] = entry
    return out


def bursts(dataset: str, label: str) -> list[dict]:
    """The bursts of one image set, newest first."""
    folder = _set_dir(dataset, label)
    out = []
    for b in (folder.iterdir() if folder.is_dir() else []):
        if not (b.is_dir() and NAME.fullmatch(b.name)):
            continue
        images = sorted(i.name for i in b.glob("*.jpg"))
        entry = {"name": b.name, "images": len(images), "cover": images[len(images) // 2] if images else None,
                 "time": b.stat().st_mtime}
        if dataset == "yogurt_detection":
            entry["labelled"] = sum(_box(b / i) is not None for i in images)
        out.append(entry)
    return sorted(out, key=lambda e: e["name"], reverse=True)


def burst_images(dataset: str, label: str, burst: str) -> list[dict]:
    folder = burst_dir(dataset, label, burst)
    out = []
    for i in sorted(folder.glob("*.jpg")):
        entry = {"name": i.name, "v": i.stat().st_mtime_ns}
        if dataset == "yogurt_detection":
            entry["box"] = _box(i)
        out.append(entry)
    return out


def thumbnail(file: Path, width: int = 320) -> bytes:
    image = cv2.imread(str(file))
    if image is None:
        raise LookupError(f"{file.name} is not a readable picture")
    h, w = image.shape[:2]
    small = cv2.resize(image, (width, max(1, round(h * width / w))), interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()


def _drop_label(image: Path) -> None:
    if image.is_relative_to(DATA / "yogurt_detection" / "images"):
        _label_file(image).unlink(missing_ok=True)


def _prune(folder: Path) -> None:
    """Remove a burst folder (and its folder of boxes) once it is empty."""
    if folder.is_dir() and not any(folder.iterdir()):
        folder.rmdir()
        if folder.is_relative_to(DATA / "yogurt_detection" / "images"):
            boxes = _label_file(folder / "x.jpg").parent
            if boxes.is_dir() and not any(boxes.iterdir()):
                boxes.rmdir()


def delete_images(dataset: str, label: str, burst: str, names) -> int:
    """Delete some pictures of a burst (with their boxes). Returns how many pictures the burst has left."""
    if not isinstance(names, list) or not names:
        raise ValueError("choose at least one picture")
    files = [image_file(dataset, label, burst, n) for n in names]
    for f in files:
        _drop_label(f)
        f.unlink()
    folder = files[0].parent
    left = len(list(folder.glob("*.jpg")))
    _prune(folder)
    return left


def delete_burst(dataset: str, label: str, burst: str) -> int:
    folder = burst_dir(dataset, label, burst)
    count = len(list(folder.glob("*.jpg")))
    if dataset == "yogurt_detection":
        shutil.rmtree(_label_file(folder / "x.jpg").parent, ignore_errors=True)
    shutil.rmtree(folder)
    return count


def move_burst(dataset: str, label: str, burst: str, to: str) -> str:
    """Give a whole burst the other label (recorded with the wrong one selected). Returns its new name."""
    folder = burst_dir(dataset, label, burst)
    if to == label or to not in IMAGE_SETS[dataset]:
        raise ValueError(f"{dataset} pictures can be labelled {' or '.join(IMAGE_SETS[dataset])}")
    target = _unique(_set_dir(dataset, to) / burst)
    target.parent.mkdir(parents=True, exist_ok=True)
    folder.rename(target)
    return target.name


# ---- box labels for the yogurt detector -----------------------------------------------------


def next_unlabelled() -> tuple[str, str] | None:
    """(burst, picture) of the first detector picture without a box, oldest burst first."""
    images = DATA / "yogurt_detection" / "images"
    for image in sorted(images.rglob("*.jpg")):
        if _box(image) is None and image.parent.parent == images:
            return image.parent.name, image.name
    return None


def save_box(burst: str, name: str, box) -> None:
    """box = [x1, y1, x2, y2] as fractions of the picture, "none" for 'no yogurt here', None to forget the label.

    Written in YOLO format (class cx cy w h); an empty file marks a picture with no yogurt.
    """
    label = _label_file(image_file("yogurt_detection", "images", burst, name))
    if box is None:
        label.unlink(missing_ok=True)
        return
    label.parent.mkdir(parents=True, exist_ok=True)
    if box == "none":
        label.write_text("")
        return
    if not (isinstance(box, list) and len(box) == 4):
        raise ValueError("a box is [x1, y1, x2, y2]")
    x1, y1, x2, y2 = (min(max(float(v), 0.0), 1.0) for v in box)
    if abs(x2 - x1) < 0.005 or abs(y2 - y1) < 0.005:
        raise ValueError("that box is too small: drag around the whole cup")
    label.write_text(f"0 {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {abs(x2 - x1):.6f} {abs(y2 - y1):.6f}\n")
