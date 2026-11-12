"""What the cameras see.

Base camera:  faces, where each one is (bearing) and how much its mouth is moving.
Wrist camera: yogurt cups (detector) and yes/no questions about the gripper (classifiers).
"""

import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

YUNET = Path(__file__).parent / "assets" / "face_detection_yunet_2023mar.onnx"
DETECT_WIDTH = 640      # faces are found on a frame shrunk to this width
PATCH = (32, 24)        # mouth / eye patches are compared at this size
MOUTH_HISTORY_S = 12.0


def bearing_deg(x: float, width: int, hfov_deg: float) -> float:
    """Angle of image column x from the optical axis; positive = right of centre."""
    focal = (width / 2) / math.tan(math.radians(hfov_deg) / 2)
    return math.degrees(math.atan((x - width / 2) / focal))


# ---- faces ----------------------------------------------------------------------------------


@dataclass
class Face:
    id: int
    box: tuple[int, int, int, int]              # x, y, w, h in frame pixels
    bearing: float                              # degrees, positive = right of the base camera's axis
    seen: float                                 # time.monotonic() of the last sighting
    mouth: float = 0.0                          # smoothed mouth movement, 0 = still
    points: np.ndarray | None = None            # smoothed landmarks: eyes, nose, mouth corners
    history: deque = field(default_factory=deque)  # (time, raw mouth movement)
    patches: tuple | None = None                # last mouth and eye patches

    def mouth_since(self, since: float) -> float:
        """Average mouth movement since a moment in the past (e.g. since the wake word began)."""
        values = [v for t, v in self.history if t >= since]
        return float(np.mean(values)) if values else 0.0


def _patch(gray: np.ndarray, cx: float, cy: float, w: float, h: float) -> np.ndarray | None:
    x0, y0, x1, y1 = int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2)
    if x0 < 0 or y0 < 0 or x1 > gray.shape[1] or y1 > gray.shape[0] or x1 - x0 < 6 or y1 - y0 < 4:
        return None
    patch = cv2.resize(gray[y0:y1, x0:x1], PATCH, interpolation=cv2.INTER_AREA).astype(np.float32)
    return (patch - patch.mean()) / (patch.std() + 10.0)


class FaceTracker:
    """Finds faces with OpenCV's YuNet, follows them from frame to frame, and measures mouth movement.

    Mouth movement = how much the mouth patch changes between frames, minus how much the eye patch
    changes (which cancels head motion and camera noise). Talking makes it rise; a still or merely
    nodding face stays near zero.
    """

    def __init__(self, hfov_deg: float):
        self.hfov_deg = hfov_deg
        self.detector = cv2.FaceDetectorYN.create(str(YUNET), "", (DETECT_WIDTH, DETECT_WIDTH), 0.7, 0.3, 20)
        self.faces: dict[int, Face] = {}
        self.lock = threading.Lock()   # update() runs in a worker thread; readers take the lock too
        self._next_id = 1
        self._last = 0.0

    def update(self, frame: np.ndarray) -> None:
        with self.lock:
            self._update(frame)

    def _update(self, frame: np.ndarray) -> None:
        now = time.monotonic()
        dt, self._last = min(now - self._last, 0.5), now
        height, width = frame.shape[:2]
        scale = DETECT_WIDTH / width
        small = cv2.resize(frame, (DETECT_WIDTH, round(height * scale)), interpolation=cv2.INTER_AREA)
        self.detector.setInputSize((small.shape[1], small.shape[0]))
        _, found = self.detector.detect(small)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        unmatched = dict(self.faces)
        for row in found if found is not None else []:
            x, y, w, h = (row[:4] / scale).tolist()
            points = row[4:14].reshape(5, 2) / scale
            centre = np.array([x + w / 2, y + h / 2])
            # Same person as last frame = the closest known face, if it is within about a face width.
            best = min(unmatched.values(), default=None,
                       key=lambda f: np.linalg.norm(centre - (f.box[0] + f.box[2] / 2, f.box[1] + f.box[3] / 2)))
            if best is not None and np.linalg.norm(
                centre - (best.box[0] + best.box[2] / 2, best.box[1] + best.box[3] / 2)
            ) < max(w, best.box[2]):
                face = unmatched.pop(best.id)
                face.points = 0.5 * face.points + 0.5 * points
            else:
                face = Face(self._next_id, (0, 0, 0, 0), 0.0, now, points=points)
                self.faces[face.id] = face
                self._next_id += 1
            face.box, face.seen = (int(x), int(y), int(w), int(h)), now
            face.bearing = bearing_deg(centre[0], width, self.hfov_deg)
            self._measure_mouth(face, gray, now, dt)

        for face in list(self.faces.values()):
            if now - face.seen > 1.5:
                del self.faces[face.id]

    def visible(self, max_age_s: float = 0.5) -> list[Face]:
        """Faces seen in the last moment, left to right."""
        now = time.monotonic()
        with self.lock:
            return sorted((f for f in self.faces.values() if now - f.seen <= max_age_s), key=lambda f: f.bearing)

    def speaker(self, audio_bearing: float | None, since: float, sigma_deg: float) -> Face | None:
        faces = self.visible()
        with self.lock:
            return pick_speaker(faces, audio_bearing, since, sigma_deg)

    def _measure_mouth(self, face: Face, gray: np.ndarray, now: float, dt: float) -> None:
        right_eye, left_eye, _, right_mouth, left_mouth = face.points
        eye_gap = float(np.linalg.norm(left_eye - right_eye))
        eyes, mouth = (right_eye + left_eye) / 2, (right_mouth + left_mouth) / 2
        patches = (
            _patch(gray, mouth[0], mouth[1] + 0.1 * eye_gap, 1.3 * eye_gap, 0.9 * eye_gap),
            _patch(gray, eyes[0], eyes[1], 1.6 * eye_gap, 0.6 * eye_gap),
        )
        raw = 0.0
        if face.patches and all(p is not None for p in patches + face.patches):
            mouth_change = float(np.abs(patches[0] - face.patches[0]).mean())
            eye_change = float(np.abs(patches[1] - face.patches[1]).mean())
            raw = max(0.0, mouth_change - eye_change)
        face.patches = patches
        face.mouth += (raw - face.mouth) * min(1.0, dt / 0.4)
        face.history.append((now, raw))
        while face.history and now - face.history[0][0] > MOUTH_HISTORY_S:
            face.history.popleft()


def pick_speaker(faces: list[Face], audio_bearing: float | None, since: float, sigma_deg: float) -> Face | None:
    """Who said it? The face whose mouth moved most while the voice was heard, and that sits
    where the microphones heard the voice. Each cue scores 0..1; the best total wins."""
    if not faces:
        return None
    mouths = [f.mouth_since(since) for f in faces]
    top = max(mouths)

    def score(face: Face, mouth: float) -> float:
        total = mouth / top if top > 0.02 else 0.0
        if audio_bearing is not None:
            total += math.exp(-0.5 * ((face.bearing - audio_bearing) / sigma_deg) ** 2)
        return total

    return max(zip(faces, mouths, strict=True), key=lambda pair: score(*pair))[0]


# ---- wrist camera models --------------------------------------------------------------------


@dataclass
class Detection:
    box: tuple[int, int, int, int]  # x1, y1, x2, y2 in frame pixels
    score: float
    label: str

    @property
    def centre(self) -> tuple[float, float]:
        return (self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2


class YogurtDetector:
    """A YOLO detector exported by yogurt.train.detector. Keeps only the labels that mean 'yogurt cup'."""

    def __init__(self, model, labels: list[str]):
        self.model, self.labels = model, set(labels)
        names = model.meta["names"]  # a list, or Ultralytics' {index: name}
        self.names = {int(k): v for k, v in (names.items() if isinstance(names, dict) else enumerate(names))}

    def detect(self, frame: np.ndarray, min_score: float) -> list[Detection]:
        height, width = frame.shape[:2]
        mw, mh = self.model.width, self.model.height
        scale = min(mw / width, mh / height)
        nw, nh = round(width * scale), round(height * scale)
        dx, dy = (mw - nw) // 2, (mh - nh) // 2
        canvas = np.full((mh, mw, 3), 114, np.uint8)
        canvas[dy:dy + nh, dx:dx + nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
        outputs = self.model.run(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        if self.model.hailo:
            # HailoRT's built-in NMS: for each class, rows of [ymin, xmin, ymax, xmax, score] in 0..1.
            rows = [(x1 * mw, y1 * mh, x2 * mw, y2 * mh, score, cls)
                    for cls, dets in enumerate(outputs[0]) for y1, x1, y2, x2, score in dets]
        else:
            rows = outputs[0][0]  # Ultralytics end-to-end export: x1, y1, x2, y2, score, class
        detections = []
        for x1, y1, x2, y2, score, cls in rows:
            label = self.names.get(int(cls), str(int(cls)))
            if score >= min_score and label in self.labels:
                box = (max(0, int((x1 - dx) / scale)), max(0, int((y1 - dy) / scale)),
                       min(width, int((x2 - dx) / scale)), min(height, int((y2 - dy) / scale)))
                detections.append(Detection(box, float(score), label))
        return sorted(detections, key=lambda d: -d.score)


class Classifier:
    """A two-class image classifier exported by yogurt.train.classifier; prob() is P(second class)."""

    def __init__(self, model):
        self.model = model
        self.classes = model.meta["classes"]

    def prob(self, frame: np.ndarray) -> float:
        small = cv2.resize(frame, (self.model.width, self.model.height), interpolation=cv2.INTER_AREA)
        logits = np.asarray(self.model.run(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))[0], dtype=np.float64).reshape(-1)
        odds = np.exp(logits - logits.max())
        return float(odds[1] / odds.sum())
