"""Drawing on the two video feeds, so a person watching can see what the robot sees and why it moves.

Base camera:  every face with its role (REQUESTER / OTHER / TARGET) and mouth movement, the direction
              the microphones hear the voice from, and where the arm is pointing.
Wrist camera: yogurt detections, how far the cup is from the centre line, and the live answers of the
              grasp-verification and hand-off models.
"""

import math

import cv2
import numpy as np

STREAM_WIDTH = 800
WHITE, GREY, BLACK = (255, 255, 255), (170, 170, 170), (0, 0, 0)
GREEN, BLUE, YELLOW = (90, 210, 90), (230, 160, 60), (0, 220, 255)
MAGENTA, ORANGE, RED = (220, 90, 230), (0, 150, 255), (60, 60, 240)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _label(img, text: str, x: int, y: int, colour=WHITE, scale: float = 0.5) -> None:
    (w, h), _ = cv2.getTextSize(text, FONT, scale, 1)
    x = max(2, min(x, img.shape[1] - w - 4))
    cv2.rectangle(img, (x - 2, y - h - 4), (x + w + 2, y + 3), BLACK, -1)
    cv2.putText(img, text, (x, y), FONT, scale, colour, 1, cv2.LINE_AA)


def _bar(img, name: str, value: float, threshold: float, row: int) -> None:
    """A labelled 0..1 gauge with its decision threshold marked."""
    x, y, w = 10, img.shape[0] - 18 - 26 * row, 220
    cv2.rectangle(img, (x, y), (x + w, y + 14), BLACK, -1)
    cv2.rectangle(img, (x, y), (x + int(w * min(max(value, 0), 1)), y + 14), GREEN if value >= threshold else GREY, -1)
    cv2.line(img, (x + int(w * threshold), y - 3), (x + int(w * threshold), y + 17), YELLOW, 2)
    _label(img, f"{name} {value:.2f}", x + w + 8, y + 12)


PLAIN = str.maketrans({"°": " deg", "“": '"', "”": '"', "…": "...", "’": "'", "‘": "'", "–": "-", "—": "-"})


def _banner(img, brain) -> None:
    """State and the current 'why' sentence across the top."""
    detail = brain.detail.translate(PLAIN).encode("ascii", "ignore").decode()  # OpenCV's font is ASCII-only
    words, lines, line = detail.split(), [], ""
    for word in words:
        if len(line) + len(word) > 88:
            lines.append(line)
            line = ""
        line = f"{line} {word}".strip()
    lines = (lines + [line])[:3]
    cv2.rectangle(img, (0, 0), (img.shape[1], 30 + 18 * len(lines)), BLACK, -1)
    cv2.putText(img, brain.state.replace("_", " "), (8, 21), FONT, 0.7, YELLOW, 2, cv2.LINE_AA)
    for i, text in enumerate(lines):
        cv2.putText(img, text, (8, 42 + 18 * i), FONT, 0.45, WHITE, 1, cv2.LINE_AA)


def _shrink(frame: np.ndarray) -> tuple[np.ndarray, float]:
    scale = STREAM_WIDTH / frame.shape[1]
    return cv2.resize(frame, (STREAM_WIDTH, round(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA), scale


def no_signal(name: str, error: str | None) -> np.ndarray:
    img = np.zeros((450, STREAM_WIDTH, 3), np.uint8)
    _label(img, f"{name} camera: {error or 'waiting for frames'}"[:95], 20, 225, RED, 0.5)
    return img


def draw_base(frame: np.ndarray, brain) -> np.ndarray:
    img, scale = _shrink(frame)
    height, width = img.shape[:2]
    focal = (width / 2) / math.tan(math.radians(brain.cfg.cameras.base.hfov_deg) / 2)

    def column(bearing: float) -> int:
        return int(width / 2 + focal * math.tan(math.radians(max(-85.0, min(85.0, bearing)))))

    request = brain.request
    others = {p.id for p in request.others}
    for face in brain.faces.visible(1.0):
        x, y, w, h = (int(v * scale) for v in face.box)
        role, colour = "", GREY
        if request.requester and face.id == request.requester.id:
            role, colour = "REQUESTER", GREEN
        elif face.id in others:
            role, colour = "OTHER", BLUE
        elif brain.view.get("speaker") == face.id:
            role, colour = "speaking?", GREEN
        target = request.target is not None and face.id == request.target.id
        cv2.rectangle(img, (x, y), (x + w, y + h), colour, 2)
        if target:
            cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), YELLOW, 2)
        _label(img, f"#{face.id} {role}{' = TARGET' if target else ''} {face.bearing:+.0f} deg", x, y - 8, colour)
        cv2.rectangle(img, (x, y + h + 4), (x + int(w * min(1.0, face.mouth * 8)), y + h + 10), colour, -1)
        _label(img, "mouth", x + w + 6, y + h + 12, colour, 0.4)

    if (voice := brain.view.get("voice")) is not None:
        cv2.line(img, (column(voice), 80), (column(voice), height), MAGENTA, 2)
        _label(img, f"voice {voice:+.0f} deg", column(voice) + 6, height - 30, MAGENTA)
    if request.target is not None and brain.state in ("CHECK", "GRASP", "VERIFY", "HANDOFF"):
        x = column(request.target.bearing)
        cv2.drawMarker(img, (x, height - 60), YELLOW, cv2.MARKER_TRIANGLE_DOWN, 22, 2)
        _label(img, f"hand-off target {request.target.bearing:+.0f} deg", x + 14, height - 56, YELLOW)
    # Where the arm points, in this camera's terms.
    aim = (brain.arm.pos["shoulder_pan"] - brain.cfg.base_camera.yaw_deg) * brain.cfg.base_camera.pan_sign
    cv2.drawMarker(img, (column(aim), height - 12), ORANGE, cv2.MARKER_TRIANGLE_UP, 20, 2)
    _label(img, "arm", column(aim) + 14, height - 6, ORANGE, 0.45)
    _banner(img, brain)
    return img


def draw_wrist(frame: np.ndarray, brain) -> np.ndarray:
    img, scale = _shrink(frame)
    height, width = img.shape[:2]
    view = brain.view
    cv2.line(img, (width // 2, 80), (width // 2, height), GREY, 1)
    for i, (box, score, label) in enumerate(view.get("detections", [])):
        x1, y1, x2, y2 = (int(v * scale) for v in box)
        colour = YELLOW if i == 0 else GREY
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)
        _label(img, f"{label} {score:.2f}", x1, y1 - 8, colour)
        if i == 0 and brain.state == "CHECK":
            centre = ((x1 + x2) // 2, (y1 + y2) // 2)
            cv2.arrowedLine(img, centre, (width // 2, centre[1]), ORANGE, 2, tipLength=0.15)
    row = 0
    if "held_p" in view:
        _bar(img, "P(held)", view["held_p"], brain.cfg.verify.threshold, row)
        row += 1
    if "grab_p" in view:
        _bar(img, "P(grabbed)", view["grab_p"], brain.cfg.handoff.threshold, row)
        row += 1
    if "wake_p" in view and brain.wake is not None:
        _bar(img, "wake word", view["wake_p"], brain.wake.threshold, row)
    if "countdown" in view:
        _label(img, f"give up in {view['countdown']:.0f} s", width - 150, height - 14, YELLOW, 0.55)
    if brain.capture.folder is not None:
        cv2.circle(img, (width - 24, 100), 9, RED, -1)
        _label(img, f"saving images: {brain.capture.saved}", width - 200, 105, RED)
    _banner(img, brain)
    return img
