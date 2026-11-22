"""The yogurt-cup detector for the wrist camera.

    uv run python -m yogurt.train.detector pretrained   # ready-made: COCO YOLO11s, its "cup"/"bowl" classes
    uv run python -m yogurt.train.detector train        # fine-tune on the boxes drawn in the UI
    uv run python -m yogurt.train.detector preview      # live wrist camera + detector; nothing else needed
    uv run --no-sync python -m yogurt.train.detector hailo   # compile for the Pi's AI HAT+ 2 (x86 Linux only)

`pretrained` needs no data but is only a starting point: it was never shown a yogurt cup as such and
also fires on mugs and glasses. `train` reads data/yogurt_detection/images/**.jpg with the matching
labels/**.txt (YOLO format, written by the UI's box-drawing panel) and learns one class, "yogurt".
Both write models/yogurt_detector.onnx (16:9 input, boxes already filtered inside the model).

`hailo` compiles the same weights to models/yogurt_detector.hef for the AI HAT+ 2 (Hailo-10H). It
needs Linux x86_64 with Hailo's Dataflow Compiler 5.x wheel installed in this environment (see the
README), and calibrates on the wrist pictures in data/yogurt_detection/images.
"""

import argparse
import json
import random
import shutil
from pathlib import Path

from .. import config

DATA = config.ROOT / "data" / "yogurt_detection"
WORK = config.ROOT / "outputs" / "yogurt_detector"
BASE_WEIGHTS = "yolo11s.pt"
TRAINED_WEIGHTS = WORK / "run" / "weights" / "best.pt"
HAILO_ARCH = "hailo10h"   # AI HAT+ 2. The original AI HAT+ is hailo8, a different chip needing Hailo's older compiler.
# The HEF's NMS drops boxes below this score on the chip and nothing can recover them afterwards, so it is
# set to the lowest score anything asks for (the Lab's box proposals use 0.1); check.confidence filters above it.
HAILO_MIN_SCORE = 0.1


def export(weights, cfg) -> None:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    exported = model.export(format="onnx", imgsz=(384, 640), nms=True, simplify=False, opset=17)
    out = config.path(cfg.models.yogurt_detector)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(exported, out)
    # The label list also goes next to the model, so a .hef compiled from it knows its labels too.
    out.with_suffix(".json").write_text(json.dumps({"names": list(model.names.values())}))
    print(f"wrote {out} - restart the app or press Reload models")


def export_hailo(weights, cfg) -> None:
    from ultralytics import YOLO

    if not weights.exists():
        raise SystemExit(f"No {weights}: train the detector first, or pass --weights "
                         f"(outputs/{BASE_WEIGHTS} for the ready-made one).")
    images = sorted((DATA / "images").rglob("*.jpg"))
    if not images:
        raise SystemExit(f"No pictures in {DATA / 'images'} to calibrate on: copy data/yogurt_detection here too.")
    model = YOLO(str(weights))
    # Calibration sets the INT8 ranges, so it must see what the robot sees: every wrist picture, cup or not.
    WORK.mkdir(parents=True, exist_ok=True)
    listing = WORK / "calibration.txt"
    listing.write_text("\n".join(map(str, images)))
    (WORK / "calibration.yaml").write_text(json.dumps(   # JSON is valid YAML
        {"train": str(listing), "val": str(listing), "names": list(model.names.values())}))
    # Square 640, the size Ultralytics validates Hailo export at. The 16:9 wrist frame is letterboxed into it,
    # so the cup gets the same pixels as in the 384x640 ONNX; the chip just also processes the grey bars.
    folder = Path(model.export(format="hailo", name=HAILO_ARCH, imgsz=640, data=str(WORK / "calibration.yaml"),
                               conf=HAILO_MIN_SCORE))
    out = config.path(cfg.models.yogurt_detector).with_suffix(".hef")
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(next(folder.glob("*.hef")), out)
    out.with_suffix(".json").write_text(json.dumps({"names": list(model.names.values())}))
    print(f"wrote {out} - copy it and {out.with_suffix('.json').name} into models/ on the Pi")


def train(cfg, epochs: int) -> None:
    from ultralytics import YOLO

    images = sorted(f for f in (DATA / "images").rglob("*.jpg")
                    if (DATA / "labels" / f.relative_to(DATA / "images")).with_suffix(".txt").exists())
    with_cup = sum(1 for f in images
                   if (DATA / "labels" / f.relative_to(DATA / "images")).with_suffix(".txt").read_text().strip())
    print(f"{len(images)} labelled images, {with_cup} of them with a yogurt cup")
    if with_cup < 50:
        raise SystemExit("Too few: label at least 50 images that contain the cup "
                         "(see data/yogurt_detection/README.md).")
    # Hold out whole capture sessions: frames from one burst look alike.
    sessions = sorted({f.parent for f in images})
    random.Random(0).shuffle(sessions)
    held_out = set(sessions[: max(1, len(sessions) // 6)]) if len(sessions) > 1 else set()
    val = [f for f in images if f.parent in held_out] or images[::6]
    training = [f for f in images if f not in set(val)]
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "train.txt").write_text("\n".join(map(str, training)))
    (WORK / "val.txt").write_text("\n".join(map(str, val)))
    (WORK / "data.yaml").write_text(f"train: {WORK / 'train.txt'}\nval: {WORK / 'val.txt'}\nnames:\n  0: yogurt\n")
    model = YOLO(str(config.ROOT / "outputs" / BASE_WEIGHTS))
    result = model.train(data=str(WORK / "data.yaml"), epochs=epochs, imgsz=640, batch=16, project=str(WORK),
                         name="run", exist_ok=True, plots=False)
    export(result.save_dir / "weights" / "best.pt", cfg)


def preview(cfg, min_score: float) -> None:
    """Live wrist camera + the exported detector, exactly as the app loads it. Needs nothing else plugged in."""
    import time

    import cv2

    from .. import infer, vision
    from ..cameras import Camera
    from ..fsm import YOGURT_LABELS

    detector = vision.YogurtDetector(infer.load(config.path(cfg.models.yogurt_detector)), YOGURT_LABELS)
    camera = Camera("wrist", cfg.cameras.wrist)
    camera.open()
    print(f"showing detections scoring >= {min_score} (the app's check.confidence); press q to quit")
    try:
        while True:
            frame = camera.read()
            if frame is None:
                if cv2.waitKey(100) & 0xFF == ord("q"):
                    break
                print(f"waiting for the wrist camera: {camera.error}", end="\r")
                continue
            start = time.perf_counter()
            found = detector.detect(frame, min_score)
            ms = (time.perf_counter() - start) * 1000
            shown = frame.copy()
            for d in found:
                x1, y1, x2, y2 = d.box
                cv2.rectangle(shown, (x1, y1), (x2, y2), (0, 220, 0), 3)
                cv2.putText(shown, f"{d.label} {d.score:.2f}", (x1, max(30, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 220, 0), 2)
            cv2.putText(shown, f"{len(found)} found, {ms:.0f} ms", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                        (255, 255, 255), 3)
            cv2.imshow("yogurt detector (q to quit)", shown)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.close()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["pretrained", "train", "preview", "hailo"])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--min-score", type=float, help="preview: score to show (default: check.confidence)")
    parser.add_argument("--weights", type=Path, default=TRAINED_WEIGHTS,
                        help="hailo: the .pt to compile (default: the last `train` run's best.pt)")
    args = parser.parse_args()
    cfg = config.load("laptop")
    (config.ROOT / "outputs").mkdir(exist_ok=True)
    if args.mode == "preview":
        preview(cfg, args.min_score if args.min_score is not None else cfg.check.confidence)
    elif args.mode == "hailo":
        export_hailo(args.weights.resolve(), cfg)
    elif args.mode == "pretrained":
        export(config.ROOT / "outputs" / BASE_WEIGHTS, cfg)
    else:
        train(cfg, args.epochs)


if __name__ == "__main__":
    main()
