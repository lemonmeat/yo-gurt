"""The yogurt-cup detector for the wrist camera.

    uv run python -m yogurt.train.detector pretrained   # ready-made: COCO YOLO11s, its "cup"/"bowl" classes
    uv run python -m yogurt.train.detector train        # fine-tune on the boxes drawn in the UI

`pretrained` needs no data but is only a starting point: it was never shown a yogurt cup as such and
also fires on mugs and glasses. `train` reads data/yogurt_detection/images/**.jpg with the matching
labels/**.txt (YOLO format, written by the UI's box-drawing panel) and learns one class, "yogurt".
Both write models/yogurt_detector.onnx (16:9 input, boxes already filtered inside the model).
"""

import argparse
import json
import random
import shutil

from .. import config

DATA = config.ROOT / "data" / "yogurt_detection"
WORK = config.ROOT / "outputs" / "yogurt_detector"
BASE_WEIGHTS = "yolo11s.pt"


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["pretrained", "train"])
    parser.add_argument("--epochs", type=int, default=80)
    args = parser.parse_args()
    cfg = config.load("laptop")
    (config.ROOT / "outputs").mkdir(exist_ok=True)
    if args.mode == "pretrained":
        export(config.ROOT / "outputs" / BASE_WEIGHTS, cfg)
    else:
        train(cfg, args.epochs)


if __name__ == "__main__":
    main()
