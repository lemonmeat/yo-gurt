"""Train one of the two yes/no wrist-camera classifiers from the images saved in the UI.

    uv run python -m yogurt.train.classifier grasp_verify    # is the yogurt in the gripper?
    uv run python -m yogurt.train.classifier handoff_grab    # has a person taken hold of it?

Reads   data/<task>/<label>/<session>/*.jpg   (two labels per task, see TASKS)
Writes  models/<task>.onnx and models/<task>.json

A MobileNetV3-small pretrained on ImageNet is fine-tuned on whole (squashed) pictures. Whole
recording sessions are held out for validation, because neighbouring frames are near-duplicates.

grasp_verify sees the wrist frame. handoff_grab sees a two-camera picture (see vision.handoff_picture):
the wrist frame on the left, the base camera's view of the gripper on the right. While training, one
of the two halves is sometimes blacked out, so the model still answers sensibly when a camera drops
out or a hand hides one view.
"""

import argparse
import json
import random

import cv2
import numpy as np
import torch
from torch import nn
from torchvision import models

from .. import config

TASKS = {"grasp_verify": ["empty", "held"], "handoff_grab": ["not_grabbed", "grabbed"]}  # second label = "yes"
SIZE = 224
SIDE_BY_SIDE = {"handoff_grab"}  # tasks whose pictures are two square views side by side
BLANK_P = 0.2                    # how often training blacks out one of the two views


class Net(nn.Module):
    """MobileNetV3-small taking RGB pixels in 0..1 (the colour normalisation lives inside the model)."""

    def __init__(self):
        super().__init__()
        self.backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.backbone.classifier[3] = nn.Linear(self.backbone.classifier[3].in_features, 2)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, image):
        return self.backbone((image - self.mean) / self.std)


def _jitter(image: np.ndarray, rng: random.Random) -> np.ndarray:
    """Lighting and small framing changes; no flips, the gripper always looks the same way round."""
    h, w = image.shape[:2]
    dx, dy = int(w * rng.uniform(0, 0.08)), int(h * rng.uniform(0, 0.08))
    image = image[dy:h - int(h * rng.uniform(0, 0.08)), dx:w - int(w * rng.uniform(0, 0.08))]
    return cv2.convertScaleAbs(image, alpha=rng.uniform(0.7, 1.3), beta=rng.uniform(-25, 25))


def load_image(file, augment: bool, rng: random.Random, halves: bool = False) -> np.ndarray:
    image = cv2.imread(str(file))
    if halves:  # each view on its own, so the seam between them stays put
        side = image.shape[1] // 2
        views = [image[:, :side], image[:, side:]]
        if augment:
            views = [_jitter(v, rng) for v in views]
            if rng.random() < BLANK_P:
                views[rng.randrange(2)] = np.zeros_like(views[0])
        image = np.hstack([cv2.resize(v, (SIZE, SIZE), interpolation=cv2.INTER_AREA) for v in views])
    else:
        if augment:
            image = _jitter(image, rng)
        image = cv2.resize(image, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.float32) / 255


def batches(items, size, augment, rng, halves=False):
    items = items[:]
    if augment:
        rng.shuffle(items)
    for i in range(0, len(items), size):
        chunk = items[i:i + size]
        yield (torch.from_numpy(np.stack([load_image(f, augment, rng, halves) for f, _ in chunk])),
               torch.tensor([label for _, label in chunk]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("task", choices=TASKS)
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()
    cfg, rng = config.load("laptop"), random.Random(0)
    torch.manual_seed(0)
    labels, root = TASKS[args.task], config.ROOT / "data" / args.task
    halves = args.task in SIDE_BY_SIDE

    train, val = [], []
    for index, label in enumerate(labels):
        sessions = sorted(p for p in (root / label).glob("*") if p.is_dir())
        files = {s: sorted(s.glob("*.jpg")) for s in sessions}
        total = sum(map(len, files.values()))
        if total < 40:
            raise SystemExit(f"Only {total} images for '{label}': record at least 40 in a few separate sessions "
                             f"(see data/{args.task}/README.md).")
        rng.shuffle(sessions)
        held_out = sessions[: max(1, len(sessions) // 5)] if len(sessions) > 1 else []
        for session in sessions:
            images = [(f, index) for f in files[session]]
            if not held_out:  # a single session: hold out its last fifth
                cut = len(images) * 4 // 5
                train += images[:cut]
                val += images[cut:]
            else:
                (val if session in held_out else train).extend(images)
        print(f"{label}: {total} images in {len(sessions)} session(s)")

    # CUDA if there is one, else CPU (Apple's MPS backend fails on this network's backward pass).
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Net().to(device)
    optimiser = torch.optim.AdamW(model.parameters(), 3e-4, weight_decay=1e-4)
    counts = np.bincount([label for _, label in train], minlength=2)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(counts.sum() / (2 * counts), dtype=torch.float32, device=device))
    best, best_state = -1.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in batches(train, 32, True, rng, halves):
            loss = loss_fn(model(x.to(device)), y.to(device))
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        model.eval()
        confusion = np.zeros((2, 2), int)
        with torch.no_grad():
            for x, y in batches(val, 64, False, rng, halves):
                for truth, guess in zip(y.tolist(), model(x.to(device)).argmax(1).tolist(), strict=True):
                    confusion[truth, guess] += 1
        balanced = float(np.mean([confusion[i, i] / max(1, confusion[i].sum()) for i in range(2)]))
        right = "  ".join(f"{labels[i]}: {confusion[i, i]}/{confusion[i].sum()}" for i in range(2))
        print(f"epoch {epoch:2d}  loss {loss.item():.3f}  validation balanced accuracy {balanced:.3f}  {right}")
        if balanced >= best:  # on a tie keep the later, better-trained epoch
            best, best_state = balanced, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.cpu().eval()
    out = config.path(cfg.models[args.task])
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, torch.zeros(1, 3, SIZE, SIZE * (2 if halves else 1)), str(out), input_names=["image"],
                      output_names=["logits"], dynamo=False)
    out.with_suffix(".json").write_text(json.dumps(
        {"classes": labels, "validation_balanced_accuracy": round(best, 3),
         "input": "wrist | base camera around the gripper" if halves else "wrist",
         "images": {"train": len(train), "validation": len(val)}}, indent=2))
    print(f"wrote {out} (validation balanced accuracy {best:.3f}) - restart the app or press Reload models")


if __name__ == "__main__":
    main()
