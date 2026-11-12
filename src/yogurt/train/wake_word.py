"""Train the "Yo, gurt" detector from the clips recorded in the UI.

    uv run python -m yogurt.train.wake_word

Reads   data/wake_word/positive/*.wav    the wake phrase, once per clip
        data/wake_word/negative/*.wav    other words and phrases, especially similar-sounding ones
        data/wake_word/background/*.wav  long recordings of the room
Writes  models/wake_word.onnx and models/wake_word.json (decision threshold + how well it validated)

The detector looks at the last 1.5 s of sound every 100 ms. Training examples are built the same way:
a positive is a window in which the whole phrase has just finished (sometimes with the first words
of a command already following it); negatives are other speech, the room, and windows where the
phrase is only half said, so the robot does not fire early.
"""

import argparse
import json
import random

import numpy as np
import torch
from torch import nn

from .. import config
from ..audio import RATE, WAKE_WINDOW_S, read_wav, wake_features

WINDOW = int(WAKE_WINDOW_S * RATE)
DATA = config.ROOT / "data" / "wake_word"


class WakeNet(nn.Module):
    """A small CNN over the log-mel picture: about 60k weights, well under a millisecond on a Pi."""

    def __init__(self):
        super().__init__()

        def block(cin, cout, pool):
            layers = [nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU()]
            return layers + ([nn.MaxPool2d(2)] if pool else [])

        self.features = nn.Sequential(*block(1, 16, True), *block(16, 32, True), *block(32, 48, True),
                                      *block(48, 64, False), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(64, 2))

    def forward(self, mel):
        return self.head(self.features(mel))


def find_phrase(clip: np.ndarray) -> np.ndarray | None:
    """Cut the spoken phrase out of a clip by loudness. None if the clip does not look like one phrase."""
    frame = RATE // 100
    loud = np.sqrt(np.mean(clip[: len(clip) // frame * frame].reshape(-1, frame) ** 2, axis=1))
    if loud.max() < 0.005:
        return None
    active = np.flatnonzero(loud > max(0.2 * loud.max(), 2.5 * np.percentile(loud, 10)))
    start, end = max(0, active[0] - 5) * frame, min(len(loud), active[-1] + 6) * frame
    return clip[start:end] if 0.25 * RATE <= end - start <= 1.3 * RATE else None


def crop(clip: np.ndarray, length: int, rng: random.Random) -> np.ndarray:
    """A random stretch of the clip, zero-padded if the clip is shorter."""
    if len(clip) <= length:
        return np.pad(clip, (length - len(clip), 0))
    start = rng.randrange(len(clip) - length)
    return clip[start:start + length]


def stretch(clip: np.ndarray, factor: float) -> np.ndarray:
    """Speak a little faster or slower."""
    return np.interp(np.arange(0, len(clip), factor), np.arange(len(clip)), clip).astype(np.float32)


class Examples:
    """Builds training windows on the fly from one split of the clips."""

    def __init__(self, phrases, negatives, backgrounds, seed):
        self.phrases, self.negatives, self.backgrounds = phrases, negatives, backgrounds
        self.rng = random.Random(seed)

    def _room(self) -> np.ndarray:
        if self.backgrounds:
            return crop(self.rng.choice(self.backgrounds), WINDOW, self.rng) * self.rng.uniform(0.3, 1.5)
        return np.random.default_rng(self.rng.randrange(1 << 30)).normal(0, 0.002, WINDOW).astype(np.float32)

    def _other_speech(self, length: int) -> np.ndarray:
        source = self.rng.choice(self.negatives or self.backgrounds or [np.zeros(WINDOW, np.float32)])
        return crop(source, length, self.rng)

    def make(self, positive: bool) -> np.ndarray:
        rng, window = self.rng, self._room()
        kind = "phrase" if positive else rng.choice(["speech", "speech", "room", "cut_off"])
        if kind == "speech":
            window = window + self._other_speech(WINDOW)
        elif kind != "room":
            phrase = stretch(rng.choice(self.phrases), rng.uniform(0.9, 1.1))[:WINDOW - 1]
            if kind == "phrase":       # the phrase ended 0 - 0.35 s ago
                end = WINDOW - rng.randrange(int(0.35 * RATE))
            else:                      # "cut_off": only the first 35 - 65 % has been said so far
                phrase = phrase[: int(len(phrase) * rng.uniform(0.35, 0.65))]
                end = WINDOW
            start = max(0, end - len(phrase))
            window[start:end] += phrase[len(phrase) - (end - start):]
            if end < WINDOW and rng.random() < 0.3:
                window[end:] += self._other_speech(WINDOW - end)   # the command that follows straight away
        return wake_features(window * 10 ** (rng.uniform(-12, 6) / 20))

    def batch(self, size: int) -> tuple[torch.Tensor, torch.Tensor]:
        labels = [i % 2 for i in range(size)]
        mels = np.stack([self.make(bool(label)) for label in labels])
        return torch.from_numpy(mels)[:, None], torch.tensor(labels)


def load(folder: str) -> list[np.ndarray]:
    return [read_wav(f) for f in sorted((DATA / folder).glob("*.wav"))]


def split(items: list, rng: random.Random) -> tuple[list, list]:
    items = items[:]
    rng.shuffle(items)
    cut = max(1, len(items) // 6) if len(items) > 1 else 0
    return items[cut:], items[:cut]


def false_alarms_per_hour(model: nn.Module, recordings: list[np.ndarray], threshold: float, frames: int) -> float:
    """Run the detector over held-out recordings exactly as the robot does and count how often it fires."""
    hop, alarms, seconds = RATE // 10, 0, 0.0
    for audio in recordings:
        seconds += len(audio) / RATE
        mels = [wake_features(audio[end - WINDOW:end]) for end in range(WINDOW, len(audio), hop)]
        if not mels:
            continue
        with torch.no_grad():
            scores = model(torch.from_numpy(np.stack(mels))[:, None]).softmax(1)[:, 1].numpy()
        run = 0
        for score in scores:
            run = run + 1 if score >= threshold else 0
            alarms += run == frames
    return alarms / max(seconds / 3600, 1e-9)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=1500, help="training batches of 64 windows")
    args = parser.parse_args()
    cfg = config.load("laptop")
    rng = random.Random(0)
    torch.manual_seed(0)

    clips = load("positive")
    phrases = [p for p in map(find_phrase, clips) if p is not None]
    negatives, backgrounds = load("negative"), load("background")
    print(f"{len(clips)} positive clips ({len(clips) - len(phrases)} unusable: silent or not one short phrase), "
          f"{len(negatives)} negative clips, {sum(len(b) for b in backgrounds) / RATE / 60:.1f} min of background")
    if len(phrases) < 30 or len(negatives) < 30 or not backgrounds:
        raise SystemExit("Not enough data: record at least 30 positive and 30 negative clips and some background "
                         "(the more the better, see data/wake_word/README.md).")

    train_p, val_p = split(phrases, rng)
    train_n, val_n = split(negatives, rng)
    train_b, val_b = split(backgrounds, rng) if len(backgrounds) > 1 else (backgrounds, backgrounds)
    train, val = Examples(train_p, train_n, train_b, 1), Examples(val_p, val_n, val_b, 2)
    val_x, val_y = val.batch(1000)

    model = WakeNet()
    optimiser = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimiser, 2e-3, total_steps=args.steps)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    for step in range(1, args.steps + 1):
        model.train()
        x, y = train.batch(64)
        loss = loss_fn(model(x), y)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        schedule.step()
        if step % 100 == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                accuracy = (model(val_x).argmax(1) == val_y).float().mean().item()
            print(f"step {step:5d}  loss {loss.item():.3f}  validation accuracy {accuracy:.3f}")

    # Threshold: as low as possible while at most 0.5 % of the held-out negative windows pass it.
    model.eval()
    with torch.no_grad():
        scores = model(val_x).softmax(1)[:, 1].numpy()
    negative_scores, positive_scores = scores[val_y.numpy() == 0], scores[val_y.numpy() == 1]
    threshold = float(min(0.95, max(0.5, np.quantile(negative_scores, 0.995) + 0.02)))
    report = {
        "threshold": round(threshold, 3),
        "recall": round(float((positive_scores >= threshold).mean()), 3),
        "false_positive_rate": round(float((negative_scores >= threshold).mean()), 4),
        "false_alarms_per_hour": round(false_alarms_per_hour(model, val_b + val_n, threshold, cfg.wake.frames), 1),
        "clips": {"positive": len(phrases), "negative": len(negatives), "background": len(backgrounds)},
    }
    print(json.dumps(report, indent=2))

    out = config.path(cfg.models.wake_word)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, torch.zeros(1, 1, *wake_features(np.zeros(WINDOW, np.float32)).shape), str(out),
                      input_names=["mel"], output_names=["logits"], dynamic_axes={"mel": {0: "batch"}}, dynamo=False)
    out.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(f"wrote {out} - restart the app (or press Reload models in Safe Stop) to use it")


if __name__ == "__main__":
    main()
