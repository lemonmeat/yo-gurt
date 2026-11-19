# Yogurt detector

Check sweeps the wrist camera across the table and stops when this model sees a yogurt cup. The
ready-made model (`uv run python -m yogurt.train.detector pretrained`) only knows "cup" in general;
this dataset teaches a model your cup, through your wide-angle lens.

## 1. Take pictures (Lab → Wrist images → Yogurt detector)

1. Press SAFE STOP, then **Survey pose** under "Put the arm in place". This is the view Check has.
2. Put the cup on the table, press **Start saving** (2 per second is fine), and turn the arm with
   **Pan −5° / Pan +5°** so the cup crosses the whole picture. Press **Stop saving**.
3. Move the cup and repeat. Work through the list the Lab shows under the buttons
   (distances, rotations, clutter, lighting, and some bursts with no cup at all).

Aim for 250+ pictures with the cup and 50+ without, in at least 6 separate bursts (each
Start/Stop makes its own folder, and whole folders are held out to test the model honestly).

## 2. Draw the boxes (Lab → Wrist images → Yogurt detector → **Draw boxes**)

Each unlabelled picture appears in turn. Drag a tight box around the cup and press **Save box**
(Enter), or **No yogurt in this picture** (N). If a detector is already loaded its guess is drawn
dashed; save it if it fits, redraw it if not. Click any picture in a burst to check or correct its
box later (**Forget the label** removes it). Boxes are stored in `labels/` in YOLO format, mirroring
`images/`; deleting a picture in the Lab deletes its box too.

## 3. Train

```bash
uv run python -m yogurt.train.detector train
```

Fine-tunes YOLO11s for 80 epochs: minutes on a GPU, about an hour on a laptop CPU for 300 pictures
(`--epochs 40` is a reasonable shortcut). It prints precision/recall on the held-out bursts and writes
`models/yogurt_detector.onnx`. `check.confidence` in `config/robot.yaml` is the score needed to count
as a cup; the wrist video shows every detection with its score during Check.

Note: Ultralytics YOLO and its weights are AGPL-3.0 licensed.
