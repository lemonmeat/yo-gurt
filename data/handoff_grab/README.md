# Hand-off: has someone taken hold of the cup?

In Handoff the arm holds the cup out and this model watches the wrist picture. When it has seen
`grabbed` for 0.4 s the gripper opens. If the gripper opens too early the yogurt lands on the floor,
so the `not_grabbed` examples that look almost like a grab are the ones that matter.

## The crop: what the model looks at

The model sees only a square of the wrist picture around the cup (`handoff.wrist_crop` in
`config/robot.yaml`), not the whole wide-angle frame: fingers on the cup get more detail, and people
in the background, which the model would otherwise learn instead of the hand, drop out. In Handoff
and Safe Stop the wrist video draws it as the white **hand-off crop** box.

Set it once, with the arm holding the cup in the handoff pose: the cup and the space a hand needs
around it on every side should fill the box. Change `x`, `y` (the centre, as fractions of the width
and height) and `size` (as a fraction of the height), restart, look again. Pictures are saved whole, so
a new crop needs only a retrain. Record every burst, `grabbed` and `not_grabbed`, in the same arm pose,
so the cup always sits in the same place in the picture.

## Record (Lab → Wrist images → Hand-off — grabbed / not grabbed)

1. Run a request through to **Handoff** with the buttons under "Put the arm in place" (**Idle**,
   **Start a request**, **Yogurt centred**, **Grasp done**, **Held — go to Handoff**) so the arm is
   stretched out holding the cup, then press **SAFE STOP**: the arm stays put and the 10 s give-up
   timer is off.
2. Choose the label, press **Start saving** (3 per second), act out the situation for 10–15 s,
   **Stop saving**. The Lab lists what to cover: for `grabbed`, hands of different people holding the
   cup from every side; for `not_grabbed`, nobody there, hands approaching, hovering, and touching
   without gripping.
3. Do not leave the arm stretched out for more than about a minute at a time: the shoulder servo is
   working hard in that pose (watch its temperature in the robot page's Arm panel). Press
   **Return (rest the arm)** to rest it.

Only record `grabbed` while the hand really is holding the cup; start and stop the burst with the
hand already in place. Aim for 300+ pictures per label, at least 5 bursts each, from several people.
Afterwards, look through each burst in the Lab and delete any `grabbed` picture where the hand is not
yet (or no longer) holding on.

## Train

```bash
uv run python -m yogurt.train.classifier handoff_grab
```

Writes `models/handoff_grab.onnx` and, in `models/handoff_grab.json`, the crop it was trained on; the
robot cuts out the same square. If `handoff.wrist_crop` changes afterwards, the robot page says the
model needs retraining. During Handoff the wrist video shows P(grabbed) against
`handoff.threshold`; raise the threshold or `handoff.hold_s` in `config/robot.yaml` if it lets go too
eagerly.
