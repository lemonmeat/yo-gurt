# Hand-off: has someone taken hold of the cup?

In Handoff the arm holds the cup out and this model watches two cameras at once. Each picture it sees
(and each one the Lab saves) is two squares side by side:

- **left**: the whole wrist picture, close up on the cup and the fingers touching it;
- **right**: the part of the base picture around the gripper, which shows best whether a hand is
  under the cup. Its column follows the arm's pan; its height and size are `handoff.base_crop` in
  `config/robot.yaml`.

When it has seen `grabbed` for 0.4 s the gripper opens. If the gripper opens too early the yogurt
lands on the floor, so the `not_grabbed` examples that look almost like a grab are the ones that matter.

## Before recording: aim the base-camera square

With the arm in the handoff pose (step 1 below), the base video draws a white **hand-off view** box.
Change `handoff.base_crop` (`row`, `size`, `shift_deg`) until the gripper and the space under it fill
the box, restart, and check again; the Lab's live picture shows exactly what will be saved. Pictures
recorded with one crop do not suit another: if you change it later, record again.

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

Writes `models/handoff_grab.onnx` (input 448 × 224: both views at 224 × 224). While training, one view
is blacked out in a fifth of the pictures, so the model still works from the other when a camera drops
out or a hand hides one view. During Handoff the wrist video shows P(grabbed) against
`handoff.threshold`; raise the threshold or `handoff.hold_s` in `config/robot.yaml` if it lets go too
eagerly.
