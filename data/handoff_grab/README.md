# Hand-off: has someone taken hold of the cup?

In Handoff the arm holds the cup out and this model watches the wrist picture. When it has seen
`grabbed` for 0.4 s the gripper opens. If the gripper opens too early the yogurt lands on the floor,
so the `not_grabbed` examples that look almost like a grab are the ones that matter.

## Record (UI → Data collection → 2 · Wrist images → `handoff_grab / grabbed` or `not_grabbed`)

1. Run a request through to **Handoff** (stand-in buttons are fine) so the arm is stretched out
   holding the cup, then press **SAFE STOP**: the arm stays put and the 10 s give-up timer is off.
2. Choose the label, press **Start saving** (3 per second), act out the situation for 10–15 s, **Stop**.
   The UI lists what to cover: for `grabbed`, hands of different people holding the cup from every
   side; for `not_grabbed`, nobody there, hands approaching, hovering, and touching without gripping.
3. Do not leave the arm stretched out for more than about a minute at a time: the shoulder servo is
   working hard in that pose (watch its temperature in the Arm panel). Click **RETURN** to rest it.

Only record `grabbed` while the hand really is holding the cup; start and stop the burst with the
hand already in place. Aim for 300+ pictures per label, at least 5 bursts each, from several people.

## Train

```bash
uv run python -m yogurt.train.classifier handoff_grab
```

Writes `models/handoff_grab.onnx`. During Handoff the wrist video shows P(grabbed) against
`handoff.threshold`; raise the threshold or `handoff.hold_s` in `config/robot.yaml` if it lets go too
eagerly.
