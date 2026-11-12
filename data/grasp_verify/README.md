# Grasp verification: is the yogurt in the gripper?

After the grasp, the arm lifts to the Verify pose (the `survey` pose, gripper left as it is) and
this model looks at the wrist picture: `held` or `empty`. `empty` sends the robot back to Check.

## Record (UI → Data collection → 2 · Wrist images → `grasp_verify / held` or `empty`)

The pictures must be taken in the Verify pose, which the robot reaches by itself:

1. Start a request (type "give me the yogurt") and step through it with the stand-in buttons
   (**Yogurt centred**, then grasp with the leader arm or by hand, **Grasp done**).
2. The robot lifts and waits in **Verify** for you to press Held / Not held. While it waits, choose
   the label, press **Start saving**, and vary the scene for 10–15 s (turn the cup slightly in the
   jaws, change the light, move things in the background). Press **Stop**.
3. Press **Not held** to go round again (Check → Grasp → Verify) for the next burst.

For `empty`, let the grasp miss on purpose: gripper closed on nothing with the cup still on the
table, gripper half open, no cup in sight. The UI lists the situations to cover.

Aim for 300+ pictures per label in at least 5 bursts each. Keep the two labels roughly balanced.

## Train

```bash
uv run python -m yogurt.train.classifier grasp_verify
```

A few minutes on a laptop. It prints accuracy on held-out bursts and writes
`models/grasp_verify.onnx`. During Verify the wrist video shows P(held) against `verify.threshold`
(`config/robot.yaml`).
