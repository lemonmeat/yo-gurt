# Grasp demonstrations for the ACT policy

The policy sees the wrist picture and the six joint angles and outputs the next joint targets. It
learns from demonstrations given with the leader arm. Each demonstration starts exactly where the
real run will: the robot itself finds the cup and centres on it (Check), then hands control to you.

## Once: set up the leader arm

```bash
lerobot-find-port                                   # with only the leader plugged in
lerobot-calibrate --teleop.type=so101_leader --teleop.port=<port> \
    --teleop.id=leader_config --teleop.calibration_dir=config
```

That writes `config/leader_config.json`. Put the leader's port in `leader.port` and the follower's
in `arm.port` in `config/hardware_laptop.yaml` (with two arms plugged in, `auto` cannot tell them apart).
Teach the `survey` pose before recording anything: every demonstration starts from it, so changing
it later means recording again.

## Record (UI → Data collection → 3 · Grasp demonstrations)

1. Put the cup on the table within reach. Press **Start demo session**. The robot sweeps, finds the
   cup and centres on it. (Without a trained detector, jog the arm until the cup is centred and
   press **Yogurt centred**.)
2. The status line now asks you to bring the leader to the follower's pose and shows which joint is
   furthest off. When they match, recording starts by itself, so the follower never jumps.
3. Grasp the cup, lift it about 5 cm, hold still for a second. Press **Save demo** (or **Discard**).
4. The leader still drives the arm but nothing is recorded: put the cup down somewhere new, open
   the gripper, back away. Press **Next demo** and go to step 2, or **Finish demos**.

What makes good demonstrations

- 50 to start with; 100+ if the cup can be anywhere within reach. Spread the cup over the whole
  area and at different rotations.
- Do it the same way every time: same approach direction, one smooth motion, no hesitating or
  re-gripping. End every demonstration the same way (lifted, still).
- Aim for 5–10 s each. Set `grasp.duration_s` in `config/robot.yaml` to a little more than your
  longest one: the policy runs for that long before Verify.
- Keep the gripper camera view clear of your hands and the leader arm.

The dataset is a standard LeRobot dataset at `data/grasp_act/dataset` (wrist video at 640×360,
30 fps, `observation.state` and `action` as the six joint values). **Finish demos** closes it properly;
you can add more later with another session.

## Train

```bash
uv sync --extra train
lerobot-train \
  --dataset.repo_id=local/yogurt_grasp --dataset.root=data/grasp_act/dataset \
  --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
  --output_dir=outputs/act_grasp --steps=60000 --batch_size=8 --wandb.enable=false
```

Use `--policy.device=mps` on an Apple laptop or `cpu` anywhere; a GPU takes a couple of hours, a
laptop much longer. To train on another machine, copy the `data/grasp_act/dataset` folder there.
Then install the result:

```bash
mkdir -p models && cp -r outputs/act_grasp/checkpoints/last/pretrained_model models/act_grasp
```

Restart the app or press **Reload models**. In Grasp the status line shows the policy's gripper
command and the time left; whatever the policy outputs still goes through the arm's speed and range
limits. If a grasp fails, Verify sends the robot round again by itself.
