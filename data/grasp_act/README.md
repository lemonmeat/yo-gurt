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

## Record (the Lab: robot page → **Lab →**, or <http://127.0.0.1:8080/lab>)

The Lab works like Hugging Face's [LeLab](https://github.com/huggingface/leLab): a session records a
set number of episodes into one dataset, each a timed **recording** phase followed by a timed
**reset** phase. Yo-gurt adds two phases in front of every episode, so that each one starts exactly
where the real run will.

1. Press **SAFE STOP** (or let the robot sit in Idle) and put the cup within reach.
2. **New dataset** (or pick a dataset and **Record more** to add to it). Choose the number of
   episodes, the episode time and the reset time, then **Start recording**.
3. Each episode then goes:
   - **Finding the cup** — Check sweeps and centres on the cup. Without a trained detector, put the
     cup in the middle of the wrist picture and press **Cup is centred** (→).
   - **Match the leader arm** — bring the leader to the follower's pose; the screen names the joint
     that is furthest off. Recording starts by itself when they match, so the follower never jumps.
   - **Recording** — grasp the cup, lift it about 5 cm, hold still for a second, then **End Episode**
     (→). **Re-record** (Delete or ←) throws the take away. A take that reaches the episode time is
     discarded and recorded again, as in LeLab: only episodes you end yourself are kept.
   - **Reset** — nothing is recorded and the leader still drives the arm: put the cup somewhere new,
     open the gripper, back away. When the reset time runs out (or you press **Start Next Episode**)
     the arm moves to the survey pose **by itself** and the next episode begins; three beeps warn you
     three seconds before. Reset time 0 waits for the button instead.
4. After the last episode, put the cup down in the reset phase and press **Finish Session**. The arm
   then holds where it is (Safe Stop).

Space (SAFE STOP) pauses the session: the take in progress is discarded and **Resume** starts that
episode again from Check. **Stop recording** ends the session early; everything saved so far is
kept. Esc (KILL) also ends it and keeps what was saved. The dataset is only readable once a session
has ended one of these ways: a session cut off by a crash or power cut leaves a dataset that LeRobot
cannot open (the Lab marks it unreadable; delete it).

## Perturbed takes (DART)

The form's **Perturbed takes (%)** (default `dart.fraction` in `config/robot.yaml`, 50 %) picks that
share of takes at random. In a perturbed take the follower is sent the leader's pose **plus a small,
smooth, slowly wandering offset** (a few degrees per joint, never the gripper), while the dataset
saves the **clean leader pose** as the action. The status line says PERTURBED. Just do the grasp as
usual and steer the follower back onto the cup when it drifts. That gives the policy examples of
"slightly off-track → what to do about it", which clean demonstrations never contain (DART, Laskey
et al. 2017), so a trained policy recovers from its own small errors instead of compounding them.

- In the episode browser the joint trace shows it: in a perturbed episode `state` wanders a few
  degrees off `action`. `data/grasp_act/<name>/dart.json` lists which episodes were perturbed
  (deleting episodes keeps it right).
- Size the noise like the errors your policy makes (`dart.sigma_deg`, per joint). Too little and it
  never leaves the clean path; too much and the demonstrations become a fight. Shoulder and elbow
  move the hand most per degree, so they get the smallest values.
- DART only covers drift as large as the noise. For bigger failures of a trained policy, record
  interventions (HG-DAgger) afterwards.

## Review and delete episodes (the Lab)

Pick a dataset to see its episodes. Each one plays back from the dataset's own video file, with a
trace of every joint underneath (hover to read the values, click to jump). ↑/↓ change episode, ←/→
step one frame.

- **Delete episode** removes the one on screen; tick several in the list and press **Delete** to
  remove them in one go (faster: each deletion rewrites the dataset and re-encodes the video files it
  touches). The episodes after a deleted one are renumbered. A dataset cannot be emptied this way:
  delete the whole dataset with the trash button instead (you type its name to confirm).
- Nothing can be deleted while a recording session is running.

What makes good demonstrations

- 50 to start with; 100+ if the cup can be anywhere within reach. Spread the cup over the whole
  area and at different rotations.
- Do it the same way every time: same approach direction, one smooth motion, no hesitating or
  re-gripping. End every demonstration the same way (lifted, still). Delete the ones that are not.
- Aim for 5–10 s each and an episode time of about 20 s. Set `grasp.duration_s` in
  `config/robot.yaml` to a little more than your longest one: the policy runs for that long before Verify.
- Keep the gripper camera view clear of your hands and the leader arm.

Each dataset is a standard LeRobot dataset in `data/grasp_act/<name>/` (repo id `local/<name>`, wrist
video at 640×360, 30 fps, `observation.state` and `action` as the six joint values). New episodes can
only be added to a dataset recorded with the same frame rate and picture size.

## Clean (the Lab)

Under each episode's video:

- **[ Trim start here** / **Trim end here ]** (keys `[` and `]`) keep only part of a take, from the
  frame on screen. Use it to cut the fumbling at the start or the drifting after the hold.
  **Keep whole take** undoes it. The orange bar and the grey shading on the joint traces show what is kept.
- **Disable** (key `D`) leaves a take out of training without deleting it. **Enable** brings it back.

These edits are stored on the robot computer in `data/grasp_act/<name>/cleaning.json`, next to
the recording, and are never applied to it, so they can all be undone. `lerobot-train` cannot read
them, so press **Make cleaned copy** (under the training command): it writes `<name>-clean`, a normal
dataset with the disabled takes left out and the trims applied. Train on that copy. The Lab says
when the copy is out of date because you edited (or recorded) more since. Making the copy re-encodes
every kept frame: about 5 minutes for 20k frames on a laptop, much longer on the Pi.

## Train

The Lab shows this command for the dataset on screen (with **Copy**); for a dataset called `<name>`:

```bash
uv sync --extra train
lerobot-train \
  --dataset.repo_id=local/<name> --dataset.root=data/grasp_act/<name> \
  --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
  --output_dir=outputs/act_<name> --steps=60000 --batch_size=8 --wandb.enable=false
```

Use `--policy.device=mps` on an Apple laptop or `cpu` anywhere; a GPU takes a couple of hours, a
laptop much longer. To train on another machine, copy the `data/grasp_act/<name>` folder there.
Then install the result:

```bash
mkdir -p models && cp -r outputs/act_<name>/checkpoints/last/pretrained_model models/act_grasp
```

Restart the app or press **Reload models**. In Grasp the status line shows the policy's gripper
command and the time left; whatever the policy outputs still goes through the arm's speed and range
limits. If a grasp fails, Verify sends the robot round again by itself.
