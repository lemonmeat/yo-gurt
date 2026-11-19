# Wake word: "Yo, gurt"

The detector listens to the last 1.5 s of sound, ten times a second, and has to tell the wake phrase
from everything else people say near the robot, including the word "yogurt".

## Record (Lab → Wake word)

The clip is recorded by the robot's microphone, so stand where people will stand at the demo
(0.5–3 m from the base camera, in front and off to both sides). Pick the label, press **Record**, then
speak; the default clip is 2 s. The Lab steps through the phrases in [prompts.json](prompts.json) by
itself (the arrows skip one), and shows the progress towards the numbers below.

| Label | What to say | How many |
|---|---|---|
| **positive** | *Only* the wake phrase, once per clip, the way the UI shows it (normal, quiet, loud, slow, fast). | 150 or more, from at least 4 people |
| **negative** | The phrase shown: "yogurt", "give me the yogurt", "yo", "your gut", … Similar-sounding phrases matter most. | 300 or more, same people |
| **background** | Nothing: set the length to 30–60 s and let the room be itself. Include talking nearby, and a few clips while the arm is moving (servo noise). | 10 minutes or more in total |

Tips

- A positive clip must contain the phrase and nothing else. Each clip can be played back right after
  recording: if you stumbled, press **Delete** (or move it to `negative`) and say it again.
- The Lab shows each clip's peak level and marks clips below 0.02 as quiet: the microphone barely
  heard them. Search and filter the list, tick clips, and **Move to label** or **Delete** them.
- More voices beat more clips from one voice. Anyone who will use the robot should record.

## Train

```bash
uv run python -m yogurt.train.wake_word
```

About five minutes on a laptop. It prints, and saves in `models/wake_word.json`:

- `recall`: the share of held-out wake phrases it caught (aim for 0.95+),
- `false_alarms_per_hour`: how often it fired on held-out negative and background recordings (aim for 0),
- `threshold`: the score at which the robot fires.

Restart the app or press **Reload models**. In Idle the wrist video shows the live score against the
threshold. If it fires when it should not, record what set it off as a **negative** and retrain; to
adjust by hand, set `wake.threshold` in `config/robot.yaml`.
