# Data

Everything here is recorded from the web UI (section **Data collection**), with the robot's own
cameras and microphones, so the models train on exactly what they will see and hear later.

| Folder | For | What to record | Minimum to start | Guide |
|---|---|---|---|---|
| `wake_word/` | Idle: hearing "Yo, gurt" | short audio clips | 150 positive, 300 negative, 10 min background | [README](wake_word/README.md) |
| `yogurt_detection/` | Check: finding the cup | wrist images + a box on each | 250 images | [README](yogurt_detection/README.md) |
| `grasp_act/` | Grasp: the ACT policy | leader-arm demonstrations | 50 demonstrations | [README](grasp_act/README.md) |
| `grasp_verify/` | Verify: is the cup in the gripper? | wrist images, `held` / `empty` | 300 of each | [README](grasp_verify/README.md) |
| `handoff_grab/` | Handoff: has someone taken hold? | wrist images, `grabbed` / `not_grabbed` | 300 of each | [README](handoff_grab/README.md) |

`wake_word/prompts.json` is the script of phrases the UI asks you to say; `shots.json` is the list of
situations the UI asks you to photograph. Edit either file to change what the UI prompts for.

Suggested order: grasp demonstrations first (they take longest and fix the `survey` pose for good),
then the two classifiers, the detector, and the wake word.

Audio, images and video are not committed to git (see `.gitignore`); back the folder up yourself.
