# Yo-gurt

Say **"Yo, gurt, give me the yogurt"** and an SO-101 arm finds the yogurt cup, picks it up, and
holds it out to you, or to the person you pointed it at ("give *him* the yogurt").

Everything runs from one web page: live video with overlays that show what the robot sees and why
it is moving, the state machine, a Safe Stop and a Kill switch, buttons to step through states by
hand. Every dataset the models need is recorded and tidied up in the **Lab**
(<http://127.0.0.1:8080/lab>), modelled on Hugging Face's [LeLab](https://github.com/huggingface/leLab):
grasp demonstrations as timed recording sessions into LeRobot datasets (replay, inspect and delete
individual episodes), wrist-camera pictures (capture bursts, view, delete, relabel, draw the
detector's boxes) and wake-word clips (record, play, relabel, delete).

## Run it

```bash
uv sync                                    # Python 3.12+, installs everything the robot needs
brew install ollama                        # the local language model server (Linux: see "Raspberry Pi")
brew services start ollama && ollama pull gemma3:1b
uv sync --extra train                      # only on the machine that trains/exports models
uv run python -m yogurt.train.detector pretrained   # writes the ready-made detector to models/

uv run yogurt laptop                       # uses config/hardware_laptop.yaml
uv run yogurt laptop --sim                 # same, with a simulated arm (no servos needed)
```

Open <http://127.0.0.1:8080> and log in with **gurt / gurt**.

The program starts in **Safe Stop**: the arm holds wherever it is and nothing moves until you click
**IDLE**. On macOS the terminal you launch from needs Camera and Microphone permission
(System Settings → Privacy & Security); without microphone permission the UI reports "pure silence".

## What each state does

| State | What happens | Model it uses | Until that model exists |
|---|---|---|---|
| **Idle** | Rests in the `idle` pose, listening for "Yo, gurt". | `wake_word.onnx` (you train it) | **Wake** button, or type a command |
| **Listen** | Records the command. Estimates who is speaking from the two microphones (time difference → bearing) and from mouth movement on each face; turns the arm to face them. No speech in 4 s → Idle. | YuNet face detector (bundled) | — |
| **Parse** | Speech → text → "is this a yogurt request, and for whom?" "me" → the requester; "him/her/a name" → the other person nearest the requester. | faster-whisper, `gemma3:1b` via Ollama | type the command |
| **Check** | Raises the wrist camera, sweeps left → right until it sees a cup, then turns until the cup is centred. That is the pre-grasp pose. Nothing found → Idle. | `yogurt_detector.onnx` | **Yogurt centred** button |
| **Grasp** | Runs the ACT policy for `grasp.duration_s`. | `models/act_grasp/` (you train it) | grasp by hand, **Grasp done** |
| **Verify** | Lifts the cup in front of the wrist camera and asks "is it held?". No → back to Check, for as long as it takes. | `grasp_verify.onnx` (you train it) | **Held / Not held** |
| **Handoff** | Turns to the target person, reaches out, waits for a hand to take the cup, lets go. Nobody in 10 s → puts the cup back where the gripper closed on it. | `handoff_grab.onnx` (you train it), watching the wrist camera and the base camera's view of the gripper | **Taken** |
| **Return** | Back to the `idle` pose, then Idle. | — | — |
| **Safe Stop** | Freezes the arm; servos stay powered and hold. (Space bar) | — | — |
| **Kill** | Releases every servo and disconnects arm, cameras and microphone. (Esc) **Reconnect hardware** brings it back, in Safe Stop. | — | — |

Click any state in the top bar to jump straight to it. The wake word can only fire in Idle, so the
word "yogurt" inside a command can never re-trigger it.

## First time with the real arm

Do these once, in Safe Stop, with a hand near the KILL button. They take about ten minutes and
replace guesses in the config with facts about your robot.

1. **Port.** `arm.port: auto` works when the follower is the only USB serial device. With the leader
   also plugged in, put both ports in `config/hardware_laptop.yaml` (`lerobot-find-port`).
2. **Calibration.** `config/arm_config.json` is the file from the brief. If the servos hold a different
   calibration, the program writes this one to them at connect (and says so in the terminal).
3. **Poses.** `config/poses.yaml` was computed from the SO-101 model, not tried on your arm. In the Arm
   panel press **Go there** for `idle`, `survey` and `handoff` one at a time and watch. To fix one:
   **Release servos**, move the arm by hand, **Engage servos**, **Save current as …**.
   - `survey`: wrist camera looking down at the table, gripper open and about 15 cm above the table.
     Check sweeps in this pose, and every grasp starts from it.
   - `handoff`: arm stretched out, gripper level so the cup stays upright (its gripper value is ignored).
     Then, still in that pose, look at the white **hand-off view** box on the base video: set
     `handoff.base_crop` in `config/robot.yaml` so the gripper and the space under it fill the box.
4. **Directions.** Jog `shoulder_pan` with **+5**. If things in the *wrist* picture slide right instead
   of left, set `check.pan_sign: -1` in `config/robot.yaml`. Run a request and look at the top view: if the
   arm turns away from the person, fix `base_camera.pan_sign` / `base_camera.yaw_deg`.
5. **Microphones.** Measure the distance between the two microphone holes on the base camera and put it
   in `mic.spacing_m`. Speak from the left: if the pink "voice" line shows on the right, set `mic.flip: true`.
6. **Camera angles.** `hfov_deg` for both cameras are estimates; they only scale the bearings.

## Safety

- Every motion goes through `Arm` ([src/yogurt/arm.py](src/yogurt/arm.py)): targets are clamped to the
  calibrated range minus 3°, the commanded position moves at a bounded speed (40°/s scripted, 150°/s
  ceiling for the policy and teleoperation), and it may lead the measured position by at most 15°, so a
  blocked arm is not pushed harder and harder. The ACT policy's output passes through the same limits.
- Servo temperatures are read every second and shown per joint; at 62 °C the servos are released.
- LeRobot's gripper protection (50 % torque, over-current limit) is applied at connect.
- If the serial link fails ten times in a row, or a move does not arrive, the robot stops and says why.
- Ctrl-C in the terminal does what Kill does. Kill lets the arm fall: support it if it is holding a cup.

## Data and training

Record everything in the **Lab** (the robot page's **Lab →** button), using the robot's own cameras and
microphones. Each Lab tab shows how much has been recorded, what is still missing, and the command
that trains its model. Each folder has its own instructions:

| What | Record | Guide | Train | Result |
|---|---|---|---|---|
| Wake word | Lab → Wake word | [data/wake_word](data/wake_word/README.md) | `uv run python -m yogurt.train.wake_word` | `models/wake_word.onnx` |
| Yogurt detector | Lab → Wrist images | [data/yogurt_detection](data/yogurt_detection/README.md) | `uv run python -m yogurt.train.detector train` | `models/yogurt_detector.onnx` |
| Grasp (ACT) | Lab → Grasp demonstrations | [data/grasp_act](data/grasp_act/README.md) | `lerobot-train …` (the Lab shows it) | `models/act_grasp/` |
| Grasp verification | Lab → Wrist images | [data/grasp_verify](data/grasp_verify/README.md) | `uv run python -m yogurt.train.classifier grasp_verify` | `models/grasp_verify.onnx` |
| Hand-off | Lab → Wrist images | [data/handoff_grab](data/handoff_grab/README.md) | `uv run python -m yogurt.train.classifier handoff_grab` | `models/handoff_grab.onnx` |

After training, press **Reload models** (in Safe Stop, on the robot page or in the Lab) or restart.
The "Models and hardware" panel shows which models are loaded.

The ready-made detector (`detector pretrained`) is COCO YOLO11s using its "cup" and "bowl" classes. It
found the yogurt in the brief's concept picture with a confidence between 0.3 and 0.7 depending on
framing, and it will also fire on mugs. Treat it as a way to get started and train your own.

## Raspberry Pi 5 + AI HAT+

```bash
sudo apt install libportaudio2
curl -LsSf https://astral.sh/uv/install.sh | sh
curl -fsSL https://ollama.com/install.sh | sh && ollama pull gemma3:1b
git clone <this repo> && cd yo-gurt && uv sync
# copy models/ from the laptop, then:
uv run yogurt pi          # http://<pi-address>:8080 from the laptop, gurt / gurt
```

`config/hardware_pi.yaml` listens on the local network and runs every model on the Pi's CPU, which
needs nothing from the HAT. **I had no Pi or Hailo hardware, so nothing in this section has been run.**

To move a vision model onto the AI HAT+, compile it to a `.hef` with Hailo's Dataflow Compiler (x86
Linux) and point the model path in `hardware_pi.yaml` at it; a path ending in `.hef` is run through
HailoRT ([src/yogurt/infer.py](src/yogurt/infer.py), written from Hailo's reference `HailoInfer` example):

- **Detector**: the code expects a HEF with HailoRT's built-in NMS, which is what
  `hailomz compile yolov11s --ckpt best.onnx --hw-arch hailo8 --calib-path data/yogurt_detection/images --classes 1`
  produces (export `best.pt` with plain `yolo export format=onnx imgsz=640` first). The pre-compiled
  COCO `yolov11s.hef` from the Hailo Model Zoo works for the ready-made detector. Keep
  `models/yogurt_detector.json` next to the `.hef`: it holds the label names.
- **Classifiers**: parse `grasp_verify.onnx` / `handoff_grab.onnx`, and add
  `normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])` to the model script so the HEF takes 0–255 pixels.
- HailoRT's Python package is installed for the system Python (`sudo apt install hailo-all`), so create
  the environment from it: `uv venv --python /usr/bin/python3 --system-site-packages`. This project needs
  Python 3.12 or newer.
- The wake word, speech-to-text, language model and ACT policy stay on the CPU. (An AI HAT+ 2 can serve
  the language model through `hailo-ollama`; set `slm.url` to it.)

## Files

```
config/hardware_laptop.yaml, hardware_pi.yaml   ports, cameras, microphone, model files, web address
config/robot.yaml                               limits, timings, thresholds (same on every computer)
config/poses.yaml                               idle / survey / handoff, rewritten when you re-teach
config/arm_config.json                          follower calibration
data/                                           datasets + how to record each
models/                                         trained models (not in git)
src/yogurt/app.py        web server, login, video streams, API
src/yogurt/fsm.py        the state machine
src/yogurt/arm.py        the arm and its safety limits
src/yogurt/audio.py      microphone, voice direction, wake word, speech-to-text
src/yogurt/vision.py     faces and mouth movement, yogurt detector, classifiers
src/yogurt/parse.py      the language-model command parser
src/yogurt/grasp.py      ACT policy runner
src/yogurt/collect.py    wake-word clips, wrist-image bursts and detector boxes: record, list, move, delete
src/yogurt/lab.py        grasp demonstration datasets: recording sessions, episode deletion
src/yogurt/static/       index.html (the robot page), lab.html (the Lab)
src/yogurt/overlay.py    what is drawn on the video
src/yogurt/infer.py      ONNX (CPU) / HEF (Hailo) model loading
src/yogurt/train/        the three training scripts
```

`uv run pytest` runs eight tests: joint limits, the motion limiter, voice direction, a full request
through the state machine followed by Safe Stop and Kill, the login, a whole Lab recording session
(re-record, time-out, pause, resume; only the takes you end are kept), browsing and deleting
episodes, and tidying up wake-word clips and wrist pictures (relabel, delete, boxes).
