"""The Lab: a LeLab-style recording session, episode management, and tidying wake-word clips and wrist images.

Both use real LeRobot datasets in a temporary folder. Every take is filmed in its own shade of grey,
so the saved videos show exactly which takes were kept.
"""

import asyncio
import base64

import av
import cv2
import numpy as np
import pytest
from aiohttp.test_utils import TestClient, TestServer
from lerobot.datasets import LeRobotDataset
from test_core import make_brain  # pytest puts tests/ on the import path

from yogurt import audio, collect, lab
from yogurt.app import make_app

AUTH = {"Authorization": "Basic " + base64.b64encode(b"gurt:gurt").decode()}


class GreyCamera:
    """A wrist camera filming a plain grey picture; `shade` says which take is being filmed."""

    error = None

    def __init__(self):
        self.shade, self.stamp = 0, 1.0

    def open(self): ...
    def close(self): ...

    def read(self, max_age_s=1.0):
        return np.full((90, 160, 3), self.shade, np.uint8)


class FakeLeader:
    """A leader arm held exactly where the follower is."""

    def __init__(self, arm):
        self.bus = self
        self.arm, self.disconnected = arm, False

    def sync_read(self, register, num_retry=0):
        return dict(self.arm.pos)

    def disconnect(self, disable_torque=True):
        self.disconnected = True


def shades(dataset_dir, episode_rows, camera="wrist"):
    """The mean grey level of every frame of every episode, decoded from the dataset's own video files."""
    out = []
    for row in episode_rows:
        v = row["videos"][camera]
        file = dataset_dir / f"videos/observation.images.{camera}/chunk-{v['chunk']:03d}/file-{v['file']:03d}.mp4"
        with av.open(str(file)) as container:
            frames = [f.to_ndarray(format="gray").mean() for f in container.decode(video=0)]
        first = round(v["from"] * 30)
        out.append(frames[first:first + row["length"]])
    return out


@pytest.fixture
def data(tmp_path, monkeypatch):
    monkeypatch.setattr(collect, "DATA", tmp_path)
    return tmp_path / "grasp_act"


def test_recording_session_keeps_only_the_episodes_you_end(monkeypatch, data):
    async def scenario():
        brain = make_brain(monkeypatch)
        brain.cfg.leader["port"] = "fake"
        camera = brain.wrist = GreyCamera()
        leader = FakeLeader(brain.arm)
        monkeypatch.setattr(lab, "open_feetech", lambda kind, port, calibration: leader)

        async with TestClient(TestServer(make_app(brain.cfg, brain))) as client:  # runs the state machine
            async def post(path, body):
                response = await client.post(path, json=body, headers=AUTH)
                return response.status, await response.json()

            async def phase(name):
                for _ in range(500):
                    status = (await (await client.get("/api/status", headers=AUTH)).json())["session"]
                    if status.get("phase") == name:
                        return status
                    await asyncio.sleep(0.02)
                raise AssertionError(f"stuck in {brain.session_status().get('phase')} ({brain.detail}), not {name}")

            async def press(action, seen):
                status, reply = await post("/api/lab/session", {"action": action, "phase": seen})
                assert status == 200, reply

            async def record(shade, seconds=0.5):
                await phase("positioning")
                camera.shade = shade                # nothing is recorded until the recording phase
                await press("next", "positioning")  # no detector in tests: "Cup is centred"
                await phase("recording")
                await asyncio.sleep(seconds)

            settings = {"dataset": "demo", "new": True, "task": "Grab the yogurt cup.",
                        "num_episodes": 2, "episode_time_s": 3, "reset_time_s": 0, "from_survey": True}
            status, reply = await post("/api/lab/record", {**settings, "num_episodes": 2.5})
            assert status == 400 and "whole number" in reply["error"]
            assert (await post("/api/lab/record", settings))[0] == 200
            status, reply = await post("/api/lab/record", {**settings, "dataset": "other"})
            assert status == 400 and "already running" in reply["error"]

            await record(30)                       # take 1: thrown away with Re-record
            await press("rerecord", "recording")
            await phase("resetting")               # reset time 0: waits for Start Next Episode
            status, reply = await post("/api/lab/session", {"action": "next", "phase": "recording"})
            assert status == 400 and "moved on" in reply["error"]  # a late double click does nothing
            await asyncio.sleep(0.3)
            assert brain.session_status()["phase"] == "resetting"
            await press("next", "resetting")
            await record(90)                       # take 2: kept
            await press("next", "recording")
            status = await phase("resetting")
            assert status["saved"] == 1 and status["episode"] == 2 and not status["done"]
            await press("next", "resetting")
            await record(150, seconds=3.4)         # take 3: runs out of time, so it is discarded
            assert (await phase("resetting"))["saved"] == 1
            await press("next", "resetting")
            await record(180)                      # take 4: interrupted by Safe Stop
            assert (await post("/api/state", {"state": "SAFE_STOP"}))[0] == 200
            status = await phase("paused")
            assert status["saved"] == 1 and brain.session.open
            status, reply = await post("/api/lab/datasets/demo/delete-episodes", {"episodes": [0]})
            assert status == 400 and "recording session is running" in reply["error"]
            await press("resume", "paused")
            await record(220)                      # take 5: kept, the last one
            await press("next", "recording")
            status = await phase("resetting")
            assert status["done"] and status["saved"] == 2
            await press("next", "resetting")       # "Finish Session"
            for _ in range(200):
                if not brain.session.open and brain.state == "SAFE_STOP":
                    break
                await asyncio.sleep(0.02)
            last = (await (await client.get("/api/status", headers=AUTH)).json())["session"]["last"]
            assert last["saved"] == 2 and last["reason"] == "all episodes recorded" and last["kept"]
            assert leader.disconnected

            # a session that saves nothing does not leave an empty dataset behind
            assert (await post("/api/lab/record", {**settings, "dataset": "empty"}))[0] == 200
            await phase("positioning")
            assert (await post("/api/lab/session", {"action": "stop"}))[0] == 200
            assert brain.state == "SAFE_STOP" and not (data / "empty").exists()
            assert "not kept" in brain.session.last["note"]

    asyncio.run(scenario())

    described = lab.describe("demo")
    assert described["problem"] is None and described["total_episodes"] == 2
    assert [e["tasks"] for e in described["episodes"]] == [["Grab the yogurt cup."]] * 2
    for kept, grey in zip(shades(data / "demo", described["episodes"]), (90, 220), strict=True):
        assert len(kept) > 5 and all(abs(g - grey) < 6 for g in kept)  # no frame of a discarded take leaked in


def test_recording_session_starts_each_episode_where_the_arm_is(monkeypatch, data):
    """Without from_survey the arm never goes to the survey pose: every take, and a resumed one, starts in place."""
    async def scenario():
        brain = make_brain(monkeypatch)
        brain.cfg.leader["port"] = "fake"
        brain.wrist = GreyCamera()
        monkeypatch.setattr(lab, "open_feetech", lambda kind, port, calibration: FakeLeader(brain.arm))
        entered = []

        async with TestClient(TestServer(make_app(brain.cfg, brain))) as client:
            async def post(path, body):
                response = await client.post(path, json=body, headers=AUTH)
                return response.status, await response.json()

            async def phase(name):
                for _ in range(500):
                    entered.append(brain.state)
                    status = brain.session_status()
                    if status.get("phase") == name:
                        return status
                    await asyncio.sleep(0.02)
                raise AssertionError(f"stuck in {status.get('phase')} ({brain.detail}), not {name}")

            async def press(action, seen):
                status, reply = await post("/api/lab/session", {"action": action, "phase": seen})
                assert status == 200, reply

            settings = {"dataset": "here", "new": True, "task": "Grab the yogurt cup.",
                        "num_episodes": 2, "episode_time_s": 5, "reset_time_s": 0,
                        "dart_percent": 0}  # no perturbed takes: the follower stays exactly where the leader holds it
            assert (await post("/api/lab/record", {**settings, "from_survey": "yes"}))[0] == 400
            assert (await post("/api/lab/record", settings))[0] == 200
            await phase("recording")
            start = dict(brain.arm.pos)
            await asyncio.sleep(0.3)
            await press("next", "recording")
            await phase("resetting")
            await press("next", "resetting")
            await phase("recording")
            assert (await post("/api/state", {"state": "SAFE_STOP"}))[0] == 200
            await phase("paused")
            await press("resume", "paused")
            await phase("recording")
            assert "CHECK" not in entered
            survey = brain.cfg.poses.survey
            assert all(abs(brain.arm.pos[j] - start[j]) < 1 for j in start)  # every take began where the last ended
            assert abs(start["shoulder_lift"] - survey.shoulder_lift) > 5   # ... which is not the survey pose
            await asyncio.sleep(0.3)
            await press("next", "recording")
            assert (await phase("resetting"))["saved"] == 2

    asyncio.run(scenario())
    assert lab.describe("here")["total_episodes"] == 2


def make_dataset(cfg, name, lengths):
    dataset = LeRobotDataset.create(lab.repo_id(name), cfg.grasp.fps, features=lab.features(cfg),
                                    root=lab.root() / name, robot_type=lab.ROBOT_TYPE, use_videos=True)
    for k, length in enumerate(lengths):
        for i in range(length):
            dataset.add_frame({"observation.state": np.full(6, k, np.float32),
                               "action": np.full(6, 10 * k + i, np.float32),
                               "observation.images.wrist": np.full((cfg.grasp.image_height, cfg.grasp.image_width, 3),
                                                                   40 + 60 * k, np.uint8),
                               "task": cfg.grasp.task})
        dataset.save_episode()
    dataset.finalize()


def test_episodes_can_be_browsed_and_deleted(monkeypatch, data):
    async def scenario():
        brain = make_brain(monkeypatch)
        make_dataset(brain.cfg, "grasps", [12, 15, 18, 9])
        (data / "broken" / "meta").mkdir(parents=True)  # a recording killed before it was finalised
        (data / "broken" / "meta" / "info.json").write_text((data / "grasps" / "meta" / "info.json").read_text())

        async with TestClient(TestServer(make_app(brain.cfg, brain))) as client:
            async def call(method, path, **kw):
                response = await client.request(method, path, headers={**AUTH, **kw.pop("headers", {})}, **kw)
                return response.status, (await response.json() if response.content_type == "application/json" else None)

            status, listing = await call("GET", "/api/lab/datasets")
            by_name = {d["name"]: d for d in listing["datasets"]}
            assert by_name["grasps"]["episodes"] == 4 and by_name["grasps"]["problem"] is None
            assert "interrupted" in by_name["broken"]["problem"]

            status, d = await call("GET", "/api/lab/datasets/grasps")
            assert status == 200 and [e["length"] for e in d["episodes"]] == [12, 15, 18, 9]
            _, joints = await call("GET", "/api/lab/datasets/grasps/episodes/2/joints")
            assert len(joints["action"]) == 18 and joints["action"][5][0] == 25 and joints["names"][-1] == "gripper"

            v = d["episodes"][1]["videos"]["wrist"]
            video = await client.get(f"/api/lab/datasets/grasps/videos/wrist/{v['chunk']}/{v['file']}",
                                     headers={**AUTH, "Range": "bytes=0-99"})
            assert video.status == 206 and len(await video.read()) == 100
            assert (await client.get("/api/lab/datasets/grasps/videos/wrist/0/7", headers=AUTH)).status == 404
            assert (await client.get("/api/lab/datasets/..%2F..%2Fconfig/videos/wrist/0/0", headers=AUTH)).status == 404
            status, err = await call("GET", "/api/lab/datasets/..%2F..%2Fconfig")
            assert status == 400 and "name" in err["error"]

            everything = {"episodes": [0, 1, 2, 3]}
            status, err = await call("POST", "/api/lab/datasets/grasps/delete-episodes", json=everything)
            assert status == 400 and "delete the whole dataset" in err["error"]
            status, err = await call("POST", "/api/lab/datasets/grasps/delete-episodes", json={"episodes": [4]})
            assert status == 400
            status, done = await call("POST", "/api/lab/datasets/grasps/delete-episodes", json={"episodes": [1, 3]})
            assert status == 200 and done["remaining"] == 2

            _, d = await call("GET", "/api/lab/datasets/grasps")
            assert [e["length"] for e in d["episodes"]] == [12, 18]   # old episode 2 is now episode 1
            _, joints = await call("GET", "/api/lab/datasets/grasps/episodes/1/joints")
            assert joints["action"][0][0] == 20 and joints["state"][0][0] == 2
            for frames, grey, length in zip(shades(data / "grasps", d["episodes"]), (40, 160), (12, 18), strict=True):
                assert len(frames) == length and all(abs(g - grey) < 6 for g in frames)
            assert not [p for p in data.iterdir() if p.name.startswith(".")]  # the staging folder is gone

            status, err = await call("POST", "/api/lab/datasets/grasps/delete", json={"confirm": "nope"})
            assert status == 400 and (data / "grasps").exists()
            status, _ = await call("POST", "/api/lab/datasets/grasps/delete", json={"confirm": "grasps"})
            assert status == 200 and not (data / "grasps").exists()

    asyncio.run(scenario())


def test_wake_clips_and_wrist_images_can_be_tidied(monkeypatch, data):
    root = data.parent  # the temporary data/ folder
    for label, name in [("positive", "20261118-100000-001_yo-gurt.wav"),
                        ("positive", "20261118-100001-002_yo-gurt.wav"),
                        ("negative", "20261118-100002-003_yogurt.wav")]:
        audio.write_wav(root / "wake_word" / label / name, 0.5 * np.sin(np.arange(16000) / 5))
    picture = np.full((90, 160, 3), 128, np.uint8)
    for dataset, label, burst, n in [("grasp_verify", "held", "20261118-100000", 3),
                                     ("grasp_verify", "held", "20261118-100100", 2),
                                     ("yogurt_detection", "images", "20261118-100200", 3)]:
        (root / dataset / label / burst).mkdir(parents=True)
        for i in range(n):
            cv2.imwrite(str(root / dataset / label / burst / f"{i:05d}.jpg"), picture)

    async def scenario():
        brain = make_brain(monkeypatch)
        brain.wrist = GreyCamera()
        async with TestClient(TestServer(make_app(brain.cfg, brain))) as client:
            async def call(method, path, body=None):
                response = await client.request(method, path, json=body, headers=AUTH)
                return response.status, (await response.json() if response.content_type == "application/json" else None)

            # ---- wake-word clips
            status, wake = await call("GET", "/api/lab/wake")
            assert status == 200 and wake["totals"]["positive"]["clips"] == 2 and wake["prompts"]["positive"]
            clip = next(c for c in wake["clips"] if c["label"] == "negative")
            assert clip["phrase"] == "yogurt" and clip["seconds"] == 1.0
            assert clip["peak"] == pytest.approx(0.5, abs=0.01)
            played = await client.get(f"/api/lab/wake/negative/{clip['name']}", headers=AUTH)
            assert played.status == 200 and (await played.read())[:4] == b"RIFF"
            assert (await client.get("/api/lab/wake/negative/..%2F..%2Fshots.json", headers=AUTH)).status == 404
            fumbled = {"label": "positive", "name": "20261118-100001-002_yo-gurt.wav"}
            assert (await call("POST", "/api/lab/wake/move", {"clips": [fumbled], "to": "negative"}))[1]["moved"] == 1
            assert (root / "wake_word" / "negative" / fumbled["name"]).exists()
            status, _ = await call("POST", "/api/lab/wake/delete", {"clips": [fumbled]})  # it is not there any more
            assert status == 400
            gone = {"label": "negative", "name": fumbled["name"]}
            assert (await call("POST", "/api/lab/wake/delete", {"clips": [gone]}))[1]["deleted"] == 1
            assert len(list((root / "wake_word").rglob("*.wav"))) == 2

            # ---- wrist images
            status, sets = await call("GET", "/api/lab/images")
            assert sets["sets"]["grasp_verify/held"] == {"images": 5, "bursts": 2, "target": 300}
            status, listing = await call("GET", "/api/lab/images/grasp_verify/held")
            assert [b["name"] for b in listing["bursts"]] == ["20261118-100100", "20261118-100000"]
            held_dir = "/api/lab/images/grasp_verify/held"
            thumb = await client.get(f"{held_dir}/20261118-100000/00001.jpg?thumb=1", headers=AUTH)
            assert cv2.imdecode(np.frombuffer(await thumb.read(), np.uint8), cv2.IMREAD_COLOR).shape[1] == 320
            assert (await client.get(f"{held_dir}/..%2Fempty/00001.jpg", headers=AUTH)).status == 404

            det = "20261118-100200"

            def box(name, value):
                return call("POST", "/api/lab/images/box", {"burst": det, "name": name, "box": value})

            assert (await box("00000.jpg", [0.1, 0.2, 0.5, 0.6]))[0] == 200
            assert (await box("00001.jpg", "none"))[0] == 200
            status, err = await box("00002.jpg", [0.1, 0.1, 0.1, 0.1])
            assert status == 400 and "too small" in err["error"]
            assert (await call("GET", "/api/lab/images/next-unlabelled"))[1] == {"burst": det, "name": "00002.jpg"}
            _, shown = await call("GET", f"/api/lab/images/yogurt_detection/images/{det}")
            assert [i["box"] for i in shown["images"]] == [[0.1, 0.2, 0.5, 0.6], "none", None]
            labels = root / "yogurt_detection" / "labels" / det
            assert (labels / "00000.txt").read_text().split()[0] == "0" and (labels / "00001.txt").read_text() == ""
            assert (await box("00001.jpg", None))[0] == 200
            assert not (labels / "00001.txt").exists()
            _, left = await call("POST", "/api/lab/images/delete", {"dataset": "yogurt_detection", "label": "images",
                                                                    "burst": det, "names": ["00000.jpg"]})
            assert left == {"deleted": 1, "left": 2} and not (labels / "00000.txt").exists()
            await call("POST", "/api/lab/images/delete", {"dataset": "yogurt_detection", "label": "images",
                                                          "burst": det, "names": ["00001.jpg", "00002.jpg"]})
            assert not (root / "yogurt_detection" / "images" / det).exists() and not labels.exists()

            held = {"dataset": "grasp_verify", "label": "held", "burst": "20261118-100100"}
            assert (await call("POST", "/api/lab/images/move", {**held, "to": "held"}))[0] == 400
            moved = await call("POST", "/api/lab/images/move", {**held, "to": "empty"})
            assert moved[1] == {"burst": "20261118-100100"}
            assert len(list((root / "grasp_verify" / "empty" / "20261118-100100").glob("*.jpg"))) == 2

            # two bursts started within the same second get their own folders; the one being filled is protected
            capture = {"dataset": "grasp_verify", "label": "held", "per_second": 5}
            first = (await call("POST", "/api/lab/images/capture", capture))[1]
            assert (await call("POST", "/api/lab/images/capture", capture))[0] == 400
            status, err = await call("POST", "/api/lab/images/delete", {**held, "burst": first["burst"]})
            assert status == 400 and "Stop" in err["error"]
            await call("POST", "/api/lab/images/capture", {"stop": True})
            second = (await call("POST", "/api/lab/images/capture", capture))[1]
            await call("POST", "/api/lab/images/capture", {"stop": True})
            assert second["burst"] != first["burst"]
            assert (await call("POST", "/api/lab/images/delete", {**held, "burst": first["burst"]}))[0] == 200

    asyncio.run(scenario())
