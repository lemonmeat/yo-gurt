"""What the microphones hear.

Mic         a rolling 30 s buffer of sound from the base camera's stereo microphones, plus the
            direction each bit of speech came from (time difference between the two microphones).
WakeWord    the "Yo, gurt" detector trained by yogurt.train.wake_word.
Transcriber speech-to-text for the command that follows the wake word.
"""

import logging
import re
import threading
import time
import wave
from collections import deque
from pathlib import Path

import numpy as np

log = logging.getLogger("yogurt.audio")

RATE = 16000            # everything downstream of the microphone works at 16 kHz mono
BLOCK_S = 0.03
SPEED_OF_SOUND = 343.0
BUFFER_S = 30

# ---- wake-word features (shared by training and by the live detector) ----------------------

WAKE_WINDOW_S = 1.5
N_FFT, WIN, HOP, N_MELS = 512, 400, 160, 40


def _mel_filters() -> np.ndarray:
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    edges_hz = 700 * (10 ** (np.linspace(to_mel(60), to_mel(7600), N_MELS + 2) / 2595) - 1)
    bins = np.fft.rfftfreq(N_FFT, 1 / RATE)
    filters = np.zeros((N_MELS, len(bins)), np.float32)
    for i in range(N_MELS):
        lo, mid, hi = edges_hz[i:i + 3]
        filters[i] = np.clip(np.minimum((bins - lo) / (mid - lo), (hi - bins) / (hi - mid)), 0, None)
    return filters


MEL = _mel_filters()
WINDOW = np.hanning(WIN).astype(np.float32)


def wake_features(audio: np.ndarray) -> np.ndarray:
    """1.5 s of 16 kHz audio -> (148, 40) log-mel picture, normalised so loudness does not matter."""
    audio = np.asarray(audio, np.float32)[-int(WAKE_WINDOW_S * RATE):]
    audio = np.pad(audio, (int(WAKE_WINDOW_S * RATE) - len(audio), 0))
    frames = np.lib.stride_tricks.sliding_window_view(audio, WIN)[::HOP] * WINDOW
    power = np.abs(np.fft.rfft(frames, N_FFT)) ** 2
    logmel = np.log(power @ MEL.T + 1e-6)
    return ((logmel - logmel.mean()) / (logmel.std() + 1e-3)).astype(np.float32)


# ---- direction of sound ---------------------------------------------------------------------


def stereo_delay(left: np.ndarray, right: np.ndarray, rate: int, max_delay_s: float) -> float:
    """Seconds by which the sound reached the LEFT microphone later than the right one (GCC-PHAT).

    Positive means the source is on the right-hand microphone's side.
    """
    n = 1 << (2 * len(left) - 1).bit_length()
    window = np.hanning(len(left))
    cross = np.fft.rfft(left * window, n) * np.conj(np.fft.rfft(right * window, n))
    cross /= np.abs(cross) + 1e-12                       # PHAT: keep phase only
    freqs = np.fft.rfftfreq(n, 1 / rate)
    cross[(freqs < 300) | (freqs > 5000)] = 0            # speech band
    upsample = 8                                         # finer than one sample
    correlation = np.fft.irfft(cross, n * upsample)
    reach = int(max_delay_s * rate * upsample) + 1
    correlation = np.concatenate([correlation[-reach:], correlation[:reach + 1]])
    return (int(np.argmax(correlation)) - reach) / (rate * upsample)


def delay_to_bearing(delay_s: float, spacing_m: float) -> float:
    """Degrees off the camera axis, positive = towards the right microphone."""
    return float(np.degrees(np.arcsin(np.clip(delay_s * SPEED_OF_SOUND / spacing_m, -1, 1))))


# ---- microphone -----------------------------------------------------------------------------


def find_mic(name) -> int:
    import sounddevice as sd

    if isinstance(name, int):
        return name
    key = re.sub(r"[^a-z0-9]", "", str(name).lower())
    inputs = [(i, d["name"]) for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0]
    matches = [i for i, device in inputs if key in re.sub(r"[^a-z0-9]", "", device.lower())]
    if len(matches) != 1:
        problem = "not found" if not matches else "matches more than one device"
        raise RuntimeError(f"microphone '{name}' {problem}; inputs present: {[n for _, n in inputs]}")
    return matches[0]


class Mic:
    def __init__(self, cfg):
        self.cfg = cfg.mic
        self.rate = self.cfg.sample_rate
        if self.rate % RATE:
            raise ValueError("mic.sample_rate must be a multiple of 16000")
        self.step = self.rate // RATE
        # Low-pass before keeping every step-th sample (windowed-sinc FIR).
        taps = np.sinc(np.arange(-24 * self.step, 24 * self.step + 1) / self.step) / self.step
        self.taps = (taps * np.hamming(len(taps))).astype(np.float32)
        self.ring = np.zeros(BUFFER_S * RATE, np.float32)
        self.written = 0                 # total 16 kHz samples ever written
        self.stamp = 0.0                 # time.monotonic() at the end of the newest sample
        self.level = 0.0                 # loudness of the newest block (RMS, 0..1)
        self.noise = 0.0                 # slow-moving estimate of the room's background level
        self.last_voice = 0.0            # time.monotonic() of the newest block that held speech
        self.voice: deque = deque()      # (time, bearing_deg or None) for every recent block of speech
        self.error: str | None = "not opened yet"
        self.channels = 0
        self._tail = np.zeros(len(self.taps) - 1, np.float32)
        self._previous: np.ndarray | None = None
        self._stream = None
        self._lock = threading.Lock()

    def open(self) -> None:
        import sounddevice as sd

        self.close()
        try:
            device = find_mic(self.cfg.device)
            self.channels = min(2, sd.query_devices(device)["max_input_channels"])
            self._stream = sd.InputStream(device=device, channels=self.channels, samplerate=self.rate,
                                          blocksize=int(self.rate * BLOCK_S), dtype="float32",
                                          callback=self._on_block)
            self._stream.start()
            self.error = None
        except Exception as e:
            self.error = str(e)
            raise

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
        self._stream, self.error = None, "closed"

    def _on_block(self, block: np.ndarray, frames: int, time_info, status) -> None:
        self.feed(block.copy(), time.monotonic())

    def feed(self, block: np.ndarray, now: float) -> None:
        """Take one block of raw microphone samples, shaped (samples, channels)."""
        mono = np.concatenate([self._tail, block.mean(axis=1)])
        self._tail = mono[-(len(self.taps) - 1):]
        samples = np.convolve(mono, self.taps, "valid")[::self.step]
        level = float(np.sqrt(np.mean(samples ** 2)))
        # Background level: drops quickly to any quieter block, creeps up slowly otherwise.
        self.noise = level if level < self.noise or self.noise == 0 else self.noise + (level - self.noise) * 0.002
        voiced = level > max(self.cfg.speech_rms, 3 * self.noise)
        with self._lock:
            at = self.written % len(self.ring)
            first = min(len(samples), len(self.ring) - at)
            self.ring[at:at + first] = samples[:first]
            self.ring[:len(samples) - first] = samples[first:]
            self.written += len(samples)
            self.stamp, self.level = now, level
            if voiced:
                bearing = None
                if block.shape[1] == 2 and self._previous is not None:
                    both = np.concatenate([self._previous, block])
                    left, right = (both[:, 1], both[:, 0]) if self.cfg.flip else (both[:, 0], both[:, 1])
                    delay = stereo_delay(left, right, self.rate, self.cfg.spacing_m / SPEED_OF_SOUND)
                    bearing = delay_to_bearing(delay, self.cfg.spacing_m)
                self.last_voice = now
                self.voice.append((now, bearing))
            while self.voice and now - self.voice[0][0] > BUFFER_S:
                self.voice.popleft()
        self._previous = block

    def audio(self, since: float, until: float | None = None) -> np.ndarray:
        """The 16 kHz mono sound between two time.monotonic() moments (until defaults to now)."""
        with self._lock:
            end = self.written - int(max(0.0, self.stamp - (until or self.stamp)) * RATE)
            start = self.written - int(max(0.0, self.stamp - since) * RATE)
            start = max(start, self.written - len(self.ring), 0)
            index = np.arange(start, max(start, end)) % len(self.ring)
            return self.ring[index].copy()

    def bearing(self, since: float) -> float | None:
        """Where the voice came from since a moment in the past: the median over the speech blocks."""
        with self._lock:
            values = [b for t, b in self.voice if t >= since and b is not None]
        return float(np.median(values)) if len(values) >= 3 else None

    def speech_seconds(self, since: float) -> float:
        """How much speech has been heard since a moment in the past."""
        with self._lock:
            return sum(1 for t, _ in self.voice if t >= since) * BLOCK_S


# ---- wake word ------------------------------------------------------------------------------


class WakeWord:
    def __init__(self, model_file: Path, threshold: float | None):
        import json

        import onnxruntime as ort

        self.session = ort.InferenceSession(str(model_file), providers=["CPUExecutionProvider"])
        saved = json.loads(model_file.with_suffix(".json").read_text())
        self.threshold = threshold if threshold is not None else saved["threshold"]

    def score(self, audio: np.ndarray) -> float:
        """P(the last 1.5 s of audio contain the wake phrase)."""
        logits = self.session.run(None, {"mel": wake_features(audio)[None, None]})[0][0].astype(np.float64)
        odds = np.exp(logits - logits.max())
        return float(odds[1] / odds.sum())


# ---- speech to text -------------------------------------------------------------------------


class Transcriber:
    """faster-whisper, loaded in the background so the UI comes up immediately."""

    def __init__(self, cfg):
        self.cfg, self.model, self.error = cfg.asr, None, None
        threading.Thread(target=self._load, name="asr-load", daemon=True).start()

    def _load(self) -> None:
        try:
            from faster_whisper import WhisperModel

            self.model = WhisperModel(self.cfg.model, device="cpu", compute_type=self.cfg.compute_type)
            log.info("speech-to-text model '%s' ready", self.cfg.model)
        except Exception as e:
            self.error = f"could not load speech-to-text model '{self.cfg.model}': {e}"
            log.error(self.error)

    def transcribe(self, audio: np.ndarray) -> str:
        if self.model is None:
            raise RuntimeError(self.error or "speech-to-text model is still loading")
        segments, _ = self.model.transcribe(
            audio, language="en", beam_size=5, vad_filter=True, condition_on_previous_text=False,
            initial_prompt="Yo, gurt, give me the yogurt.",  # spells the words we expect
        )
        return " ".join(segment.text.strip() for segment in segments).strip()


# ---- wav files (data collection and training) -----------------------------------------------


def write_wav(file: Path, audio: np.ndarray) -> None:
    file.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(file), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(RATE)
        out.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())


def read_wav(file: Path) -> np.ndarray:
    with wave.open(str(file), "rb") as source:
        if source.getframerate() != RATE or source.getsampwidth() != 2:
            raise ValueError(f"{file}: expected 16 kHz 16-bit audio (record it from the UI)")
        audio = np.frombuffer(source.readframes(source.getnframes()), np.int16).astype(np.float32) / 32767
        return audio.reshape(-1, source.getnchannels()).mean(axis=1)
