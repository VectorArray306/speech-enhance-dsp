"""High-pass filter and voice-activity detection for live audio."""

from __future__ import annotations

import math

import numpy as np
import torch


class Highpass:
    """One-pole high-pass. It removes low rumble while audio is still arriving."""

    def __init__(self, sample_rate: int, cutoff_hz: float) -> None:
        self._x1 = 0.0
        self._y1 = 0.0
        if cutoff_hz <= 0:
            self._alpha = 1.0
        else:
            dt = 1.0 / sample_rate
            rc = 1.0 / (2.0 * math.pi * cutoff_hz)
            self._alpha = rc / (rc + dt)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        out = np.empty(len(frame), dtype=np.float32)
        x1, y1, alpha = self._x1, self._y1, self._alpha
        for index, sample in enumerate(np.asarray(frame, dtype=np.float32)):
            value = float(sample)
            y1 = alpha * (y1 + value - x1)
            x1 = value
            out[index] = y1
        self._x1, self._y1 = x1, y1
        return out


class SpeechSegmenter:
    """Collect speech and return one utterance after a long enough pause.

    Silero scores 32 ms blocks. Incoming frames can be any length; they are
    buffered until a full block is ready. A pause inside a word is kept,
    because dysarthric speech often has gaps longer than ordinary silence.
    """

    def __init__(self, config: dict) -> None:
        audio = config["audio"]
        vad = config["vad"]
        self.sample_rate = int(audio["sample_rate"])
        self.highpass = Highpass(self.sample_rate, float(audio.get("highpass_hz", 0)))
        self.threshold = float(vad["threshold"])
        self.window = 512 if self.sample_rate == 16000 else 256
        self.min_speech = int(self.sample_rate * float(vad["min_speech_ms"]) / 1000)
        self.min_silence = int(self.sample_rate * float(vad["min_silence_ms"]) / 1000)
        self.pad = int(self.sample_rate * float(vad["speech_pad_ms"]) / 1000)
        self.max_samples = int(self.sample_rate * float(vad["max_utterance_s"]))
        self._model = None
        self._pending = np.zeros(0, dtype=np.float32)
        self._pre_roll = np.zeros(0, dtype=np.float32)
        self._candidate = np.zeros(0, dtype=np.float32)
        self._utterance = np.zeros(0, dtype=np.float32)
        self._speech_run = 0
        self._silence_run = 0
        self._active = False

    def _load(self) -> None:
        if self._model is not None:
            return
        model, _utils = torch.hub.load(
            "snakers4/silero-vad",
            "silero_vad",
            trust_repo=True,
            verbose=False,
        )
        self._model = model
        self._model.reset_states()

    def _speech_probability(self, block: np.ndarray) -> float:
        self._load()
        tensor = torch.from_numpy(block).unsqueeze(0)
        with torch.inference_mode():
            return float(self._model(tensor, self.sample_rate).item())

    def _remember_preroll(self, block: np.ndarray) -> None:
        if self.pad <= 0:
            return
        joined = np.concatenate([self._pre_roll, block])
        self._pre_roll = joined[-self.pad :]

    def push(self, frame: np.ndarray) -> list[np.ndarray]:
        filtered = self.highpass.apply(frame)
        self._pending = np.concatenate([self._pending, filtered])
        ready: list[np.ndarray] = []
        while len(self._pending) >= self.window:
            block = self._pending[: self.window]
            self._pending = self._pending[self.window :]
            utterance = self._consume(block)
            if utterance is not None:
                ready.append(utterance)
        return ready

    def finish(self) -> list[np.ndarray]:
        ready: list[np.ndarray] = []
        if len(self._pending):
            padded = np.pad(self._pending, (0, self.window - len(self._pending)))
            self._pending = np.zeros(0, dtype=np.float32)
            utterance = self._consume(padded)
            if utterance is not None:
                ready.append(utterance)
        if self._active and len(self._utterance):
            ready.append(self._utterance.copy())
            self._reset_utterance()
        return ready

    def _consume(self, block: np.ndarray) -> np.ndarray | None:
        probability = self._speech_probability(block)
        if probability >= self.threshold:
            self._silence_run = 0
            self._speech_run += len(block)
            if self._active:
                self._utterance = np.concatenate([self._utterance, block])
            else:
                # Keep every block of this onset. The old path only kept a
                # rolling pre-roll, so the first part of a slow start was lost
                # once speech was finally confirmed.
                self._candidate = np.concatenate([self._candidate, block])
                if self._speech_run >= self.min_speech:
                    self._active = True
                    self._utterance = np.concatenate([self._pre_roll, self._candidate])
                    self._candidate = np.zeros(0, dtype=np.float32)
        else:
            if len(self._candidate):
                self._remember_preroll(self._candidate)
                self._candidate = np.zeros(0, dtype=np.float32)
            self._speech_run = 0
            self._remember_preroll(block)
            if self._active:
                self._silence_run += len(block)
                self._utterance = np.concatenate([self._utterance, block])
                if self._silence_run >= self.min_silence:
                    finished = self._utterance.copy()
                    self._reset_utterance()
                    return finished
        if self._active and len(self._utterance) >= self.max_samples:
            finished = self._utterance[: self.max_samples].copy()
            self._reset_utterance()
            return finished
        return None

    def _reset_utterance(self) -> None:
        self._utterance = np.zeros(0, dtype=np.float32)
        self._candidate = np.zeros(0, dtype=np.float32)
        self._active = False
        self._speech_run = 0
        self._silence_run = 0
        if self._model is not None:
            self._model.reset_states()
