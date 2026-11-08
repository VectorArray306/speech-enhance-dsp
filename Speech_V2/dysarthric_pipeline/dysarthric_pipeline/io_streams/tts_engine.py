"""Turn recognized text into a clear preset voice with Piper."""

from __future__ import annotations

import urllib.request
import wave
from pathlib import Path

import numpy as np

from dysarthric_pipeline.config import PACKAGE_ROOT

_VOICE_FILES = {
    "en_US-lessac-medium": (
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
    )
}


class PresetVoice:
    """Piper voice. The first call downloads the voice files if they are missing."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.voice_name = config["tts"]["voice"]
        self.sample_rate = int(config["tts"]["sample_rate"])
        self.voice_dir = PACKAGE_ROOT.parent / "voices"
        self._voice = None

    def model_path(self) -> Path:
        return self.voice_dir / f"{self.voice_name}.onnx"

    def ensure_voice(self) -> Path:
        model = self.model_path()
        config_path = model.with_suffix(".onnx.json")
        if model.is_file() and config_path.is_file():
            return model
        urls = _VOICE_FILES.get(self.voice_name)
        if urls is None:
            raise FileNotFoundError(
                f"Voice {self.voice_name} is not downloaded. Place {model.name} and its .onnx.json in {self.voice_dir}."
            )
        self.voice_dir.mkdir(parents=True, exist_ok=True)
        for url, destination in ((urls[0], model), (urls[1], config_path)):
            if destination.is_file():
                continue
            print(f"downloading {destination.name}", flush=True)
            urllib.request.urlretrieve(url, destination)
        return model

    def _load(self):
        if self._voice is None:
            from piper import PiperVoice

            self._voice = PiperVoice.load(self.ensure_voice())
            self.sample_rate = int(self._voice.config.sample_rate)
        return self._voice

    def synthesize(self, text: str) -> np.ndarray:
        cleaned = " ".join(text.split())
        if not cleaned:
            return np.zeros(0, dtype=np.float32)
        voice = self._load()
        chunks = [np.asarray(piece.audio_float_array, dtype=np.float32) for piece in voice.synthesize(cleaned)]
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks)

    def write_wav(self, text: str, path: str | Path) -> Path:
        audio = self.synthesize(text)
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        pcm = np.clip(audio, -1.0, 1.0)
        with wave.open(str(destination), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(self.sample_rate)
            handle.writeframes((pcm * 32767.0).astype("<i2").tobytes())
        return destination

    def play(self, text: str) -> None:
        audio = self.synthesize(text)
        if len(audio) == 0:
            return
        import sounddevice as sd

        sd.play(audio, self.sample_rate)
        sd.wait()
