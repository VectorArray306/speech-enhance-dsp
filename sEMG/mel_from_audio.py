"""HiFi-GAN-compatible log-mel from waveform."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

_HIFI_ROOT = Path(__file__).resolve().parents[1] / "hifi-gan"
if str(_HIFI_ROOT) not in sys.path:
    sys.path.insert(0, str(_HIFI_ROOT))

from meldataset import mel_spectrogram  # noqa: E402


def load_hifigan_config(config_path: Path | None = None) -> dict:
    if config_path is None:
        config_path = _HIFI_ROOT / "config_v1.json"
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


def load_audio_for_mel(
    path: Path,
    target_sr: int,
    prefer_clean: bool = True,
) -> np.ndarray:
    """Load FLAC/WAV, mono float32 in [-1, 1] at ``target_sr``."""
    if prefer_clean and path.name.endswith("_audio.flac"):
        clean = path.parent / path.name.replace("_audio.flac", "_audio_clean.flac")
        if clean.is_file():
            path = clean
    audio, sr = sf.read(str(path), always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    peak = np.max(np.abs(audio)) + 1e-8
    audio = (audio / peak) * 0.95
    return audio


def waveform_to_mel(waveform: np.ndarray, h: dict, device: torch.device | None = None) -> torch.Tensor:
    """Return log-mel ``(num_mels, time_frames)``."""
    if device is None:
        device = torch.device("cpu")
    wav = torch.from_numpy(waveform).float().to(device)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    mel = mel_spectrogram(
        wav,
        h["n_fft"],
        h["num_mels"],
        h["sampling_rate"],
        h["hop_size"],
        h["win_size"],
        h["fmin"],
        h["fmax"],
        center=False,
    )
    return mel.squeeze(0)


def audio_path_to_mel(path: Path, h: dict, device: torch.device | None = None) -> torch.Tensor:
    wav = load_audio_for_mel(path, h["sampling_rate"])
    return waveform_to_mel(wav, h, device)
