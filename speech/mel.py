"""HiFi-GAN v1 log-mel used by the dysarthria pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
_HIFI = _REPO / "hifi-gan"
if str(_HIFI) not in sys.path:
    sys.path.insert(0, str(_HIFI))

from meldataset import mel_spectrogram  # noqa: E402


def load_hifigan_config(config_path: Path | None = None) -> dict:
    path = config_path or (_HIFI / "config_v1.json")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def waveform_to_mel(waveform: np.ndarray, h: dict, device: torch.device | None = None) -> np.ndarray:
    """Return log-mel float32 array shaped (n_mels, time)."""
    device = device or torch.device("cpu")
    wav = torch.from_numpy(np.asarray(waveform, dtype=np.float32)).to(device)
    if wav.numel() < h["n_fft"]:
        wav = torch.nn.functional.pad(wav, (0, h["n_fft"] - wav.numel()))
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
    return mel.squeeze(0).detach().cpu().numpy().astype(np.float32)
