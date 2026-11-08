"""Load frozen HiFi-GAN generator and vocode log-mel."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.io.wavfile import write

_HIFI_ROOT = Path(__file__).resolve().parents[1] / "hifi-gan"
if str(_HIFI_ROOT) not in sys.path:
    sys.path.insert(0, str(_HIFI_ROOT))

from env import AttrDict  # noqa: E402
from models import Generator  # noqa: E402


def load_generator(checkpoint_path: Path, device: torch.device) -> tuple[Generator, AttrDict]:
    checkpoint_path = Path(checkpoint_path)
    config_path = checkpoint_path.parent / "config.json"
    if not config_path.is_file():
        config_path = _HIFI_ROOT / "config_v1.json"
    with open(config_path, encoding="utf-8") as f:
        h = AttrDict(json.load(f))
    generator = Generator(h).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    generator.load_state_dict(state["generator"])
    generator.eval()
    generator.remove_weight_norm()
    return generator, h


def mel_to_waveform(
    mel: torch.Tensor,
    generator: Generator,
) -> np.ndarray:
    """
    Args:
        mel: ``(1, n_mels, time)`` or ``(n_mels, time)`` log-mel.
    Returns:
        float waveform in [-1, 1] approximate.
    """
    if mel.dim() == 2:
        mel = mel.unsqueeze(0)
    with torch.no_grad():
        audio = generator(mel).squeeze()
    return audio.cpu().numpy()


def save_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767.0).astype(np.int16)
    write(str(path), sample_rate, pcm)
