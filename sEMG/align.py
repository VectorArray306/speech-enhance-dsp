"""Time-align filtered EMG samples to HiFi-GAN mel frames."""

from __future__ import annotations

import numpy as np


def emg_to_mel_frames(
    emg: np.ndarray,
    n_mel_frames: int,
    emg_sample_rate: float,
    audio_sample_rate: int,
    hop_size: int,
) -> np.ndarray:
    """
    Resample EMG along time to one row per mel frame.

    Returns ``(n_mel_frames, n_channels)``.
    """
    if emg.ndim != 2:
        raise ValueError(f"expected EMG (time, channels), got {emg.shape}")
    emg_times = np.arange(emg.shape[0], dtype=np.float64) / emg_sample_rate
    mel_times = np.arange(n_mel_frames, dtype=np.float64) * hop_size / audio_sample_rate
    mel_times = np.clip(mel_times, emg_times[0], emg_times[-1])
    aligned = np.zeros((n_mel_frames, emg.shape[1]), dtype=np.float32)
    for ch in range(emg.shape[1]):
        aligned[:, ch] = np.interp(mel_times, emg_times, emg[:, ch])
    return aligned
