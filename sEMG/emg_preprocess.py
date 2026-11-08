"""EMG filtering to mirror planned MCU DSP (bandpass + notch)."""

from __future__ import annotations

import numpy as np
from scipy import signal


def bandpass_emg(
    x: np.ndarray,
    sample_rate: float,
    low_hz: float = 20.0,
    high_hz: float = 450.0,
    order: int = 4,
) -> np.ndarray:
    """Filter each column of ``x`` (time, channels)."""
    nyq = sample_rate * 0.5
    low = low_hz / nyq
    high = min(high_hz / nyq, 0.99)
    b, a = signal.butter(order, [low, high], btype="band")
    out = np.zeros_like(x, dtype=np.float64)
    for ch in range(x.shape[1]):
        out[:, ch] = signal.filtfilt(b, a, x[:, ch])
    return out


def notch_emg(
    x: np.ndarray,
    sample_rate: float,
    freq_hz: float = 50.0,
    q: float = 30.0,
) -> np.ndarray:
    b, a = signal.iirnotch(freq_hz, q, sample_rate)
    out = np.zeros_like(x, dtype=np.float64)
    for ch in range(x.shape[1]):
        out[:, ch] = signal.filtfilt(b, a, x[:, ch])
    return out


def preprocess_emg(
    raw: np.ndarray,
    sample_rate: float = 1000.0,
    notch_hz: float = 50.0,
) -> np.ndarray:
    """DC removal via high-pass drift filter, bandpass, mains notch."""
    x = raw.astype(np.float64)
    b, a = signal.butter(3, 2.0, btype="highpass", fs=sample_rate)
    for ch in range(x.shape[1]):
        x[:, ch] = signal.filtfilt(b, a, x[:, ch])
    x = bandpass_emg(x, sample_rate)
    x = notch_emg(x, sample_rate, freq_hz=notch_hz)
    return x.astype(np.float32)
