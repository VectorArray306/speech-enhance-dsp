"""Shared waveform front-end for training clips and a finished mic buffer."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt


TARGET_SR = 22050
HIGHPASS_HZ = 80.0


def highpass(audio: np.ndarray, sample_rate: int, cutoff_hz: float = HIGHPASS_HZ) -> np.ndarray:
    if audio.size < int(sample_rate * 0.05) or cutoff_hz <= 0:
        return audio
    sos = butter(4, cutoff_hz, btype="highpass", fs=sample_rate, output="sos")
    return sosfiltfilt(sos, audio).astype(np.float32)


def preprocess_waveform(audio: np.ndarray, sample_rate: int, target_sr: int = TARGET_SR) -> np.ndarray:
    """Mono, high-pass, resample to 22.05 kHz, peak-normalize."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = highpass(audio, sample_rate)
    if sample_rate != target_sr:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sr).astype(np.float32)
    peak = float(np.max(np.abs(audio))) + 1e-8
    return (audio / peak) * 0.95


def load_preprocessed(path: Path, target_sr: int = TARGET_SR) -> np.ndarray:
    audio, sample_rate = sf.read(str(path), always_2d=False)
    return preprocess_waveform(audio, int(sample_rate), target_sr)
