"""Warp a TTS log-mel onto the dysarthric utterance's frame grid."""

from __future__ import annotations

import numpy as np


def _zscore_utterance(mel: np.ndarray) -> np.ndarray:
    mean = float(mel.mean())
    std = float(mel.std()) + 1e-6
    return ((mel - mean) / std).astype(np.float32)


def align_source_to_target(
    source: np.ndarray,
    target: np.ndarray,
    diagonal_penalty: float = 0.0,
) -> np.ndarray:
    """Warp ``source`` (n_mels, Ts) onto the time axis of ``target`` (n_mels, Td)."""
    import librosa
    from scipy.spatial.distance import cdist

    if source.ndim != 2 or target.ndim != 2:
        raise ValueError("mels must be (n_mels, time)")
    if target.shape[1] <= 0 or source.shape[1] <= 0:
        raise ValueError("mel time dimension must be positive")
    src_norm = _zscore_utterance(source)
    tgt_norm = _zscore_utterance(target)
    cost = cdist(tgt_norm.T, src_norm.T, metric="euclidean").astype(np.float32)
    if diagonal_penalty > 0.0:
        ti = np.linspace(0.0, 1.0, target.shape[1], dtype=np.float32)[:, None]
        si = np.linspace(0.0, 1.0, source.shape[1], dtype=np.float32)[None, :]
        cost = cost + float(diagonal_penalty) * np.abs(ti - si) * float(np.median(cost))
    _, path = librosa.sequence.dtw(C=cost)
    return _apply_path(source, path, target.shape[1])


def _apply_path(source: np.ndarray, path: np.ndarray, target_frames: int) -> np.ndarray:
    """``path`` rows are (target_index, source_index)."""
    buckets: list[list[int]] = [[] for _ in range(target_frames)]
    for row in path:
        target_index = int(row[0])
        source_index = int(row[1])
        if 0 <= target_index < target_frames and 0 <= source_index < source.shape[1]:
            buckets[target_index].append(source_index)
    warped = np.zeros((source.shape[0], target_frames), dtype=np.float32)
    filled = -1
    for index, choices in enumerate(buckets):
        if choices:
            warped[:, index] = source[:, choices].mean(axis=1)
            filled = index
        elif filled >= 0:
            warped[:, index] = warped[:, filled]
    if filled < 0:
        column = source.mean(axis=1, keepdims=True)
        return np.repeat(column, target_frames, axis=1).astype(np.float32)
    first = next(index for index, choices in enumerate(buckets) if choices)
    if first > 0:
        warped[:, :first] = warped[:, first : first + 1]
    return warped
