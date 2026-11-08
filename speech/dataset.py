"""Load cached mel pairs for CNN-BiLSTM training."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


def active_mask(dys: np.ndarray) -> np.ndarray:
    """True on frames above the utterance noise floor."""
    energy = dys.mean(axis=0)
    floor = float(np.percentile(energy, 20))
    peak = float(np.percentile(energy, 90))
    if peak - floor < 0.5:
        active = np.ones(dys.shape[1], dtype=bool)
    else:
        active = energy >= floor + 0.35 * (peak - floor)
    if int(active.sum()) < 8:
        active[:] = True
    return active


def articulation_target(dys: np.ndarray, aligned: np.ndarray) -> np.ndarray:
    """Ask for the aligned clear mel's shape, on this speaker's level, and keep their silence.

    Each mel bin of ``aligned`` is shifted so its active-frame mean matches ``dys``.
    Frames where ``dys`` is near its own noise floor are copied through unchanged.
    """
    frames = min(dys.shape[1], aligned.shape[1])
    dys = dys[:, :frames]
    aligned = aligned[:, :frames]
    active = active_mask(dys)
    matched = np.empty_like(aligned)
    for bin_index in range(dys.shape[0]):
        source = aligned[bin_index]
        active_source = source[active]
        active_dys = dys[bin_index, active]
        matched[bin_index] = source - float(active_source.mean()) + float(active_dys.mean())
    matched[:, ~active] = dys[:, ~active]
    return matched.astype(np.float32)


def spectral_tilt(mel: np.ndarray) -> float:
    """Low-band minus high-band level on active frames. Higher means a steeper spectrum."""
    speech = mel[:, active_mask(mel)]
    if speech.shape[1] == 0:
        speech = mel
    return float(speech[:20].mean() - speech[40:].mean())


def severity_blend_target(
    dys: np.ndarray,
    aligned: np.ndarray,
    low: float = 2.15,
    high: float = 2.64,
) -> np.ndarray:
    """Blend the input toward the aligned clear mel in proportion to spectral tilt.

    ``low`` and ``high`` are the training-speaker medians for mild ``torgo_F03`` and
    severe ``torgo_M02``. Tilt at or below ``low`` keeps the input. Tilt at or above
    ``high`` uses the clear mel on active frames. Silence always stays the input.
    """
    frames = min(dys.shape[1], aligned.shape[1])
    dys = dys[:, :frames]
    aligned = aligned[:, :frames]
    span = max(float(high) - float(low), 1e-6)
    alpha = float(np.clip((spectral_tilt(dys) - float(low)) / span, 0.0, 1.0))
    active = active_mask(dys)
    target = np.array(dys, dtype=np.float32, copy=True)
    target[:, active] = (1.0 - alpha) * dys[:, active] + alpha * aligned[:, active]
    return target


def active_control_target(dys: np.ndarray, aligned: np.ndarray) -> np.ndarray:
    """Keep the aligned clear mel on voiced frames and the input on silence."""
    frames = min(dys.shape[1], aligned.shape[1])
    dys = dys[:, :frames]
    target = np.array(aligned[:, :frames], dtype=np.float32, copy=True)
    silence = ~active_mask(dys)
    target[:, silence] = dys[:, silence]
    return target


def load_split(manifest_dir: Path, split: str, corpus_filter: str = "all") -> list[dict]:
    path = manifest_dir / f"{split}.jsonl"
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    if corpus_filter != "all":
        allowed = {item.strip() for item in corpus_filter.split(",") if item.strip()}
        rows = [row for row in rows if row.get("corpus") in allowed]
    return rows


class MelPairDataset(Dataset):
    def __init__(
        self,
        rows: list[dict],
        cache_dir: Path,
        target_mode: str = "cached",
        severity_low: float = 2.15,
        severity_high: float = 2.64,
    ):
        if target_mode not in ("cached", "articulation", "active_control", "severity_blend"):
            raise ValueError(
                "target_mode must be 'cached', 'articulation', 'active_control', or 'severity_blend'"
            )
        self.rows = rows
        self.inputs = cache_dir / "inputs"
        self.targets = cache_dir / "targets"
        self.target_mode = target_mode
        self.severity_low = severity_low
        self.severity_high = severity_high

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        dys = np.load(self.inputs / f"{row['id']}.npy")
        target = np.load(self.targets / f"{row['id']}.npy")
        frames = min(dys.shape[1], target.shape[1])
        dys = dys[:, :frames]
        target = target[:, :frames]
        if self.target_mode == "articulation":
            target = articulation_target(dys, target)
        elif self.target_mode == "active_control":
            target = active_control_target(dys, target)
        elif self.target_mode == "severity_blend":
            target = severity_blend_target(dys, target, self.severity_low, self.severity_high)
        return {
            "input": torch.from_numpy(np.ascontiguousarray(dys)),
            "target": torch.from_numpy(np.ascontiguousarray(target)),
            "length": frames,
            "id": row["id"],
            "text": row["text"],
        }


def collate_mels(batch: list[dict]) -> dict:
    lengths = torch.tensor([item["length"] for item in batch], dtype=torch.long)
    inputs = pad_sequence([item["input"].transpose(0, 1) for item in batch], batch_first=True)
    targets = pad_sequence([item["target"].transpose(0, 1) for item in batch], batch_first=True)
    return {
        "input": inputs.transpose(1, 2),
        "target": targets.transpose(1, 2),
        "lengths": lengths,
        "ids": [item["id"] for item in batch],
        "texts": [item["text"] for item in batch],
    }
