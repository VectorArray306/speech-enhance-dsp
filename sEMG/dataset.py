"""PyTorch dataset: filtered EMG -> HiFi-GAN target mel."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from align import emg_to_mel_frames
from emg_preprocess import preprocess_emg
from gaddy_io import UtteranceRef, discover_utterances, train_val_split
from mel_from_audio import audio_path_to_mel, load_hifigan_config


class EMGNormStats:
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (self.std + 1e-6)

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> EMGNormStats:
        return cls(np.array(d["mean"]), np.array(d["std"]))

    def save(self, path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: Path) -> EMGNormStats:
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


def compute_emg_norm_stats(
    refs: list[UtteranceRef],
    h: dict,
    emg_sr: float,
    notch_hz: float,
    max_utterances: int = 200,
) -> EMGNormStats:
    all_vecs = []
    for ref in refs[:max_utterances]:
        raw = np.load(ref.emg_path)
        filt = preprocess_emg(raw, emg_sr, notch_hz=notch_hz)
        mel = audio_path_to_mel(ref.audio_path, h)
        aligned = emg_to_mel_frames(
            filt,
            mel.shape[-1],
            emg_sr,
            h["sampling_rate"],
            h["hop_size"],
        )
        all_vecs.append(aligned.reshape(-1, aligned.shape[-1]))
    stacked = np.concatenate(all_vecs, axis=0)
    return EMGNormStats(stacked.mean(axis=0), stacked.std(axis=0))


class GaddyEMGMelDataset(Dataset):
    def __init__(
        self,
        refs: list[UtteranceRef],
        h: dict,
        emg_norm: EMGNormStats | None = None,
        segment_frames: int = 128,
        emg_sample_rate: float = 1000.0,
        notch_hz: float = 50.0,
        train: bool = True,
    ):
        self.refs = refs
        self.h = h
        self.emg_norm = emg_norm
        self.segment_frames = segment_frames
        self.emg_sr = emg_sample_rate
        self.notch_hz = notch_hz
        self.train = train
        self._cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def _load_pair(self, ref: UtteranceRef) -> tuple[np.ndarray, np.ndarray]:
        key = hash((str(ref.directory), ref.index))
        if key in self._cache:
            return self._cache[key]

        raw = np.load(ref.emg_path)
        filt = preprocess_emg(raw, self.emg_sr, notch_hz=self.notch_hz)
        mel_t = audio_path_to_mel(ref.audio_path, self.h)
        mel = mel_t.cpu().numpy()
        aligned = emg_to_mel_frames(
            filt,
            mel.shape[-1],
            self.emg_sr,
            self.h["sampling_rate"],
            self.h["hop_size"],
        )
        if self.emg_norm is not None:
            aligned = self.emg_norm.normalize(aligned)
        self._cache[key] = (aligned, mel)
        return aligned, mel

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ref = self.refs[idx]
        emg, mel = self._load_pair(ref)
        n_frames = mel.shape[-1]
        seg = self.segment_frames
        if n_frames <= seg:
            pad = seg - n_frames
            emg = np.pad(emg, ((0, pad), (0, 0)), mode="constant")
            mel = np.pad(mel, ((0, 0), (0, pad)), mode="constant")
            start = 0
        elif self.train:
            start = np.random.randint(0, n_frames - seg)
            emg = emg[start : start + seg]
            mel = mel[:, start : start + seg]
        else:
            start = 0
            emg = emg[:seg]
            mel = mel[:, :seg]

        emg_t = torch.from_numpy(emg.T.copy()).float()
        mel_t = torch.from_numpy(mel.copy()).float()
        return {"emg": emg_t, "mel": mel_t, "ref": ref.text}


def build_datasets(
    emg_data_root: Path,
    segment_frames: int = 128,
    val_fraction: float = 0.1,
    emg_sample_rate: float = 1000.0,
    notch_hz: float = 50.0,
    norm_stats_path: Path | None = None,
    norm_stats_load: Path | None = None,
    recompute_norm: bool = False,
    hifi_config: Path | None = None,
) -> tuple[GaddyEMGMelDataset, GaddyEMGMelDataset, EMGNormStats, dict]:
    h = load_hifigan_config(hifi_config)
    refs = discover_utterances(emg_data_root)
    if not refs:
        raise FileNotFoundError(
            f"No Gaddy utterances under {emg_data_root}. "
            "Expected */<index>_emg.npy and *_info.json with sentence_index >= 0."
        )
    train_refs, val_refs = train_val_split(refs, val_fraction=val_fraction)
    load_path = norm_stats_load or norm_stats_path
    if not recompute_norm and load_path and load_path.is_file():
        emg_norm = EMGNormStats.load(load_path)
        if norm_stats_path and norm_stats_path != load_path:
            norm_stats_path.parent.mkdir(parents=True, exist_ok=True)
            if not norm_stats_path.is_file():
                emg_norm.save(norm_stats_path)
    else:
        emg_norm = compute_emg_norm_stats(train_refs, h, emg_sample_rate, notch_hz)
        if norm_stats_path:
            norm_stats_path.parent.mkdir(parents=True, exist_ok=True)
            emg_norm.save(norm_stats_path)
    train_ds = GaddyEMGMelDataset(
        train_refs, h, emg_norm, segment_frames, emg_sample_rate, notch_hz, train=True
    )
    val_ds = GaddyEMGMelDataset(
        val_refs, h, emg_norm, segment_frames, emg_sample_rate, notch_hz, train=False
    )
    return train_ds, val_ds, emg_norm, h
