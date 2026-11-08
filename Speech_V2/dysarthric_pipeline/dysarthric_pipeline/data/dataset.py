"""Manifest dataset with speed perturbation and SpecAugment."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio


def spec_augment(
    mel: torch.Tensor,
    freq_masks: int = 2,
    time_masks: int = 2,
    freq_width: int = 15,
    time_width: int = 30,
) -> torch.Tensor:
    """Zero out random frequency and time bands of a ``[mel, time]`` spectrogram."""
    out = mel.clone()
    n_freq, n_time = out.shape
    for _ in range(freq_masks):
        width = int(torch.randint(0, min(freq_width, n_freq) + 1, ()).item()) if n_freq else 0
        if width <= 0 or width >= n_freq:
            continue
        start = int(torch.randint(0, n_freq - width + 1, ()).item())
        out[start : start + width, :] = 0
    for _ in range(time_masks):
        width = int(torch.randint(0, min(time_width, n_time) + 1, ()).item()) if n_time else 0
        if width <= 0 or width >= n_time:
            continue
        start = int(torch.randint(0, n_time - width + 1, ()).item())
        out[:, start : start + width] = 0
    return out


def normalize_loudness(audio: torch.Tensor, target_dbfs: float, peak: float = 0.95) -> torch.Tensor:
    """Scale a waveform to a target RMS, then cap the peak."""
    if audio.numel() == 0:
        return audio
    rms = torch.sqrt(torch.mean(audio.float().pow(2)) + 1e-8)
    target = 10 ** (float(target_dbfs) / 20.0)
    scaled = audio * (target / rms)
    max_abs = float(scaled.abs().max())
    limit = float(peak)
    if max_abs > limit > 0:
        scaled = scaled * (limit / max_abs)
    return scaled


def trim_silence(audio: torch.Tensor, sample_rate: int, top_db: float, pad_ms: int) -> torch.Tensor:
    """Keep the energetic region of a waveform, plus a short pad on each side."""
    if top_db <= 0 or audio.numel() < sample_rate // 10:
        return audio
    frame = max(1, int(round(sample_rate * 0.02)))
    hop = max(1, int(round(sample_rate * 0.01)))
    if audio.numel() <= frame:
        return audio
    power = audio.unfold(0, frame, hop).pow(2).mean(dim=1)
    peak = float(power.max())
    if peak < 1e-8:
        return audio
    threshold = peak * (10 ** (-float(top_db) / 10.0))
    active = torch.nonzero(power >= threshold, as_tuple=False).flatten()
    if active.numel() == 0:
        return audio
    start = int(active[0].item()) * hop
    end = int(active[-1].item()) * hop + frame
    pad = int(round(sample_rate * pad_ms / 1000))
    start = max(0, start - pad)
    end = min(int(audio.numel()), end + pad)
    trimmed = audio[start:end]
    return trimmed if trimmed.numel() else audio


def _speed_perturb(audio: torch.Tensor, sample_rate: int, factors: tuple[float, ...]) -> torch.Tensor:
    factor = factors[int(torch.randint(0, len(factors), ()).item())]
    if abs(factor - 1.0) < 1e-6:
        return audio
    changed = torchaudio.functional.resample(audio, int(round(sample_rate * factor)), sample_rate)
    return changed


class SpeechManifestDataset(torch.utils.data.Dataset):
    """One manifest row is one utterance.

    Training picks a single microphone from ``audio_paths`` and applies speed
    perturbation plus SpecAugment on the log-mel. Evaluation uses ``audio_path``.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        train: bool = False,
        sample_rate: int = 16000,
        speed_factors: tuple[float, ...] = (0.9, 1.0, 1.1),
        apply_specaugment: bool = True,
        compute_mel: bool = True,
        seed: int = 17,
        trim_db: float = 0.0,
        trim_pad_ms: int = 200,
        normalize_dbfs: float | None = None,
        normalize_peak: float = 0.95,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.train = train
        self.sample_rate = sample_rate
        self.speed_factors = speed_factors
        self.apply_specaugment = apply_specaugment
        self.compute_mel = compute_mel
        self.trim_db = float(trim_db)
        self.trim_pad_ms = int(trim_pad_ms)
        self.normalize_dbfs = None if normalize_dbfs is None else float(normalize_dbfs)
        self.normalize_peak = float(normalize_peak)
        self.random = random.Random(seed)
        self.rows = [
            json.loads(line)
            for line in self.manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.mel = (
            torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=400,
                hop_length=160,
                n_mels=80,
            )
            if compute_mel
            else None
        )

    def __len__(self) -> int:
        return len(self.rows)

    def _read(self, path: str) -> torch.Tensor:
        audio, file_rate = sf.read(path, dtype="float32", always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        waveform = torch.from_numpy(np.ascontiguousarray(audio))
        if file_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, file_rate, self.sample_rate)
        return waveform

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        if self.train:
            choices = row.get("audio_paths") or [row["audio_path"]]
            path = self.random.choice(choices)
        else:
            path = row["audio_path"]
        audio = self._read(path)
        if self.trim_db > 0:
            audio = trim_silence(audio, self.sample_rate, self.trim_db, self.trim_pad_ms)
        if self.normalize_dbfs is not None:
            audio = normalize_loudness(audio, self.normalize_dbfs, self.normalize_peak)
        max_samples = self.sample_rate * 30
        if audio.numel() > max_samples:
            audio = audio[:max_samples]
        if self.train:
            audio = _speed_perturb(audio, self.sample_rate, self.speed_factors)
        item = {
            "audio": audio,
            "text": row["text"],
            "speaker": row["speaker"],
            "corpus": row.get("corpus", ""),
            "condition": row.get("condition", ""),
            "severity": row.get("severity", ""),
            "sample_rate": self.sample_rate,
            "utterance_id": row["utterance_id"],
            "audio_path": path,
        }
        if self.mel is not None:
            mel = torch.log(self.mel(audio).clamp(min=1e-5))
            if self.train and self.apply_specaugment:
                mel = spec_augment(mel)
            item["mel"] = mel
        return item
