#!/usr/bin/env python3
"""Validate EMG->mel model: L1 on val set + vocode with frozen HiFi-GAN."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from align import emg_to_mel_frames
from dataset import EMGNormStats, build_datasets
from emg_preprocess import preprocess_emg
from gaddy_io import discover_utterances, train_val_split
from hifigan_vocode import load_generator, mel_to_waveform, save_wav
from mel_from_audio import audio_path_to_mel, load_audio_for_mel, load_hifigan_config, waveform_to_mel
from model import EMGToMelCNN


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--emg_data_dir", type=Path, required=True)
    p.add_argument("--emg_checkpoint", type=Path, required=True)
    p.add_argument("--hifigan_checkpoint", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, default=Path("sEMG/validation_out"))
    p.add_argument("--max_utterances", type=int, default=5)
    p.add_argument(
        "--min_audio_peak",
        type=float,
        default=0.05,
        help="Skip utterances whose raw audio peak is below this. "
        "Silent-speech sessions are near-zero and sound like noise after gain.",
    )
    p.add_argument("--emg_sample_rate", type=float, default=1000.0)
    p.add_argument("--notch_hz", type=float, default=50.0)
    p.add_argument("--norm_stats", type=Path, default=None, help="emg_norm.json from training.")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def raw_audio_peak(path: Path) -> float:
    audio, _sr = sf.read(str(path), always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if audio.size == 0:
        return 0.0
    return float(np.max(np.abs(audio)))


def select_audible(refs, max_utterances: int, min_audio_peak: float):
    """Keep utterances that contain acoustic speech, not mouthed silence."""
    chosen = []
    skipped = 0
    for ref in refs:
        if raw_audio_peak(ref.audio_path) < min_audio_peak:
            skipped += 1
            continue
        chosen.append(ref)
        if len(chosen) >= max_utterances:
            break
    return chosen, skipped


def load_emg_model(path: Path, device: torch.device) -> EMGToMelCNN:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = EMGToMelCNN(
        n_emg_channels=ckpt["n_emg_channels"],
        n_mels=ckpt["n_mels"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


@torch.no_grad()
def predict_mel_for_ref(
    model: EMGToMelCNN,
    ref,
    emg_norm: EMGNormStats,
    h: dict,
    emg_sr: float,
    notch_hz: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    raw = np.load(ref.emg_path)
    filt = preprocess_emg(raw, emg_sr, notch_hz=notch_hz)
    target_mel = audio_path_to_mel(ref.audio_path, h, device)
    n_frames = target_mel.shape[-1]
    aligned = emg_to_mel_frames(
        filt, n_frames, emg_sr, h["sampling_rate"], h["hop_size"]
    )
    aligned = emg_norm.normalize(aligned)
    emg_t = torch.from_numpy(aligned.T.copy()).float().unsqueeze(0).to(device)
    pred = model(emg_t)
    return pred.squeeze(0), target_mel


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    h = load_hifigan_config()
    norm_path = args.norm_stats or args.emg_checkpoint.parent / "emg_norm.json"
    emg_norm = EMGNormStats.load(norm_path)

    refs = discover_utterances(args.emg_data_dir)
    _, val_refs = train_val_split(refs)
    val_refs, n_silent = select_audible(val_refs, args.max_utterances, args.min_audio_peak)
    if not val_refs:
        raise SystemExit(
            "No validation utterances with audible audio. "
            f"Lower --min_audio_peak (now {args.min_audio_peak})."
        )
    print(
        f"Listening set: {len(val_refs)} audible utterances "
        f"(skipped {n_silent} near-silent silent-speech recordings)"
    )

    emg_model = load_emg_model(args.emg_checkpoint, device)
    generator, h_g = load_generator(args.hifigan_checkpoint, device)

    l1_total = 0.0
    for i, ref in enumerate(val_refs):
        pred_mel, target_mel = predict_mel_for_ref(
            emg_model, ref, emg_norm, h, args.emg_sample_rate, args.notch_hz, device
        )
        l1 = F.l1_loss(pred_mel, target_mel).item()
        l1_total += l1
        print(
            f"{i}: peak={raw_audio_peak(ref.audio_path):.3f}  "
            f"{ref.directory.parent.name}/{ref.directory.name}  "
            f"{ref.text[:60]!r}  mel_l1={l1:.4f}"
        )

        pred_wav = mel_to_waveform(pred_mel.unsqueeze(0), generator)
        target_wav = mel_to_waveform(target_mel.unsqueeze(0), generator)
        save_wav(args.output_dir / f"{i:02d}_emg_pred.wav", pred_wav, h_g.sampling_rate)
        save_wav(args.output_dir / f"{i:02d}_mel_target.wav", target_wav, h_g.sampling_rate)

        orig = load_audio_for_mel(ref.audio_path, h_g.sampling_rate)
        save_wav(
            args.output_dir / f"{i:02d}_original.wav",
            orig,
            h_g.sampling_rate,
        )
        orig_mel = waveform_to_mel(orig, h, device)
        save_wav(
            args.output_dir / f"{i:02d}_hifigan_roundtrip.wav",
            mel_to_waveform(orig_mel.unsqueeze(0), generator),
            h_g.sampling_rate,
        )

    print(f"mean mel L1 on {len(val_refs)} utterances: {l1_total / max(len(val_refs), 1):.4f}")
    print(f"Wrote wavs to {args.output_dir}")


if __name__ == "__main__":
    main()
