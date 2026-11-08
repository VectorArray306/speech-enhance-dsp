#!/usr/bin/env python3
"""Train 1D U-Net EMG -> log-mel (Phase 2)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import build_datasets
from mel_losses import MelLossWeights, mel_training_loss
from model import EMGToMelCNN


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train EMG-to-mel 1D U-Net on Gaddy & Klein.")
    p.add_argument(
        "--emg_data_dir",
        type=Path,
        required=True,
        help="Root of Gaddy data (e.g. emg_data or a session folder).",
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        default=Path("sEMG/checkpoints"),
        help="Checkpoints and logs.",
    )
    p.add_argument("--segment_frames", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--emg_sample_rate", type=float, default=1000.0)
    p.add_argument("--notch_hz", type=float, default=50.0, help="Use 60 for US mains.")
    p.add_argument(
        "--hifi_config",
        type=Path,
        default=None,
        help="HiFi-GAN config (default: ../hifi-gan/config_v1.json).",
    )
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_workers", type=int, default=0 if os.name == "nt" else 2)
    p.add_argument("--w_log_l1", type=float, default=1.0, help="Weight for log-mel L1.")
    p.add_argument("--w_ssim", type=float, default=0.2, help="Weight for 1-SSIM on mel map.")
    p.add_argument(
        "--w_multi_scale",
        type=float,
        default=0.5,
        help="Weight for multi-scale log-mel L1 (temporal pooling).",
    )
    p.add_argument(
        "--init_checkpoint",
        type=Path,
        default=None,
        help="Fine-tune: load CNN weights from a prior run (e.g. voiced_parallel best.pt).",
    )
    p.add_argument(
        "--norm_stats",
        type=Path,
        default=None,
        help="EMG norm JSON to load (default: beside --init_checkpoint, else output_dir).",
    )
    p.add_argument(
        "--recompute_emg_norm",
        action="store_true",
        help="Recompute emg_norm on this dataset instead of reusing pretrain stats.",
    )
    return p.parse_args()


def load_model_from_checkpoint(
    path: Path,
    n_emg_channels: int,
    n_mels: int,
    device: torch.device,
) -> EMGToMelCNN:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    ckpt_ch = int(ckpt["n_emg_channels"])
    ckpt_mels = int(ckpt["n_mels"])
    if ckpt_ch != n_emg_channels or ckpt_mels != n_mels:
        raise ValueError(
            f"Checkpoint expects {ckpt_ch} EMG channels and {ckpt_mels} mels, "
            f"but data has {n_emg_channels} channels and config has {n_mels} mels."
        )
    model = EMGToMelCNN(n_emg_channels=ckpt_ch, n_mels=ckpt_mels).to(device)
    model.load_state_dict(ckpt["model"])
    return model


def train_one_epoch(
    model: EMGToMelCNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_weights: MelLossWeights,
) -> dict[str, float]:
    model.train()
    sums = {"total": 0.0, "log_l1": 0.0, "ssim": 0.0, "multi_scale": 0.0}
    n = 0
    for batch in loader:
        emg = batch["emg"].to(device)
        mel = batch["mel"].to(device)
        pred = model(emg)
        loss, parts = mel_training_loss(pred, mel, loss_weights)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        for k in sums:
            sums[k] += parts[k]
        n += 1
    return {k: v / max(n, 1) for k, v in sums.items()}


@torch.no_grad()
def validate(
    model: EMGToMelCNN,
    loader: DataLoader,
    device: torch.device,
    loss_weights: MelLossWeights,
) -> dict[str, float]:
    model.eval()
    sums = {"total": 0.0, "log_l1": 0.0, "ssim": 0.0, "multi_scale": 0.0}
    n = 0
    for batch in loader:
        emg = batch["emg"].to(device)
        mel = batch["mel"].to(device)
        pred = model(emg)
        _loss, parts = mel_training_loss(pred, mel, loss_weights)
        for k in sums:
            sums[k] += parts[k]
        n += 1
    return {k: v / max(n, 1) for k, v in sums.items()}


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    norm_save = args.output_dir / "emg_norm.json"
    norm_load = args.norm_stats
    if norm_load is None and args.init_checkpoint is not None:
        sibling = args.init_checkpoint.parent / "emg_norm.json"
        if sibling.is_file():
            norm_load = sibling

    train_ds, val_ds, emg_norm, h = build_datasets(
        args.emg_data_dir,
        segment_frames=args.segment_frames,
        val_fraction=args.val_fraction,
        emg_sample_rate=args.emg_sample_rate,
        notch_hz=args.notch_hz,
        norm_stats_path=norm_save,
        norm_stats_load=norm_load,
        recompute_norm=args.recompute_emg_norm,
        hifi_config=args.hifi_config,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    n_ch = int(np.load(train_ds.refs[0].emg_path).shape[1])
    n_mels = int(h["num_mels"])
    if args.init_checkpoint is not None:
        model = load_model_from_checkpoint(
            args.init_checkpoint, n_ch, n_mels, device
        )
        print(f"Loaded weights from {args.init_checkpoint}")
    else:
        model = EMGToMelCNN(n_emg_channels=n_ch, n_mels=n_mels).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_weights = MelLossWeights(
        log_l1=args.w_log_l1,
        ssim=args.w_ssim,
        multi_scale=args.w_multi_scale,
    )
    best_val = float("inf")

    meta = {
        "hifi_config": h,
        "emg_data_dir": str(args.emg_data_dir),
        "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
        "emg_sample_rate": args.emg_sample_rate,
        "notch_hz": args.notch_hz,
        "segment_frames": args.segment_frames,
        "loss_weights": loss_weights.as_dict(),
    }
    with open(args.output_dir / "train_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    for epoch in range(1, args.epochs + 1):
        train_m = train_one_epoch(model, train_loader, optimizer, device, loss_weights)
        val_m = validate(model, val_loader, device, loss_weights)
        print(
            f"epoch {epoch:03d}  train total={train_m['total']:.4f}  "
            f"log_l1={train_m['log_l1']:.4f}  ssim={train_m['ssim']:.4f}  "
            f"ms={train_m['multi_scale']:.4f}  |  val total={val_m['total']:.4f}  "
            f"log_l1={val_m['log_l1']:.4f}"
        )
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_loss": val_m,
            "loss_weights": loss_weights.as_dict(),
            "n_emg_channels": model.n_emg_channels,
            "n_mels": h["num_mels"],
        }
        torch.save(ckpt, args.output_dir / "emg_to_mel_last.pt")
        if val_m["total"] < best_val:
            best_val = val_m["total"]
            torch.save(ckpt, args.output_dir / "emg_to_mel_best.pt")
            print(f"  saved best (val total={best_val:.4f})")


if __name__ == "__main__":
    main()
