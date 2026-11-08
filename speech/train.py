"""Train the CNN-BiLSTM. Not invoked by the data-prep scripts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import MelPairDataset, collate_mels, load_split, spectral_tilt
from model import CnnBiLstm


def filter_by_tilt(rows: list[dict], cache_dir: Path, minimum_tilt: float) -> list[dict]:
    kept = []
    for row in rows:
        mel = np.load(cache_dir / "inputs" / f"{row['id']}.npy")
        if spectral_tilt(mel) >= minimum_tilt:
            kept.append(row)
    return kept


def filter_by_frames(rows: list[dict], cache_dir: Path, max_frames: int) -> list[dict]:
    if max_frames <= 0:
        return rows
    kept = []
    for row in rows:
        frames = int(np.load(cache_dir / "inputs" / f"{row['id']}.npy", mmap_mode="r").shape[1])
        if frames <= max_frames:
            kept.append(row)
    return kept


def _frame_mask(lengths: torch.Tensor, frames: int) -> torch.Tensor:
    return torch.arange(frames, device=lengths.device)[None, :] < lengths[:, None]


def masked_l1(prediction: torch.Tensor, target: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    mask = _frame_mask(lengths, prediction.shape[-1])[:, None, :].expand_as(prediction)
    return (prediction - target).abs()[mask].mean()


def per_item_l1(prediction: torch.Tensor, target: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    mask = _frame_mask(lengths, prediction.shape[-1])[:, None, :].expand_as(prediction)
    denom = mask.sum(dim=(1, 2)).clamp(min=1)
    return ((prediction - target).abs() * mask).sum(dim=(1, 2)) / denom


def run_epoch(model, loader, optimizer, device, gate_weight: float, severity: bool) -> tuple[float, float, float]:
    training = optimizer is not None
    model.train(training)
    total = 0.0
    identity = 0.0
    gate_total = 0.0
    seen = 0
    for batch in loader:
        source = batch["input"].to(device)
        target = batch["target"].to(device)
        lengths = batch["lengths"].to(device)
        prediction = model(source)
        item = per_item_l1(prediction, target, lengths)
        loss = item.mean()
        if training and severity:
            gap = per_item_l1(source, target, lengths).detach()
            weights = gap / gap.mean().clamp(min=1e-6)
            loss = (item * weights).mean()
        gate_mean = prediction.new_zeros(())
        if getattr(model, "gated", False):
            gate = model.last_gate
            time_mask = _frame_mask(lengths, gate.shape[-1])
            gate_mean = gate.squeeze(1)[time_mask].mean()
            if training and gate_weight > 0:
                loss = loss + gate_weight * gate_mean
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        batch_size = source.shape[0]
        total += float(item.detach().mean().item()) * batch_size
        identity += float(masked_l1(prediction.detach(), source, lengths).item()) * batch_size
        gate_total += float(gate_mean.detach().item()) * batch_size
        seen += batch_size
    scale = max(seen, 1)
    return total / scale, identity / scale, gate_total / scale


def main() -> None:
    parser = argparse.ArgumentParser(description="Train dysarthric mel to clear mel.")
    parser.add_argument("--manifest_dir", type=Path, default=Path("speech/manifests"))
    parser.add_argument("--cache_dir", type=Path, default=Path("speech/cache"))
    parser.add_argument("--output_dir", type=Path, default=Path("speech/checkpoints"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--init_checkpoint",
        type=Path,
        default=None,
        help="Fine-tune from a previous mel_mapper checkpoint.",
    )
    parser.add_argument(
        "--residual",
        action="store_true",
        help="Predict a residual log-mel delta added to the input mel (mel + delta).",
    )
    parser.add_argument(
        "--train_corpus",
        default="all",
        help="Comma-separated corpus filter for training rows (e.g. 'torgo' or 'all').",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=1600,
        help="Drop utterances longer than this many mel frames. 0 keeps every row.",
    )
    parser.add_argument(
        "--target_mode",
        choices=("cached", "articulation", "active_control", "severity_blend"),
        default="cached",
        help="cached uses the DTW target. articulation matches its level to the input. active_control keeps the clear mel on voiced frames. severity_blend keeps mild speech and moves steeper spectra toward the clear mel.",
    )
    parser.add_argument("--severity_low", type=float, default=2.15, help="Spectral tilt that stays at the input. torgo_F03 median.")
    parser.add_argument("--severity_high", type=float, default=2.64, help="Spectral tilt that uses the clear mel. torgo_M02 median.")
    parser.add_argument(
        "--min_train_tilt",
        type=float,
        default=0.0,
        help="Drop training utterances below this spectral tilt. 0 keeps every row.",
    )
    parser.add_argument(
        "--gated",
        action="store_true",
        help="Multiply the residual by a learned per-frame gate, initialized nearly closed.",
    )
    parser.add_argument("--gate_weight", type=float, default=0.02, help="Penalty on the mean open gate.")
    parser.add_argument(
        "--gate_warmup",
        type=int,
        default=8,
        help="Epochs with no gate penalty, so the residual can start moving before the gate closes.",
    )
    parser.add_argument(
        "--severity_weight",
        action="store_true",
        help="Up-weight training utterances whose target is far from the input.",
    )
    parser.add_argument(
        "--identity_weight",
        type=float,
        default=0.35,
        help="Checkpoint score is val_l1 + identity_weight * val distance to the input.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_rows = filter_by_frames(
        load_split(args.manifest_dir, "train", corpus_filter=args.train_corpus),
        args.cache_dir,
        args.max_frames,
    )
    if args.min_train_tilt > 0:
        train_rows = filter_by_tilt(train_rows, args.cache_dir, args.min_train_tilt)
    val_rows = filter_by_frames(
        load_split(args.manifest_dir, "validation"),
        args.cache_dir,
        args.max_frames,
    )
    print(
        f"train_rows={len(train_rows)} val_rows={len(val_rows)} residual={args.residual} "
        f"gated={args.gated} target_mode={args.target_mode} max_frames={args.max_frames}",
        flush=True,
    )
    train_loader = DataLoader(
        MelPairDataset(
            train_rows,
            args.cache_dir,
            target_mode=args.target_mode,
            severity_low=args.severity_low,
            severity_high=args.severity_high,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_mels,
    )
    val_loader = DataLoader(
        MelPairDataset(
            val_rows,
            args.cache_dir,
            target_mode=args.target_mode,
            severity_low=args.severity_low,
            severity_high=args.severity_high,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_mels,
    )
    use_residual = bool(args.residual)
    use_gated = bool(args.gated)
    if args.init_checkpoint is not None:
        checkpoint = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        if not use_residual and bool(checkpoint.get("residual", False)):
            use_residual = True
        if not use_gated and bool(checkpoint.get("gated", False)):
            use_gated = True
        model = CnnBiLstm(
            n_mels=int(checkpoint.get("n_mels", 80)),
            residual=use_residual,
            gated=use_gated,
        ).to(device)
        model.load_state_dict(checkpoint["model"])
        print(f"loaded {args.init_checkpoint} (residual={use_residual} gated={use_gated})", flush=True)
    else:
        model = CnnBiLstm(residual=use_residual, gated=use_gated).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_l1 = float("inf")
    best_score = float("inf")
    for epoch in range(1, args.epochs + 1):
        gate_weight = 0.0 if epoch <= args.gate_warmup else args.gate_weight
        train_loss, train_identity, train_gate = run_epoch(
            model, train_loader, optimizer, device, gate_weight, args.severity_weight
        )
        with torch.no_grad():
            val_loss, val_identity, val_gate = run_epoch(
                model, val_loader, None, device, args.gate_weight, False
            )
        score = val_loss + args.identity_weight * val_identity
        history.append(
            {
                "epoch": epoch,
                "train_l1": train_loss,
                "train_identity_l1": train_identity,
                "train_gate": train_gate,
                "val_l1": val_loss,
                "val_identity_l1": val_identity,
                "val_gate": val_gate,
                "val_score": score,
            }
        )
        print(
            f"epoch {epoch} train_l1 {train_loss:.4f} val_l1 {val_loss:.4f} "
            f"val_identity_l1 {val_identity:.4f} val_gate {val_gate:.3f} val_score {score:.4f}",
            flush=True,
        )
        checkpoint = {
            "model": model.state_dict(),
            "n_mels": model.n_mels,
            "residual": model.residual,
            "gated": model.gated,
            "target_mode": args.target_mode,
            "epoch": epoch,
            "val_l1": val_loss,
            "val_identity_l1": val_identity,
            "val_score": score,
        }
        torch.save(checkpoint, args.output_dir / "mel_mapper_last.pt")
        if val_loss < best_l1:
            best_l1 = val_loss
            torch.save(checkpoint, args.output_dir / "mel_mapper_best.pt")
        if score < best_score:
            best_score = score
            torch.save(checkpoint, args.output_dir / "mel_mapper_conservative.pt")
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
