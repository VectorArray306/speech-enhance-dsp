"""Continue the general LoRA on one speaker's calibration utterances."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from dysarthric_pipeline.config import resolve_under
from dysarthric_pipeline.models.asr_model import WhisperLoRA
from dysarthric_pipeline.models.train_asr import (
    WhisperCollator,
    _dataset,
    _manifest,
    _print_report,
    evaluate_manifest,
    learning_rate_scale,
)


def _calibration_manifest(speaker: str, calibration_dir: str, destination: Path) -> Path:
    folder = Path(calibration_dir)
    rows = []
    for audio in sorted(folder.glob("*.wav")):
        transcript = audio.with_suffix(".txt")
        if not transcript.is_file():
            raise FileNotFoundError(f"Missing transcript for {audio.name}: {transcript}")
        text = transcript.read_text(encoding="utf-8").strip().lower()
        if not text:
            continue
        rows.append(
            {
                "audio_path": str(audio),
                "audio_paths": [str(audio)],
                "text": text,
                "speaker": speaker,
                "corpus": "calibration",
                "condition": "dysarthric",
                "severity": "unknown",
                "mic": "",
                "block": "",
                "word_code": "",
                "split": "adapt",
                "utterance_id": f"calibration:{audio.stem}",
                "session": "",
            }
        )
    if not rows:
        raise FileNotFoundError(f"No wav/txt pairs in {folder}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return destination


def _speaker_manifest(config: dict, speaker: str, destination: Path) -> Path:
    source = _manifest(config, "adapt")
    rows = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["speaker"] == speaker:
            rows.append(row)
    if not rows:
        known = sorted({json.loads(line)["speaker"] for line in source.read_text().splitlines() if line.strip()})
        raise ValueError(
            f"No adapt-split utterances for {speaker}. "
            f"Use a manifest speaker id such as torgo:F01 or uaspeech:M05. "
            f"Available example: {', '.join(known[:8])}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    print(f"{speaker}: {len(rows)} calibration utterances")
    return destination


def adapt(config: dict, speaker: str | None, calibration_dir: str | None = None) -> dict:
    if not speaker:
        raise ValueError("adapt requires --speaker, for example uaspeech:M05 or torgo:F01.")
    checkpoint_dir = resolve_under(config, "checkpoint_dir")
    general = checkpoint_dir / "adapter_best"
    if not general.is_dir():
        raise FileNotFoundError(f"General adapter not found at {general}. Train it first.")
    speaker_dir = resolve_under(config, "adapter_dir") / speaker.replace(":", "_")
    manifest = (
        _calibration_manifest(speaker, calibration_dir, speaker_dir / "calibration.jsonl")
        if calibration_dir
        else _speaker_manifest(config, speaker, speaker_dir / "calibration.jsonl")
    )

    held_out = _manifest(config, "adapt_test")
    speaker_test = speaker_dir / "adapt_test.jsonl"
    rows = [
        json.loads(line)
        for line in held_out.read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line)["speaker"] == speaker
    ]
    if rows:
        speaker_test.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    asr = WhisperLoRA(config)
    asr.load(adapter_path=general, train_adapter=True)
    report = {"speaker": speaker, "adapter": str(speaker_dir), "steps": 0}
    if rows:
        report["before"] = evaluate_manifest(config, asr, speaker_test, final=True)
        _print_report(f"{speaker} before", report["before"])

    from torch.utils.data import DataLoader

    dataset = _dataset(config, manifest, train=True)
    loader = DataLoader(
        dataset,
        batch_size=int(config["adapt"]["batch_size"]),
        shuffle=True,
        num_workers=0,
        collate_fn=WhisperCollator(asr.processor, train=True, apply_specaugment=True),
    )
    base_lr = float(config["adapt"]["learning_rate"])
    optimizer = torch.optim.AdamW(
        (param for param in asr.model.parameters() if param.requires_grad),
        lr=base_lr,
        weight_decay=float(config["adapt"].get("weight_decay", 0.0)),
    )
    steps = int(config["adapt"]["steps"])
    warmup = int(config["adapt"].get("warmup_steps", 0))
    min_ratio = float(config["adapt"].get("min_lr_ratio", 0.1))
    completed = 0
    asr.model.train()
    while completed < steps:
        for batch in loader:
            scale = learning_rate_scale(completed, warmup, steps, min_ratio)
            for group in optimizer.param_groups:
                group["lr"] = base_lr * scale
            features = batch["input_features"].to(device=asr.device, dtype=asr.dtype)
            labels = batch["labels"].to(asr.device)
            loss = asr.model(input_features=features, labels=labels).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            completed += 1
            if completed % 10 == 0 or completed == steps:
                print(f"adapt step {completed}/{steps} loss {float(loss.detach()):.4f}", flush=True)
            if completed >= steps:
                break
    asr.save(speaker_dir)
    report["steps"] = completed
    if rows:
        report["adapt_test"] = evaluate_manifest(config, asr, speaker_test, final=True)
        _print_report(f"{speaker} adapt_test", report["adapt_test"])
    (speaker_dir / "adapt_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
