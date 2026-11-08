"""Multi-speaker LoRA training and WER/CER evaluation."""

from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path

import torch
from jiwer import cer, wer
from torch.utils.data import DataLoader, WeightedRandomSampler

from dysarthric_pipeline.config import resolve_under
from dysarthric_pipeline.data.dataset import SpeechManifestDataset, spec_augment
from dysarthric_pipeline.models.asr_model import WhisperLoRA, generation_kwargs

logging.getLogger("transformers").setLevel(logging.ERROR)

_SPACE = re.compile(r"\s+")
_PUNCT = re.compile(r"[^a-z0-9' ]+")


def normalize_transcript(text: str) -> str:
    cleaned = _PUNCT.sub(" ", text.lower())
    return _SPACE.sub(" ", cleaned).strip()


class WhisperCollator:
    def __init__(self, processor, train: bool, apply_specaugment: bool) -> None:
        self.processor = processor
        self.train = train
        self.apply_specaugment = apply_specaugment

    def __call__(self, batch: list[dict]) -> dict:
        audios = [item["audio"].numpy() for item in batch]
        encoded = self.processor(
            audios,
            sampling_rate=16000,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        features = encoded.input_features
        if self.train and self.apply_specaugment:
            for index in range(features.shape[0]):
                features[index] = spec_augment(features[index])
        tokenized = self.processor.tokenizer(
            [item["text"] for item in batch],
            return_tensors="pt",
            padding=True,
        )
        labels = tokenized.input_ids.masked_fill(tokenized.attention_mask.ne(1), -100)
        bos = self.processor.tokenizer.bos_token_id
        if bos is not None and labels.shape[1] > 1 and bool((labels[:, 0] == bos).all()):
            labels = labels[:, 1:]
        return {
            "input_features": features,
            "labels": labels,
            "text": [item["text"] for item in batch],
            "severity": [item["severity"] for item in batch],
            "corpus": [item["corpus"] for item in batch],
            "condition": [item["condition"] for item in batch],
        }


def _dataset(config: dict, manifest: Path, train: bool) -> SpeechManifestDataset:
    audio = config["audio"]
    return SpeechManifestDataset(
        manifest,
        train=train,
        sample_rate=int(audio["sample_rate"]),
        compute_mel=False,
        apply_specaugment=False,
        trim_db=float(audio.get("trim_db") or 0),
        trim_pad_ms=int(audio.get("trim_pad_ms", 200)),
        normalize_dbfs=audio.get("normalize_dbfs"),
        normalize_peak=float(audio.get("normalize_peak", 0.95)),
    )


def _loader(config: dict, asr: WhisperLoRA, manifest: Path, train: bool) -> DataLoader:
    dataset = _dataset(config, manifest, train)
    workers = int(config["train"]["num_workers"]) if train else 0
    # Windows starts DataLoader workers with spawn. The Whisper collator is safer in-process.
    if os.name == "nt":
        workers = 0
    sampler = None
    shuffle = train
    weights = config["train"].get("severity_weights") if train else None
    if weights:
        sample_weights = [
            float(weights.get(row.get("severity") or "", 1.0))
            for row in dataset.rows
        ]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(dataset), replacement=True)
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=shuffle,
        sampler=sampler,
        num_workers=workers,
        collate_fn=WhisperCollator(
            asr.processor,
            train=train,
            apply_specaugment=bool(config["train"].get("spec_augment", True)),
        ),
        pin_memory=asr.device == "cuda",
    )


def _rates(references: list[str], hypotheses: list[str]) -> dict[str, float]:
    refs = [normalize_transcript(text) or "<empty>" for text in references]
    hyps = [normalize_transcript(text) for text in hypotheses]
    return {
        "wer": round(float(wer(refs, hyps)) * 100, 2),
        "cer": round(float(cer(refs, hyps)) * 100, 2),
        "utterances": len(refs),
    }


def _grouped(references, hypotheses, tags) -> dict[str, dict]:
    buckets: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for ref, hyp, tag in zip(references, hypotheses, tags):
        buckets[tag or "unknown"].append((ref, hyp))
    return {
        tag: _rates([ref for ref, _ in pairs], [hyp for _, hyp in pairs])
        for tag, pairs in sorted(buckets.items())
    }


def evaluate_manifest(config: dict, asr: WhisperLoRA, manifest: Path, final: bool = False) -> dict:
    if asr.model is None or asr.processor is None:
        raise RuntimeError("Load a model before evaluation.")
    was_training = asr.model.training
    asr.model.eval()
    loader = _loader(config, asr, manifest, train=False)
    references: list[str] = []
    hypotheses: list[str] = []
    severities: list[str] = []
    corpora: list[str] = []
    conditions: list[str] = []
    for batch in loader:
        features = batch["input_features"].to(device=asr.device, dtype=asr.dtype)
        with torch.inference_mode():
            predicted = asr.model.generate(features, **generation_kwargs(config, final=final))
        hypotheses.extend(asr.processor.batch_decode(predicted, skip_special_tokens=True))
        references.extend(batch["text"])
        severities.extend(batch["severity"])
        corpora.extend(batch["corpus"])
        conditions.extend(batch["condition"])
        if len(references) % 200 == 0:
            print(f"  evaluated {len(references)}", flush=True)
    if was_training:
        asr.model.train()
    return {
        "overall": _rates(references, hypotheses),
        "by_severity": _grouped(references, hypotheses, severities),
        "by_corpus": _grouped(references, hypotheses, corpora),
        "by_condition": _grouped(references, hypotheses, conditions),
    }


def _manifest(config: dict, name: str) -> Path:
    path = resolve_under(config, "manifest_dir") / f"{name}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing manifest {path}. Run prepare first.")
    return path


def _print_report(title: str, report: dict) -> None:
    overall = report["overall"]
    print(
        f"{title}: WER {overall['wer']:.2f}%  CER {overall['cer']:.2f}%  "
        f"n={overall['utterances']}",
        flush=True,
    )
    dysarthric = report.get("by_condition", {}).get("dysarthric")
    if dysarthric:
        print(
            f"  dysarthric: WER {dysarthric['wer']:.2f}%  CER {dysarthric['cer']:.2f}%  "
            f"n={dysarthric['utterances']}"
        )
    for group, rows in report["by_severity"].items():
        print(f"  {group}: WER {rows['wer']:.2f}%  CER {rows['cer']:.2f}%  n={rows['utterances']}")


def evaluate(config: dict, adapter_path: str | Path | None = None) -> dict:
    asr = WhisperLoRA(config)
    asr.load(adapter_path=adapter_path, train_adapter=False)
    reports = {}
    for split in ("validation", "test"):
        reports[split] = evaluate_manifest(config, asr, _manifest(config, split))
        _print_report(split, reports[split])
    return reports


def learning_rate_scale(step: int, warmup: int, total_steps: int, min_ratio: float) -> float:
    """Linear warmup, then cosine decay down to ``min_ratio`` of the base rate."""
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup
    span = max(total_steps - warmup, 1)
    progress = min(max((step - warmup) / span, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_ratio + (1.0 - min_ratio) * cosine


def _set_lr(optimizer: torch.optim.Optimizer, step: int, warmup: int, total_steps: int, base_lr: float, min_ratio: float) -> None:
    scale = learning_rate_scale(step, warmup, total_steps, min_ratio)
    for group in optimizer.param_groups:
        group["lr"] = base_lr * scale


def _archive_previous(checkpoint_dir: Path) -> None:
    history = checkpoint_dir / "history.json"
    archived_history = checkpoint_dir / "history_v1.json"
    if history.is_file() and not archived_history.exists():
        shutil.copy2(history, archived_history)
    best = checkpoint_dir / "adapter_best"
    archived_best = checkpoint_dir / "adapter_v1"
    if best.is_dir() and not archived_best.exists():
        shutil.copytree(best, archived_best)


def _selection_score(config: dict, validation: dict) -> tuple[str, float]:
    metric = str(config["train"].get("select_metric", "wer"))
    if metric == "dysarthric_wer":
        dysarthric = validation.get("by_condition", {}).get("dysarthric")
        if dysarthric:
            return metric, float(dysarthric["wer"])
    return "wer", float(validation["overall"]["wer"])


def train(config: dict, baseline_only: bool = False, skip_baseline: bool = False) -> dict:
    checkpoint_dir = resolve_under(config, "checkpoint_dir")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _archive_previous(checkpoint_dir)
    history: dict = {"baseline": None, "epochs": []}
    if skip_baseline and (checkpoint_dir / "baseline_wer.json").is_file():
        history["baseline"] = json.loads((checkpoint_dir / "baseline_wer.json").read_text(encoding="utf-8"))
        history["baseline_note"] = "Zero-shot score from the phase 3 run, before silence trimming."
    if not skip_baseline:
        print("Zero-shot baseline", flush=True)
        history["baseline"] = evaluate(config)
        (checkpoint_dir / "baseline_wer.json").write_text(
            json.dumps(history["baseline"], indent=2) + "\n",
            encoding="utf-8",
        )
    if baseline_only:
        return history

    asr = WhisperLoRA(config)
    asr.load(train_adapter=True)
    train_loader = _loader(config, asr, _manifest(config, "train"), train=True)
    optimizer = torch.optim.AdamW(
        (param for param in asr.model.parameters() if param.requires_grad),
        lr=float(config["train"]["learning_rate"]),
        weight_decay=float(config["train"].get("weight_decay", 0.0)),
    )
    epochs = int(config["train"]["epochs"])
    accum = int(config["train"]["grad_accum"])
    warmup = int(config["train"]["warmup_steps"])
    base_lr = float(config["train"]["learning_rate"])
    min_ratio = float(config["train"].get("min_lr_ratio", 0.1))
    updates_per_epoch = max((len(train_loader) + accum - 1) // accum, 1)
    total_updates = updates_per_epoch * epochs
    best_score = float("inf")
    update_step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, epochs + 1):
        asr.model.train()
        running = 0.0
        seen = 0
        for index, batch in enumerate(train_loader, start=1):
            _set_lr(optimizer, update_step, warmup, total_updates, base_lr, min_ratio)
            features = batch["input_features"].to(device=asr.device, dtype=asr.dtype)
            labels = batch["labels"].to(asr.device)
            loss = asr.model(input_features=features, labels=labels).loss / accum
            loss.backward()
            running += float(loss.detach()) * accum
            seen += 1
            if index % accum == 0 or index == len(train_loader):
                torch.nn.utils.clip_grad_norm_(
                    (param for param in asr.model.parameters() if param.requires_grad),
                    1.0,
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                update_step += 1
            if index % 50 == 0:
                print(
                    f"epoch {epoch} step {index}/{len(train_loader)} "
                    f"loss {running / seen:.4f}",
                    flush=True,
                )
        validation = evaluate_manifest(config, asr, _manifest(config, "validation"))
        _print_report(f"epoch {epoch} validation", validation)
        epoch_row = {"epoch": epoch, "train_loss": round(running / max(seen, 1), 4), "validation": validation}
        history["epochs"].append(epoch_row)
        asr.save(checkpoint_dir / "adapter_last")
        metric, score = _selection_score(config, validation)
        if score < best_score:
            best_score = score
            asr.save(checkpoint_dir / "adapter_best")
            print(f"saved best adapter at {metric} {score:.2f}%", flush=True)
        (checkpoint_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")

    print("Beam-search evaluation with the best adapter", flush=True)
    tuned = WhisperLoRA(config)
    tuned.load(adapter_path=checkpoint_dir / "adapter_best", train_adapter=False)
    history["validation_beam"] = evaluate_manifest(config, tuned, _manifest(config, "validation"), final=True)
    _print_report("validation beam", history["validation_beam"])
    history["test"] = evaluate_manifest(config, tuned, _manifest(config, "test"), final=True)
    _print_report("test beam", history["test"])
    (checkpoint_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    return history
