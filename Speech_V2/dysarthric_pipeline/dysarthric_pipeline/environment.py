"""Phase 1 environment check: data root, lexicon, and GPU visibility."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dysarthric_pipeline.data.uaspeech_wordlist import (
    DIGITS,
    RADIO_ALPHABET,
    parse_filename,
    transcript_for_code,
)


def summarize_data_root(data_root: Path) -> dict[str, Any]:
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")
    summary: dict[str, Any] = {"root": str(data_root), "groups": {}}
    for name in ("F", "FC", "M", "MC"):
        group = data_root / name
        speakers = sorted(path.name for path in group.iterdir() if path.is_dir()) if group.is_dir() else []
        summary["groups"][name] = speakers
    raw = data_root / "RAWDysPeech"
    summary["rawdyspeech"] = {}
    if raw.is_dir():
        for label in ("0", "1"):
            folder = raw / label
            if folder.is_dir():
                summary["rawdyspeech"][label] = sum(1 for _ in os.scandir(folder))
    return summary


def lexicon_status() -> dict[str, int]:
    checks = {
        "D0": transcript_for_code("B1", "D0"),
        "D9": transcript_for_code("B2", "D9"),
        "LA": transcript_for_code("B3", "LA"),
        "LZ": transcript_for_code("B1", "LZ"),
        "C1": transcript_for_code("B1", "C1"),
        "CW1": transcript_for_code("B1", "CW1"),
        "UW1": transcript_for_code("B1", "UW1"),
    }
    if checks["D0"] != "zero" or checks["D9"] != "nine":
        raise RuntimeError("Digit codes D0–D9 did not resolve.")
    if checks["LA"] != "alpha" or checks["LZ"] != "zulu":
        raise RuntimeError("Radio alphabet codes LA–LZ did not resolve.")
    if len(DIGITS) != 10 or len(RADIO_ALPHABET) != 26:
        raise RuntimeError("Closed-set lexicon is incomplete.")
    sample = parse_filename("F02_B1_C10_M2.wav")
    flat = parse_filename("wav_headMic_M05S02_0345.wav")
    if sample is None or sample.speaker != "F02" or sample.word_code != "C10":
        raise RuntimeError("UA-Speech filename parser failed.")
    if flat is None or flat.speaker != "M05" or flat.session != "S02":
        raise RuntimeError("Flattened TORGO filename parser failed.")
    pending = sum(1 for key in ("C1", "CW1", "UW1") if checks[key] is None)
    return {"closed_set": len(DIGITS) + len(RADIO_ALPHABET), "pending_open_classes": pending}


def torch_status() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"installed": False, "cuda": False, "device": None}
    cuda = bool(torch.cuda.is_available())
    device = torch.cuda.get_device_name(0) if cuda else None
    return {"installed": True, "version": torch.__version__, "cuda": cuda, "device": device}
