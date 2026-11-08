"""Shared transcript normalization for TTS targets and WER."""

from __future__ import annotations

import re


_PUNCT = re.compile(r"[^a-z0-9'\s]+")
_SPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Lowercase, drop punctuation other than apostrophes, collapse whitespace."""
    lowered = text.strip().lower().replace("’", "'")
    cleaned = _PUNCT.sub(" ", lowered)
    return _SPACE.sub(" ", cleaned).strip()


def usable_transcript(raw: str) -> bool:
    """Keep words and prompted sentences. Drop noise markers, image cards, and stage directions."""
    stripped = raw.strip()
    if not stripped:
        return False
    lowered = stripped.lower()
    if lowered == "xxx" or lowered.startswith("xxx"):
        return False
    if ".jpg" in lowered or ".jpeg" in lowered or ".png" in lowered:
        return False
    if stripped.startswith("[") and stripped.endswith("]"):
        return False
    return bool(normalize_text(stripped))
