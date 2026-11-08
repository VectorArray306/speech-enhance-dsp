"""Turn raw TORGO prompt files into ASR transcripts."""

from __future__ import annotations

import re

_GLOSS = re.compile(r"\[[^\]]*\]")
_SPACE = re.compile(r"\s+")
_PUNCT = re.compile(r"[^a-z0-9' ]+")


def clean_torgo_prompt(raw: str) -> str | None:
    """Return a spoken transcript, or None when the prompt is not lexical speech.

    Bracket-only instructions such as ``[say Ah]`` and image paths are dropped.
    A gloss in brackets is removed: ``tear [as in "tear up that paper"]``
    becomes ``tear``.
    """
    text = raw.replace("\x00", " ").strip()
    if not text or "\x1b" in text or "..." in text:
        return None
    lowered = text.lower()
    if lowered.startswith("input/") or lowered.startswith("http"):
        return None
    if text.startswith("[") and text.endswith("]"):
        return None
    text = _GLOSS.sub(" ", text)
    text = _SPACE.sub(" ", text).strip().lower()
    text = _PUNCT.sub("", text)
    text = _SPACE.sub(" ", text).strip()
    if not re.search(r"[a-z]", text):
        return None
    return text


def vote_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())
