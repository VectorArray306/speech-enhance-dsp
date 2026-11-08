"""Discover Gaddy & Klein silent-speech EMG utterances on disk."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class UtteranceRef:
    directory: Path
    index: int
    text: str
    book: int | str
    sentence_index: int

    @property
    def emg_path(self) -> Path:
        return self.directory / f"{self.index}_emg.npy"

    @property
    def audio_path(self) -> Path:
        clean = self.directory / f"{self.index}_audio_clean.flac"
        if clean.is_file():
            return clean
        return self.directory / f"{self.index}_audio.flac"

    @property
    def info_path(self) -> Path:
        return self.directory / f"{self.index}_info.json"


def _parse_book(raw: object) -> int | str:
    """Gaddy metadata uses int book ids or strings such as ``datedata.txt``."""
    if raw is None:
        return 0
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str):
        try:
            return int(raw)
        except ValueError:
            return raw
    return str(raw)


def _parse_sentence_index(raw: object) -> int | None:
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _scan_session_dir(session_dir: Path) -> list[UtteranceRef]:
    refs: list[UtteranceRef] = []
    for info_path in sorted(session_dir.glob("*_info.json")):
        m = re.match(r"(\d+)_info\.json$", info_path.name)
        if not m:
            continue
        idx = int(m.group(1))
        try:
            with open(info_path, encoding="utf-8") as f:
                info = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        sentence_index = _parse_sentence_index(info.get("sentence_index", -1))
        if sentence_index is None or sentence_index < 0:
            continue
        emg_path = session_dir / f"{idx}_emg.npy"
        if not emg_path.is_file():
            continue
        refs.append(
            UtteranceRef(
                directory=session_dir,
                index=idx,
                text=str(info.get("text", "")),
                book=_parse_book(info.get("book", 0)),
                sentence_index=sentence_index,
            )
        )
    return refs


def discover_utterances(emg_data_root: Path) -> list[UtteranceRef]:
    """
    Walk Gaddy layout under ``emg_data_root`` (e.g. voiced_parallel_data/session_*).

    Also accepts a single session directory that already contains ``*_info.json``.
    """
    root = Path(emg_data_root)
    refs: list[UtteranceRef] = []

    if any(root.glob("*_info.json")):
        refs.extend(_scan_session_dir(root))
        return refs

    for sub in sorted(root.rglob("*")):
        if not sub.is_dir():
            continue
        if any(sub.glob("*_info.json")):
            refs.extend(_scan_session_dir(sub))

    # Deduplicate (book, sentence_index, index) — keep first
    seen: set[tuple[int | str, int, int]] = set()
    unique: list[UtteranceRef] = []
    for r in refs:
        key = (r.book, r.sentence_index, r.index)
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)
    return unique


def train_val_split(
    refs: list[UtteranceRef],
    val_fraction: float = 0.1,
    seed: int = 1234,
) -> tuple[list[UtteranceRef], list[UtteranceRef]]:
    import random

    rng = random.Random(seed)
    shuffled = list(refs)
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_fraction))
    return shuffled[n_val:], shuffled[:n_val]
