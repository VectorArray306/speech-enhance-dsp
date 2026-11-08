"""Index the local dysarthric_speech folder into utterance records."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

from normalize_text import normalize_text, usable_transcript


@dataclass(frozen=True)
class Utterance:
    id: str
    wav: str
    text: str
    speaker: str
    corpus: str

    def to_dict(self) -> dict:
        return asdict(self)


def _read_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace").strip()


def discover_torgo(root: Path) -> tuple[list[Utterance], dict]:
    """Dysarthric speakers only, wav_headMic paired with a usable prompt."""
    utterances: list[Utterance] = []
    skipped = {"control_speakers": 0, "unusable_prompt": 0, "missing_prompt": 0}
    for group in ("F", "M"):
        group_dir = root / group
        if not group_dir.is_dir():
            continue
        for speaker_dir in sorted(p for p in group_dir.iterdir() if p.is_dir()):
            speaker = f"torgo_{speaker_dir.name}"
            for session_dir in sorted(p for p in speaker_dir.iterdir() if p.is_dir()):
                wav_dir = session_dir / "wav_headMic"
                prompt_dir = session_dir / "prompts"
                if not wav_dir.is_dir() or not prompt_dir.is_dir():
                    continue
                prompts = {p.stem: p for p in prompt_dir.iterdir() if p.is_file()}
                for wav_path in sorted(wav_dir.glob("*.wav")):
                    prompt_path = prompts.get(wav_path.stem)
                    if prompt_path is None:
                        skipped["missing_prompt"] += 1
                        continue
                    raw = _read_prompt(prompt_path)
                    if not usable_transcript(raw):
                        skipped["unusable_prompt"] += 1
                        continue
                    utterances.append(
                        Utterance(
                            id=f"{speaker}_{session_dir.name}_{wav_path.stem}",
                            wav=str(wav_path.resolve()),
                            text=normalize_text(raw),
                            speaker=speaker,
                            corpus="torgo",
                        )
                    )
    for group in ("FC", "MC"):
        group_dir = root / group
        if group_dir.is_dir():
            skipped["control_speakers"] += sum(1 for p in group_dir.iterdir() if p.is_dir())
    return utterances, skipped


def discover_mendeley_synthetic(root: Path) -> list[Utterance]:
    dataset = root / "Synthetic Dysarthric Speech Dataset for Robust ASR" / "dataset"
    lines = (dataset / "text.txt").read_text(encoding="utf-8", errors="replace").splitlines()
    utterances: list[Utterance] = []
    for gender in ("f", "m"):
        gender_dir = dataset / gender
        if not gender_dir.is_dir():
            continue
        for speaker_dir in sorted(p for p in gender_dir.iterdir() if p.is_dir()):
            speaker = f"synth_{speaker_dir.name}"
            for index, raw in enumerate(lines):
                if not usable_transcript(raw):
                    continue
                wav_path = speaker_dir / f"{index}.wav"
                if not wav_path.is_file():
                    continue
                utterances.append(
                    Utterance(
                        id=f"{speaker}_{index:04d}",
                        wav=str(wav_path.resolve()),
                        text=normalize_text(raw),
                        speaker=speaker,
                        corpus="mendeley_synth",
                    )
                )
    return utterances


def _append_augmented(
    utterances: list[Utterance],
    speaker: str,
    index: int,
    raw: str,
    wav_by_index: dict[int, Path],
) -> None:
    if not usable_transcript(raw):
        return
    wav_path = wav_by_index.get(index)
    if wav_path is None:
        return
    utterances.append(
        Utterance(
            id=f"{speaker}_{index:04d}",
            wav=str(wav_path.resolve()),
            text=normalize_text(raw),
            speaker=speaker,
            corpus="augmented_synth",
        )
    )


def discover_augmented(root: Path) -> list[Utterance]:
    utterances: list[Utterance] = []
    base = root / "augmented_dysarthric_speech"
    folders = {
        "aug_female": base / "synthetic_dataset_female",
        "aug_male": base / "synthetic_dataset_male",
    }
    for speaker, folder in folders.items():
        meta = folder / "metadata.csv"
        if not meta.is_file():
            continue
        wav_by_index = {
            int(path.name.split("_", 1)[0]): path
            for path in folder.glob("*.wav")
        }
        with meta.open(newline="", encoding="utf-8", errors="replace") as handle:
            rows = list(csv.DictReader(handle))
        seen: set[int] = set()
        extras: list[str] = []
        for row in rows:
            raw = row["transcript"]
            index = int(Path(row["wav_filename"]).name.split("_", 1)[0])
            if index in seen:
                extras.append(raw)
                continue
            seen.add(index)
            _append_augmented(utterances, speaker, index, raw, wav_by_index)
        # One hundred metadata rows repeat wav ids 0001-0100 with a second sentence.
        # Those sentences line up with the wavs that have no metadata row.
        leftover = sorted(set(wav_by_index) - seen)
        for index, raw in zip(leftover, extras):
            _append_augmented(utterances, speaker, index, raw, wav_by_index)
    return utterances


def discover_all(root: Path) -> tuple[list[Utterance], dict]:
    torgo, skipped = discover_torgo(root)
    items = torgo + discover_mendeley_synthetic(root) + discover_augmented(root)
    return items, skipped
