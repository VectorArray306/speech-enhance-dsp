"""Scan TORGO and RAWDysPeech and write train, validation, and test manifests."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from dysarthric_pipeline.config import resolve_under
from dysarthric_pipeline.data.build_lexicon import ensure_lexicon
from dysarthric_pipeline.data.prompts import clean_torgo_prompt
from dysarthric_pipeline.data.speakers import (
    TORGO_SEVERITY,
    UASPEECH_INTELLIGIBILITY,
    intelligibility_band,
    is_control_speaker,
)
from dysarthric_pipeline.data.uaspeech_wordlist import load_lexicon, parse_filename, transcript_for_code

MANIFEST_FIELDS = (
    "audio_path",
    "audio_paths",
    "text",
    "speaker",
    "corpus",
    "condition",
    "severity",
    "mic",
    "block",
    "word_code",
    "split",
    "utterance_id",
    "session",
)

_SESSION_CODE = {
    "Session1": "S01",
    "Session2": "S02",
    "Session3": "S03",
    "Session2_3": "S02",
}
_MIN_BYTES = 1000


def _fraction(seed: int, salt: str, utterance_id: str) -> float:
    digest = hashlib.sha1(f"{seed}:{salt}:{utterance_id}".encode()).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def _severity(corpus: str, speaker: str, condition: str) -> str:
    if condition == "control":
        return "control"
    if corpus == "torgo":
        return TORGO_SEVERITY.get(speaker, "unknown")
    percent = UASPEECH_INTELLIGIBILITY.get(speaker)
    if percent is None:
        return "unknown"
    return intelligibility_band(percent)


def _held_out(config: dict[str, Any]) -> set[str]:
    speakers = set()
    for item in config["splits"]["held_out_speakers"]:
        speakers.add(item if ":" in item else f"torgo:{item}")
    return speakers


def _collect_torgo(data_root: Path) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, Path]] = defaultdict(dict)
    prompts: dict[tuple[str, str, str], Path] = {}
    for group in ("F", "FC", "M", "MC"):
        group_dir = data_root / group
        if not group_dir.is_dir():
            continue
        for speaker_dir in sorted(path for path in group_dir.iterdir() if path.is_dir()):
            for session_dir in sorted(path for path in speaker_dir.iterdir() if path.is_dir()):
                if session_dir.name not in _SESSION_CODE:
                    continue
                key_session = session_dir.name
                prompt_dir = session_dir / "prompts"
                for mic_name in ("wav_headMic", "wav_arrayMic"):
                    mic_dir = session_dir / mic_name
                    if not mic_dir.is_dir():
                        continue
                    for audio in mic_dir.glob("*.wav"):
                        if audio.stat().st_size < _MIN_BYTES:
                            continue
                        utt = audio.stem
                        grouped[(speaker_dir.name, key_session, utt)][mic_name] = audio
                if prompt_dir.is_dir():
                    for prompt in prompt_dir.glob("*.txt"):
                        prompts[(speaker_dir.name, key_session, prompt.stem)] = prompt

    records = []
    for (speaker, session, utt), mics in grouped.items():
        prompt_path = prompts.get((speaker, session, utt))
        if prompt_path is None:
            continue
        text = clean_torgo_prompt(prompt_path.read_text(encoding="utf-8", errors="ignore"))
        if text is None:
            continue
        canonical_mic = "wav_headMic" if "wav_headMic" in mics else "wav_arrayMic"
        condition = "control" if is_control_speaker(speaker) else "dysarthric"
        records.append(
            {
                "corpus": "torgo",
                "speaker": f"torgo:{speaker}",
                "raw_speaker": speaker,
                "condition": condition,
                "severity": _severity("torgo", speaker, condition),
                "text": text,
                "audio_path": str(mics[canonical_mic]),
                "audio_paths": [str(mics[name]) for name in sorted(mics)],
                "mic": canonical_mic,
                "block": "",
                "word_code": "",
                "session": f"{session}/{_SESSION_CODE[session]}",
                "utterance_id": f"torgo:{speaker}:{session}:{utt}",
            }
        )
    return records


def _collect_uaspeech(data_root: Path, lexicon: dict) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], dict[str, Path]] = defaultdict(dict)
    raw = data_root / "RAWDysPeech"
    for label, condition in (("0", "control"), ("1", "dysarthric")):
        folder = raw / label
        if not folder.is_dir():
            continue
        for audio in folder.iterdir():
            parsed = parse_filename(audio.name)
            if parsed is None or parsed.kind != "uaspeech":
                continue
            if audio.stat().st_size < _MIN_BYTES or parsed.block is None or parsed.word_code is None:
                continue
            grouped[(condition, parsed.speaker, parsed.block, parsed.word_code)][parsed.mic or ""] = audio

    records = []
    for (condition, speaker, block, code), mics in grouped.items():
        text = transcript_for_code(block, code, lexicon)
        if not text:
            continue
        canonical_mic = "M5" if "M5" in mics else sorted(mics)[0]
        records.append(
            {
                "corpus": "uaspeech",
                "speaker": f"uaspeech:{speaker}",
                "raw_speaker": speaker,
                "condition": condition,
                "severity": _severity("uaspeech", speaker, condition),
                "text": text,
                "audio_path": str(mics[canonical_mic]),
                "audio_paths": [str(mics[name]) for name in sorted(mics)],
                "mic": canonical_mic,
                "block": block,
                "word_code": code,
                "session": block,
                "utterance_id": f"uaspeech:{speaker}:{block}:{code}",
            }
        )
    return records


def _assign_split(record: dict[str, Any], config: dict[str, Any], held_out: set[str]) -> str:
    splits = config["splits"]
    seed = int(splits["seed"])
    if record["speaker"] in held_out:
        return "test"
    adapt_pool = False
    if record["condition"] == "dysarthric" and record["corpus"] == "uaspeech" and record["block"] == "B3":
        adapt_pool = True
    elif record["condition"] == "dysarthric" and record["corpus"] == "torgo":
        adapt_pool = _fraction(seed, "torgo-adapt", record["utterance_id"]) < float(splits["torgo_adapt_fraction"])
    if adapt_pool:
        cutoff = float(splits["adaptation_train_fraction"])
        salt = "adapt"
        return "adapt" if _fraction(seed, salt, record["utterance_id"]) < cutoff else "adapt_test"
    cutoff = float(splits["validation_fraction"])
    return "validation" if _fraction(seed, "val", record["utterance_id"]) < cutoff else "train"


def _validate(records: list[dict[str, Any]], held_out: set[str]) -> None:
    by_id: dict[str, set[str]] = defaultdict(set)
    for record in records:
        by_id[record["utterance_id"]].add(record["split"])
        if not record["text"] or not Path(record["audio_path"]).is_file():
            raise RuntimeError(f"Broken manifest row {record['utterance_id']}")
        if record["speaker"] in held_out and record["split"] != "test":
            raise RuntimeError(f"Held-out speaker leaked into {record['split']}: {record['speaker']}")
        if record["split"] == "test" and record["speaker"] not in held_out:
            raise RuntimeError(f"Unexpected test speaker {record['speaker']}")
    leaked = [utt for utt, splits in by_id.items() if len(splits) > 1]
    if leaked:
        raise RuntimeError(f"Utterance appears in multiple splits: {leaked[:5]}")


def build_manifests(config: dict[str, Any]) -> dict[str, Any]:
    data_root = Path(config["paths"]["data_root"])
    manifest_dir = resolve_under(config, "manifest_dir")
    manifest_dir.mkdir(parents=True, exist_ok=True)
    lexicon = ensure_lexicon(data_root)
    lexicon = load_lexicon()

    records = _collect_torgo(data_root) + _collect_uaspeech(data_root, lexicon)
    held_out = _held_out(config)
    for record in records:
        record["split"] = _assign_split(record, config, held_out)
    _validate(records, held_out)

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        row = {field: record.get(field, "") for field in MANIFEST_FIELDS}
        buckets[record["split"]].append(row)

    for split, rows in buckets.items():
        rows.sort(key=lambda item: item["utterance_id"])
        destination = manifest_dir / f"{split}.jsonl"
        with destination.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    counts: dict[str, Any] = {"splits": {}, "by_condition": {}, "by_severity": {}}
    for split, rows in sorted(buckets.items()):
        counts["splits"][split] = len(rows)
        counts["by_condition"][split] = dict(Counter(row["condition"] for row in rows))
        counts["by_severity"][split] = dict(Counter(row["severity"] for row in rows))
    counts["utterances"] = len(records)
    counts["held_out_speakers"] = sorted(held_out)
    counts["train_mic_policy"] = (
        "Each train row is one utterance. Training draws one microphone from audio_paths. "
        "validation, test, adapt, and adapt_test use the canonical microphone in audio_path "
        "(TORGO wav_headMic, UA-Speech M5)."
    )
    summary_path = manifest_dir / "split_summary.json"
    summary_path.write_text(json.dumps(counts, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {counts['utterances']} utterances to {manifest_dir}")
    for split, count in counts["splits"].items():
        print(f"  {split}: {count}")
    return counts
