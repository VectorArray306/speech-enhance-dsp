"""Cache dysarthric log-mels, espeak-ng TTS mels, and DTW-warped targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from align import align_source_to_target
from audio import load_preprocessed
from mel import load_hifigan_config, waveform_to_mel
from normalize_text import normalize_text, usable_transcript


DEFAULT_DATASET_ROOT = Path.home() / "Downloads" / "dysarthric_speech"


def espeak_executable() -> str:
    found = shutil.which("espeak-ng")
    if found:
        return found
    if os.name == "nt":
        program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
        program_files_x86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
        for candidate in (
            Path(program_files) / "eSpeak NG" / "espeak-ng.exe",
            Path(program_files_x86) / "eSpeak NG" / "espeak-ng.exe",
        ):
            if candidate.is_file():
                return str(candidate)
    raise FileNotFoundError(
        "espeak-ng was not found. On Windows, install eSpeak NG and add its folder to PATH "
        "(https://github.com/espeak-ng/espeak-ng/releases)."
    )


def discover_torgo_controls(dataset_root: Path) -> dict[str, dict[str, Path]]:
    """Map gender group ('FC' or 'MC') -> normalized prompt text -> best control wav path.

    Prefers FC02/FC03 for female controls and MC01/MC03 for male controls so reference
    voices stay consistent across prompts.
    """
    controls: dict[str, dict[str, Path]] = {"FC": {}, "MC": {}}
    preferred_order = {
        "FC": ["FC02", "FC03", "FC01"],
        "MC": ["MC01", "MC03", "MC04", "MC02"],
    }
    for group in ("FC", "MC"):
        group_dir = dataset_root / group
        if not group_dir.is_dir():
            continue
        speaker_dirs = {p.name: p for p in group_dir.iterdir() if p.is_dir()}
        ordered = [speaker_dirs[name] for name in preferred_order[group] if name in speaker_dirs]
        ordered.extend(p for name, p in sorted(speaker_dirs.items()) if name not in preferred_order[group])
        for speaker_dir in ordered:
            for session in sorted(p for p in speaker_dir.iterdir() if p.is_dir()):
                wav_dir = session / "wav_headMic"
                prompt_dir = session / "prompts"
                if not wav_dir.is_dir() or not prompt_dir.is_dir():
                    continue
                prompts = {p.stem: p for p in prompt_dir.iterdir() if p.is_file()}
                for wav in sorted(wav_dir.glob("*.wav")):
                    prompt_path = prompts.get(wav.stem)
                    if prompt_path is None:
                        continue
                    raw = prompt_path.read_text(encoding="utf-8", errors="replace").strip()
                    if not usable_transcript(raw):
                        continue
                    text = normalize_text(raw)
                    if text and text not in controls[group]:
                        controls[group][text] = wav
    return controls


def speaker_control_group(speaker: str) -> str:
    upper = speaker.upper()
    if "_F" in upper or "FEMALE" in upper:
        return "FC"
    return "MC"


def load_manifests(manifest_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for name in ("train.jsonl", "validation.jsonl", "test.jsonl"):
        path = manifest_dir / name
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def text_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def synthesize_espeak(text: str, wav_path: Path, voice: str) -> None:
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as handle:
        handle.write(text + "\n")
        script = handle.name
    try:
        subprocess.run(
            [espeak_executable(), "-v", voice, "-s", "150", "-w", str(wav_path), "-f", script],
            check=True,
            capture_output=True,
        )
    finally:
        Path(script).unlink(missing_ok=True)


def resolve_reference_mel(row: dict, root: Path) -> np.ndarray:
    key = text_key(row["text"])
    group = speaker_control_group(str(row.get("speaker", "")))
    other_group = "MC" if group == "FC" else "FC"
    for cand in (
        root / "controls" / f"{group}_{key}.npy",
        root / "controls" / f"{other_group}_{key}.npy",
        root / "tts" / f"{key}.npy",
    ):
        if cand.is_file():
            return np.load(cand)
    raise FileNotFoundError(f"no reference mel found for text={row['text']!r} (key={key})")


def cache_utterance(payload: tuple[dict, str, bool]) -> str:
    row, cache_dir, force_targets = payload
    root = Path(cache_dir)
    input_path = root / "inputs" / f"{row['id']}.npy"
    target_path = root / "targets" / f"{row['id']}.npy"
    if input_path.is_file() and target_path.is_file() and not force_targets:
        return row["id"]
    config = load_hifigan_config()
    if not input_path.is_file():
        mel = waveform_to_mel(load_preprocessed(Path(row["wav"])), config)
        np.save(input_path, mel)
    dys_mel = np.load(input_path)
    ref_mel = resolve_reference_mel(row, root)
    np.save(target_path, align_source_to_target(ref_mel, dys_mel))
    return row["id"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build input and target log-mel caches.")
    parser.add_argument("--manifest_dir", type=Path, default=Path(__file__).resolve().parent / "manifests")
    parser.add_argument("--cache_dir", type=Path, default=Path(__file__).resolve().parent / "cache")
    parser.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--voice", default="en-us", help="espeak-ng fallback voice when no TORGO control exists.")
    parser.add_argument(
        "--disable_torgo_controls",
        action="store_true",
        help="Use espeak-ng TTS for all prompts instead of TORGO FC/MC control recordings.",
    )
    parser.add_argument(
        "--rebuild_targets",
        action="store_true",
        help="Recompute targets/*.npy even when they already exist on disk.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N utterances. 0 means all.")
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    args = parser.parse_args()

    rows = load_manifests(args.manifest_dir)
    if args.limit > 0:
        rows = rows[: args.limit]
    config = load_hifigan_config()
    input_dir = args.cache_dir / "inputs"
    tts_dir = args.cache_dir / "tts"
    control_dir = args.cache_dir / "controls"
    target_dir = args.cache_dir / "targets"
    for folder in (input_dir, tts_dir, control_dir, target_dir):
        folder.mkdir(parents=True, exist_ok=True)

    unique_texts = sorted({row["text"] for row in rows})
    controls = {"FC": {}, "MC": {}} if args.disable_torgo_controls else discover_torgo_controls(args.dataset_root)
    control_built = 0
    for group in ("FC", "MC"):
        group_map = controls.get(group, {})
        for text in unique_texts:
            wav_path = group_map.get(text)
            if wav_path is None:
                continue
            key = text_key(text)
            mel_path = control_dir / f"{group}_{key}.npy"
            if not mel_path.is_file():
                mel = waveform_to_mel(load_preprocessed(wav_path), config)
                np.save(mel_path, mel)
            control_built += 1
    if not args.disable_torgo_controls:
        print(f"controls cached: {control_built} (FC={len(controls['FC'])}, MC={len(controls['MC'])})", flush=True)

    for index, text in enumerate(unique_texts, start=1):
        key = text_key(text)
        mel_path = tts_dir / f"{key}.npy"
        if mel_path.is_file():
            continue
        wav_path = tts_dir / f"{key}.wav"
        synthesize_espeak(text, wav_path, args.voice)
        mel = waveform_to_mel(load_preprocessed(wav_path), config)
        np.save(mel_path, mel)
        wav_path.unlink(missing_ok=True)
        if index % 100 == 0 or index == len(unique_texts):
            print(f"tts {index}/{len(unique_texts)}", flush=True)

    payloads = [(row, str(args.cache_dir), bool(args.rebuild_targets)) for row in rows]
    done = 0
    mp.set_start_method("spawn", force=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for _utt_id in pool.map(cache_utterance, payloads, chunksize=8):
            done += 1
            if done % 200 == 0 or done == len(payloads):
                print(f"utterances {done}/{len(payloads)}", flush=True)


if __name__ == "__main__":
    main()
