"""Fill UA-Speech command, common-word, and uncommon-word transcripts.

The released filenames encode the prompt (``C10``, ``CW3``, ``UW12``) but not
the orthography. Digits and the radio alphabet are fixed. The remaining
codes are labeled by a majority vote of Whisper on intelligible control
speakers, microphone M5, and cached in ``uaspeech_lexicon.json``.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import soundfile as sf

from dysarthric_pipeline.data.prompts import vote_key
from dysarthric_pipeline.data.uaspeech_wordlist import (
    DIGITS,
    LEXICON_PATH,
    RADIO_ALPHABET,
    load_lexicon,
    parse_filename,
    transcript_for_code,
)

CONTROL_SPEAKERS = ("CF02", "CF03", "CM01")
WHISPER_MODEL = "openai/whisper-small.en"
LEXICON_VERSION = 2
_GATE_CODES = ("D0", "D5", "D9", "LA", "LM", "LZ")


def _open_codes(data_root: Path) -> dict[str, set[str]]:
    found: dict[str, set[str]] = defaultdict(set)
    for label in ("0", "1"):
        folder = data_root / "RAWDysPeech" / label
        if not folder.is_dir():
            continue
        for entry in folder.iterdir():
            parsed = parse_filename(entry.name)
            if parsed is None or parsed.word_code is None or parsed.block is None:
                continue
            code = parsed.word_code
            if code in DIGITS or code in RADIO_ALPHABET:
                continue
            if code.startswith(("C", "CW", "UW")):
                found[parsed.block].add(code)
    return found


def _missing(codes: dict[str, set[str]], lexicon: dict) -> list[tuple[str, str]]:
    missing = []
    for block, block_codes in sorted(codes.items()):
        for code in sorted(block_codes, key=_code_sort):
            if transcript_for_code(block, code, lexicon) is None:
                missing.append((block, code))
    return missing


def _code_sort(code: str) -> tuple[str, int]:
    prefix = "".join(ch for ch in code if ch.isalpha())
    number = "".join(ch for ch in code if ch.isdigit())
    return prefix, int(number or 0)


def _control_path(data_root: Path, speaker: str, block: str, code: str) -> Path | None:
    path = data_root / "RAWDysPeech" / "0" / f"{speaker}_{block}_{code}_M5.wav"
    if path.is_file() and path.stat().st_size >= 1000:
        return path
    return None


class _WhisperLabeler:
    def __init__(self) -> None:
        import warnings

        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        warnings.filterwarnings("ignore", message=".*max_new_tokens.*")
        warnings.filterwarnings("ignore", message=".*attention mask is not set.*")
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = WhisperProcessor.from_pretrained(WHISPER_MODEL)
        self.model = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL).to(self.device)
        self.model.eval()

    def score(self, features, text: str) -> float:
        tokenizer = self.processor.tokenizer
        start = tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
        no_timestamps = tokenizer.convert_tokens_to_ids("<|notimestamps|>")
        end = tokenizer.eos_token_id
        body = tokenizer(text, add_special_tokens=False).input_ids
        sequence = [start, no_timestamps, *body, end]
        decoder_input = self.torch.tensor([sequence[:-1]], device=self.device)
        target = self.torch.tensor(sequence[1:], device=self.device)
        with self.torch.inference_mode():
            logits = self.model(input_features=features, decoder_input_ids=decoder_input).logits
        log_probs = self.torch.log_softmax(logits, dim=-1)
        index = self.torch.arange(target.shape[0], device=self.device)
        return float(log_probs[0, index, target].sum())

    def transcribe(self, paths: list[Path], batch_size: int = 8) -> list[str]:
        texts: list[str] = []
        for start in range(0, len(paths), batch_size):
            batch = paths[start : start + batch_size]
            waves = [_load_audio(path) for path in batch]
            inputs = self.processor(waves, sampling_rate=16000, return_tensors="pt", padding=True)
            features = inputs.input_features.to(self.device)
            generate_kwargs = {"max_new_tokens": 16}
            attention_mask = getattr(inputs, "attention_mask", None)
            if attention_mask is not None:
                generate_kwargs["attention_mask"] = attention_mask.to(self.device)
            with self.torch.inference_mode():
                predicted = self.model.generate(features, **generate_kwargs)
            decoded = self.processor.batch_decode(predicted, skip_special_tokens=True)
            texts.extend(item.strip().lower() for item in decoded)
            done = min(start + batch_size, len(paths))
            if done % 80 == 0 or done == len(paths):
                print(f"  transcribed {done}/{len(paths)}", flush=True)
        return texts


def _load_audio(path: Path):
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if getattr(audio, "ndim", 1) == 2:
        audio = audio.mean(axis=1)
    if sample_rate != 16000:
        raise ValueError(f"{path} is {sample_rate} Hz, expected 16000")
    return audio


_NUMBER_WORDS = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
}


def _normalize_hypothesis(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9' ]+", " ", text.lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return _NUMBER_WORDS.get(cleaned, cleaned)


def _edit_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            replace = previous[j - 1] + (left_char != right_char)
            current.append(min(insert, delete, replace))
        previous = current
    return previous[-1]


def _agree(hypotheses: list[str], min_votes: int) -> str | None:
    hypotheses = [_normalize_hypothesis(item) for item in hypotheses]
    keyed = [(vote_key(item), item) for item in hypotheses if vote_key(item)]
    if len(keyed) < min_votes:
        return None
    counts = Counter(key for key, _ in keyed)
    winner, votes = counts.most_common(1)[0]
    if votes >= min_votes:
        surfaces = [text for key, text in keyed if key == winner]
        return min(surfaces, key=len)
    keys = [key for key, _ in keyed]
    mean_distance = [
        sum(_edit_distance(key, other) for other in keys) / len(keys)
        for key in keys
    ]
    medoid_index = min(range(len(keys)), key=mean_distance.__getitem__)
    if mean_distance[medoid_index] > 2:
        return None
    medoid = keys[medoid_index]
    cluster = [text for key, text in keyed if _edit_distance(key, medoid) <= 2]
    if len(cluster) < min_votes:
        return None
    return min(cluster, key=len)


def _check_gate(data_root: Path, labeler: _WhisperLabeler) -> None:
    jobs: list[tuple[str, str, Path]] = []
    for code in _GATE_CODES:
        expected = DIGITS.get(code) or RADIO_ALPHABET[code]
        for speaker in CONTROL_SPEAKERS:
            path = _control_path(data_root, speaker, "B1", code)
            if path is None:
                raise FileNotFoundError(f"Missing gate recording {speaker} B1 {code}")
            jobs.append((code, expected, path))
    print(f"Checking Whisper on {len(jobs)} known control words")
    hypotheses = labeler.transcribe([path for _, _, path in jobs])
    by_code: dict[str, list[str]] = defaultdict(list)
    expected_by_code = {}
    for (code, expected, _), hypothesis in zip(jobs, hypotheses):
        by_code[code].append(hypothesis)
        expected_by_code[code] = expected
    wrong = []
    for code, expected in expected_by_code.items():
        agreed = _agree(by_code[code], min_votes=2)
        if agreed is None or vote_key(agreed) != vote_key(expected):
            wrong.append((code, expected, by_code[code]))
    if wrong:
        detail = "; ".join(f"{code} expected {expected} got {hyp}" for code, expected, hyp in wrong)
        raise RuntimeError(f"Control-speech transcripts failed the known-word check: {detail}")
    print("Known digits and radio words matched")


# Whisper confuses these with the Brown-corpus word at the same CW code.
# The surrounding codes already match that list, so these replacements are applied
# after the vote instead of leaving the near-miss.
_COMMON_FIXES = {
    "CW4": "a",
    "CW13": "for",
    "CW22": "be",
    "CW27": "had",
    "CW28": "by",
    "CW33": "all",
    "CW42": "an",
    "CW44": "which",
    "CW48": "their",
    "CW62": "would",
    "CW79": "my",
    "CW80": "than",
    "CW83": "been",
    "CW87": "its",
}


def _apply_common_fixes(lexicon: dict) -> bool:
    changed = False
    for code, word in _COMMON_FIXES.items():
        if lexicon.get("common", {}).get(code) != word:
            lexicon.setdefault("common", {})[code] = word
            changed = True
    return changed


def _choose_command(labeler: _WhisperLabeler, data_root: Path, code: str, options: tuple[str, ...]) -> str | None:
    totals = {option: 0.0 for option in options}
    used = 0
    for speaker in CONTROL_SPEAKERS:
        audio_path = _control_path(data_root, speaker, "B1", code)
        if audio_path is None:
            continue
        wave = _load_audio(audio_path)
        inputs = labeler.processor([wave], sampling_rate=16000, return_tensors="pt")
        features = inputs.input_features.to(labeler.device)
        used += 1
        for option in options:
            totals[option] += labeler.score(features, option)
    if used == 0:
        return None
    return max(totals, key=totals.get)


def ensure_lexicon(data_root: Path, lexicon_path: Path | None = None) -> dict:
    path = lexicon_path or LEXICON_PATH
    lexicon = load_lexicon(path)
    if lexicon.get("version") != LEXICON_VERSION:
        lexicon = {
            "version": LEXICON_VERSION,
            "commands": {},
            "common": {},
            "uncommon": {"B1": {}, "B2": {}, "B3": {}},
        }
    fixes_changed = _apply_common_fixes(lexicon)
    needed = _open_codes(data_root)
    missing = _missing(needed, lexicon)
    skipped = set()
    for item in lexicon.get("unresolved", []):
        if item["block"] == "*":
            skipped.update((block, item["code"]) for block in ("B1", "B2", "B3"))
        else:
            skipped.add((item["block"], item["code"]))
    missing = [item for item in missing if item not in skipped]
    if not missing:
        if fixes_changed:
            path.write_text(json.dumps(lexicon, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(f"Updated common-word corrections in {path.name}")
        else:
            print(f"UA-Speech lexicon already complete ({path.name})")
        return lexicon

    labeler = _WhisperLabeler()
    _check_gate(data_root, labeler)
    jobs = []
    for block, code in missing:
        for speaker in CONTROL_SPEAKERS:
            audio = _control_path(data_root, speaker, block, code)
            if audio is not None:
                jobs.append((block, code, speaker, audio))
    print(f"Labeling {len(missing)} UA-Speech codes from {len(jobs)} control recordings")
    hypotheses = labeler.transcribe([item[3] for item in jobs])
    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    for (block, code, _, _), hypothesis in zip(jobs, hypotheses):
        grouped[(block, code)].append(hypothesis)

    unresolved = []
    blocks = sorted({block for block, _ in missing})
    codes = sorted({code for _, code in missing}, key=_code_sort)
    for code in codes:
        if code.startswith("UW"):
            for block in blocks:
                heard = grouped.get((block, code), [])
                if not heard:
                    continue
                agreed = _agree(heard, min_votes=2)
                if agreed is None:
                    unresolved.append({"block": block, "code": code, "hypotheses": heard})
                else:
                    lexicon["uncommon"].setdefault(block, {})[code] = agreed
            continue
        heard = []
        for block in blocks:
            heard.extend(grouped.get((block, code), []))
        agreed = _agree(heard, min_votes=3)
        if agreed is None:
            unresolved.append({"block": "*", "code": code, "hypotheses": heard})
            continue
        if code.startswith("CW"):
            lexicon["common"][code] = agreed
        else:
            lexicon["commands"][code] = agreed

    for code, options in (("C7", ("alt", "halt", "hold")), ("C10", ("line", "fine", "find"))):
        chosen = _choose_command(labeler, data_root, code, options)
        if chosen is not None:
            print(f"  {code} likelihood choice: {chosen}")
            lexicon["commands"][code] = chosen

    _apply_common_fixes(lexicon)
    lexicon["version"] = LEXICON_VERSION
    lexicon["source"] = (
        f"version {LEXICON_VERSION}: pooled Whisper vote of {WHISPER_MODEL} on "
        + ", ".join(CONTROL_SPEAKERS)
        + " microphone M5. Repeated words use all three blocks. "
        + "C7 and C10 were chosen by transcript likelihood. "
        + "Digits and the radio alphabet are the fixed closed set. "
        + "Unresolved codes are omitted rather than guessed."
    )
    lexicon["unresolved"] = unresolved
    path.write_text(json.dumps(lexicon, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {path} with {len(unresolved)} unresolved codes")
    return lexicon
