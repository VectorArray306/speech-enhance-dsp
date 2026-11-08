"""UA-Speech filename codes.

Digits (D0–D9) and the radio alphabet (LA–LZ) are the same word in every
block. Computer commands, common words, and uncommon words are loaded from
``uaspeech_lexicon.json`` when that file contains them. Missing entries
return None so a transcript is never guessed.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

LEXICON_PATH = Path(__file__).with_name("uaspeech_lexicon.json")
_LEXICON_PATH = LEXICON_PATH

DIGITS: dict[str, str] = {
    "D0": "zero",
    "D1": "one",
    "D2": "two",
    "D3": "three",
    "D4": "four",
    "D5": "five",
    "D6": "six",
    "D7": "seven",
    "D8": "eight",
    "D9": "nine",
}

RADIO_ALPHABET: dict[str, str] = {
    "LA": "alpha",
    "LB": "bravo",
    "LC": "charlie",
    "LD": "delta",
    "LE": "echo",
    "LF": "foxtrot",
    "LG": "golf",
    "LH": "hotel",
    "LI": "india",
    "LJ": "juliet",
    "LK": "kilo",
    "LL": "lima",
    "LM": "mike",
    "LN": "november",
    "LO": "oscar",
    "LP": "papa",
    "LQ": "quebec",
    "LR": "romeo",
    "LS": "sierra",
    "LT": "tango",
    "LU": "uniform",
    "LV": "victor",
    "LW": "whiskey",
    "LX": "xray",
    "LY": "yankee",
    "LZ": "zulu",
}

_UA_NAME = re.compile(
    r"^(?P<speaker>C?[FM]\d{2})_(?P<block>B[123])_(?P<code>[A-Z]+\d*)_(?P<mic>M\d+)\.wav$"
)
_TORGO_FLAT_NAME = re.compile(
    r"^wav_(?P<mic>headMic|arrayMic)_(?P<speaker>(?:MC|FC|M|F)\d{2})(?P<session>S\d{2})_(?P<utt>\d+)\.wav$"
)


@dataclass(frozen=True)
class ParsedName:
    kind: str
    speaker: str
    filename: str
    block: str | None = None
    word_code: str | None = None
    mic: str | None = None
    session: str | None = None
    utterance: str | None = None


def load_lexicon(path: str | Path | None = None) -> dict:
    lexicon_path = Path(path) if path is not None else _LEXICON_PATH
    if not lexicon_path.is_file():
        return {"commands": {}, "common": {}, "uncommon": {}}
    with lexicon_path.open(encoding="utf-8") as handle:
        loaded = json.load(handle)
    loaded.setdefault("commands", {})
    loaded.setdefault("common", {})
    loaded.setdefault("uncommon", {})
    return loaded


def transcript_for_code(block: str, word_code: str, lexicon: dict | None = None) -> str | None:
    if word_code in DIGITS:
        return DIGITS[word_code]
    if word_code in RADIO_ALPHABET:
        return RADIO_ALPHABET[word_code]
    table = lexicon if lexicon is not None else load_lexicon()
    if word_code.startswith("C") and word_code[1:].isdigit():
        return table["commands"].get(word_code)
    if word_code.startswith("CW"):
        return table["common"].get(word_code)
    if word_code.startswith("UW"):
        block_table = table["uncommon"].get(block, {})
        return block_table.get(word_code)
    return None


def parse_filename(filename: str) -> ParsedName | None:
    ua_match = _UA_NAME.match(filename)
    if ua_match is not None:
        return ParsedName(
            kind="uaspeech",
            speaker=ua_match.group("speaker"),
            filename=filename,
            block=ua_match.group("block"),
            word_code=ua_match.group("code"),
            mic=ua_match.group("mic"),
        )
    torgo_match = _TORGO_FLAT_NAME.match(filename)
    if torgo_match is not None:
        return ParsedName(
            kind="torgo",
            speaker=torgo_match.group("speaker"),
            filename=filename,
            mic=torgo_match.group("mic"),
            session=torgo_match.group("session"),
            utterance=torgo_match.group("utt"),
        )
    return None
