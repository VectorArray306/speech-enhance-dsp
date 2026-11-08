"""Speaker tables used by later splits.

UA-Speech intelligibility percentages are the subjective scores reported
with that corpus (very low 0–25, low 26–50, mid 51–75, high 76–100).
TORGO severities follow the corpus speaker descriptions.
"""

from __future__ import annotations

UASPEECH_INTELLIGIBILITY = {
    "M04": 2,
    "F03": 6,
    "M12": 7,
    "M01": 15,
    "M07": 28,
    "F02": 29,
    "M16": 43,
    "M05": 58,
    "M11": 62,
    "F04": 62,
    "M09": 86,
    "M14": 90,
    "M08": 93,
    "M10": 93,
    "F05": 95,
}

TORGO_SEVERITY = {
    "F01": "severe",
    "F03": "moderate",
    "F04": "mild",
    "M01": "severe",
    "M02": "severe",
    "M03": "moderate",
    "M04": "mild",
    "M05": "moderate",
}

TORGO_CONTROL_SPEAKERS = ("FC01", "FC02", "FC03", "MC01", "MC02", "MC03", "MC04")


def intelligibility_band(percent: int) -> str:
    if percent <= 25:
        return "very_low"
    if percent <= 50:
        return "low"
    if percent <= 75:
        return "mid"
    return "high"


def is_control_speaker(speaker: str) -> bool:
    if speaker in TORGO_CONTROL_SPEAKERS:
        return True
    return speaker.startswith("C") and len(speaker) >= 3 and speaker[1] in {"F", "M"}
