"""Write speaker-disjoint train/validation/test manifests.

The requested 90/10/10 ratio sums to 110%. This script uses 80/10/10 of usable
utterances. Validation and test are filled from TORGO dysarthric speakers
first, so those splits stay on real speech when the counts allow it.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from discover import discover_all


# One female and one male TORGO speaker for domain adaptation.
# The original test speakers stay held out. The other two TORGO speakers are validation.
DEFAULT_TORGO_TRAIN = ("torgo_F03", "torgo_M02")
DEFAULT_TORGO_TEST = ("torgo_F01", "torgo_M01", "torgo_M03", "torgo_M05")


def assign_speakers(
    counts: dict[str, int],
    torgo_train: set[str],
    torgo_test: set[str],
) -> dict[str, str]:
    """Speaker-disjoint split. Synthetic speakers always train.

    Real TORGO speakers listed in ``torgo_train`` train with the synthetic data.
    ``torgo_test`` stays unseen. Every other TORGO speaker is validation.
    """
    overlap = torgo_train & torgo_test
    if overlap:
        raise ValueError(f"Speaker cannot be both train and test: {sorted(overlap)}")
    missing = (torgo_train | torgo_test) - set(counts)
    if missing:
        raise ValueError(f"Unknown speakers: {sorted(missing)}")
    assignment: dict[str, str] = {}
    for speaker in counts:
        if not speaker.startswith("torgo_") or speaker in torgo_train:
            assignment[speaker] = "train"
        elif speaker in torgo_test:
            assignment[speaker] = "test"
        else:
            assignment[speaker] = "validation"
    return assignment


def main() -> None:
    parser = argparse.ArgumentParser(description="Build speech manifests from dysarthric_speech.")
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path.home() / "Downloads" / "dysarthric_speech",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(__file__).resolve().parent / "manifests",
    )
    parser.add_argument(
        "--torgo_train_speakers",
        default=",".join(DEFAULT_TORGO_TRAIN),
        help="Comma-separated TORGO speakers included in training.",
    )
    parser.add_argument(
        "--torgo_test_speakers",
        default=",".join(DEFAULT_TORGO_TEST),
        help="Comma-separated TORGO speakers held out for test.",
    )
    args = parser.parse_args()

    utterances, skipped = discover_all(args.data_root)
    counts = Counter(item.speaker for item in utterances)
    torgo_train = {item for item in args.torgo_train_speakers.split(",") if item}
    torgo_test = {item for item in args.torgo_test_speakers.split(",") if item}
    assignment = assign_speakers(dict(counts), torgo_train, torgo_test)

    grouped: dict[str, list[dict]] = defaultdict(list)
    for item in utterances:
        record = item.to_dict()
        record["split"] = assignment[item.speaker]
        grouped[record["split"]].append(record)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in grouped.items():
        rows.sort(key=lambda row: row["id"])
        path = args.output_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    by_split = Counter(assignment[item.speaker] for item in utterances)
    speaker_rows = [
        {
            "speaker": speaker,
            "split": assignment[speaker],
            "utterances": counts[speaker],
            "corpus": next(item.corpus for item in utterances if item.speaker == speaker),
        }
        for speaker in sorted(assignment)
    ]
    summary = {
        "ratio": "speaker-disjoint with TORGO in train",
        "ratio_note": (
            "Synthetic speakers train. "
            f"TORGO train: {sorted(torgo_train)}. "
            f"TORGO test: {sorted(torgo_test)}. "
            "Remaining TORGO speakers are validation."
        ),
        "usable_utterances": len(utterances),
        "split_counts": dict(by_split),
        "skipped": skipped,
        "speakers": speaker_rows,
    }
    (args.output_dir / "split_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"split_counts": summary["split_counts"], "speakers": speaker_rows}, indent=2))


if __name__ == "__main__":
    main()
