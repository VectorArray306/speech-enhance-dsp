"""Word error rate for a finished split. Requires a HiFi-GAN checkpoint and Whisper."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_SPEECH = Path(__file__).resolve().parent
_REPO = _SPEECH.parent
sys.path.insert(0, str(_SPEECH))
_sem = str(_REPO / "sEMG")
if _sem not in sys.path:
    sys.path.append(_sem)

from dataset import load_split  # noqa: E402
from hifigan_vocode import load_generator, mel_to_waveform, save_wav  # noqa: E402
from model import CnnBiLstm  # noqa: E402
from normalize_text import normalize_text  # noqa: E402


def word_error_rate(reference: str, hypothesis: str) -> tuple[int, int]:
    ref = reference.split()
    hyp = hypothesis.split()
    if not ref:
        return (0, 0) if not hyp else (len(hyp), 0)
    rows = len(ref) + 1
    cols = len(hyp) + 1
    distance = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        distance[i][0] = i
    for j in range(cols):
        distance[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            distance[i][j] = min(
                distance[i - 1][j] + 1,
                distance[i][j - 1] + 1,
                distance[i - 1][j - 1] + cost,
            )
    return distance[-1][-1], len(ref)


def load_mel(cache_dir: Path, kind: str, utt_id: str) -> torch.Tensor:
    array = np.load(cache_dir / kind / f"{utt_id}.npy")
    return torch.from_numpy(array)


def predict_mel(model: CnnBiLstm, dys_mel: torch.Tensor, device: torch.device) -> torch.Tensor:
    with torch.no_grad():
        prediction = model(dys_mel.unsqueeze(0).to(device))
    return prediction.squeeze(0).cpu()


def main() -> None:
    parser = argparse.ArgumentParser(description="Vocode a split and score WER with Whisper.")
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--system", choices=("dysarthric", "tts", "model"), required=True)
    parser.add_argument("--manifest_dir", type=Path, default=Path("speech/manifests"))
    parser.add_argument("--cache_dir", type=Path, default=Path("speech/cache"))
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--hifigan_checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Required when --system model.")
    parser.add_argument("--whisper_model", default="base")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if args.system == "model" and args.checkpoint is None:
        parser.error("--checkpoint is required when --system model")

    import whisper

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator, hifi = load_generator(args.hifigan_checkpoint, device)
    asr = whisper.load_model(args.whisper_model, device=str(device))
    rows = load_split(args.manifest_dir, args.split)
    if args.limit > 0:
        rows = rows[: args.limit]

    model = None
    if args.system == "model":
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model = CnnBiLstm(
            n_mels=int(checkpoint.get("n_mels", 80)),
            residual=bool(checkpoint.get("residual", False)),
            gated=bool(checkpoint.get("gated", False)),
        ).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    errors = 0
    words = 0
    details = []
    kind = "inputs" if args.system == "dysarthric" else "targets"
    for row in rows:
        if args.system == "model":
            mel = predict_mel(model, load_mel(args.cache_dir, "inputs", row["id"]), device)
        else:
            mel = load_mel(args.cache_dir, kind, row["id"])
        wav_path = args.output_dir / f"{row['id']}.wav"
        waveform = mel_to_waveform(mel.to(device), generator)
        save_wav(wav_path, waveform, hifi.sampling_rate)
        hypothesis = normalize_text(asr.transcribe(str(wav_path), fp16=False, language="en")["text"])
        distance, ref_words = word_error_rate(row["text"], hypothesis)
        errors += distance
        words += ref_words
        details.append(
            {
                "id": row["id"],
                "reference": row["text"],
                "hypothesis": hypothesis,
                "speaker": row["speaker"],
            }
        )
        print(f"{row['id']} ref_words={ref_words} edit_distance={distance}", flush=True)

    wer = errors / words if words else 0.0
    summary = {"split": args.split, "system": args.system, "wer": wer, "errors": errors, "words": words}
    (args.output_dir / "wer.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "hypotheses.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in details),
        encoding="utf-8",
    )
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
