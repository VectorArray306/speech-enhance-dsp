"""Run the frozen pipeline on one finished waveform (file or microphone)."""

from __future__ import annotations

import argparse
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

from audio import load_preprocessed, preprocess_waveform  # noqa: E402
from hifigan_vocode import load_generator, mel_to_waveform, save_wav  # noqa: E402
from mel import load_hifigan_config, waveform_to_mel  # noqa: E402
from model import CnnBiLstm  # noqa: E402


def record_until_silence(sample_rate: int, silence_ms: int, max_seconds: float) -> np.ndarray:
    import sounddevice as sd

    block = int(sample_rate * 0.03)
    silence_blocks = max(1, int(silence_ms / 30))
    max_blocks = int(max_seconds * sample_rate / block)
    collected: list[np.ndarray] = []
    quiet = 0
    started = False
    with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32", blocksize=block) as stream:
        for _ in range(max_blocks):
            audio, _overflow = stream.read(block)
            chunk = audio[:, 0]
            collected.append(chunk.copy())
            level = float(np.sqrt(np.mean(chunk**2) + 1e-12))
            if level > 0.01:
                started = True
                quiet = 0
            elif started:
                quiet += 1
                if quiet >= silence_blocks:
                    break
    if not collected:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(collected).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess one utterance, map mel, vocode.")
    parser.add_argument("--wav", type=Path, default=None, help="Finished recording. Omit to use the microphone.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hifigan_checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("speech/live_out.wav"))
    parser.add_argument("--silence_ms", type=int, default=600)
    parser.add_argument("--max_seconds", type=float, default=20.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_hifigan_config()
    if args.wav is not None:
        waveform = load_preprocessed(args.wav, config["sampling_rate"])
    else:
        recorded = record_until_silence(16_000, args.silence_ms, args.max_seconds)
        waveform = preprocess_waveform(recorded, 16_000, config["sampling_rate"])

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = CnnBiLstm(
        n_mels=int(checkpoint.get("n_mels", 80)),
        residual=bool(checkpoint.get("residual", False)),
        gated=bool(checkpoint.get("gated", False)),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    mel = torch.from_numpy(waveform_to_mel(waveform, config, device))
    with torch.no_grad():
        predicted = model(mel.unsqueeze(0).to(device))
    generator, hifi = load_generator(args.hifigan_checkpoint, device)
    audio = mel_to_waveform(predicted, generator)
    save_wav(args.output, audio, hifi.sampling_rate)
    print(args.output)


if __name__ == "__main__":
    main()
