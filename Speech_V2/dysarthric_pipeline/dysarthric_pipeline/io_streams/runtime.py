"""Live path: audio in, speech detection, recognition, optional preset voice."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from dysarthric_pipeline.config import resolve_under
from dysarthric_pipeline.io_streams.audio_sources import (
    AudioSource,
    FileStreamSource,
    MicrophoneSource,
    SerialSource,
    resolve_serial_port,
)
from dysarthric_pipeline.io_streams.dsp_vad import SpeechSegmenter
from dysarthric_pipeline.io_streams.tts_engine import PresetVoice
from dysarthric_pipeline.models.asr_model import WhisperLoRA

logging.getLogger("transformers").setLevel(logging.ERROR)


def resolve_adapter(config: dict, speaker: str | None) -> Path | None:
    if speaker:
        path = resolve_under(config, "adapter_dir") / speaker.replace(":", "_")
        if not path.is_dir():
            raise FileNotFoundError(f"No speaker adapter at {path}. Run adapt for {speaker} first.")
        return path
    configured = config["asr"].get("adapter_dir")
    if configured:
        return Path(configured)
    general = resolve_under(config, "checkpoint_dir") / "adapter_best"
    if general.is_dir():
        return general
    return None


def open_source(config: dict, source: str, wav_path: str | None, device: str | None = None) -> AudioSource:
    sample_rate = int(config["audio"]["sample_rate"])
    frame_ms = int(config["audio"]["frame_ms"])
    if source == "file":
        if not wav_path:
            raise ValueError("File input needs --input /path/to/audio.wav")
        return FileStreamSource(wav_path, sample_rate=sample_rate, frame_ms=frame_ms)
    if source == "mic":
        chosen = device or config["audio"].get("input_device")
        return MicrophoneSource(sample_rate=sample_rate, frame_ms=frame_ms, device=chosen)
    if source == "serial":
        serial = config["serial"]
        return SerialSource(
            port=resolve_serial_port(str(serial["port"])),
            baud=int(serial["baud"]),
            mode=serial["mode"],
            sample_rate=sample_rate,
            frame_ms=frame_ms,
            magic=tuple(serial["magic"]),
        )
    raise ValueError(f"Unknown source {source!r}. Use file, mic, or serial.")


def run_session(
    config: dict,
    source: AudioSource,
    speaker: str | None = None,
    output_mode: str = "text",
    play: bool = False,
    save_dir: str | None = None,
    segment: bool = True,
) -> list[dict]:
    adapter = resolve_adapter(config, speaker)
    recognizer = WhisperLoRA(config)
    recognizer.load(adapter_path=adapter, train_adapter=False)
    print(f"recognizer: {adapter or config['asr']['base_model']}", flush=True)
    voice = PresetVoice(config) if output_mode in {"voice", "both"} else None
    if voice is not None:
        voice.ensure_voice()
    segmenter = SpeechSegmenter(config) if segment else None
    folder = Path(save_dir) if save_dir else None
    if folder is not None:
        folder.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    whole: list[np.ndarray] = []
    try:
        for frame in source.chunks():
            if segmenter is None:
                if len(frame):
                    whole.append(np.asarray(frame, dtype=np.float32))
                continue
            for utterance in segmenter.push(frame):
                _emit(results, recognizer, voice, utterance, output_mode, play, folder)
        if segmenter is None:
            if whole:
                _emit(results, recognizer, voice, np.concatenate(whole), output_mode, play, folder)
        else:
            for utterance in segmenter.finish():
                _emit(results, recognizer, voice, utterance, output_mode, play, folder)
    except KeyboardInterrupt:
        print("stopped", flush=True)
    finally:
        source.close()
    print(f"utterances: {len(results)}", flush=True)
    return results


def _emit(results, recognizer, voice, utterance: np.ndarray, output_mode: str, play: bool, folder: Path | None) -> None:
    if utterance.size < recognizer.config["audio"]["sample_rate"] // 10:
        return
    text = recognizer.transcribe(utterance)
    index = len(results) + 1
    print(f"[{index}] {text}", flush=True)
    voice_path = ""
    if voice is not None and text:
        if folder is not None:
            voice_path = str(voice.write_wav(text, folder / f"utterance_{index:03d}.wav"))
        if play:
            try:
                voice.play(text)
            except Exception as exc:
                print(f"playback skipped: {exc}", flush=True)
                if not voice_path and folder is None:
                    voice_path = str(voice.write_wav(text, Path("voices") / f"utterance_{index:03d}.wav"))
    results.append({"index": index, "text": text, "voice_path": voice_path})
