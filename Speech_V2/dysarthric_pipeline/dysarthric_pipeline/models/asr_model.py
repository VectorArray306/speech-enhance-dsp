"""Whisper-small with a LoRA adapter for dysarthric speech."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from dysarthric_pipeline.data.dataset import normalize_loudness, trim_silence


class WhisperLoRA:
    """Base Whisper model plus an optional LoRA adapter."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.processor: WhisperProcessor | None = None
        self.model: WhisperForConditionalGeneration | PeftModel | None = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

    def load(self, adapter_path: str | Path | None = None, train_adapter: bool = False) -> None:
        name = self.config["asr"]["base_model"]
        self.processor = WhisperProcessor.from_pretrained(name)
        base = WhisperForConditionalGeneration.from_pretrained(name, dtype=self.dtype)
        base.config.use_cache = not train_adapter
        if adapter_path is not None:
            self.model = PeftModel.from_pretrained(
                base,
                str(adapter_path),
                is_trainable=train_adapter,
            )
        elif train_adapter:
            asr = self.config["asr"]
            self.model = get_peft_model(
                base,
                LoraConfig(
                    r=int(asr["lora_r"]),
                    lora_alpha=int(asr["lora_alpha"]),
                    lora_dropout=float(asr["lora_dropout"]),
                    target_modules=list(asr["target_modules"]),
                    bias="none",
                ),
            )
        else:
            self.model = base
        self.model.to(self.device)
        if self.model.generation_config is not None:
            self.model.generation_config.max_length = None
        if train_adapter:
            self.model.train()
            self.model.print_trainable_parameters()
        else:
            self.model.eval()

    def save(self, directory: str | Path) -> None:
        if self.model is None or self.processor is None:
            raise RuntimeError("Call load() before save().")
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.processor.save_pretrained(path)

    def transcribe_batch(self, audios: list[np.ndarray]) -> list[str]:
        if self.model is None or self.processor is None:
            raise RuntimeError("Call load() before transcribe().")
        inputs = self.processor(
            audios,
            sampling_rate=16000,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        features = inputs.input_features.to(device=self.device, dtype=self.dtype)
        with torch.inference_mode():
            predicted = self.model.generate(features, **generation_kwargs(self.config, final=True))
        return [text.strip() for text in self.processor.batch_decode(predicted, skip_special_tokens=True)]

    def transcribe(self, audio) -> str:
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().numpy()
        wave = np.asarray(audio, dtype=np.float32)
        audio_cfg = self.config.get("audio", {})
        top_db = float(audio_cfg.get("trim_db") or 0)
        if top_db > 0:
            trimmed = trim_silence(
                torch.from_numpy(wave),
                int(audio_cfg.get("sample_rate", 16000)),
                top_db,
                int(audio_cfg.get("trim_pad_ms", 200)),
            )
            wave = trimmed.numpy()
        target_dbfs = audio_cfg.get("normalize_dbfs")
        if target_dbfs is not None:
            wave = normalize_loudness(
                torch.from_numpy(wave),
                float(target_dbfs),
                float(audio_cfg.get("normalize_peak", 0.95)),
            ).numpy()
        return self.transcribe_batch([wave])[0]


def generation_kwargs(config: dict, final: bool = False) -> dict:
    """Greedy decoding during training; beam search for the reported evaluation."""
    train = config["train"]
    beams = int(train.get("eval_beams" if final else "num_beams", 1))
    kwargs = {"max_new_tokens": int(train["max_new_tokens"])}
    if beams > 1:
        kwargs["num_beams"] = beams
        ngram = int(train.get("no_repeat_ngram_size", 0))
        if ngram > 1:
            kwargs["no_repeat_ngram_size"] = ngram
    if final:
        penalty = float(train.get("repetition_penalty", 1.0))
        if penalty > 1.0:
            kwargs["repetition_penalty"] = penalty
    return kwargs
