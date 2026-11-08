# Dysarthric speech to clear mel

Utterance-level pipeline: preprocessed dysarthric audio, CNN–BiLSTM log-mel mapper, frozen HiFi-GAN v1. Training is a separate command and is not started by the prep scripts.

Data root: `~\Downloads\dysarthric_speech` (your user Downloads folder). Override it with `--data_root`.

| Folder | Corpus |
|--------|--------|
| `Synthetic Dysarthric Speech Dataset for Robust ASR` | Eight synthetic speakers, shared `text.txt` |
| `augmented_dysarthric_speech` | Augmented female and male synthetic speakers |
| `F`, `M` | TORGO dysarthric, `wav_headMic` |
| `FC`, `MC` | TORGO controls, left out of every split |

Splits are speaker-disjoint. Synthetic speakers always train. Two TORGO speakers (`torgo_F03`, `torgo_M02`) also train so the mapper sees real dysarthric speech. The original test speakers stay held out. The previous synth-only manifests are in `speech/manifests_synth_only`.

| Split | Speakers |
|-------|----------|
| Train | synthetic speakers plus `torgo_F03`, `torgo_M02` |
| Validation | `torgo_F04`, `torgo_M04` |
| Test | `torgo_F01`, `torgo_M01`, `torgo_M03`, `torgo_M05` |

The augmented metadata repeats wav ids 0001–0100 with a second sentence. Those extra sentences are paired with wavs 1064–1163, which have no row of their own. Seven TORGO control speakers are excluded. 145 prompts were dropped (`xxx`, image cards, or bracketed instructions) and 3 head-mic files had no prompt.

From the repository root, after `python -m pip install -r requirements.txt`. `prepare_cache.py` needs eSpeak NG on PATH. `eval_wer.py` needs ffmpeg on PATH. Put the corpus in `%USERPROFILE%\Downloads\dysarthric_speech`, or pass `--data_root`.

## Prep

```bat
python speech\build_manifests.py --output_dir speech\manifests
python speech\prepare_cache.py --manifest_dir speech\manifests --cache_dir speech\cache
```

`prepare_cache.py` high-passes at 80 Hz, resamples to 22.05 kHz, writes HiFi-GAN v1 log-mels, synthesizes targets with `espeak-ng -v en-us`, and warps each target onto the dysarthric frames.

## Train

```bat
python speech\train.py --manifest_dir speech\manifests --cache_dir speech\cache --output_dir speech\checkpoints --epochs 20 --batch_size 8 --lr 1e-3
```

Early stopping in this script keeps the lowest validation mel L1. Choose the checkpoint you report with validation WER below.

## Validation and test

HiFi-GAN universal v1 weights are required (`config.json` may sit beside the checkpoint; otherwise `hifi-gan/config_v1.json` is used). Whisper is required for WER:

Whisper is installed by the root `requirements.txt` (`openai-whisper`). ffmpeg must be on PATH.

Validation, three systems:

```bat
python speech\eval_wer.py --split validation --system dysarthric --manifest_dir speech\manifests --cache_dir speech\cache --hifigan_checkpoint C:\path\to\generator_v1 --output_dir speech\validation_out\dysarthric
python speech\eval_wer.py --split validation --system tts --manifest_dir speech\manifests --cache_dir speech\cache --hifigan_checkpoint C:\path\to\generator_v1 --output_dir speech\validation_out\tts
python speech\eval_wer.py --split validation --system model --checkpoint speech\checkpoints\mel_mapper_best.pt --manifest_dir speech\manifests --cache_dir speech\cache --hifigan_checkpoint C:\path\to\generator_v1 --output_dir speech\validation_out\model
```

Test, run once:

```bat
python speech\eval_wer.py --split test --system dysarthric --manifest_dir speech\manifests --cache_dir speech\cache --hifigan_checkpoint C:\path\to\generator_v1 --output_dir speech\test_out\dysarthric
python speech\eval_wer.py --split test --system tts --manifest_dir speech\manifests --cache_dir speech\cache --hifigan_checkpoint C:\path\to\generator_v1 --output_dir speech\test_out\tts
python speech\eval_wer.py --split test --system model --checkpoint speech\checkpoints\mel_mapper_best.pt --manifest_dir speech\manifests --cache_dir speech\cache --hifigan_checkpoint C:\path\to\generator_v1 --output_dir speech\test_out\model
```

## One finished recording

```bat
python speech\infer_utterance.py --wav C:\path\to\finished.wav --checkpoint speech\checkpoints\mel_mapper_best.pt --hifigan_checkpoint C:\path\to\generator_v1 --output speech\live_out.wav
```

Omit `--wav` to record from the microphone until about 600 ms of silence. That path needs `sounddevice`.
