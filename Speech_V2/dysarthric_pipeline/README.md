# Dysarthric speech pipeline

This program listens to speech that is hard to understand, writes down the words, and can speak those words back in a clear preset voice.

It does not try to copy the speaker's own voice. The spoken reply always uses the same Piper voice, `en_US-lessac-medium`.

## How the pipeline works

```text
audio in  ->  clean the sound  ->  find the speech  ->  read the words  ->  text
                                                                      \->  clear voice (optional)
```

1. **Audio in.** The sound can come from a WAV file, the computer microphone, or the FRDM-MCXN236 board over USB serial.
2. **Clean the sound.** A high-pass filter removes low rumble (default cutoff: 80 Hz).
3. **Find the speech.** Silero voice detection waits through pauses. Dysarthric speech often has longer gaps, so the pause has to last 700 ms before an utterance is finished. Each utterance also keeps 300 ms of sound before the detected start.
4. **Trim silence.** Extra silence at the ends is cut off before recognition. This matters for short words, where leftover silence makes the recognizer guess extra words.
5. **Read the words.** Whisper Small English, with a LoRA adapter trained on TORGO and UA-Speech, turns the utterance into text. Beam search (3 beams) is used at run time, and the answer is limited to 24 new tokens.
6. **Optional voice.** Piper speaks the text. You can print the text, play the voice, or save a WAV file.

The general adapter is `checkpoints/adapter_best`. A person-specific adapter, if you have trained one, can be loaded with `--speaker`.

### What the model was trained on

The recordings live in your Downloads folder as `dysarthric_speech` (`paths.data_root` in the config, or the `DYSARTHRIC_DATA_ROOT` environment variable).

- **TORGO** has real dysarthric speech and control speech, with written prompts.
- **UA-Speech** has repeated words and commands. The word list was built by listening to clear control speakers. 27 uncommon word codes were left out because the listeners did not agree.
- Speaker names include the corpus, so `torgo:F03` and `uaspeech:F03` are different people.
- The test speakers, who were never used for training, are `torgo:F03` (moderate) and `torgo:M02` (severe).

On that held-out test set the current adapter has about **23% word error**. Moderate speech is about **17%**. Severe speech is about **31%**. Very hard speech is still weak (about **62%** word error on the very-low validation band).

## Setup

From the repository root, with Python 3.10 or newer:

```bat
python -m pip install -r requirements.txt
cd Speech_V2\dysarthric_pipeline
python -m pip install -e .
python -m dysarthric_pipeline check
```

`requirements.txt` installs the recognizer, microphone, serial port, and Piper voice. On an NVIDIA GPU, install the CUDA build of PyTorch from pytorch.org before that requirements file.

All commands below are run from `Speech_V2\dysarthric_pipeline`:

```bat
python -m dysarthric_pipeline <command>
```

Settings live in `dysarthric_pipeline/configs/default.yaml`. Pass `--config other.yaml` to use a copy.

## Commands

### check

Confirms that the data folder is there, the closed word list loads, and the GPU is visible.

```bash
python -m dysarthric_pipeline check
python -m dysarthric_pipeline check --data-root /path/to/dysarthric_speech
```

### prepare

Builds the training lists in `manifests/`. The first run downloads Whisper and labels UA-Speech words from control speakers `CF02`, `CF03`, and `CM01` (microphone M5). Later runs reuse the saved word list.

```bash
python -m dysarthric_pipeline prepare
```

Files written:

| File | Meaning |
|---|---|
| `manifests/train.jsonl` | Clips used to train |
| `manifests/validation.jsonl` | Clips used to pick the best checkpoint |
| `manifests/test.jsonl` | Held-out TORGO speakers only |
| `manifests/adapt.jsonl` | Clips used to fit one speaker |
| `manifests/adapt_test.jsonl` | Held-out clips for that speaker |
| `dysarthric_pipeline/data/uaspeech_lexicon.json` | Word code to text |

### train

Fine-tunes the general adapter and writes scores.

```bash
python -m dysarthric_pipeline train
python -m dysarthric_pipeline train --baseline-only
python -m dysarthric_pipeline train --skip-baseline
python -m dysarthric_pipeline train --epochs 4
```

- `--baseline-only` scores the original Whisper model and stops.
- `--skip-baseline` starts training immediately. If an older baseline file exists, it is copied into the new history.
- The best checkpoint is the one with the lowest word error on **dysarthric** validation clips, not the overall score. Control clips would otherwise dominate the choice.

Files written in `checkpoints/`:

| File | Meaning |
|---|---|
| `adapter_best/` | Adapter to use for recognition |
| `adapter_last/` | Adapter from the final epoch |
| `adapter_v1/` | The earlier 2-epoch adapter, kept for comparison |
| `baseline_wer.json` | Word error before fine-tuning |
| `history.json` | Loss and scores for each epoch |
| `history_v1.json` | History from the earlier run |

Training uses speed changes (0.9×, 1.0×, 1.1×), SpecAugment, and extra draws of severe and very-low clips. The learning rate warms up for 200 steps and then falls on a cosine curve. The run uses 4 epochs, batch size 4, and gradient accumulation 4.

### adapt

Continues training on one speaker, starting from `adapter_best`.

```bash
python -m dysarthric_pipeline adapt --speaker uaspeech:M08
python -m dysarthric_pipeline adapt --speaker torgo:F01 --calibration-dir /path/to/wav_and_txt
```

`--speaker` must look like `uaspeech:M05` or `torgo:F01`. A calibration folder needs one `.txt` file with the same name as each `.wav`. The text should be the words that were spoken.

The adapter is saved in `adapters/`, with `:` replaced by `_`. Example: `adapters/uaspeech_M08/`. `adapt_report.json` in that folder shows word error before and after adaptation.

On the speakers we tried, adaptation helped the low band, barely changed the high and mid bands, and made the very-low speaker `uaspeech:M04` worse. Do not use a speaker adapter when its report shows a higher word error than the "before" score.

### run

Listens, prints the text, and can speak it.

```bash
python -m dysarthric_pipeline run --source file --input /path/to/clip.wav --output text
python -m dysarthric_pipeline run --source file --input /path/to/clip.wav --output both --save-dir /tmp/spoken
python -m dysarthric_pipeline run --source mic --output text
python -m dysarthric_pipeline run --source mic --output both --play
python -m dysarthric_pipeline run --source serial --output text
python -m dysarthric_pipeline run --source file --input clip.wav --speaker uaspeech:M08 --output text
```

| Option | Meaning |
|---|---|
| `--source file` | Read a WAV file. `--input` is required. Any sample rate is converted to 16 kHz. |
| `--source mic` | Use the default microphone. Press Ctrl+C to stop. |
| `--source serial` | Read the board on `serial.port` (default `auto`: a USB COM port on Windows, `/dev/ttyACM0` on Linux, 921600 baud). |
| `--output text` | Print text only. |
| `--output voice` | Speak or save audio, and still print the text. |
| `--output both` | Same as voice for this program: text plus the preset voice. This is the config default. |
| `--play` | Play the voice through the speakers. |
| `--save-dir` | Write `utterance_001.wav`, `utterance_002.wav`, and so on. |
| `--speaker` | Load that speaker's adapter instead of the general one. |

Each finished utterance is printed as `[1] the words`. The first voice run downloads the Piper files into `voices/` (about 60 MB).

A checked example: TORGO `F03` clip `0005.wav`, whose prompt is "beta", printed `[1] beta` and wrote a 22,050 Hz WAV when `--output both` was used.

## Board audio format

`serial.mode: framed` (the default) expects this packet:

```text
0xAA 0x55 | length (uint16, little endian) | PCM samples (int16, little endian) | CRC-16
```

The CRC is CRC-16/CCITT-FALSE over the PCM bytes only. The audio inside the packet is 16 kHz, mono. `serial.mode: raw` is a plain stream of those int16 samples, with no header.

## Config, in plain language

Edit `dysarthric_pipeline/configs/default.yaml`.

| Setting | What it changes |
|---|---|
| `audio.sample_rate` | 16000. Everything is converted to this rate. |
| `audio.highpass_hz` | Low-rumble cutoff. `80` is the default. `0` turns it off. |
| `audio.frame_ms` | Size of each live audio block. `30` ms. |
| `audio.trim_db` / `trim_pad_ms` | How much edge silence to cut, and how much to keep. |
| `vad.threshold` | How sure the detector must be. Lower catches quieter speech and also more noise. |
| `vad.min_speech_ms` | Ignore bursts shorter than this. |
| `vad.min_silence_ms` | Pause length that ends an utterance. |
| `vad.speech_pad_ms` | Sound kept before the detected start. |
| `vad.max_utterance_s` | Force a cut after this many seconds. |
| `serial.*` | Port, speed, framed or raw. |
| `asr.base_model` | `openai/whisper-small.en`. |
| `asr.lora_*` and `target_modules` | Size and location of the trained adapter. |
| `train.*` | Epochs, batch size, learning rate, sampling weights, and decoding. |
| `adapt.*` | Steps and learning rate for one speaker. |
| `tts.voice` | Piper voice name. |
| `paths.data_root` | Folder of the original recordings. |
| `output.mode` | `text`, `voice`, or `both`, when `--output` is not passed. |
| `splits.held_out_speakers` | Speakers kept out of training. |

## Folders

| Path | What is in it |
|---|---|
| `dysarthric_pipeline/` | The program |
| `manifests/` | Training lists. |
| `checkpoints/` | The general adapter and scores. |
| `adapters/` | One folder per adapted speaker. |
| `voices/` | The Piper voice files. |

## Limits

- Severe and very-low speech still has a high word error.
- A speaker adapter can get worse. Read `adapt_report.json` before using it.
- The microphone and the board were implemented and the framed serial format was checked in software. They were not run on the physical FRDM-MCXN236 in this session.
- Live recognition loads the model first, so the first utterance waits until Whisper and Silero are in memory.
