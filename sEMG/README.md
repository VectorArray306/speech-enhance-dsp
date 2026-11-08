# Phase 2: EMG → mel (1D-CNN)

Trains a **1D-CNN** on [Gaddy & Klein](https://doi.org/10.5281/zenodo.4064408) EMG to predict **HiFi-GAN log-mel** targets. HiFi-GAN stays **frozen** for listening tests in `validate.py`.

## Layout

| File | Role |
|------|------|
| `emg_preprocess.py` | Bandpass 20–450 Hz, 50 Hz notch, drift high-pass |
| `mel_from_audio.py` | Same mel as `../hifi-gan/meldataset.py` (`config_v1.json`) |
| `gaddy_io.py` | Find `*_emg.npy` + `*_info.json` (skip `sentence_index: -1`) |
| `align.py` | EMG samples → one vector per mel frame |
| `dataset.py` | PyTorch dataset + EMG normalization stats |
| `model.py` | `EMGToMelCNN` |
| `train.py` | L1 training |
| `validate.py` | Val L1 + vocode pred/target/original wavs |
| `hifigan_vocode.py` | Load generator checkpoint |

## Data

Download Gaddy EMG into e.g. `emg_data/` with the usual layout:

```
emg_data/voiced_parallel_data/<session>/
  0_emg.npy
  0_audio_clean.flac
  0_info.json
  ...
```

## Train

From the repository root:

```bat
python -m pip install -r requirements.txt
python sEMG\train.py --emg_data_dir C:\path\to\emg_data --output_dir sEMG\checkpoints
```

Use `--notch_hz 60` if recordings have US mains hum.

### Fine-tune on non-parallel voiced data

After training on `voiced_parallel_data`, adapt on Gaddy **non-parallel** read speech (folder `nonparallel_data/`). Reuses EMG normalization from the parallel run by default; use a lower learning rate.

```bash
python sEMG/train.py \
  --emg_data_dir /path/to/emg_data/nonparallel_data \
  --output_dir sEMG/checkpoints_nonparallel \
  --init_checkpoint sEMG/checkpoints_voiced/emg_to_mel_best.pt \
  --lr 1e-4 \
  --epochs 20
```

Add `--recompute_emg_norm` to fit new mean/std on non-parallel EMG instead.

## Validate + listen

Download **UNIVERSAL_V1** generator weights into e.g. `hifi-gan/checkpoints/` (with `config.json` beside the `.pt` file).

```bash
python sEMG/validate.py \
  --emg_data_dir /path/to/emg_data \
  --emg_checkpoint sEMG/checkpoints/emg_to_mel_best.pt \
  --hifigan_checkpoint /path/to/generator_v1 \
  --output_dir sEMG/validation_out
```

Compare `*_emg_pred.wav` (EMG → CNN → mel → HiFi-GAN) vs `*_mel_target.wav` and `*_original.wav`.
