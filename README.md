# AML Music Tagging Project (MTAT, GTZAN)

The main task is multi-label music tagging on MagnaTagATune (MTAT). GTZAN is used only as a small
transfer/sanity benchmark.

## Team

All team members contributed equally:

- Oleksandr Adamov
- Vladislav Konjushenko
- Costel Gutu

## Central problem

The project asks a simple question: can we build a model that listens to music
clips and predicts useful tags such as genre, instruments, mood, or other
descriptors?

This is not plain genre classification. MTAT is a multi-label dataset, so one
clip can have many correct tags at the same time. That changes the setup:

- outputs are sigmoid probabilities, one per tag
- the loss is binary cross-entropy
- ROC-AUC, PR-AUC, and mAP are more useful than plain accuracy
- class imbalance matters, especially for rare tags

The second practical problem is variable length. Training is easier on fixed
short chunks, but real music files can be longer. We handle this by training on
short windows and using sliding-window inference with pooling across windows.

## Data

### MTAT: main dataset

MTAT is the main benchmark because it is a standard multi-label music tagging
dataset. We use the Top-50 tag setup.

In our manifest:

| Split | Clips |
|---|---:|
| Train | 16,960 |
| Validation | 2,115 |
| Test | 2,036 |

The manifest builder keeps split hygiene at track level. This matters because
segments from the same song must not leak across train, validation, and test.

### GTZAN: auxiliary dataset

GTZAN is used for the transfer experiment. It has 1,000 30-second clips across
10 genres, but it is known to contain duplicates, mislabeled examples, and some
broken audio. Because of that, we do not use it as the main source of claims.

The transfer script filters unreadable GTZAN files before embedding extraction.
In the current run, one GTZAN file failed the precheck and was excluded.

## Central method

The main model is:

```text
audio -> log-mel spectrogram -> short-chunk CNN -> sigmoid tag probabilities
```

Why this setup:

- Log-mel spectrograms are a strong default for music tagging. They give the CNN
  a stable time-frequency view instead of forcing it to learn everything from
  raw waveform.
- Short chunks create more training examples and work well for many tags such as
  instruments, vocals, texture, and genre.
- The CNN is small enough to train locally and still strong enough to act as a
  serious baseline.
- Track-level predictions are made by running a sliding window and pooling
  chunk predictions with mean, max, or attention pooling.

The main implementation pieces are:

- `src/aml_music/audio.py`: audio loading, padding/cropping, chunking
- `src/aml_music/features/logmel.py`: log-mel frontend
- `src/aml_music/models/short_chunk_cnn.py`: main CNN model
- `src/aml_music/models/waveform_cnn.py`: raw-waveform ablation model
- `src/aml_music/evaluation/metrics.py`: ROC-AUC, PR-AUC, mAP, genre metrics
- `scripts/train_mtat_cnn.py`: main baseline training
- `scripts/evaluate_mtat.py`: track-level evaluation and robustness tests
- `scripts/train_transfer_gtzan.py`: MTAT-to-GTZAN transfer probe

## Key experiments and results

### 1. Main MTAT baseline

The baseline: log-mel input, short chunks, CNN backbone.

| Model | Validation mAP | Best epoch |
|---|---:|---:|
| Short-chunk CNN, log-mel | 0.408 | 16 |

The baseline is stable and clearly better than the raw-waveform branch under the
same local training budget.

### 2. Representation ablation

Here we change the input representation while keeping the task the same.

| Representation | Model | Validation mAP |
|---|---|---:|
| Log-mel | Short-chunk CNN | 0.408 |
| Raw waveform | Waveform CNN | 0.255 |

The result matches the literature we reviewed: waveform models can work, but they usually
need much more data or stronger pretraining. For this project scale, log-mel is
the safer representation.

### 3. Track-level pooling and duration sweep

The evaluation script tests several chunk durations and pooling methods. Best
observed mAP in the current run:

| Duration | Pooling | ROC-AUC | PR-AUC | mAP |
|---:|---|---:|---:|---:|
| 0.5 s | max | 0.872 | 0.544 | 0.552 |
| 1.0 s | max | 0.893 | 0.593 | 0.601 |
| 2.0 s | max | 0.902 | 0.602 | 0.610 |
| 4.0 s | max | 0.907 | 0.615 | 0.623 |
| 8.0 s | max | 0.909 | 0.613 | 0.621 |

The best result comes from 4-second chunks with max pooling. Longer context
helps up to a point, but the 8-second setting does not improve mAP further.

### 4. Robustness checks

The same trained model is evaluated after simple audio perturbations.

| Condition | ROC-AUC | PR-AUC | mAP |
|---|---:|---:|---:|
| Clean | 0.902 | 0.586 | 0.595 |
| Noise, SNR 20 dB | 0.880 | 0.539 | 0.548 |
| Time-stretch 1.1x | 0.895 | 0.566 | 0.576 |
| Dynamic compression | 0.869 | 0.519 | 0.528 |

Noise and compression hurt more than mild time stretching. This suggests the
model relies strongly on local timbre and spectral energy patterns.

### 5. Pretrained encoder branch

The pretrained branch uses PANNs embeddings and trains a lightweight classifier.

| Backend | Split | ROC-AUC | PR-AUC | mAP |
|---|---|---:|---:|---:|
| PANNs | Validation | 0.911 | 0.466 | 0.471 |
| PANNs | Test | 0.906 | 0.445 | 0.451 |

This branch performs well without training a full audio model from scratch. It
also gives a useful comparison point for transfer learning.

### 6. GTZAN transfer experiment

This tests whether the MTAT encoder learns reusable music features. We compare a
frozen MTAT encoder against a random CNN encoder, then train the same linear
classifier on GTZAN genre labels.

| Encoder | Split | Accuracy | Balanced accuracy | Macro F1 |
|---|---|---:|---:|---:|
| MTAT encoder | Validation | 0.750 | 0.750 | 0.752 |
| Random encoder | Validation | 0.190 | 0.190 | 0.077 |
| MTAT encoder | Test | 0.730 | 0.730 | 0.723 |
| Random encoder | Test | 0.210 | 0.210 | 0.088 |

The MTAT encoder transfers much better than a random encoder. This is the
clearest evidence that the main model learns reusable musical structure, not
just MTAT-specific labels.

## Discussion

The strongest result is that the simple log-mel short-chunk CNN is a good
baseline. It trains reliably, handles variable-length inference with pooling,
and transfers well enough to GTZAN to beat a random encoder by a large margin.

The representation ablation is also clear: raw waveform underperforms log-mel
in our local setup. That does not mean waveform is bad in general. It means that
with MTAT-sized training and limited compute, the inductive bias of log-mel is
helpful.

Pooling matters. Max pooling gave the best track-level result in the duration
sweep, probably because many tags only need one strong local cue somewhere in a
track. Mean pooling is more conservative, and attention pooling did not beat max
pooling in the current run.

The robustness results show where the model is fragile. Dynamic compression and
noise reduce mAP noticeably. A next version should add augmentation during
training, especially noise, compression, time stretching, and possibly
SpecAugment or mixup.

The pretrained PANNs branch is useful but not a perfect replacement for the
main model. It performs strongly and is practical, but it depends on external
pretrained assets.

## What can be improved

- Add training-time augmentations and compare clean vs robust performance again.
- Add per-tag plots for rare tags, because aggregate mAP hides long-tail issues.
- Add CQT or harmonic stacking as a music-specific representation ablation.
- Try AST or PaSST if more compute is available.

## How to run

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Check the dataset folders:

```bash
python scripts/check_setup.py
```

Download MTAT annotations if needed:

```bash
python scripts/download_mtat_annotations.py
```

Build manifests:

```bash
python scripts/build_mtat_manifest.py --mtat-root MTAT --annotations MTAT/annotations_final.csv
python scripts/build_gtzan_manifest.py --gtzan-root GTZAN
```

Train the MTAT baseline:

```bash
python scripts/train_mtat_cnn.py \
  --manifest artifacts/manifests/mtat_top50_manifest.csv \
  --tags artifacts/manifests/mtat_top50_tags.json \
  --output-dir artifacts/runs/mtat_cnn_baseline
```

Evaluate the baseline:

```bash
python scripts/evaluate_mtat.py \
  --manifest artifacts/manifests/mtat_top50_manifest.csv \
  --tags artifacts/manifests/mtat_top50_tags.json \
  --checkpoint artifacts/runs/mtat_cnn_baseline/best.pt \
  --output-dir artifacts/reports/mtat_eval
```

Run the waveform ablation:

```bash
python scripts/train_mtat_waveform.py \
  --manifest artifacts/manifests/mtat_top50_manifest.csv \
  --tags artifacts/manifests/mtat_top50_tags.json \
  --output-dir artifacts/runs/mtat_waveform
```

Run the pretrained branch:

```bash
python scripts/train_mtat_pretrained.py \
  --manifest artifacts/manifests/mtat_top50_manifest.csv \
  --tags artifacts/manifests/mtat_top50_tags.json \
  --backend panns \
  --output-dir artifacts/runs/mtat_pretrained
```

Run the GTZAN transfer probe:

```bash
python scripts/train_transfer_gtzan.py \
  --gtzan-root GTZAN \
  --mtat-checkpoint artifacts/runs/mtat_cnn_baseline/best.pt \
  --output-dir artifacts/runs/gtzan_transfer
```

The notebook version of the pipeline is in:

```text
notebooks/aml_music_pipeline.ipynb
```

## Project layout

```text
src/aml_music/
  audio.py
  data/{mtat.py, gtzan.py}
  features/logmel.py
  models/{short_chunk_cnn.py, waveform_cnn.py, pooling.py}
  evaluation/{metrics.py, robustness.py}
  training.py
scripts/
  build_mtat_manifest.py
  extract_logmel.py
  train_mtat_cnn.py
  evaluate_mtat.py
  train_mtat_waveform.py
  train_mtat_pretrained.py
  train_transfer_gtzan.py
  export_saliency.py
configs/
  mtat_baseline.yaml
  mtat_waveform.yaml
  mtat_pretrained.yaml
  gtzan_transfer.yaml
```

## Literature used

The project design follows the music tagging literature rather than starting
from a generic image or NLP recipe.

- MTAT is treated as the main benchmark because it is a standard multi-label
  music tagging dataset. We follow the common Top-50 setup and use ROC-AUC,
  PR-AUC, and mAP.
- GTZAN is used cautiously because published analysis documents duplicates,
  label errors, and audio defects.
- Log-mel is the main representation because it is a strong and reproducible
  baseline for medium-sized audio tagging datasets.
- Raw waveform is included as an ablation because waveform models become more
  competitive when data scale and compute are much larger.
- Short-chunk CNNs are used because comparative work shows they are very strong
  baselines for music tagging.
- PANNs is used as the pretrained-transfer branch because large-scale audio
  pretraining can transfer well to downstream tagging tasks.

Main references:

- MARBLE: Music Audio Representation Benchmark for Universal Evaluation. https://arxiv.org/pdf/2306.10548
- End-to-end learning for music audio tagging at scale. https://ismir2018.ircam.fr/doc/pdfs/191_Paper.pdf
- An analysis of the GTZAN music genre dataset. https://vbn.aau.dk/ws/files/74499095/GTZANDB.pdf
- FMA: A Dataset for Music Analysis. https://archives.ismir.net/ismir2017/paper/000075.pdf
- Evaluation of CNN-based Automatic Music Tagging Models. https://arxiv.org/pdf/2006.00751
- Automatic Tagging Using Deep Convolutional Neural Networks. https://arxiv.org/abs/1606.00298
- musicnn: Pre-trained Convolutional Neural Networks for Music Audio Tagging. https://arxiv.org/abs/1909.06654
- AST: Audio Spectrogram Transformer. https://sls.csail.mit.edu/publications/2021/YuanGong_Interspeech-2021.pdf
- PaSST: Efficient Training of Audio Transformers with Patchout. https://www.isca-archive.org/interspeech_2022/koutini22_interspeech.pdf
- PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition. https://arxiv.org/pdf/1912.10211
- LEAF: A Learnable Frontend for Audio Classification. https://openreview.net/pdf?id=jM76BCb6F9m
- SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition. https://www.isca-archive.org/interspeech_2019/park19e_interspeech.pdf
