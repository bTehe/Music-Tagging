# AML Music Tagging Project (MTAT-first, GTZAN auxiliary)

This repository is structured around the requested design:

1. MTAT as the main dataset (`multi-label tagging`).
2. Log-mel spectrograms as the main representation.
3. Short-chunk CNN as the first serious baseline.
4. One-factor-at-a-time ablations (representation, architecture/pooling, duration, transfer).
5. GTZAN only as auxiliary transfer/sanity benchmark.
6. No FMA usage in this implementation.

## 0) Prerequisites

```bash
python -m pip install -r requirements.txt
```

MTAT labels are required: you need `annotations_final.csv` in `MTAT/` (or pass `--annotations` explicitly).
Current folder already has `clip_info_final.csv` + audio clips, but labels were not present by default.

Quick download helper:

```bash
python scripts/download_mtat_annotations.py
```

## 1) Data loaders and split audit (Step 1)

```bash
python scripts/build_mtat_manifest.py --mtat-root MTAT --annotations MTAT/annotations_final.csv
```

Outputs:

- `artifacts/manifests/mtat_top50_manifest.csv`
- `artifacts/manifests/mtat_top50_tags.json`
- `artifacts/manifests/mtat_split_audit.json`
- dropped rows logs for missing audio/labels

Split hygiene:

- Track-level grouping (no segment leakage across train/val/test)
- Uses official `train/valid/test clip ids` when available
- Falls back to deterministic grouped split if official lists are missing

## 2) MTAT log-mel extraction pipeline (Step 2)

```bash
python scripts/extract_logmel.py --manifest artifacts/manifests/mtat_top50_manifest.csv
```

This is optional because training also supports on-the-fly log-mel.

## 3) Short-chunk CNN baseline (Step 3)

```bash
python scripts/train_mtat_cnn.py \
  --manifest artifacts/manifests/mtat_top50_manifest.csv \
  --tags artifacts/manifests/mtat_top50_tags.json \
  --output-dir artifacts/runs/mtat_cnn_baseline
```

## 4) Track-level pooling + metric suite + edge cases (Steps 4, 5, 6)

```bash
python scripts/evaluate_mtat.py \
  --manifest artifacts/manifests/mtat_top50_manifest.csv \
  --tags artifacts/manifests/mtat_top50_tags.json \
  --checkpoint artifacts/runs/mtat_cnn_baseline/best.pt \
  --output-dir artifacts/reports/mtat_eval
```

Includes:

- Pooling: `mean`, `max`, `attention`
- Metrics: ROC-AUC, PR-AUC, mAP + per-tag stats
- Duration sweep: `0.5s, 1s, 2s, 4s, 8s` by default
- Rare-tag bucket analysis
- Robustness perturbations: noise, time-stretch, compression

Optional qualitative explanation maps:

```bash
python scripts/export_saliency.py \
  --manifest artifacts/manifests/mtat_top50_manifest.csv \
  --tags artifacts/manifests/mtat_top50_tags.json \
  --checkpoint artifacts/runs/mtat_cnn_baseline/best.pt
```

## 5) Raw waveform branch (Step 7)

```bash
python scripts/train_mtat_waveform.py \
  --manifest artifacts/manifests/mtat_top50_manifest.csv \
  --tags artifacts/manifests/mtat_top50_tags.json \
  --output-dir artifacts/runs/mtat_waveform
```

## 6) Pretrained branch: PANNs/musicnn probe (Step 8)

```bash
python scripts/train_mtat_pretrained.py \
  --manifest artifacts/manifests/mtat_top50_manifest.csv \
  --tags artifacts/manifests/mtat_top50_tags.json \
  --backend panns \
  --output-dir artifacts/runs/mtat_pretrained
```

Backend options:

- `panns`
- `musicnn`

Compatibility note:

- On Python `3.13`, `musicnn` is typically unavailable due legacy dependency constraints.
- If `--backend musicnn` is requested and `musicnn` cannot be loaded, the script falls back to `panns` by default.
- Use `--strict-backend` to disable fallback and fail fast instead.

## 7) GTZAN transfer experiment (Step 10)

```bash
python scripts/train_transfer_gtzan.py \
  --gtzan-root GTZAN \
  --mtat-checkpoint artifacts/runs/mtat_cnn_baseline/best.pt \
  --output-dir artifacts/runs/gtzan_transfer
```

This implements transfer as representation probing:

- Encoder pretrained on MTAT
- Frozen feature extraction on GTZAN
- Linear classifier on GTZAN labels
- Comparison against a random-initialized encoder baseline

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
