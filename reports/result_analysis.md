# Result analysis report (MTAT, GTZAN)

## Scope and evidence policy

This report analyzes the current repository outputs with explicit evidence only.
Each claim below is tied to a generated artifact in `artifacts/analysis/`.
Language is intentionally cautious: observed associations are not treated as
proven causal effects.

Primary sources used in this report:

- `artifacts/analysis/tables/unified_evaluation_table.csv`
- `artifacts/analysis/tables/tag_performance_analysis.csv`
- `artifacts/analysis/tables/pooling_duration_overall.csv`
- `artifacts/analysis/tables/pooling_duration_per_tag.csv`
- `artifacts/analysis/tables/robustness_overall.csv`
- `artifacts/analysis/tables/robustness_per_tag.csv`
- `artifacts/analysis/tables/tag_split_counts.csv`
- `artifacts/analysis/tables/split_prevalence_shift.csv`
- `artifacts/analysis/tables/tag_cooccurrence_stats.csv`
- `artifacts/analysis/tables/failure_cases.csv`
- `artifacts/analysis/tables/gtzan_per_genre_f1.csv`
- `artifacts/analysis/tables/gtzan_confusion_matrix.csv`

## Experimental context

- Main task: MTAT Top-50 multi-label tagging.
- Train/val/test clips: 16,960 / 2,115 / 2,036.
- Track-level test evaluation: 529 tracks (grouped by `track_id`).
- Compared branches:
  - `ShortChunkCNN` with log-mel input.
  - `WaveformCNN` with raw waveform input.
  - PANNs embedding + logistic regression probe.
- Pooling note:
  - In-model pooling: `AdaptiveAvgPool2d` (ShortChunkCNN) and
    `AdaptiveAvgPool1d` (WaveformCNN).
  - Track aggregation pooling across chunk predictions: `mean`, `max`, and
    the repo's current `attention` mode.
  - Important: current `attention` is heuristic confidence weighting from
    [`pooling.py`](../src/aml_music/models/pooling.py), not a learned attention
    module. In this report it is referred to as
    `heuristic_attention`.

## A) Representation: why log-mel outperformed waveform

Evidence:

- MTAT validation best mAP:
  - log-mel CNN: 0.408 (`artifacts/runs/mtat_cnn_baseline/summary.json`)
  - waveform CNN: 0.255 (`artifacts/runs/mtat_waveform/summary.json`)
- Best-epoch validation ROC/PR:
  - log-mel ROC-AUC 0.879, PR-AUC 0.403
  - waveform ROC-AUC 0.791, PR-AUC 0.250
  from [`training_dynamics_comparison.png`](../artifacts/analysis/plots/training_dynamics_comparison.png)
  and `unified_evaluation_table.csv`.
- Convergence:
  - log-mel peaked earlier (best epoch 16, early-stopped at 21),
  - waveform improved slower and plateaued lower by epoch 25.
- Overall metric comparison figure:
  [`overall_metric_comparison_bars.png`](../artifacts/analysis/plots/overall_metric_comparison_bars.png)

Interpretation:

- The gap is consistent with stronger inductive bias of log-mel for MTAT scale.
- The gap is also consistent with optimization difficulty in raw waveform.
- Implementation simplicity likely contributes:
  - log-mel frontend compresses dynamic range and stabilizes frequency structure,
    while waveform model must learn this from scratch.
- Capacity is not a likely main explanation alone:
  - waveform model is smaller (0.84M params) than log-mel CNN (1.19M params),
    so this is not a matched-capacity test.
  - See `model_architecture_summary.csv`.

What is justified:
- "Log-mel is the stronger representation in this repo under this training
  budget."

What is not justified:
- "Waveform is intrinsically worse for music tagging."

## B) Architecture effects

Evidence:

- ShortChunkCNN structure:
  - 4 conv blocks, each with two 3x3 convs + max pooling.
  - global average pooling over final map before classifier.
  - params: 1,185,490 (`model_architecture_summary.csv`).
  - approximate receptive field at classifier input is about `76 x 76`
    (time x mel bins) in spectrogram coordinates, based on 8 conv(3x3) layers
    and four 2x2 pooling stages.
  - with `hop_length=512` at 16 kHz (~32 ms/frame), 76 time frames correspond
    to roughly 2.4 s local context before track-level aggregation.
- WaveformCNN:
  - stacked 1D conv blocks with stride-2 downsampling + global average pooling.
- Tag behavior:
  - strong AP on local/acoustic tags (for example `rock`, `guitar`, `classical`)
    and weaker AP on ambiguous voice-negation tags (`no voice`, `no vocal`).
  - See `tag_best12.csv` and `tag_worst12.csv`.

Interpretation:

- The architecture likely favors local timbral evidence:
  repeated local convolution and global averaging are well-suited to texture-like
  cues.
- Global average pooling can blur temporal arrangement cues after feature
  extraction. This is consistent with stronger performance on tags needing local
  spectral cues vs weaker performance on broad/semantic/absence tags.
- Distinction required by supervisor:
  - Backbone pooling (`AdaptiveAvgPool`) compresses feature maps inside model.
  - Track pooling (`mean`/`max`/`heuristic_attention`) aggregates chunk outputs
    across windows.

What is justified:
- "Current model likely underuses long-range temporal structure relative to what
  a stronger temporal backend could capture."

What is not justified:
- "Global average pooling is the sole cause of weak tags."

## C) Pooling and duration effects

Evidence:

- Best test setting: `4.0s + max`, mAP 0.623, PR-AUC 0.615, ROC-AUC 0.907
  (`pooling_duration_overall.csv`).
- `8.0s + max` was close (mAP 0.621) but not better.
- `max` consistently dominated `mean` and `heuristic_attention` at each tested
  duration in mAP.
- Plot:
  [`duration_pooling_metrics.png`](../artifacts/analysis/plots/duration_pooling_metrics.png)
- Per-tag heatmap:
  [`duration_pooling_per_tag_ap_heatmap.png`](../artifacts/analysis/plots/duration_pooling_per_tag_ap_heatmap.png)
- Tags with largest long-context gain (`8.0s max - 0.5s mean`):
  `male vocal`, `male voice`, `ambient`, `man`, `female`.

Interpretation:

- Max pooling performing best is consistent with "one strong local cue is
  enough" for many tags.
- Duration helps up to moderate context (around 4s), then saturates.
- Some tags are context-sensitive (especially several vocal identity tags), while
  others are stable with short context or even slightly worse at long context.

What is justified:
- "A mix of cue-local and context-sensitive tags exists; max pooling benefits
  cue-local tags."

What is not justified:
- "All tags should use the same optimal context length."

## D) Robustness

Evidence:

- Overall drop from clean:
  - clean mAP 0.595
  - noise mAP 0.548 (drop 0.047)
  - time-stretch mAP 0.576 (drop 0.019)
  - compression mAP 0.528 (drop 0.067)
  from `robustness_overall.csv`.
- Plot:
  [`robustness_overall_metrics.png`](../artifacts/analysis/plots/robustness_overall_metrics.png)
- Most fragile tags by mean AP drop:
  `female`, `male voice`, `male vocal`, `female voice`, `woman`, `female vocal`,
  `ambient`, `sitar`.
  from `robustness_per_tag.csv` and
  [`robustness_top_tag_degradation_heatmap.png`](../artifacts/analysis/plots/robustness_top_tag_degradation_heatmap.png).

Interpretation:

- Compression caused the largest degradation, then additive noise, then
  time-stretch. This pattern is consistent with reliance on spectral envelope
  and energy cues.
- Mild time-stretch hurting least suggests some tolerance to tempo deformation
  but lower tolerance to timbre/dynamics distortion.

What is justified:
- "Current classifier is more sensitive to timbre/dynamics perturbation than to
  mild timing perturbation."

What is not justified:
- "Model explicitly learned semantic rhythm invariance."

## E) Underperforming tags and their characteristics

Worst tags by AP (`tag_worst12.csv`):

- `no voice` (vocal)
- `no vocal` (vocal)
- `voice` (vocal)
- `no vocals` (vocal)
- `weird` (other)
- `solo` (production)
- `soft` (mood)
- `sitar` (instrument)

Observed explanatory factors:

- Low or moderate support for several weak tags, but not all:
  scarcity alone does not explain all failures (`tag_split_counts.csv`).
- Strong co-occurrence ambiguity:
  high co-occurrence entropy is associated with lower AP
  (`pearson_co_entropy_vs_ap = -0.289`).
- Negative vocal tags are semantically tricky:
  `no voice` / `no vocal` depend on absence detection and can conflict with
  weakly supervised segment-level labels.
- Split sensitivity for some tags:
  `sitar` has strong prevalence shift (test/train ratio 0.56,
  max shift 0.44), which can destabilize estimates.
- Robustness sensitivity contributes for specific tags:
  `sitar`, `soft`, `quiet` show notable perturbation drops.

Tag-type grouped performance:

- Group plot:
  [`tag_group_ap_boxplot.png`](../artifacts/analysis/plots/tag_group_ap_boxplot.png)
- Frequency/performance scatter:
  [`tag_frequency_vs_ap_scatter.png`](../artifacts/analysis/plots/tag_frequency_vs_ap_scatter.png)
- Correlation summary:
  - train_count vs AP (pearson/spearman): ~0.546 / ~0.546
  - co-occurrence degree vs AP (spearman): ~-0.356
  from `tag_correlation_stats.csv`.

## F) Dataset and split effects

Evidence:

- Split prevalence shifts are non-trivial for a subset of tags.
- Largest shift examples:
  `sitar`, `ambient`, `choir`, `choral`, `harpsichord`.
  from `split_prevalence_shift.csv`.
- Co-occurrence matrix:
  [`tag_cooccurrence_heatmap.png`](../artifacts/analysis/plots/tag_cooccurrence_heatmap.png)
- GTZAN transfer:
  transfer probe test accuracy 0.73 vs random probe 0.21
  (`artifacts/runs/gtzan_transfer/summary.json`).

Interpretation:

- MTAT Top-50 outcomes are partly driven by label prevalence and co-occurrence
  geometry, not just model capacity.
- Some tag-level wins/losses likely reflect support and split composition.
- GTZAN transfer supports reusable encoder structure, but not universal music
  understanding.
- Genre difficulty on GTZAN is uneven:
  hardest test genres: `rock`, `reggae`, `pop`;
  easiest: `classical`, `jazz`, `country`
  (`gtzan_per_genre_f1.csv`,
  [`gtzan_transfer_confusion_matrix.png`](../artifacts/analysis/plots/gtzan_transfer_confusion_matrix.png)).

What is justified:
- "Transfer success indicates useful reusable cues."

What is not justified:
- "Transfer result proves broad domain-general semantic understanding."

## G) Failure cases

Evidence artifacts:

- Case table: `failure_cases.csv`
- Saliency gallery (selected files):
  - `failure_saliency_104_no_voice.png`
  - `failure_saliency_447_no_vocal.png`
  - `failure_saliency_31_weird.png`
  - `failure_saliency_327_soft.png`

Observed patterns from mined cases:

- High-confidence false positives often occur on weak/ambiguous tags (`soft`,
  `solo`, `sitar`, `voice`) where cues can overlap with correlated labels.
- False negatives for `no vocal`/`no vocals` and `weird` appear consistent with
  weak supervision and context mismatch.

Interpretation caution:

- Saliency plots are diagnostic only. They show sensitivity patterns, not causal
  proof of semantic reasoning.

## Calibration evidence

- Reliability data in `calibration_bins.csv`.
- Plot:
  [`calibration_reliability_plot.png`](../artifacts/analysis/plots/calibration_reliability_plot.png)
- Flattened-tag ECE (test, best duration/pooling): ~0.0187
  (`analysis_master_summary.json`).

Interpretation:

- Global calibration appears reasonable on average, but this can hide per-tag
  miscalibration under heavy class imbalance.

## Main answers to supervisor questions

1. Why current results were obtained:
   dominant factors are representation bias (log-mel), moderate-context pooling
   behavior (4s + max), and label/co-occurrence structure.
2. Most influential factors:
   representation, pooling choice, tag frequency, and co-occurrence ambiguity.
3. Architecture vs representation vs protocol:
   representation and data structure explain more variance than architecture
   differences tested here, with architecture still contributing to context
   limits.
4. Why some tags underperform:
   weak supervision + ambiguity + split shifts + perturbation fragility, varying
   by tag.
5. Underperforming tag profile:
   concentrated in vocal-negation, broad semantic descriptors, and some sparse
   instrument/style tags.
6. Justified vs not justified:
   strong local conclusions are justified for this repo/protocol; global claims
   about universal model superiority are not.

## Reproducibility commands

```bash
python scripts/analyze_split_and_label_structure.py --output-dir artifacts/analysis
python scripts/analyze_pooling_duration.py --output-dir artifacts/analysis
python scripts/analyze_robustness.py --output-dir artifacts/analysis --device cpu
python scripts/analyze_tags.py --output-dir artifacts/analysis
python scripts/analyze_failure_cases.py --output-dir artifacts/analysis --device cpu
python scripts/analyze_results.py --output-dir artifacts/analysis --skip-subscripts --device cpu
```

## Created/modified files

- `reports/result_analysis.md`
- `scripts/analyze_results.py`
- `scripts/analyze_tags.py`
- `scripts/analyze_pooling_duration.py`
- `scripts/analyze_robustness.py`
- `scripts/analyze_split_and_label_structure.py`
- `scripts/analyze_failure_cases.py`
- `src/aml_music/analysis/common.py`
- `src/aml_music/analysis/__init__.py`

Generated outputs are indexed in:
- `artifacts/analysis/json/generated_artifacts_index.json`

## Top 5 strongest conclusions

1. Under the current training budget and implementation, log-mel + ShortChunkCNN is clearly stronger than the waveform branch on MTAT validation metrics and convergence behavior.
2. For track-level MTAT evaluation, `4.0s + max` is the strongest tested pooling/duration setting; gains from longer windows saturate after moderate context.
3. Tag frequency is important but not sufficient: AP rises with train support, but co-occurrence complexity and split shifts still explain additional weak tags.
4. Robustness losses are largest under compression and noise, consistent with reliance on spectral/timbral cues.
5. GTZAN transfer indicates reusable encoder features (clear gap vs random encoder), but this is not evidence of universal music semantics.

## Top 5 remaining uncertainties

1. Representation fairness is limited by unmatched model capacity and optimization settings between log-mel and waveform branches.
2. Weak supervision and possible label noise in MTAT prevent clean causal attribution for low-performing semantic/negation tags.
3. The current `heuristic_attention` is not learned attention, so conclusions about attention-based aggregation are limited.
4. Robustness analysis uses a compact perturbation suite; broader distortions and codec artifacts may change rank order of fragile tags.
5. GTZAN conclusions remain sensitive to dataset artifacts and genre overlap; cross-dataset generalization beyond GTZAN is still uncertain.
