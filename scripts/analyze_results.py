from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from aml_music.analysis.common import (  # noqa: E402
    count_trainable_parameters,
    display_pooling_name,
    ensure_analysis_dirs,
    evaluate_track_level,
    load_manifest_and_tags,
    load_model_from_checkpoint,
)
from aml_music.models.short_chunk_cnn import ShortChunkCNN  # noqa: E402
from aml_music.models.waveform_cnn import WaveformCNN  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full result analysis pipeline.")
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/manifests/mtat_top50_manifest.csv"))
    parser.add_argument("--tags", type=Path, default=Path("artifacts/manifests/mtat_top50_tags.json"))
    parser.add_argument(
        "--logmel-checkpoint",
        type=Path,
        default=Path("artifacts/runs/mtat_cnn_baseline/best.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/analysis"),
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports"),
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--skip-subscripts", action="store_true")
    return parser.parse_args()


def _run_cmd(args: list[str]) -> None:
    proc = subprocess.run(args, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(args)}")


def _ensure_sub_analyses(python_exe: str, output_dir: Path) -> None:
    scripts = [
        "scripts/analyze_split_and_label_structure.py",
        "scripts/analyze_pooling_duration.py",
        "scripts/analyze_robustness.py",
        "scripts/analyze_tags.py",
        "scripts/analyze_failure_cases.py",
    ]
    for script in scripts:
        _run_cmd([python_exe, script, "--output-dir", str(output_dir)])


def _extract_best_val_metrics(history_path: Path, summary_path: Path) -> dict[str, float]:
    hist = pd.read_csv(history_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    best_epoch = int(summary["best_epoch"])
    row = hist.loc[hist["epoch"] == best_epoch].iloc[0]
    return {
        "val_map": float(row["val_map"]),
        "val_macro_pr_auc": float(row["val_macro_pr_auc"]),
        "val_macro_roc_auc": float(row["val_macro_roc_auc"]),
        "val_loss": float(row["val_loss"]),
    }


def _plot_training_dynamics(cnn_hist: pd.DataFrame, wave_hist: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(cnn_hist["epoch"], cnn_hist["train_loss"], label="logmel train")
    axes[0].plot(cnn_hist["epoch"], cnn_hist["val_loss"], label="logmel val")
    axes[0].plot(wave_hist["epoch"], wave_hist["train_loss"], label="waveform train")
    axes[0].plot(wave_hist["epoch"], wave_hist["val_loss"], label="waveform val")
    axes[0].set_title("Training dynamics: loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.2)
    axes[0].legend()

    axes[1].plot(cnn_hist["epoch"], cnn_hist["val_map"], marker="o", label="logmel val mAP")
    axes[1].plot(wave_hist["epoch"], wave_hist["val_map"], marker="o", label="waveform val mAP")
    axes[1].set_title("Validation mAP by epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("mAP")
    axes[1].grid(alpha=0.2)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_dir / "training_dynamics_comparison.png", dpi=180)
    plt.close(fig)


def _plot_metric_bars(metric_df: pd.DataFrame, out_path: Path) -> None:
    # Plot only MTAT rows and available metrics.
    mtat = metric_df.loc[metric_df["task"] == "mtat"].copy()
    mtat = mtat.loc[mtat["metric_name"].isin(["map", "macro_pr_auc", "macro_roc_auc"])]
    mtat = mtat.loc[mtat["split"].isin(["val", "test"])]

    if mtat.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    metric_order = ["map", "macro_pr_auc", "macro_roc_auc"]
    for ax, metric_name in zip(axes, metric_order):
        m = mtat.loc[mtat["metric_name"] == metric_name].copy()
        labels = m["experiment_name"] + "_" + m["split"]
        ax.bar(np.arange(len(m)), m["metric_value"])
        ax.set_xticks(np.arange(len(m)))
        ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=8)
        ax.set_title(metric_name)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Overall metric comparison across MTAT experiments")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _compute_calibration_plot(
    manifest: pd.DataFrame,
    tags: list[str],
    checkpoint: Path,
    pooling_duration_df: pd.DataFrame,
    output_dir: Path,
    sample_rate: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    best = pooling_duration_df.sort_values("map", ascending=False).iloc[0]
    duration = float(best["duration"])
    pooling = str(best["pooling"])
    hop = max(0.1, duration / 2.0)

    model = load_model_from_checkpoint(
        checkpoint_path=checkpoint,
        representation="logmel",
        num_tags=len(tags),
        device=device,
    )
    frame = manifest.loc[manifest["split"] == "test"].copy()
    y_true, y_pred_modes, _ = evaluate_track_level(
        model=model,
        frame=frame,
        tags=tags,
        representation="logmel",
        sample_rate=sample_rate,
        chunk_seconds=duration,
        hop_seconds=hop,
        pooling_modes=[pooling],
        batch_size=batch_size,
        device=device,
        desc="calibration_eval",
    )
    y_prob = y_pred_modes[pooling]

    probs = y_prob.reshape(-1)
    labels = y_true.reshape(-1)
    bins = np.linspace(0.0, 1.0, 11)
    bin_ids = np.digitize(probs, bins) - 1
    rows = []
    ece = 0.0
    for b in range(10):
        mask = bin_ids == b
        if mask.sum() == 0:
            rows.append({"bin": b, "confidence": float((bins[b] + bins[b + 1]) / 2), "accuracy": np.nan, "count": 0})
            continue
        conf = float(probs[mask].mean())
        acc = float(labels[mask].mean())
        count = int(mask.sum())
        ece += (count / len(probs)) * abs(acc - conf)
        rows.append({"bin": b, "confidence": conf, "accuracy": acc, "count": count})

    calib_df = pd.DataFrame(rows)
    calib_df.to_csv(output_dir / "tables" / "calibration_bins.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 6))
    valid = calib_df["count"] > 0
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", label="perfect calibration")
    ax.plot(calib_df.loc[valid, "confidence"], calib_df.loc[valid, "accuracy"], marker="o", label="model")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Empirical positive rate")
    ax.set_title("Reliability plot (all tags flattened)")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "plots" / "calibration_reliability_plot.png", dpi=180)
    plt.close(fig)

    return {"ece": float(ece), "duration": duration, "pooling": pooling}


def _gtzan_transfer_confusion(output_dir: Path) -> dict[str, object]:
    train = np.load("artifacts/runs/gtzan_transfer/transfer_embeddings_train.npz")
    val = np.load("artifacts/runs/gtzan_transfer/transfer_embeddings_val.npz")
    test = np.load("artifacts/runs/gtzan_transfer/transfer_embeddings_test.npz")
    summary = json.loads(Path("artifacts/runs/gtzan_transfer/summary.json").read_text(encoding="utf-8"))
    label_map = summary["label_mapping"]
    inv_map = {int(v): str(k) for k, v in label_map.items()}
    labels_order = [inv_map[i] for i in sorted(inv_map.keys())]

    clf = LogisticRegression(max_iter=2000, solver="lbfgs")
    clf.fit(train["x"], train["y"])
    pred_val = clf.predict(val["x"])
    pred_test = clf.predict(test["x"])

    report_val = classification_report(val["y"], pred_val, output_dict=True, zero_division=0)
    report_test = classification_report(test["y"], pred_test, output_dict=True, zero_division=0)
    cm = confusion_matrix(test["y"], pred_test, labels=np.arange(len(labels_order)))

    per_genre_rows = []
    for idx, genre in enumerate(labels_order):
        key = str(idx)
        per_genre_rows.append(
            {
                "genre": genre,
                "f1_val": float(report_val.get(key, {}).get("f1-score", np.nan)),
                "f1_test": float(report_test.get(key, {}).get("f1-score", np.nan)),
                "support_test": int(report_test.get(key, {}).get("support", 0)),
            }
        )
    per_genre_df = pd.DataFrame(per_genre_rows).sort_values("f1_test", ascending=False)
    per_genre_df.to_csv(output_dir / "tables" / "gtzan_per_genre_f1.csv", index=False)
    pd.DataFrame(cm, index=labels_order, columns=labels_order).to_csv(
        output_dir / "tables" / "gtzan_confusion_matrix.csv"
    )

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    ax.set_xticks(np.arange(len(labels_order)))
    ax.set_yticks(np.arange(len(labels_order)))
    ax.set_xticklabels(labels_order, rotation=45, ha="right")
    ax.set_yticklabels(labels_order)
    ax.set_xlabel("Predicted genre")
    ax.set_ylabel("True genre")
    ax.set_title("GTZAN transfer confusion matrix")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Count")
    fig.tight_layout()
    fig.savefig(output_dir / "plots" / "gtzan_transfer_confusion_matrix.png", dpi=180)
    plt.close(fig)

    return {
        "val_accuracy_recomputed": float((pred_val == val["y"]).mean()),
        "test_accuracy_recomputed": float((pred_test == test["y"]).mean()),
        "hardest_genres_test": per_genre_df.tail(3)["genre"].tolist(),
        "easiest_genres_test": per_genre_df.head(3)["genre"].tolist(),
    }


def main() -> None:
    args = parse_args()
    out = ensure_analysis_dirs(args.output_dir)
    reports_dir = args.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    if not args.skip_subscripts:
        _ensure_sub_analyses(sys.executable, out.base)

    manifest, tags = load_manifest_and_tags(args.manifest, args.tags)

    cnn_hist = pd.read_csv("artifacts/runs/mtat_cnn_baseline/history.csv")
    wave_hist = pd.read_csv("artifacts/runs/mtat_waveform/history.csv")
    cnn_best_val = _extract_best_val_metrics(
        history_path=Path("artifacts/runs/mtat_cnn_baseline/history.csv"),
        summary_path=Path("artifacts/runs/mtat_cnn_baseline/summary.json"),
    )
    wave_best_val = _extract_best_val_metrics(
        history_path=Path("artifacts/runs/mtat_waveform/history.csv"),
        summary_path=Path("artifacts/runs/mtat_waveform/summary.json"),
    )
    pretrained_summary = json.loads(Path("artifacts/runs/mtat_pretrained/summary.json").read_text(encoding="utf-8"))
    transfer_summary = json.loads(Path("artifacts/runs/gtzan_transfer/summary.json").read_text(encoding="utf-8"))
    pooling_duration_overall = pd.read_csv(out.tables / "pooling_duration_overall.csv")
    robustness_overall = pd.read_csv(out.tables / "robustness_overall.csv")

    # Model size / architecture table.
    n_tags = len(tags)
    short_params = count_trainable_parameters(ShortChunkCNN(num_tags=n_tags))
    wave_params = count_trainable_parameters(WaveformCNN(num_tags=n_tags))
    arch_df = pd.DataFrame(
        [
            {
                "model": "ShortChunkCNN",
                "representation": "logmel",
                "trainable_params": short_params,
                "backbone_pooling": "AdaptiveAvgPool2d(1,1)",
                "track_pooling": "mean/max/heuristic_attention",
            },
            {
                "model": "WaveformCNN",
                "representation": "waveform",
                "trainable_params": wave_params,
                "backbone_pooling": "AdaptiveAvgPool1d(1)",
                "track_pooling": "N/A (chunk-level val)",
            },
        ]
    )
    arch_df.to_csv(out.tables / "model_architecture_summary.csv", index=False)

    _plot_training_dynamics(cnn_hist=cnn_hist, wave_hist=wave_hist, out_dir=out.plots)

    # Unified evaluation table.
    rows: list[dict[str, object]] = []

    for metric_name, value in cnn_best_val.items():
        if not metric_name.startswith("val_"):
            continue
        rows.append(
            {
                "experiment_name": "mtat_cnn_logmel",
                "task": "mtat",
                "representation": "logmel",
                "model": "ShortChunkCNN",
                "duration": 3.0,
                "pooling": "chunk_train_eval",
                "split": "val",
                "metric_name": metric_name.replace("val_", ""),
                "metric_value": float(value),
            }
        )

    for metric_name, value in wave_best_val.items():
        if not metric_name.startswith("val_"):
            continue
        rows.append(
            {
                "experiment_name": "mtat_cnn_waveform",
                "task": "mtat",
                "representation": "waveform",
                "model": "WaveformCNN",
                "duration": 3.0,
                "pooling": "chunk_train_eval",
                "split": "val",
                "metric_name": metric_name.replace("val_", ""),
                "metric_value": float(value),
            }
        )

    for row in pooling_duration_overall.itertuples(index=False):
        for metric_name in ("map", "macro_pr_auc", "macro_roc_auc"):
            rows.append(
                {
                    "experiment_name": "mtat_cnn_logmel_duration_pooling",
                    "task": "mtat",
                    "representation": "logmel",
                    "model": "ShortChunkCNN",
                    "duration": float(row.duration),
                    "pooling": display_pooling_name(str(row.pooling)),
                    "split": "test",
                    "metric_name": metric_name,
                    "metric_value": float(getattr(row, metric_name)),
                }
            )

    for row in robustness_overall.itertuples(index=False):
        for metric_name in ("map", "macro_pr_auc", "macro_roc_auc"):
            rows.append(
                {
                    "experiment_name": "mtat_cnn_logmel_robustness",
                    "task": "mtat",
                    "representation": "logmel",
                    "model": "ShortChunkCNN",
                    "duration": 3.0,
                    "pooling": "mean",
                    "split": "test",
                    "condition": str(row.condition),
                    "metric_name": metric_name,
                    "metric_value": float(getattr(row, metric_name)),
                }
            )

    for split in ("val", "test"):
        for metric_name in ("map", "macro_pr_auc", "macro_roc_auc"):
            rows.append(
                {
                    "experiment_name": "mtat_pretrained_panns",
                    "task": "mtat",
                    "representation": "pretrained_embedding",
                    "model": "PANNs+LogReg",
                    "duration": np.nan,
                    "pooling": "embedding_head",
                    "split": split,
                    "metric_name": metric_name,
                    "metric_value": float(pretrained_summary[split][metric_name]),
                }
            )

    for split in ("val", "test"):
        for metric_name in ("accuracy", "balanced_accuracy", "macro_f1"):
            rows.append(
                {
                    "experiment_name": "gtzan_transfer_probe",
                    "task": "gtzan",
                    "representation": "mtat_encoder_embedding",
                    "model": "ShortChunkCNN(frozen)+LogReg",
                    "duration": 3.0,
                    "pooling": "feature_mean",
                    "split": split,
                    "metric_name": metric_name,
                    "metric_value": float(transfer_summary["transfer_encoder"][split][metric_name]),
                }
            )
            rows.append(
                {
                    "experiment_name": "gtzan_random_probe",
                    "task": "gtzan",
                    "representation": "random_encoder_embedding",
                    "model": "ShortChunkCNN(random)+LogReg",
                    "duration": 3.0,
                    "pooling": "feature_mean",
                    "split": split,
                    "metric_name": metric_name,
                    "metric_value": float(transfer_summary["random_encoder_baseline"][split][metric_name]),
                }
            )

    unified_df = pd.DataFrame(rows)
    unified_df.to_csv(out.tables / "unified_evaluation_table.csv", index=False)
    _plot_metric_bars(unified_df, out.plots / "overall_metric_comparison_bars.png")

    # Calibration evidence.
    calibration_summary = _compute_calibration_plot(
        manifest=manifest,
        tags=tags,
        checkpoint=args.logmel_checkpoint,
        pooling_duration_df=pooling_duration_overall,
        output_dir=out.base,
        sample_rate=args.sample_rate,
        batch_size=args.batch_size,
        device=device,
    )

    # GTZAN per-genre diagnostics.
    gtzan_diag = _gtzan_transfer_confusion(out.base)

    final_summary = {
        "model_params": {
            "short_chunk_cnn": int(short_params),
            "waveform_cnn": int(wave_params),
            "param_ratio_waveform_vs_logmel": float(wave_params / max(1, short_params)),
        },
        "calibration": calibration_summary,
        "gtzan_transfer_diagnostics": gtzan_diag,
    }
    with (out.json / "analysis_master_summary.json").open("w", encoding="utf-8") as fp:
        json.dump(final_summary, fp, indent=2)

    # Save a lightweight machine-readable checklist of generated core files.
    generated = {
        "tables": sorted([p.name for p in out.tables.glob("*.csv")]),
        "plots": sorted([p.name for p in out.plots.glob("*.png")]),
        "json": sorted([p.name for p in out.json.glob("*.json")]),
    }
    with (out.json / "generated_artifacts_index.json").open("w", encoding="utf-8") as fp:
        json.dump(generated, fp, indent=2)

    print(f"Wrote master analysis outputs to: {out.base}")
    print(f"Report path reserved at: {reports_dir / 'result_analysis.md'}")


if __name__ == "__main__":
    main()
