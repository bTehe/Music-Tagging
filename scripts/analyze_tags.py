from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from aml_music.analysis.common import (  # noqa: E402
    build_tag_group_map,
    ensure_analysis_dirs,
    load_manifest_and_tags,
    split_tag_counts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze per-tag strengths and failures.")
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/manifests/mtat_top50_manifest.csv"))
    parser.add_argument("--tags", type=Path, default=Path("artifacts/manifests/mtat_top50_tags.json"))
    parser.add_argument(
        "--pooling-duration-per-tag",
        type=Path,
        default=Path("artifacts/analysis/tables/pooling_duration_per_tag.csv"),
    )
    parser.add_argument(
        "--pooling-duration-best",
        type=Path,
        default=Path("artifacts/analysis/tables/pooling_duration_best_per_tag.csv"),
    )
    parser.add_argument(
        "--cooccurrence-stats",
        type=Path,
        default=Path("artifacts/analysis/tables/tag_cooccurrence_stats.csv"),
    )
    parser.add_argument(
        "--robustness-per-tag",
        type=Path,
        default=Path("artifacts/analysis/tables/robustness_per_tag.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/analysis"))
    return parser.parse_args()


def _fit_line(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(x) & np.isfinite(y)
    x1 = x[valid]
    y1 = y[valid]
    if len(x1) < 2:
        return x1, y1
    m, b = np.polyfit(x1, y1, deg=1)
    xs = np.linspace(x1.min(), x1.max(), num=100)
    ys = m * xs + b
    return xs, ys


def _plot_freq_vs_ap(df: pd.DataFrame, out_path: Path) -> None:
    x = df["train_count"].to_numpy(dtype=float)
    y = df["ap"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(x, y, alpha=0.8)
    xs, ys = _fit_line(x, y)
    if len(xs) > 1:
        ax.plot(xs, ys, color="black", linewidth=1.5, label="linear trend")
        ax.legend()

    worst = df.sort_values("ap", ascending=True).head(5)
    best = df.sort_values("ap", ascending=False).head(5)
    for _, row in pd.concat([worst, best]).iterrows():
        ax.annotate(row["tag"], (row["train_count"], row["ap"]), fontsize=8)

    ax.set_xlabel("Train positive count")
    ax.set_ylabel("AP (best setting per tag)")
    ax.set_title("Tag frequency vs performance")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_tag_group_performance(df: pd.DataFrame, out_path: Path) -> None:
    groups = ["genre", "instrument", "vocal", "mood", "production", "other"]
    data = [df.loc[df["tag_group"] == g, "ap"].to_numpy(dtype=float) for g in groups]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(data, tick_labels=groups, showmeans=True)
    ax.set_ylabel("AP (best setting per tag)")
    ax.set_title("Per-tag AP by semantic tag group")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _infer_reason(row: pd.Series) -> str:
    reasons: list[str] = []
    if row["train_count"] < 250:
        reasons.append("data_scarcity")
    if row["co_occurrence_entropy"] > 0.72:
        reasons.append("cooccurrence_ambiguity")
    if row["max_prevalence_shift"] > 0.35:
        reasons.append("split_shift")
    if row["mean_ap_drop"] > 0.08:
        reasons.append("robustness_fragility")
    if row["duration_gain_long_short"] > 0.08:
        reasons.append("context_length_dependence")
    if len(reasons) == 0:
        reasons.append("mixed_or_label_noise")
    return ";".join(reasons)


def main() -> None:
    args = parse_args()
    out = ensure_analysis_dirs(args.output_dir)
    manifest, tags = load_manifest_and_tags(args.manifest, args.tags)
    tag_groups = build_tag_group_map(tags)

    per_tag_duration = pd.read_csv(args.pooling_duration_per_tag)
    best_by_tag = pd.read_csv(args.pooling_duration_best)
    co_stats = pd.read_csv(args.cooccurrence_stats)
    robust = pd.read_csv(args.robustness_per_tag)
    split_counts = split_tag_counts(manifest, tags)

    split_sizes = manifest["split"].value_counts().to_dict()
    for split in ("train", "val", "test"):
        split_counts[f"{split}_prevalence"] = split_counts[f"{split}_count"] / max(1, split_sizes.get(split, 1))
    eps = 1e-8
    split_counts["val_train_ratio"] = (split_counts["val_prevalence"] + eps) / (
        split_counts["train_prevalence"] + eps
    )
    split_counts["test_train_ratio"] = (split_counts["test_prevalence"] + eps) / (
        split_counts["train_prevalence"] + eps
    )
    split_counts["max_prevalence_shift"] = split_counts[["val_train_ratio", "test_train_ratio"]].apply(
        lambda row: float(max(abs(row["val_train_ratio"] - 1.0), abs(row["test_train_ratio"] - 1.0))),
        axis=1,
    )

    # Long-short context gain proxy from AP.
    pivot_ap = per_tag_duration.pivot_table(index="tag", columns="setting_display", values="ap")
    col_short = "0.5s_mean"
    col_long = "8.0s_max"
    duration_gain = pd.Series(0.0, index=pivot_ap.index)
    if col_short in pivot_ap.columns and col_long in pivot_ap.columns:
        duration_gain = pivot_ap[col_long] - pivot_ap[col_short]
    duration_gain_df = pd.DataFrame(
        {"tag": duration_gain.index.tolist(), "duration_gain_long_short": duration_gain.to_numpy(dtype=float)}
    )

    # Assemble main per-tag table.
    base_best = best_by_tag.rename(
        columns={
            "best_ap": "ap",
            "best_pr_auc": "pr_auc",
            "best_roc_auc": "roc_auc",
            "best_pooling": "pooling",
        }
    ).copy()
    if "train_count" in base_best.columns:
        base_best = base_best.drop(columns=["train_count"])

    tag_df = (
        base_best
        .merge(split_counts, on="tag", how="left")
        .merge(co_stats[["tag", "co_occurrence_degree", "co_occurrence_entropy"]], on="tag", how="left")
        .merge(robust[["tag", "drop_noise_ap", "drop_stretch_ap", "drop_compression_ap", "mean_ap_drop"]], on="tag", how="left")
        .merge(duration_gain_df, on="tag", how="left")
    )
    tag_df["tag_group"] = tag_df["tag"].map(tag_groups)
    tag_df["best_duration"] = tag_df["best_duration"].astype(float)
    tag_df["best_pooling"] = tag_df["pooling"].astype(str)
    tag_df["inferred_failure_reason"] = tag_df.apply(_infer_reason, axis=1)

    # Correlations.
    corr = {
        "pearson_train_count_vs_ap": float(tag_df["train_count"].corr(tag_df["ap"], method="pearson")),
        "spearman_train_count_vs_ap": float(tag_df["train_count"].corr(tag_df["ap"], method="spearman")),
        "pearson_co_entropy_vs_ap": float(tag_df["co_occurrence_entropy"].corr(tag_df["ap"], method="pearson")),
        "spearman_co_degree_vs_ap": float(tag_df["co_occurrence_degree"].corr(tag_df["ap"], method="spearman")),
        "pearson_robustness_drop_vs_ap": float(tag_df["mean_ap_drop"].corr(tag_df["ap"], method="pearson")),
    }

    tag_df = tag_df[
        [
            "tag",
            "tag_group",
            "train_count",
            "val_count",
            "test_count",
            "ap",
            "pr_auc",
            "roc_auc",
            "best_duration",
            "best_pooling",
            "drop_noise_ap",
            "drop_stretch_ap",
            "drop_compression_ap",
            "mean_ap_drop",
            "co_occurrence_degree",
            "co_occurrence_entropy",
            "max_prevalence_shift",
            "duration_gain_long_short",
            "inferred_failure_reason",
        ]
    ].sort_values("ap", ascending=True)

    tag_df.to_csv(out.tables / "tag_performance_analysis.csv", index=False)
    pd.DataFrame([corr]).to_csv(out.tables / "tag_correlation_stats.csv", index=False)
    tag_df.head(12).to_csv(out.tables / "tag_worst12.csv", index=False)
    tag_df.tail(12).sort_values("ap", ascending=False).to_csv(out.tables / "tag_best12.csv", index=False)

    _plot_freq_vs_ap(tag_df, out.plots / "tag_frequency_vs_ap_scatter.png")
    _plot_tag_group_performance(tag_df, out.plots / "tag_group_ap_boxplot.png")

    summary = {
        "correlations": corr,
        "worst_tags": tag_df.head(10)[["tag", "ap", "inferred_failure_reason"]].to_dict(orient="records"),
        "best_tags": tag_df.sort_values("ap", ascending=False)
        .head(10)[["tag", "ap"]]
        .to_dict(orient="records"),
    }
    with (out.json / "tag_analysis_summary.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    print(f"Wrote tag analysis to: {out.base}")


if __name__ == "__main__":
    main()
