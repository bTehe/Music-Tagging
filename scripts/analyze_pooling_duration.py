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
    display_pooling_name,
    ensure_analysis_dirs,
    load_manifest_and_tags,
    parse_duration_pooling_key,
    split_tag_counts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze MTAT duration and pooling results.")
    parser.add_argument(
        "--detailed-metrics",
        type=Path,
        default=Path("artifacts/reports/mtat_eval/detailed_duration_metrics.json"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("artifacts/reports/mtat_eval/summary.json"),
    )
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/manifests/mtat_top50_manifest.csv"))
    parser.add_argument("--tags", type=Path, default=Path("artifacts/manifests/mtat_top50_tags.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/analysis"))
    return parser.parse_args()


def _plot_duration_lines(overall_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True)
    metrics = ["map", "macro_pr_auc", "macro_roc_auc"]
    titles = ["mAP", "PR-AUC", "ROC-AUC"]
    for ax, metric, title in zip(axes, metrics, titles):
        for pooling in sorted(overall_df["pooling"].unique()):
            data = overall_df.loc[overall_df["pooling"] == pooling].sort_values("duration")
            ax.plot(
                data["duration"],
                data[metric],
                marker="o",
                linewidth=1.8,
                label=display_pooling_name(pooling),
            )
        ax.set_title(f"Duration vs {title}")
        ax.set_xlabel("Chunk duration (s)")
        ax.set_ylabel(title)
        ax.grid(alpha=0.2)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_per_tag_heatmap(
    per_tag_df: pd.DataFrame,
    tag_order: list[str],
    col_order: list[str],
    out_path: Path,
) -> None:
    pivot = per_tag_df.pivot_table(index="tag", columns="setting", values="ap")
    pivot = pivot.reindex(index=tag_order, columns=col_order)
    mat = pivot.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(16, 12))
    im = ax.imshow(mat, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_yticks(np.arange(len(tag_order)))
    ax.set_yticklabels(tag_order, fontsize=8)
    ax.set_xticks(np.arange(len(col_order)))
    ax.set_xticklabels(col_order, rotation=90, fontsize=8)
    ax.set_title("Per-tag AP sensitivity by duration and pooling")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Average precision (AP)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out = ensure_analysis_dirs(args.output_dir)
    manifest, tags = load_manifest_and_tags(args.manifest, args.tags)

    with args.detailed_metrics.open("r", encoding="utf-8") as fp:
        detailed = json.load(fp)
    with args.summary.open("r", encoding="utf-8") as fp:
        summary = json.load(fp)

    overall_rows: list[dict[str, object]] = []
    per_tag_rows: list[dict[str, object]] = []
    for key, metric in detailed.items():
        duration, pooling = parse_duration_pooling_key(key)
        overall_rows.append(
            {
                "setting": key,
                "duration": duration,
                "pooling": pooling,
                "pooling_display": display_pooling_name(pooling),
                "map": float(metric["map"]),
                "macro_pr_auc": float(metric["macro_pr_auc"]),
                "macro_roc_auc": float(metric["macro_roc_auc"]),
            }
        )
        for row in metric["per_tag"]:
            per_tag_rows.append(
                {
                    "setting": key,
                    "duration": duration,
                    "pooling": pooling,
                    "pooling_display": display_pooling_name(pooling),
                    "tag": str(row["tag"]),
                    "ap": float(row["ap"]),
                    "pr_auc": float(row["pr_auc"]),
                    "roc_auc": float(row["roc_auc"]),
                    "support_pos_eval": int(row["support_pos"]),
                }
            )

    overall_df = pd.DataFrame(overall_rows).sort_values(["duration", "pooling"]).reset_index(drop=True)
    per_tag_df = pd.DataFrame(per_tag_rows).sort_values(["tag", "duration", "pooling"]).reset_index(drop=True)

    tag_counts = split_tag_counts(manifest, tags)
    per_tag_df = per_tag_df.merge(tag_counts[["tag", "train_count"]], on="tag", how="left")
    per_tag_df["setting_display"] = per_tag_df.apply(
        lambda r: f"{r['duration']:.1f}s_{display_pooling_name(str(r['pooling']))}",
        axis=1,
    )

    # Best setting per tag (selected by AP, while preserving PR/ROC values at that setting).
    best_tag = (
        per_tag_df.sort_values("ap", ascending=False)
        .groupby("tag", as_index=False)
        .head(1)
        .rename(
            columns={
                "duration": "best_duration",
                "pooling": "best_pooling",
                "ap": "best_ap",
                "pr_auc": "best_pr_auc",
                "roc_auc": "best_roc_auc",
            }
        )
    )
    best_tag = best_tag[
        [
            "tag",
            "best_duration",
            "best_pooling",
            "best_ap",
            "best_pr_auc",
            "best_roc_auc",
            "support_pos_eval",
            "train_count",
        ]
    ]

    # Global best settings.
    best_global = overall_df.sort_values("map", ascending=False).head(1).iloc[0].to_dict()

    overall_df.to_csv(out.tables / "pooling_duration_overall.csv", index=False)
    per_tag_df.to_csv(out.tables / "pooling_duration_per_tag.csv", index=False)
    best_tag.sort_values("best_ap", ascending=False).to_csv(
        out.tables / "pooling_duration_best_per_tag.csv",
        index=False,
    )

    _plot_duration_lines(overall_df, out.plots / "duration_pooling_metrics.png")
    setting_order = [
        f"{row['duration']:.1f}s_{display_pooling_name(str(row['pooling']))}"
        for _, row in overall_df.sort_values(["duration", "pooling"]).iterrows()
    ]
    tag_order = (
        tag_counts.sort_values("train_count", ascending=False)["tag"].tolist()
    )
    per_tag_df = per_tag_df.copy()
    per_tag_df["setting"] = per_tag_df["setting_display"]
    _plot_per_tag_heatmap(
        per_tag_df=per_tag_df,
        tag_order=tag_order,
        col_order=setting_order,
        out_path=out.plots / "duration_pooling_per_tag_ap_heatmap.png",
    )

    out_json = {
        "best_global_by_map": {
            "setting": str(best_global["setting"]),
            "duration": float(best_global["duration"]),
            "pooling": str(best_global["pooling"]),
            "map": float(best_global["map"]),
            "macro_pr_auc": float(best_global["macro_pr_auc"]),
            "macro_roc_auc": float(best_global["macro_roc_auc"]),
        },
        "robustness_summary_from_eval": summary.get("robustness_summary", {}),
    }
    with (out.json / "pooling_duration_summary.json").open("w", encoding="utf-8") as fp:
        json.dump(out_json, fp, indent=2)

    print(f"Wrote pooling/duration analysis to: {out.base}")


if __name__ == "__main__":
    main()
