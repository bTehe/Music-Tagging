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
    compute_cooccurrence_stats,
    ensure_analysis_dirs,
    get_label_matrix_by_split,
    load_manifest_and_tags,
    split_tag_counts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze split protocol and MTAT label structure.")
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/manifests/mtat_top50_manifest.csv"))
    parser.add_argument("--tags", type=Path, default=Path("artifacts/manifests/mtat_top50_tags.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/analysis"))
    return parser.parse_args()


def _save_split_count_plot(counts_df: pd.DataFrame, out_path: Path) -> None:
    idx = np.arange(len(counts_df))
    width = 0.25
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.bar(idx - width, counts_df["train_count"], width=width, label="train")
    ax.bar(idx, counts_df["val_count"], width=width, label="val")
    ax.bar(idx + width, counts_df["test_count"], width=width, label="test")
    ax.set_xticks(idx)
    ax.set_xticklabels(counts_df["tag"], rotation=80, ha="right")
    ax.set_ylabel("Positive labels")
    ax.set_title("MTAT Top-50 positive counts by split")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _save_prevalence_shift_plot(shift_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 7))
    x = np.arange(len(shift_df))
    ax.plot(x, shift_df["val_train_ratio"], marker="o", linewidth=1.5, label="val/train")
    ax.plot(x, shift_df["test_train_ratio"], marker="o", linewidth=1.5, label="test/train")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(shift_df["tag"], rotation=80, ha="right")
    ax.set_ylabel("Prevalence ratio")
    ax.set_title("Tag prevalence shift across splits")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _save_cooccurrence_heatmap(cond_prob: np.ndarray, tags: list[str], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cond_prob, cmap="magma", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(tags)))
    ax.set_yticks(np.arange(len(tags)))
    ax.set_xticklabels(tags, rotation=90, fontsize=8)
    ax.set_yticklabels(tags, fontsize=8)
    ax.set_title("Train split co-occurrence heatmap: P(tag_j=1 | tag_i=1)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Conditional probability")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out = ensure_analysis_dirs(args.output_dir)
    manifest, tags = load_manifest_and_tags(args.manifest, args.tags)
    tag_groups = build_tag_group_map(tags)

    counts_df = split_tag_counts(manifest, tags)
    counts_df["tag_group"] = counts_df["tag"].map(tag_groups)

    split_sizes = manifest["split"].value_counts().to_dict()
    for split in ("train", "val", "test"):
        denom = float(split_sizes.get(split, 1))
        counts_df[f"{split}_prevalence"] = counts_df[f"{split}_count"] / denom

    eps = 1e-8
    counts_df["val_train_ratio"] = (counts_df["val_prevalence"] + eps) / (
        counts_df["train_prevalence"] + eps
    )
    counts_df["test_train_ratio"] = (counts_df["test_prevalence"] + eps) / (
        counts_df["train_prevalence"] + eps
    )
    counts_df["max_prevalence_shift"] = counts_df[["val_train_ratio", "test_train_ratio"]].apply(
        lambda row: float(max(abs(row["val_train_ratio"] - 1.0), abs(row["test_train_ratio"] - 1.0))),
        axis=1,
    )

    train_y = get_label_matrix_by_split(manifest, tags, split="train")
    cond_prob, co_stats = compute_cooccurrence_stats(train_y, tags, cond_threshold=0.25)
    co_stats["tag_group"] = co_stats["tag"].map(tag_groups)
    co_stats = co_stats.merge(
        counts_df[["tag", "train_count", "val_count", "test_count"]],
        on="tag",
        how="left",
    )

    co_df = pd.DataFrame(cond_prob, columns=tags, index=tags)
    co_df.index.name = "tag"

    counts_df.to_csv(out.tables / "tag_split_counts.csv", index=False)
    counts_df.sort_values("max_prevalence_shift", ascending=False).to_csv(
        out.tables / "split_prevalence_shift.csv",
        index=False,
    )
    co_stats.sort_values("co_occurrence_degree", ascending=False).to_csv(
        out.tables / "tag_cooccurrence_stats.csv",
        index=False,
    )
    co_df.to_csv(out.tables / "tag_cooccurrence_matrix.csv")

    _save_split_count_plot(counts_df, out.plots / "split_tag_positive_counts.png")
    _save_prevalence_shift_plot(counts_df, out.plots / "split_prevalence_shift.png")
    _save_cooccurrence_heatmap(cond_prob, tags, out.plots / "tag_cooccurrence_heatmap.png")

    summary = {
        "n_rows_manifest": int(len(manifest)),
        "split_clip_counts": {k: int(v) for k, v in split_sizes.items()},
        "n_tags": int(len(tags)),
        "highest_shift_tags": counts_df.sort_values("max_prevalence_shift", ascending=False)
        .head(10)[["tag", "max_prevalence_shift", "val_train_ratio", "test_train_ratio"]]
        .to_dict(orient="records"),
        "highest_cooccurrence_degree_tags": co_stats.sort_values("co_occurrence_degree", ascending=False)
        .head(10)[["tag", "co_occurrence_degree", "co_occurrence_entropy"]]
        .to_dict(orient="records"),
    }
    with (out.json / "split_and_label_structure_summary.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    print(f"Wrote split/label analysis to: {out.base}")


if __name__ == "__main__":
    main()

