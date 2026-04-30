from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from aml_music.analysis.common import (  # noqa: E402
    ensure_analysis_dirs,
    evaluate_track_level,
    load_manifest_and_tags,
    load_model_from_checkpoint,
)
from aml_music.evaluation.metrics import multilabel_metrics  # noqa: E402
from aml_music.evaluation.robustness import (  # noqa: E402
    add_noise,
    dynamic_range_compression,
    time_stretch_resample,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze robustness effects on MTAT baseline.")
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/manifests/mtat_top50_manifest.csv"))
    parser.add_argument("--tags", type=Path, default=Path("artifacts/manifests/mtat_top50_tags.json"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/runs/mtat_cnn_baseline/best.pt"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/analysis"))
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--chunk-seconds", type=float, default=3.0)
    parser.add_argument("--hop-seconds", type=float, default=1.5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-cache", action="store_true")
    return parser.parse_args()


def _plot_overall_robustness(overall_df: pd.DataFrame, out_path: Path) -> None:
    cond_order = ["clean", "noise_snr20", "time_stretch_1p1", "compression_gamma0p6"]
    metrics = ["map", "macro_pr_auc", "macro_roc_auc"]
    labels = ["mAP", "PR-AUC", "ROC-AUC"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True)
    x = np.arange(len(cond_order))
    for ax, metric, label in zip(axes, metrics, labels):
        vals = [overall_df.loc[overall_df["condition"] == c, metric].iloc[0] for c in cond_order]
        ax.bar(x, vals, width=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(cond_order, rotation=30, ha="right")
        ax.set_ylabel(label)
        ax.set_title(f"Robustness comparison ({label})")
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_per_tag_degradation_heatmap(
    per_tag_df: pd.DataFrame,
    out_path: Path,
    top_k: int = 20,
) -> None:
    tmp = per_tag_df.copy()
    tmp["mean_drop"] = tmp[["drop_noise_ap", "drop_stretch_ap", "drop_compression_ap"]].mean(axis=1)
    tags = tmp.sort_values("mean_drop", ascending=False).head(top_k)["tag"].tolist()
    mat_df = tmp.set_index("tag").loc[tags, ["drop_noise_ap", "drop_stretch_ap", "drop_compression_ap"]]
    mat = mat_df.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8, max(6, 0.35 * top_k)))
    im = ax.imshow(mat, cmap="Reds", aspect="auto")
    ax.set_yticks(np.arange(len(tags)))
    ax.set_yticklabels(tags, fontsize=9)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(["noise", "time_stretch", "compression"])
    ax.set_title("Top degraded tags by AP drop")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("AP drop vs clean")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out = ensure_analysis_dirs(args.output_dir)
    manifest, tags = load_manifest_and_tags(args.manifest, args.tags)
    frame = manifest.loc[manifest["split"] == args.split].copy()

    device = torch.device(args.device)
    cache_path = out.json / f"robustness_track_preds_{args.split}.npz"

    perturbations = {
        "clean": None,
        "noise_snr20": lambda x: add_noise(x, snr_db=20.0),
        "time_stretch_1p1": lambda x: time_stretch_resample(x, rate=1.1),
        "compression_gamma0p6": lambda x: dynamic_range_compression(x, gamma=0.6),
    }

    y_true_ref: np.ndarray | None = None
    pred_by_condition: dict[str, np.ndarray] = {}

    use_cache = (not args.no_cache) and cache_path.exists()
    if use_cache:
        arr = np.load(cache_path, allow_pickle=True)
        y_true_ref = arr["y_true"]
        for name in perturbations.keys():
            pred_by_condition[name] = arr[f"pred_{name}"]
    else:
        model = load_model_from_checkpoint(
            checkpoint_path=args.checkpoint,
            representation="logmel",
            num_tags=len(tags),
            device=device,
        )
        for cond_name, perturb_fn in perturbations.items():
            y_true, preds, _ = evaluate_track_level(
                model=model,
                frame=frame,
                tags=tags,
                representation="logmel",
                sample_rate=args.sample_rate,
                chunk_seconds=args.chunk_seconds,
                hop_seconds=args.hop_seconds,
                pooling_modes=["mean"],
                batch_size=args.batch_size,
                device=device,
                perturb_fn=perturb_fn,
                desc=f"robust_{cond_name}",
            )
            if y_true_ref is None:
                y_true_ref = y_true
            pred_by_condition[cond_name] = preds["mean"]

        assert y_true_ref is not None
        np.savez_compressed(
            cache_path,
            y_true=y_true_ref,
            **{f"pred_{k}": v for k, v in pred_by_condition.items()},
        )

    assert y_true_ref is not None

    overall_rows: list[dict[str, object]] = []
    per_tag_by_cond: dict[str, pd.DataFrame] = {}
    for cond_name, pred in pred_by_condition.items():
        metric = multilabel_metrics(y_true_ref, pred, tag_names=tags)
        overall_rows.append(
            {
                "condition": cond_name,
                "map": float(metric["map"]),
                "macro_pr_auc": float(metric["macro_pr_auc"]),
                "macro_roc_auc": float(metric["macro_roc_auc"]),
            }
        )
        per_tag_by_cond[cond_name] = pd.DataFrame(metric["per_tag"]).rename(
            columns={
                "ap": f"ap_{cond_name}",
                "pr_auc": f"pr_auc_{cond_name}",
                "roc_auc": f"roc_auc_{cond_name}",
                "support_pos": "support_pos_eval",
            }
        )

    overall_df = pd.DataFrame(overall_rows)
    per_tag_df = per_tag_by_cond["clean"][["tag", "support_pos_eval", "ap_clean", "pr_auc_clean", "roc_auc_clean"]].copy()
    for cond_name in ("noise_snr20", "time_stretch_1p1", "compression_gamma0p6"):
        cols = ["tag", f"ap_{cond_name}", f"pr_auc_{cond_name}", f"roc_auc_{cond_name}"]
        per_tag_df = per_tag_df.merge(per_tag_by_cond[cond_name][cols], on="tag", how="left")

    per_tag_df["drop_noise_ap"] = per_tag_df["ap_clean"] - per_tag_df["ap_noise_snr20"]
    per_tag_df["drop_stretch_ap"] = per_tag_df["ap_clean"] - per_tag_df["ap_time_stretch_1p1"]
    per_tag_df["drop_compression_ap"] = per_tag_df["ap_clean"] - per_tag_df["ap_compression_gamma0p6"]
    per_tag_df["mean_ap_drop"] = per_tag_df[
        ["drop_noise_ap", "drop_stretch_ap", "drop_compression_ap"]
    ].mean(axis=1)

    overall_df.to_csv(out.tables / "robustness_overall.csv", index=False)
    per_tag_df.sort_values("mean_ap_drop", ascending=False).to_csv(
        out.tables / "robustness_per_tag.csv",
        index=False,
    )

    _plot_overall_robustness(overall_df, out.plots / "robustness_overall_metrics.png")
    _plot_per_tag_degradation_heatmap(
        per_tag_df=per_tag_df,
        out_path=out.plots / "robustness_top_tag_degradation_heatmap.png",
        top_k=20,
    )

    summary = {
        "split": args.split,
        "best_clean_metrics": overall_df.loc[overall_df["condition"] == "clean"].iloc[0].to_dict(),
        "largest_mean_ap_drop_tags": per_tag_df.sort_values("mean_ap_drop", ascending=False)
        .head(10)[["tag", "mean_ap_drop", "drop_noise_ap", "drop_stretch_ap", "drop_compression_ap"]]
        .to_dict(orient="records"),
    }
    with (out.json / "robustness_summary_extended.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    print(f"Wrote robustness analysis to: {out.base}")


if __name__ == "__main__":
    main()

