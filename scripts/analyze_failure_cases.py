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
from aml_music.audio import load_audio, pad_or_crop  # noqa: E402
from aml_music.features.logmel import LogMelFrontend  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze concrete model failure cases.")
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/manifests/mtat_top50_manifest.csv"))
    parser.add_argument("--tags", type=Path, default=Path("artifacts/manifests/mtat_top50_tags.json"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/runs/mtat_cnn_baseline/best.pt"),
    )
    parser.add_argument(
        "--pooling-duration-overall",
        type=Path,
        default=Path("artifacts/analysis/tables/pooling_duration_overall.csv"),
    )
    parser.add_argument(
        "--tag-analysis",
        type=Path,
        default=Path("artifacts/analysis/tables/tag_performance_analysis.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/analysis"))
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--hop-seconds", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-target-tags", type=int, default=8)
    return parser.parse_args()


def _top_tags(prob: np.ndarray, tags: list[str], k: int = 5) -> list[str]:
    idx = np.argsort(-prob)[:k]
    return [tags[int(i)] for i in idx]


def _true_tags(row: np.ndarray, tags: list[str]) -> list[str]:
    return [tags[i] for i in range(len(tags)) if row[i] > 0.5]


def _make_saliency(
    model: torch.nn.Module,
    audio_path: str,
    tag_idx: int,
    tag_name: str,
    out_path: Path,
    sample_rate: int,
    chunk_seconds: float,
    device: torch.device,
) -> None:
    chunk_samples = int(chunk_seconds * sample_rate)
    frontend = LogMelFrontend(sample_rate=sample_rate)

    audio, _ = load_audio(audio_path, sample_rate=sample_rate)
    chunk = pad_or_crop(audio, length_samples=chunk_samples, random_crop=False)
    mel = frontend(chunk)
    x = torch.from_numpy(mel).float().unsqueeze(0).unsqueeze(0).to(device)
    x.requires_grad_(True)

    logits = model(x)
    score = logits[0, tag_idx]
    model.zero_grad(set_to_none=True)
    score.backward()

    saliency = x.grad.detach().abs().squeeze().cpu().numpy()
    saliency = saliency / (saliency.max() + 1e-8)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].imshow(mel, origin="lower", aspect="auto")
    axes[0].set_title(f"log-mel ({tag_name})")
    axes[1].imshow(saliency, origin="lower", aspect="auto", cmap="hot")
    axes[1].set_title("saliency")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out = ensure_analysis_dirs(args.output_dir)
    manifest, tags = load_manifest_and_tags(args.manifest, args.tags)
    frame = manifest.loc[manifest["split"] == args.split].copy()
    device = torch.device(args.device)

    duration_df = pd.read_csv(args.pooling_duration_overall)
    best_row = duration_df.sort_values("map", ascending=False).iloc[0]
    best_duration = float(best_row["duration"])
    best_pooling = str(best_row["pooling"])
    hop_seconds = args.hop_seconds if args.hop_seconds is not None else max(0.1, best_duration / 2.0)

    model = load_model_from_checkpoint(
        checkpoint_path=args.checkpoint,
        representation="logmel",
        num_tags=len(tags),
        device=device,
    )
    y_true, y_pred_modes, meta = evaluate_track_level(
        model=model,
        frame=frame,
        tags=tags,
        representation="logmel",
        sample_rate=args.sample_rate,
        chunk_seconds=best_duration,
        hop_seconds=hop_seconds,
        pooling_modes=[best_pooling],
        batch_size=args.batch_size,
        device=device,
        perturb_fn=None,
        desc=f"failure_eval_{best_duration:.1f}s_{best_pooling}",
    )
    y_pred = y_pred_modes[best_pooling]

    tag_perf = pd.read_csv(args.tag_analysis).sort_values("ap", ascending=True)
    target_tags = tag_perf.head(args.num_target_tags)["tag"].tolist()
    target_idx = [tags.index(t) for t in target_tags if t in tags]

    cases: list[dict[str, object]] = []
    for idx in target_idx:
        tag = tags[idx]

        pos_idx = np.where(y_true[:, idx] > 0.5)[0]
        if len(pos_idx) > 0:
            worst_fn = int(pos_idx[np.argmin(y_pred[pos_idx, idx])])
            cases.append(
                {
                    "case_type": "false_negative_like",
                    "tag": tag,
                    "track_row_index": worst_fn,
                    "tag_probability": float(y_pred[worst_fn, idx]),
                    "true_label": 1,
                }
            )

        neg_idx = np.where(y_true[:, idx] < 0.5)[0]
        if len(neg_idx) > 0:
            worst_fp = int(neg_idx[np.argmax(y_pred[neg_idx, idx])])
            cases.append(
                {
                    "case_type": "false_positive_like",
                    "tag": tag,
                    "track_row_index": worst_fp,
                    "tag_probability": float(y_pred[worst_fp, idx]),
                    "true_label": 0,
                }
            )

    # Keep unique track/tag/type rows and strongest confidence.
    case_df = pd.DataFrame(cases).drop_duplicates(subset=["case_type", "tag", "track_row_index"])
    case_df = case_df.sort_values("tag_probability", ascending=False).reset_index(drop=True)

    enriched_rows: list[dict[str, object]] = []
    for row in case_df.itertuples(index=False):
        track_idx = int(row.track_row_index)
        prob_vec = y_pred[track_idx]
        true_vec = y_true[track_idx]
        meta_row = meta.iloc[track_idx]
        tag_idx = tags.index(str(row.tag))
        saliency_name = f"failure_saliency_{track_idx}_{str(row.tag).replace(' ', '_')}.png"
        saliency_path = out.plots / saliency_name

        _make_saliency(
            model=model,
            audio_path=str(meta_row["audio_path"]),
            tag_idx=tag_idx,
            tag_name=str(row.tag),
            out_path=saliency_path,
            sample_rate=args.sample_rate,
            chunk_seconds=best_duration,
            device=device,
        )

        enriched_rows.append(
            {
                "case_type": str(row.case_type),
                "tag": str(row.tag),
                "track_index": track_idx,
                "track_id": str(meta_row["track_id"]),
                "clip_id": int(meta_row["clip_id"]),
                "audio_path": str(meta_row["audio_path"]),
                "tag_probability": float(row.tag_probability),
                "true_label": int(row.true_label),
                "top_pred_tags": ",".join(_top_tags(prob_vec, tags, k=5)),
                "true_tags": ",".join(_true_tags(true_vec, tags)),
                "saliency_plot": str(saliency_path),
                "likely_reason": (
                    "insufficient_context_or_weak_supervision"
                    if row.case_type == "false_negative_like"
                    else "cooccurrence_or_timbre_bias"
                ),
            }
        )

    out_df = pd.DataFrame(enriched_rows)
    out_df.to_csv(out.tables / "failure_cases.csv", index=False)
    with (out.json / "failure_cases.json").open("w", encoding="utf-8") as fp:
        json.dump(out_df.to_dict(orient="records"), fp, indent=2)

    print(f"Wrote failure-case analysis to: {out.base}")


if __name__ == "__main__":
    main()

