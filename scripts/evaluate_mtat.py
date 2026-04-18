from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from aml_music.audio import chunk_audio, load_audio
from aml_music.evaluation.metrics import multilabel_metrics, rare_tag_buckets
from aml_music.evaluation.robustness import (
    add_noise,
    dynamic_range_compression,
    time_stretch_resample,
)
from aml_music.features.logmel import LogMelFrontend
from aml_music.models.pooling import aggregate_probs
from aml_music.models.short_chunk_cnn import ShortChunkCNN
from aml_music.models.waveform_cnn import WaveformCNN
from aml_music.utils import ensure_dir, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MTAT model with track-level pooling.")
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/manifests/mtat_top50_manifest.csv"))
    parser.add_argument("--tags", type=Path, default=Path("artifacts/manifests/mtat_top50_tags.json"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--representation", type=str, default="logmel", choices=["logmel", "waveform"])
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--chunk-seconds", type=float, default=3.0)
    parser.add_argument("--hop-seconds", type=float, default=1.5)
    parser.add_argument("--durations", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0, 8.0])
    parser.add_argument("--pooling", nargs="+", default=["mean", "max", "attention"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-tracks", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/reports/mtat_eval"))
    return parser.parse_args()


def _build_model(representation: str, num_tags: int) -> torch.nn.Module:
    if representation == "logmel":
        return ShortChunkCNN(num_tags=num_tags)
    if representation == "waveform":
        return WaveformCNN(num_tags=num_tags)
    raise ValueError(f"Unsupported representation: {representation}")


def _load_checkpoint(path: Path, device: torch.device):
    """Load trusted project checkpoints across PyTorch versions."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        # Older PyTorch versions may not support the weights_only argument.
        return torch.load(path, map_location=device)


def _chunk_probs(
    model: torch.nn.Module,
    chunks: list[np.ndarray],
    representation: str,
    frontend: LogMelFrontend | None,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    if not chunks:
        return np.empty((0, 0), dtype=np.float32)

    probs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(chunks), batch_size):
            batch_chunks = chunks[start : start + batch_size]
            if representation == "logmel":
                assert frontend is not None
                feats = [frontend(ch) for ch in batch_chunks]
                x = torch.from_numpy(np.stack(feats, axis=0)).float().unsqueeze(1)
            else:
                x = torch.from_numpy(np.stack(batch_chunks, axis=0)).float().unsqueeze(1)
            x = x.to(device)
            logits = model(x)
            probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs, axis=0)


def _collect_track_probs(
    model: torch.nn.Module,
    frame: pd.DataFrame,
    tag_names: list[str],
    representation: str,
    sample_rate: int,
    chunk_seconds: float,
    hop_seconds: float,
    pooling_modes: list[str],
    batch_size: int,
    device: torch.device,
    perturb: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    frontend = LogMelFrontend(sample_rate=sample_rate) if representation == "logmel" else None
    chunk_size = int(chunk_seconds * sample_rate)
    hop_size = max(1, int(hop_seconds * sample_rate))

    y_true: list[np.ndarray] = []
    y_prob_by_mode: dict[str, list[np.ndarray]] = {mode: [] for mode in pooling_modes}

    grouped = frame.groupby("track_id", sort=False)
    for _, track_df in tqdm(grouped, total=grouped.ngroups, desc=f"eval {chunk_seconds:.1f}s"):
        all_chunks: list[np.ndarray] = []
        labels = track_df[tag_names].max(axis=0).to_numpy(dtype=np.float32)
        y_true.append(labels)

        for row in track_df.itertuples(index=False):
            audio, _ = load_audio(row.audio_path, sample_rate=sample_rate)
            if perturb is not None:
                audio = perturb(audio)
            all_chunks.extend(chunk_audio(audio, chunk_size=chunk_size, hop_size=hop_size))

        chunk_prob = _chunk_probs(
            model=model,
            chunks=all_chunks,
            representation=representation,
            frontend=frontend,
            batch_size=batch_size,
            device=device,
        )
        if chunk_prob.shape[0] == 0:
            chunk_prob = np.zeros((1, len(tag_names)), dtype=np.float32)

        for mode in pooling_modes:
            y_prob_by_mode[mode].append(aggregate_probs(chunk_prob, mode=mode))

    y_true_arr = np.stack(y_true, axis=0)
    y_prob_out = {mode: np.stack(values, axis=0) for mode, values in y_prob_by_mode.items()}
    return y_true_arr, y_prob_out


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)

    manifest = pd.read_csv(args.manifest)
    with args.tags.open("r", encoding="utf-8") as fp:
        tag_names = json.load(fp)

    frame = manifest.loc[manifest["split"] == args.split].copy()
    if args.max_tracks is not None:
        keep_tracks = frame["track_id"].drop_duplicates().head(args.max_tracks)
        frame = frame.loc[frame["track_id"].isin(set(keep_tracks))].copy()

    device = torch.device(args.device)
    model = _build_model(args.representation, num_tags=len(tag_names))
    ckpt = _load_checkpoint(args.checkpoint, device=device)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    duration_results: dict[str, dict[str, float]] = {}
    detailed: dict[str, dict[str, object]] = {}

    for duration in args.durations:
        y_true, y_prob_modes = _collect_track_probs(
            model=model,
            frame=frame,
            tag_names=tag_names,
            representation=args.representation,
            sample_rate=args.sample_rate,
            chunk_seconds=duration,
            hop_seconds=max(duration / 2, 0.1),
            pooling_modes=args.pooling,
            batch_size=args.batch_size,
            device=device,
        )
        for mode in args.pooling:
            key = f"dur_{duration:.1f}_{mode}"
            metric = multilabel_metrics(y_true, y_prob_modes[mode], tag_names=tag_names)
            metric["rare_tag_buckets"] = rare_tag_buckets(y_true, y_prob_modes[mode], tag_names=tag_names)
            duration_results[key] = {
                "macro_roc_auc": metric["macro_roc_auc"],
                "macro_pr_auc": metric["macro_pr_auc"],
                "map": metric["map"],
            }
            detailed[key] = metric
            print(key, duration_results[key])

    # Robustness suite (mean pooling at default chunk length).
    robustness = {}
    perturbations = {
        "clean": None,
        "noise_snr20": lambda x: add_noise(x, snr_db=20.0),
        "time_stretch_1p1": lambda x: time_stretch_resample(x, rate=1.1),
        "compression_gamma0p6": lambda x: dynamic_range_compression(x, gamma=0.6),
    }
    for name, perturb_fn in perturbations.items():
        y_true, y_prob_modes = _collect_track_probs(
            model=model,
            frame=frame,
            tag_names=tag_names,
            representation=args.representation,
            sample_rate=args.sample_rate,
            chunk_seconds=args.chunk_seconds,
            hop_seconds=args.hop_seconds,
            pooling_modes=["mean"],
            batch_size=args.batch_size,
            device=device,
            perturb=perturb_fn,
        )
        metric = multilabel_metrics(y_true, y_prob_modes["mean"], tag_names=tag_names)
        robustness[name] = {
            "macro_roc_auc": metric["macro_roc_auc"],
            "macro_pr_auc": metric["macro_pr_auc"],
            "map": metric["map"],
        }
        print("robustness", name, robustness[name])

    write_json(
        {
            "duration_pooling_summary": duration_results,
            "robustness_summary": robustness,
        },
        out_dir / "summary.json",
    )
    write_json(detailed, out_dir / "detailed_duration_metrics.json")
    print(f"Wrote evaluation to: {out_dir}")


if __name__ == "__main__":
    main()
