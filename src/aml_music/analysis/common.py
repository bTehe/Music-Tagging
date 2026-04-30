"""Shared utilities for analysis scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from aml_music.audio import chunk_audio, load_audio
from aml_music.features.logmel import LogMelFrontend
from aml_music.models.pooling import aggregate_probs
from aml_music.models.short_chunk_cnn import ShortChunkCNN
from aml_music.models.waveform_cnn import WaveformCNN
from aml_music.utils import ensure_dir, write_json


@dataclass
class AnalysisDirs:
    base: Path
    plots: Path
    tables: Path
    json: Path


def ensure_analysis_dirs(base_dir: str | Path = "artifacts/analysis") -> AnalysisDirs:
    base = ensure_dir(base_dir)
    plots = ensure_dir(base / "plots")
    tables = ensure_dir(base / "tables")
    json_dir = ensure_dir(base / "json")
    return AnalysisDirs(base=base, plots=plots, tables=tables, json=json_dir)


def load_manifest_and_tags(manifest_path: str | Path, tags_path: str | Path) -> tuple[pd.DataFrame, list[str]]:
    manifest = pd.read_csv(manifest_path)
    tags = pd.read_json(tags_path, typ="series").tolist()
    return manifest, tags


def load_json(path: str | Path) -> dict:
    return pd.read_json(path, typ="series").to_dict()


def save_json(payload: dict, path: str | Path) -> None:
    write_json(payload, path)


def parse_duration_pooling_key(key: str) -> tuple[float, str]:
    # Expected key format: dur_4.0_max
    parts = key.split("_")
    if len(parts) != 3 or parts[0] != "dur":
        raise ValueError(f"Unexpected duration key format: {key}")
    return float(parts[1]), str(parts[2])


def display_pooling_name(pooling: str) -> str:
    if pooling == "attention":
        return "heuristic_attention"
    return pooling


def infer_tag_group(tag: str) -> str:
    t = tag.lower().strip()
    if t in {
        "rock",
        "pop",
        "metal",
        "country",
        "classical",
        "classic",
        "techno",
        "electronic",
        "ambient",
        "new age",
        "dance",
        "indian",
        "opera",
    }:
        return "genre"
    if t in {
        "guitar",
        "strings",
        "drums",
        "piano",
        "violin",
        "synth",
        "harpsichord",
        "flute",
        "sitar",
        "harp",
        "cello",
    }:
        return "instrument"
    if t in {
        "vocal",
        "vocals",
        "male vocal",
        "female vocal",
        "male voice",
        "female voice",
        "voice",
        "no voice",
        "no vocal",
        "no vocals",
        "singing",
        "choir",
        "choral",
        "male",
        "female",
        "man",
        "woman",
    }:
        return "vocal"
    if t in {"slow", "fast", "soft", "loud", "quiet"}:
        return "mood"
    if t in {"beat", "beats", "solo"}:
        return "production"
    return "other"


def build_tag_group_map(tags: list[str]) -> dict[str, str]:
    return {tag: infer_tag_group(tag) for tag in tags}


def split_tag_counts(manifest: pd.DataFrame, tags: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for tag in tags:
        train_count = int(manifest.loc[manifest["split"] == "train", tag].sum())
        val_count = int(manifest.loc[manifest["split"] == "val", tag].sum())
        test_count = int(manifest.loc[manifest["split"] == "test", tag].sum())
        rows.append(
            {
                "tag": tag,
                "train_count": train_count,
                "val_count": val_count,
                "test_count": test_count,
            }
        )
    return pd.DataFrame(rows).sort_values("train_count", ascending=False).reset_index(drop=True)


def get_label_matrix_by_split(manifest: pd.DataFrame, tags: list[str], split: str) -> np.ndarray:
    subset = manifest.loc[manifest["split"] == split, tags]
    return subset.to_numpy(dtype=np.float32)


def compute_cooccurrence_stats(
    binary_labels: np.ndarray,
    tags: list[str],
    cond_threshold: float = 0.25,
) -> tuple[np.ndarray, pd.DataFrame]:
    # binary_labels: [n_samples, n_tags] with 0/1 values.
    y = (binary_labels > 0.5).astype(np.float32)
    support = y.sum(axis=0)
    n_tags = y.shape[1]
    co_counts = y.T @ y
    cond = np.zeros((n_tags, n_tags), dtype=np.float32)
    degrees = np.zeros(n_tags, dtype=np.int32)
    co_entropy = np.zeros(n_tags, dtype=np.float32)

    for i in range(n_tags):
        denom = max(support[i], 1.0)
        cond[i, :] = co_counts[i, :] / denom
        cond[i, i] = 1.0

        # Degree proxy: number of tags with strong conditional co-occurrence.
        strong = (cond[i, :] >= cond_threshold).astype(np.int32)
        strong[i] = 0
        degrees[i] = int(strong.sum())

        # Co-occurrence entropy proxy over other tags.
        p = cond[i, :].copy()
        p[i] = 0.0
        p_sum = float(p.sum())
        if p_sum <= 0:
            co_entropy[i] = 0.0
        else:
            p = p / p_sum
            p = np.clip(p, 1e-8, 1.0)
            co_entropy[i] = float(-(p * np.log2(p)).sum() / np.log2(max(2, n_tags - 1)))

    stats = pd.DataFrame(
        {
            "tag": tags,
            "support_pos_train": support.astype(int),
            "co_occurrence_degree": degrees,
            "co_occurrence_entropy": co_entropy,
        }
    )
    return cond, stats


def _load_checkpoint(path: str | Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def build_model(representation: str, num_tags: int) -> torch.nn.Module:
    if representation == "logmel":
        return ShortChunkCNN(num_tags=num_tags)
    if representation == "waveform":
        return WaveformCNN(num_tags=num_tags)
    raise ValueError(f"Unsupported representation: {representation}")


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    representation: str,
    num_tags: int,
    device: torch.device,
) -> torch.nn.Module:
    model = build_model(representation=representation, num_tags=num_tags)
    ckpt = _load_checkpoint(checkpoint_path, device=device)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _batch_chunk_predict(
    model: torch.nn.Module,
    chunk_list: list[np.ndarray],
    representation: str,
    sample_rate: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    if len(chunk_list) == 0:
        return np.empty((0, 0), dtype=np.float32)

    frontend = LogMelFrontend(sample_rate=sample_rate) if representation == "logmel" else None
    outs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(chunk_list), batch_size):
            batch_chunks = chunk_list[start : start + batch_size]
            if representation == "logmel":
                assert frontend is not None
                feats = [frontend(ch) for ch in batch_chunks]
                x = torch.from_numpy(np.stack(feats, axis=0)).float().unsqueeze(1).to(device)
            else:
                x = torch.from_numpy(np.stack(batch_chunks, axis=0)).float().unsqueeze(1).to(device)
            logits = model(x)
            outs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(outs, axis=0)


def evaluate_track_level(
    model: torch.nn.Module,
    frame: pd.DataFrame,
    tags: list[str],
    representation: str,
    sample_rate: int,
    chunk_seconds: float,
    hop_seconds: float,
    pooling_modes: list[str],
    batch_size: int,
    device: torch.device,
    perturb_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    desc: str = "eval",
) -> tuple[np.ndarray, dict[str, np.ndarray], pd.DataFrame]:
    chunk_size = int(chunk_seconds * sample_rate)
    hop_size = max(1, int(hop_seconds * sample_rate))

    y_true_rows: list[np.ndarray] = []
    y_pred_rows: dict[str, list[np.ndarray]] = {mode: [] for mode in pooling_modes}
    meta_rows: list[dict[str, object]] = []

    grouped = frame.groupby("track_id", sort=False)
    for track_id, track_df in tqdm(grouped, total=grouped.ngroups, desc=desc):
        chunks: list[np.ndarray] = []
        labels = track_df[tags].max(axis=0).to_numpy(dtype=np.float32)
        y_true_rows.append(labels)

        first_row = track_df.iloc[0]
        meta_rows.append(
            {
                "track_id": str(track_id),
                "clip_id": int(first_row["clip_id"]),
                "audio_path": str(first_row["audio_path"]),
                "num_segments": int(len(track_df)),
            }
        )

        for row in track_df.itertuples(index=False):
            audio, _ = load_audio(row.audio_path, sample_rate=sample_rate)
            if perturb_fn is not None:
                audio = perturb_fn(audio)
            chunks.extend(chunk_audio(audio, chunk_size=chunk_size, hop_size=hop_size))

        probs = _batch_chunk_predict(
            model=model,
            chunk_list=chunks,
            representation=representation,
            sample_rate=sample_rate,
            batch_size=batch_size,
            device=device,
        )
        if probs.shape[0] == 0:
            probs = np.zeros((1, len(tags)), dtype=np.float32)
        for mode in pooling_modes:
            y_pred_rows[mode].append(aggregate_probs(probs, mode=mode))

    y_true = np.stack(y_true_rows, axis=0)
    y_pred = {mode: np.stack(rows, axis=0) for mode, rows in y_pred_rows.items()}
    meta = pd.DataFrame(meta_rows)
    return y_true, y_pred, meta

