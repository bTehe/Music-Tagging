"""MTAT data protocol, manifest builder, and dataset classes."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from aml_music.audio import chunk_audio, load_audio, pad_or_crop
from aml_music.features.logmel import LogMelFrontend


def _sniff_delimiter(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="ignore") as fp:
        sample = fp.read(4096)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
        return dialect.delimiter
    except Exception:
        if "\t" in sample and sample.count("\t") > sample.count(","):
            return "\t"
        return ","


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    sep = _sniff_delimiter(path)
    return pd.read_csv(path, sep=sep)


def build_track_id(row: pd.Series) -> str:
    for key in ("original_url", "url"):
        value = row.get(key, "")
        if isinstance(value, str) and value.strip():
            return value.strip()

    parts = [
        str(row.get("artist", "")).strip(),
        str(row.get("album", "")).strip(),
        str(row.get("title", "")).strip(),
        str(row.get("track_number", "")).strip(),
    ]
    return "|".join(parts)


def infer_annotation_id_column(df: pd.DataFrame) -> str:
    lower_map = {c.lower(): c for c in df.columns}
    for candidate in ("clip_id", "clipid", "id"):
        if candidate in lower_map:
            return lower_map[candidate]
    raise ValueError(
        "Could not find clip id column in annotations. Expected one of: clip_id, clipid, id."
    )


def infer_tag_columns(df: pd.DataFrame, id_column: str) -> list[str]:
    candidate_cols = [c for c in df.columns if c != id_column]
    numeric_cols = [
        c for c in candidate_cols if pd.api.types.is_numeric_dtype(df[c]) or df[c].dropna().isin([0, 1]).all()
    ]
    if not numeric_cols:
        raise ValueError(
            "No binary/numeric tag columns found in annotations. Verify annotation file format."
        )
    return numeric_cols


def read_clip_id_list(path: str | Path) -> set[int]:
    table = read_table(path)
    if table.shape[1] == 1:
        series = table.iloc[:, 0]
    elif "clip_id" in table.columns:
        series = table["clip_id"]
    else:
        series = table.iloc[:, 0]
    return set(series.astype(int).tolist())


def assign_grouped_splits(
    frame: pd.DataFrame,
    seed: int = 42,
    train_size: float = 0.8,
    val_size: float = 0.1,
) -> pd.Series:
    if train_size <= 0 or val_size <= 0 or (train_size + val_size) >= 1.0:
        raise ValueError("Invalid split ratios. Use train > 0, val > 0, and train+val < 1.")
    test_size = 1.0 - (train_size + val_size)

    gss_main = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    idx = np.arange(len(frame))
    groups = frame["track_id"].values
    train_val_idx, test_idx = next(gss_main.split(idx, groups=groups))

    train_val = frame.iloc[train_val_idx]
    gss_val = GroupShuffleSplit(
        n_splits=1,
        test_size=val_size / (train_size + val_size),
        random_state=seed,
    )
    train_rel, val_rel = next(gss_val.split(np.arange(len(train_val)), groups=train_val["track_id"]))
    train_idx = train_val.index.values[train_rel]
    val_idx = train_val.index.values[val_rel]
    test_idx_abs = frame.index.values[test_idx]

    split = pd.Series(index=frame.index, dtype="string")
    split.loc[train_idx] = "train"
    split.loc[val_idx] = "val"
    split.loc[test_idx_abs] = "test"
    return split


def audit_split_hygiene(frame: pd.DataFrame) -> dict[str, object]:
    split_tracks: dict[str, set[str]] = {
        split: set(frame.loc[frame["split"] == split, "track_id"].astype(str).tolist())
        for split in ("train", "val", "test")
    }
    overlaps = {
        "train_val": sorted(list(split_tracks["train"] & split_tracks["val"]))[:10],
        "train_test": sorted(list(split_tracks["train"] & split_tracks["test"]))[:10],
        "val_test": sorted(list(split_tracks["val"] & split_tracks["test"]))[:10],
    }
    overlap_counts = {k: len(v) for k, v in overlaps.items()}

    clip_counts = frame["split"].value_counts().to_dict()
    track_counts = frame.groupby("split")["track_id"].nunique().to_dict()
    return {
        "clip_counts": clip_counts,
        "track_counts": track_counts,
        "track_overlap_counts": overlap_counts,
        "track_overlap_examples": overlaps,
    }


@dataclass
class MTATManifestResult:
    manifest: pd.DataFrame
    tag_columns: list[str]
    dropped_missing_audio: pd.DataFrame
    dropped_missing_labels: pd.DataFrame
    audit: dict[str, object]


def build_manifest(
    mtat_root: str | Path,
    annotations_path: str | Path,
    top_k_tags: int = 50,
    top_tags_file: str | Path | None = None,
    train_ids_file: str | Path | None = None,
    val_ids_file: str | Path | None = None,
    test_ids_file: str | Path | None = None,
    seed: int = 42,
    drop_all_negative: bool = True,
) -> MTATManifestResult:
    mtat_root = Path(mtat_root)
    clip_info_path = mtat_root / "clip_info_final.csv"
    if not clip_info_path.exists():
        raise FileNotFoundError(f"Missing MTAT metadata: {clip_info_path}")

    clip_df = read_table(clip_info_path)
    if "clip_id" not in clip_df.columns or "mp3_path" not in clip_df.columns:
        raise ValueError("clip_info_final.csv must contain clip_id and mp3_path columns.")

    clip_df["clip_id"] = clip_df["clip_id"].astype(int)
    clip_df["track_id"] = clip_df.apply(build_track_id, axis=1)
    clip_df["audio_path"] = clip_df["mp3_path"].astype(str).map(lambda p: str((mtat_root / p).resolve()) if p else "")

    has_audio = clip_df["audio_path"].map(lambda p: Path(p).exists() if p else False)
    dropped_missing_audio = clip_df.loc[~has_audio].copy()
    clip_df = clip_df.loc[has_audio].copy()

    ann_path = Path(annotations_path)
    if not ann_path.exists():
        raise FileNotFoundError(
            f"Missing annotations file: {ann_path}. Provide standard MTAT annotations_final.csv."
        )

    ann_df = read_table(ann_path)
    ann_id_col = infer_annotation_id_column(ann_df)
    tag_cols = infer_tag_columns(ann_df, ann_id_col)
    ann_df = ann_df[[ann_id_col] + tag_cols].copy()
    if ann_id_col != "clip_id":
        ann_df = ann_df.rename(columns={ann_id_col: "clip_id"})
    ann_df["clip_id"] = ann_df["clip_id"].astype(int)

    merged = clip_df.merge(ann_df, on="clip_id", how="left")

    missing_label_mask = merged[tag_cols].isna().all(axis=1)
    dropped_missing_labels = merged.loc[missing_label_mask].copy()
    merged = merged.loc[~missing_label_mask].copy()
    merged[tag_cols] = merged[tag_cols].fillna(0).astype(np.float32)

    if top_tags_file:
        with Path(top_tags_file).open("r", encoding="utf-8") as fp:
            top_tags = [line.strip() for line in fp if line.strip()]
        unknown = sorted(set(top_tags) - set(tag_cols))
        if unknown:
            raise ValueError(f"Top tags file contains tags not present in annotations: {unknown[:5]}")
    else:
        freq = merged[tag_cols].sum(axis=0).sort_values(ascending=False)
        top_tags = freq.head(top_k_tags).index.tolist()

    manifest = merged[
        [
            "clip_id",
            "track_number",
            "title",
            "artist",
            "album",
            "url",
            "segmentStart",
            "segmentEnd",
            "original_url",
            "track_id",
            "audio_path",
        ]
        + top_tags
    ].copy()

    manifest[top_tags] = manifest[top_tags].astype(np.float32)

    if drop_all_negative:
        has_positive = manifest[top_tags].sum(axis=1) > 0
        manifest = manifest.loc[has_positive].copy()

    # Assign split.
    if train_ids_file and val_ids_file and test_ids_file:
        train_ids = read_clip_id_list(train_ids_file)
        val_ids = read_clip_id_list(val_ids_file)
        test_ids = read_clip_id_list(test_ids_file)
        split = pd.Series("unassigned", index=manifest.index)
        split.loc[manifest["clip_id"].isin(train_ids)] = "train"
        split.loc[manifest["clip_id"].isin(val_ids)] = "val"
        split.loc[manifest["clip_id"].isin(test_ids)] = "test"
        unassigned = manifest.loc[split == "unassigned"]
        if not unassigned.empty:
            # Fallback assignment for clips missing from provided lists.
            fallback = assign_grouped_splits(unassigned, seed=seed)
            split.loc[unassigned.index] = fallback
    else:
        split = assign_grouped_splits(manifest, seed=seed)

    manifest["split"] = split.astype("string")
    if manifest["split"].isna().any():
        raise RuntimeError("Some clips did not receive a split assignment.")

    audit = audit_split_hygiene(manifest)
    return MTATManifestResult(
        manifest=manifest.sort_values("clip_id").reset_index(drop=True),
        tag_columns=top_tags,
        dropped_missing_audio=dropped_missing_audio,
        dropped_missing_labels=dropped_missing_labels,
        audit=audit,
    )


class MTATChunkDataset:
    """Torch dataset for chunk-based MTAT training/evaluation."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        tag_columns: Sequence[str],
        split: str,
        sample_rate: int = 16000,
        chunk_seconds: float = 3.0,
        representation: str = "logmel",
        random_crop: bool = True,
        n_mels: int = 96,
        n_fft: int = 1024,
        hop_length: int = 512,
        seed: int = 42,
        skip_bad_audio: bool = True,
        max_decode_retries: int = 10,
    ) -> None:
        self.df = manifest.loc[manifest["split"] == split].reset_index(drop=True).copy()
        self.tag_columns = list(tag_columns)
        self.sample_rate = sample_rate
        self.chunk_samples = int(chunk_seconds * sample_rate)
        self.representation = representation
        self.random_crop = random_crop
        self.rng = np.random.default_rng(seed)
        self.skip_bad_audio = skip_bad_audio
        self.max_decode_retries = max_decode_retries
        self.bad_indices: set[int] = set()
        self.logmel = (
            LogMelFrontend(
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
            )
            if representation == "logmel"
            else None
        )

    def __len__(self) -> int:
        return len(self.df)

    def _load_item(self, index: int):
        import torch

        row = self.df.iloc[index]
        audio, _ = load_audio(row["audio_path"], sample_rate=self.sample_rate)
        chunk = pad_or_crop(
            audio=audio,
            length_samples=self.chunk_samples,
            random_crop=self.random_crop,
            rng=self.rng,
        )

        if self.representation == "logmel":
            assert self.logmel is not None
            feat = self.logmel(chunk)
            x = torch.from_numpy(feat).float().unsqueeze(0)  # [1, mel, time]
        elif self.representation == "waveform":
            x = torch.from_numpy(chunk).float().unsqueeze(0)  # [1, time]
        else:
            raise ValueError(f"Unsupported representation: {self.representation}")

        y = torch.from_numpy(row[self.tag_columns].to_numpy(dtype=np.float32))
        return x, y, int(row["clip_id"])

    def __getitem__(self, index: int):
        if not self.skip_bad_audio:
            return self._load_item(index)

        if len(self.bad_indices) >= len(self.df):
            raise RuntimeError("All dataset items are marked as bad/unreadable.")

        current = int(index)
        last_exc: Exception | None = None
        for _ in range(max(1, self.max_decode_retries)):
            if current in self.bad_indices:
                current = (current + 1) % len(self.df)
                continue
            try:
                return self._load_item(current)
            except Exception as exc:
                self.bad_indices.add(current)
                last_exc = exc
                current = (current + 1) % len(self.df)

        raise RuntimeError(
            f"Failed to load a valid audio sample after {self.max_decode_retries} retries. "
            f"Bad indices tracked: {len(self.bad_indices)}."
        ) from last_exc


def iter_track_chunks(
    audio_path: str | Path,
    sample_rate: int,
    chunk_seconds: float,
    hop_seconds: float,
) -> list[np.ndarray]:
    audio, _ = load_audio(audio_path, sample_rate=sample_rate)
    chunk_size = int(chunk_seconds * sample_rate)
    hop_size = max(1, int(hop_seconds * sample_rate))
    return chunk_audio(audio, chunk_size=chunk_size, hop_size=hop_size)
