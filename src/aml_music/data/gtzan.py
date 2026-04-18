"""GTZAN auxiliary dataset utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


GTZAN_GENRES = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock",
]


def build_gtzan_manifest(
    gtzan_root: str | Path,
    seed: int = 42,
    train_size: float = 0.8,
    val_size: float = 0.1,
) -> pd.DataFrame:
    gtzan_root = Path(gtzan_root)
    audio_root = gtzan_root / "genres_original"
    if not audio_root.exists():
        raise FileNotFoundError(f"Missing GTZAN folder: {audio_root}")

    rows: list[dict[str, object]] = []
    for genre in GTZAN_GENRES:
        genre_dir = audio_root / genre
        if not genre_dir.exists():
            continue
        for wav_path in sorted(genre_dir.glob("*.wav")):
            rows.append(
                {
                    "audio_path": str(wav_path.resolve()),
                    "genre": genre,
                    "track_id": wav_path.stem,
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("No GTZAN audio files found.")

    # Stratified split on file-level labels.
    train_val, test = train_test_split(
        frame,
        test_size=1.0 - (train_size + val_size),
        random_state=seed,
        stratify=frame["genre"],
    )
    val_ratio = val_size / (train_size + val_size)
    train, val = train_test_split(
        train_val,
        test_size=val_ratio,
        random_state=seed,
        stratify=train_val["genre"],
    )

    train = train.assign(split="train")
    val = val.assign(split="val")
    test = test.assign(split="test")
    return pd.concat([train, val, test], axis=0).reset_index(drop=True)
