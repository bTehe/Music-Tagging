from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from aml_music.audio import load_audio
from aml_music.features.logmel import LogMelFrontend
from aml_music.utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute log-mel features for MTAT manifest.")
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/manifests/mtat_top50_manifest.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/features/logmel"))
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--n-mels", type=int, default=96)
    parser.add_argument("--split", type=str, default="", choices=["", "train", "val", "test"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--on-error", type=str, default="skip", choices=["skip", "raise"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.manifest)
    if args.split:
        frame = frame.loc[frame["split"] == args.split].reset_index(drop=True)
    if args.limit:
        frame = frame.head(args.limit).copy()

    out_dir = ensure_dir(args.output_dir)
    frontend = LogMelFrontend(
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
    )

    kept_indices: list[int] = []
    feature_paths: list[str] = []
    failures: list[dict[str, object]] = []

    for row in tqdm(frame.itertuples(index=True), total=len(frame), desc="extract"):
        try:
            audio, _ = load_audio(row.audio_path, sample_rate=args.sample_rate)
            feat = frontend(audio)
            feat_path = out_dir / f"{int(row.clip_id)}.npy"
            np.save(feat_path, feat)
            feature_paths.append(str(feat_path.resolve()))
            kept_indices.append(int(row.Index))
        except Exception as exc:
            if args.on_error == "raise":
                raise
            failures.append(
                {
                    "index": int(row.Index),
                    "clip_id": int(row.clip_id),
                    "audio_path": str(row.audio_path),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )

    frame = frame.loc[kept_indices].copy().reset_index(drop=True)
    frame["logmel_path"] = feature_paths
    output_manifest = out_dir / "manifest_with_logmel.csv"
    frame.to_csv(output_manifest, index=False)
    print(f"Wrote: {output_manifest}")
    print(f"Successfully extracted: {len(feature_paths)}")
    print(f"Failed extractions: {len(failures)}")

    if failures:
        failures_path = out_dir / "failed_audio.csv"
        pd.DataFrame(failures).to_csv(failures_path, index=False)
        print(f"Wrote failure log: {failures_path}")


if __name__ == "__main__":
    main()
