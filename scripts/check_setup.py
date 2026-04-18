from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick dataset setup checks.")
    parser.add_argument("--mtat-root", type=Path, default=Path("MTAT"))
    parser.add_argument("--gtzan-root", type=Path, default=Path("GTZAN"))
    parser.add_argument("--annotations", type=Path, default=Path("MTAT/annotations_final.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    mtat_mp3 = len(list(args.mtat_root.rglob("*.mp3"))) if args.mtat_root.exists() else 0
    gtzan_wav = len(list((args.gtzan_root / "genres_original").rglob("*.wav"))) if args.gtzan_root.exists() else 0

    print(f"MTAT root exists: {args.mtat_root.exists()}")
    print(f"MTAT mp3 files: {mtat_mp3}")
    print(f"MTAT clip_info_final.csv exists: {(args.mtat_root / 'clip_info_final.csv').exists()}")
    print(f"MTAT annotations file exists: {args.annotations.exists()}")
    print(f"GTZAN root exists: {args.gtzan_root.exists()}")
    print(f"GTZAN wav files: {gtzan_wav}")

    if not args.annotations.exists():
        print(
            "Missing MTAT annotations_final.csv. Add it to MTAT/ or pass --annotations path when building manifest."
        )


if __name__ == "__main__":
    main()
