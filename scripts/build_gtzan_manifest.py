from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from aml_music.data.gtzan import build_gtzan_manifest
from aml_music.utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build GTZAN auxiliary manifest.")
    parser.add_argument("--gtzan-root", type=Path, default=Path("GTZAN"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/manifests"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)
    df = build_gtzan_manifest(args.gtzan_root, seed=args.seed)
    out_path = out_dir / "gtzan_manifest.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote: {out_path}")
    print(df["split"].value_counts().to_dict())


if __name__ == "__main__":
    main()
