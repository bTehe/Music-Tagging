from __future__ import annotations

import argparse
from pathlib import Path

import requests


DEFAULT_URL = "https://mirg.city.ac.uk/datasets/magnatagatune/annotations_final.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download MTAT annotations_final.csv from MIRG.")
    parser.add_argument("--url", type=str, default=DEFAULT_URL)
    parser.add_argument("--output", type=Path, default=Path("MTAT/annotations_final.csv"))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and not args.force:
        print(f"File already exists: {args.output}")
        print("Use --force to overwrite.")
        return

    response = requests.get(args.url, timeout=120)
    response.raise_for_status()
    args.output.write_bytes(response.content)
    print(f"Downloaded: {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
