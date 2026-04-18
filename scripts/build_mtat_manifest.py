from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from aml_music.data.mtat import build_manifest
from aml_music.utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build MTAT Top-K manifest with split hygiene audit."
    )
    parser.add_argument("--mtat-root", type=Path, default=Path("MTAT"))
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("MTAT/annotations_final.csv"),
        help="Path to MTAT annotations file (required for labels).",
    )
    parser.add_argument("--top-k-tags", type=int, default=50)
    parser.add_argument("--top-tags-file", type=Path, default=None)
    parser.add_argument("--train-ids", type=Path, default=Path("MTAT/train_clipids.csv"))
    parser.add_argument("--val-ids", type=Path, default=Path("MTAT/valid_clipids.csv"))
    parser.add_argument("--test-ids", type=Path, default=Path("MTAT/test_clipids.csv"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--keep-all-negative",
        action="store_true",
        help="Keep clips with no positive labels in selected top tags.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/manifests"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)

    train_ids = args.train_ids if args.train_ids.exists() else None
    val_ids = args.val_ids if args.val_ids.exists() else None
    test_ids = args.test_ids if args.test_ids.exists() else None

    result = build_manifest(
        mtat_root=args.mtat_root,
        annotations_path=args.annotations,
        top_k_tags=args.top_k_tags,
        top_tags_file=args.top_tags_file,
        train_ids_file=train_ids,
        val_ids_file=val_ids,
        test_ids_file=test_ids,
        seed=args.seed,
        drop_all_negative=not args.keep_all_negative,
    )

    manifest_path = out_dir / "mtat_top50_manifest.csv"
    tags_path = out_dir / "mtat_top50_tags.json"
    audit_path = out_dir / "mtat_split_audit.json"
    missing_audio_path = out_dir / "dropped_missing_audio.csv"
    missing_labels_path = out_dir / "dropped_missing_labels.csv"

    result.manifest.to_csv(manifest_path, index=False)
    result.dropped_missing_audio.to_csv(missing_audio_path, index=False)
    result.dropped_missing_labels.to_csv(missing_labels_path, index=False)

    with tags_path.open("w", encoding="utf-8") as fp:
        json.dump(result.tag_columns, fp, indent=2)
    with audit_path.open("w", encoding="utf-8") as fp:
        json.dump(result.audit, fp, indent=2)

    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote tags: {tags_path}")
    print(f"Wrote split audit: {audit_path}")
    print("Split clip counts:", result.audit.get("clip_counts", {}))
    print("Split track counts:", result.audit.get("track_counts", {}))
    print("Track overlap counts:", result.audit.get("track_overlap_counts", {}))


if __name__ == "__main__":
    main()
