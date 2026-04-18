from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from aml_music.data.mtat import MTATChunkDataset
from aml_music.evaluation.metrics import multilabel_metrics
from aml_music.models.waveform_cnn import WaveformCNN
from aml_music.training import run_epoch
from aml_music.utils import ensure_dir, set_seed, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train waveform CNN ablation on MTAT.")
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/manifests/mtat_top50_manifest.csv"))
    parser.add_argument("--tags", type=Path, default=Path("artifacts/manifests/mtat_top50_tags.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/runs/mtat_waveform"))
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--chunk-seconds", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--on-bad-audio", type=str, default="skip", choices=["skip", "raise"])
    parser.add_argument("--max-decode-retries", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = ensure_dir(args.output_dir)
    device = torch.device(args.device)
    use_pin_memory = device.type == "cuda"

    manifest = pd.read_csv(args.manifest)
    with args.tags.open("r", encoding="utf-8") as fp:
        tag_names = json.load(fp)

    train_ds = MTATChunkDataset(
        manifest=manifest,
        tag_columns=tag_names,
        split="train",
        sample_rate=args.sample_rate,
        chunk_seconds=args.chunk_seconds,
        representation="waveform",
        random_crop=True,
        seed=args.seed,
        skip_bad_audio=(args.on_bad_audio == "skip"),
        max_decode_retries=args.max_decode_retries,
    )
    val_ds = MTATChunkDataset(
        manifest=manifest,
        tag_columns=tag_names,
        split="val",
        sample_rate=args.sample_rate,
        chunk_seconds=args.chunk_seconds,
        representation="waveform",
        random_crop=False,
        seed=args.seed,
        skip_bad_audio=(args.on_bad_audio == "skip"),
        max_decode_retries=args.max_decode_retries,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=use_pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_pin_memory,
    )

    model = WaveformCNN(num_tags=len(tag_names)).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history: list[dict[str, float]] = []
    best_map = -1.0
    best_epoch = -1
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        train_out = run_epoch(model, train_loader, criterion, optimizer, device)
        val_out = run_epoch(model, val_loader, criterion, None, device)
        val_metrics = multilabel_metrics(val_out.y_true, val_out.y_prob, tag_names)
        row = {
            "epoch": epoch,
            "train_loss": train_out.loss,
            "val_loss": val_out.loss,
            "val_macro_roc_auc": val_metrics["macro_roc_auc"],
            "val_macro_pr_auc": val_metrics["macro_pr_auc"],
            "val_map": val_metrics["map"],
        }
        history.append(row)
        print(row)

        if row["val_map"] > best_map:
            best_map = row["val_map"]
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "tag_names": tag_names,
                    "args": vars(args),
                },
                out_dir / "best.pt",
            )
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
                break

    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    write_json(
        {
            "best_epoch": best_epoch,
            "best_val_map": best_map,
            "num_tags": len(tag_names),
            "train_examples": len(train_ds),
            "val_examples": len(val_ds),
        },
        out_dir / "summary.json",
    )
    print(f"Wrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()
