from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from aml_music.audio import load_audio, pad_or_crop
from aml_music.features.logmel import LogMelFrontend
from aml_music.models.short_chunk_cnn import ShortChunkCNN
from aml_music.utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export simple saliency maps for qualitative analysis.")
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/manifests/mtat_top50_manifest.csv"))
    parser.add_argument("--tags", type=Path, default=Path("artifacts/manifests/mtat_top50_tags.json"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/reports/saliency"))
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--num-examples", type=int, default=12)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--chunk-seconds", type=float, default=3.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _load_checkpoint(path: Path, device: torch.device):
    """Load trusted project checkpoints across PyTorch versions."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)

    manifest = pd.read_csv(args.manifest)
    with args.tags.open("r", encoding="utf-8") as fp:
        tags = json.load(fp)
    frame = manifest.loc[manifest["split"] == args.split].head(args.num_examples).copy()

    device = torch.device(args.device)
    model = ShortChunkCNN(num_tags=len(tags)).to(device)
    ckpt = _load_checkpoint(args.checkpoint, device=device)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    frontend = LogMelFrontend(sample_rate=args.sample_rate)
    chunk_samples = int(args.chunk_seconds * args.sample_rate)

    for row in frame.itertuples(index=False):
        audio, _ = load_audio(row.audio_path, sample_rate=args.sample_rate)
        chunk = pad_or_crop(audio, length_samples=chunk_samples, random_crop=False)
        mel = frontend(chunk)
        x = torch.from_numpy(mel).float().unsqueeze(0).unsqueeze(0).to(device)
        x.requires_grad_(True)
        logits = model(x)
        top_idx = int(torch.argmax(logits, dim=1).item())
        score = logits[0, top_idx]

        model.zero_grad(set_to_none=True)
        score.backward()
        saliency = x.grad.detach().abs().squeeze().cpu().numpy()
        saliency = saliency / (saliency.max() + 1e-8)

        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        axes[0].imshow(mel, origin="lower", aspect="auto")
        axes[0].set_title(f"clip_id={int(row.clip_id)} top_tag={tags[top_idx]}")
        axes[1].imshow(saliency, origin="lower", aspect="auto", cmap="hot")
        axes[1].set_title("Saliency")
        plt.tight_layout()
        fig.savefig(out_dir / f"saliency_{int(row.clip_id)}.png", dpi=150)
        plt.close(fig)

    print(f"Wrote saliency maps to: {out_dir}")


if __name__ == "__main__":
    main()
