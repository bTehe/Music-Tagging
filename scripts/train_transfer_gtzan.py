from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from aml_music.audio import chunk_audio, load_audio
from aml_music.data.gtzan import build_gtzan_manifest
from aml_music.evaluation.metrics import genre_metrics
from aml_music.features.logmel import LogMelFrontend
from aml_music.models.short_chunk_cnn import ShortChunkCNN
from aml_music.utils import ensure_dir, set_seed, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transfer experiment: MTAT encoder -> GTZAN linear probe."
    )
    parser.add_argument("--gtzan-root", type=Path, default=Path("GTZAN"))
    parser.add_argument("--mtat-checkpoint", type=Path, default=Path("artifacts/runs/mtat_cnn_baseline/best.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/runs/gtzan_transfer"))
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--chunk-seconds", type=float, default=3.0)
    parser.add_argument("--hop-seconds", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--on-bad-audio", type=str, default="skip", choices=["skip", "raise"])
    parser.add_argument(
        "--skip-precheck-audio",
        action="store_true",
        help="Disable GTZAN precheck that filters unreadable audio files before embedding.",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _load_checkpoint(path: Path, device: torch.device):
    """Load trusted project checkpoints across PyTorch versions."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _load_encoder(checkpoint: Path, device: torch.device) -> ShortChunkCNN:
    ckpt = _load_checkpoint(checkpoint, device=device)
    model = ShortChunkCNN(num_tags=len(ckpt.get("tag_names", [])) or 50)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


def _is_wav_readable(path: str) -> tuple[bool, str | None]:
    try:
        import soundfile as sf

        sf.info(path)
        return True, None
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _prefilter_manifest_readable_audio(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    keep_indices: list[int] = []
    failures: list[dict[str, object]] = []

    for idx, row in frame.iterrows():
        ok, err = _is_wav_readable(str(row["audio_path"]))
        if ok:
            keep_indices.append(idx)
        else:
            failures.append(
                {
                    "split": str(row["split"]),
                    "track_id": str(row["track_id"]),
                    "audio_path": str(row["audio_path"]),
                    "genre": str(row["genre"]),
                    "error": err,
                }
            )

    filtered = frame.loc[keep_indices].reset_index(drop=True)
    return filtered, failures


def _extract_embedding_pair(
    frame: pd.DataFrame,
    transfer_model: ShortChunkCNN,
    random_model: ShortChunkCNN,
    sample_rate: int,
    chunk_seconds: float,
    hop_seconds: float,
    device: torch.device,
    on_bad_audio: str,
    split_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]]]:
    frontend = LogMelFrontend(sample_rate=sample_rate)
    chunk_size = int(chunk_seconds * sample_rate)
    hop_size = max(1, int(hop_seconds * sample_rate))

    transfer_rows: list[np.ndarray] = []
    random_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    failures: list[dict[str, object]] = []

    with torch.no_grad():
        for item in tqdm(frame.itertuples(index=False), total=len(frame), desc=f"embed {split_name}"):
            try:
                audio, _ = load_audio(item.audio_path, sample_rate=sample_rate)
                chunks = chunk_audio(audio, chunk_size=chunk_size, hop_size=hop_size)
                feats = np.stack([frontend(ch) for ch in chunks], axis=0)
                x = torch.from_numpy(feats).float().unsqueeze(1).to(device)
                z_transfer = transfer_model.forward_features(x).mean(dim=0).cpu().numpy().astype(np.float32)
                z_random = random_model.forward_features(x).mean(dim=0).cpu().numpy().astype(np.float32)
                transfer_rows.append(z_transfer)
                random_rows.append(z_random)
                y_rows.append(int(item.genre_id))
            except Exception as exc:
                if on_bad_audio == "raise":
                    raise RuntimeError(
                        f"Failed to decode audio in split={split_name}: {item.audio_path}"
                    ) from exc
                failures.append(
                    {
                        "split": split_name,
                        "track_id": str(item.track_id),
                        "audio_path": str(item.audio_path),
                        "genre": str(item.genre),
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                )

    if not transfer_rows:
        raise RuntimeError(f"No readable audio found in GTZAN split '{split_name}'.")

    return (
        np.stack(transfer_rows, axis=0),
        np.stack(random_rows, axis=0),
        np.asarray(y_rows, dtype=np.int64),
        failures,
    )


def _fit_probe(x_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    clf = LogisticRegression(max_iter=2000, solver="lbfgs", multi_class="auto")
    clf.fit(x_train, y_train)
    return clf


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = ensure_dir(args.output_dir)

    manifest = build_gtzan_manifest(args.gtzan_root, seed=args.seed)
    if args.max_files:
        manifest = manifest.groupby("split").head(args.max_files).reset_index(drop=True)

    precheck_failures: list[dict[str, object]] = []
    if not args.skip_precheck_audio:
        manifest, precheck_failures = _prefilter_manifest_readable_audio(manifest)
        if precheck_failures:
            pd.DataFrame(precheck_failures).to_csv(out_dir / "gtzan_bad_audio_precheck.csv", index=False)
            print(f"Filtered unreadable GTZAN files in precheck: {len(precheck_failures)}")

    manifest.to_csv(out_dir / "gtzan_manifest.csv", index=False)

    le = LabelEncoder()
    manifest["genre_id"] = le.fit_transform(manifest["genre"])

    device = torch.device(args.device)
    transfer_encoder = _load_encoder(args.mtat_checkpoint, device=device)
    random_encoder = ShortChunkCNN(num_tags=50).to(device).eval()

    train_df = manifest.loc[manifest["split"] == "train"].copy()
    val_df = manifest.loc[manifest["split"] == "val"].copy()
    test_df = manifest.loc[manifest["split"] == "test"].copy()

    x_train_transfer, x_train_random, y_train, fail_train = _extract_embedding_pair(
        train_df,
        transfer_encoder,
        random_encoder,
        args.sample_rate,
        args.chunk_seconds,
        args.hop_seconds,
        device=device,
        on_bad_audio=args.on_bad_audio,
        split_name="train",
    )
    x_val_transfer, x_val_random, y_val, fail_val = _extract_embedding_pair(
        val_df,
        transfer_encoder,
        random_encoder,
        args.sample_rate,
        args.chunk_seconds,
        args.hop_seconds,
        device=device,
        on_bad_audio=args.on_bad_audio,
        split_name="val",
    )
    x_test_transfer, x_test_random, y_test, fail_test = _extract_embedding_pair(
        test_df,
        transfer_encoder,
        random_encoder,
        args.sample_rate,
        args.chunk_seconds,
        args.hop_seconds,
        device=device,
        on_bad_audio=args.on_bad_audio,
        split_name="test",
    )

    transfer_probe = _fit_probe(x_train_transfer, y_train)
    random_probe = _fit_probe(x_train_random, y_train)

    val_pred_transfer = transfer_probe.predict(x_val_transfer)
    test_pred_transfer = transfer_probe.predict(x_test_transfer)
    val_pred_random = random_probe.predict(x_val_random)
    test_pred_random = random_probe.predict(x_test_random)

    val_metric_transfer = genre_metrics(y_val, val_pred_transfer)
    test_metric_transfer = genre_metrics(y_test, test_pred_transfer)
    val_metric_random = genre_metrics(y_val, val_pred_random)
    test_metric_random = genre_metrics(y_test, test_pred_random)

    np.savez_compressed(out_dir / "transfer_embeddings_train.npz", x=x_train_transfer, y=y_train)
    np.savez_compressed(out_dir / "transfer_embeddings_val.npz", x=x_val_transfer, y=y_val)
    np.savez_compressed(out_dir / "transfer_embeddings_test.npz", x=x_test_transfer, y=y_test)
    all_failures = fail_train + fail_val + fail_test
    if all_failures:
        pd.DataFrame(all_failures).to_csv(out_dir / "failed_audio.csv", index=False)

    summary = {
        "label_mapping": {cls: int(idx) for idx, cls in enumerate(le.classes_)},
        "precheck_failed_audio_count": len(precheck_failures),
        "failed_audio_count": len(all_failures),
        "transfer_encoder": {
            "val": {
                "accuracy": val_metric_transfer["accuracy"],
                "macro_f1": val_metric_transfer["macro_f1"],
                "balanced_accuracy": val_metric_transfer["balanced_accuracy"],
            },
            "test": {
                "accuracy": test_metric_transfer["accuracy"],
                "macro_f1": test_metric_transfer["macro_f1"],
                "balanced_accuracy": test_metric_transfer["balanced_accuracy"],
            },
        },
        "random_encoder_baseline": {
            "val": {
                "accuracy": val_metric_random["accuracy"],
                "macro_f1": val_metric_random["macro_f1"],
                "balanced_accuracy": val_metric_random["balanced_accuracy"],
            },
            "test": {
                "accuracy": test_metric_random["accuracy"],
                "macro_f1": test_metric_random["macro_f1"],
                "balanced_accuracy": test_metric_random["balanced_accuracy"],
            },
        },
    }
    write_json(summary, out_dir / "summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
