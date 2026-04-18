from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from aml_music.audio import load_audio
from aml_music.evaluation.metrics import multilabel_metrics
from aml_music.utils import ensure_dir, write_json


PANN_LABELS_URL = "https://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"
PANN_CHECKPOINT_URL = "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"
PANN_CHECKPOINT_MIN_BYTES = int(3e8)


def _download_file(url: str, path: Path, timeout_sec: int = 120) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with requests.get(url, stream=True, timeout=timeout_sec) as resp:
        resp.raise_for_status()
        with tmp_path.open("wb") as fp:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fp.write(chunk)
    tmp_path.replace(path)


def _ensure_panns_assets(data_root: Path) -> tuple[Path, Path]:
    labels_path = data_root / "class_labels_indices.csv"
    ckpt_path = data_root / "Cnn14_mAP=0.431.pth"

    if (not labels_path.exists()) or labels_path.stat().st_size < 1024:
        _download_file(PANN_LABELS_URL, labels_path)

    if (not ckpt_path.exists()) or ckpt_path.stat().st_size < PANN_CHECKPOINT_MIN_BYTES:
        _download_file(PANN_CHECKPOINT_URL, ckpt_path)

    return labels_path, ckpt_path


class PANNExtractor:
    def __init__(self, device: str = "cpu") -> None:
        data_root = Path.home() / "panns_data"
        try:
            _, ckpt_path = _ensure_panns_assets(data_root)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download/prepare PANN assets in {data_root}. "
                "Check internet access and write permissions."
            ) from exc

        try:
            from panns_inference import AudioTagging
        except ImportError as exc:
            raise RuntimeError(
                "panns_inference is not installed. Install with: pip install panns-inference"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                "Failed to import panns_inference. Package exists but initialization failed. "
                "Ensure ~/.panns_data assets are readable and compatible."
            ) from exc

        self.model = AudioTagging(device=device, checkpoint_path=str(ckpt_path))

    def embed(self, audio_path: str) -> np.ndarray:
        audio, _ = load_audio(audio_path, sample_rate=32000)
        audio = np.expand_dims(audio, axis=0)
        _, embedding = self.model.inference(audio)
        return np.asarray(embedding[0], dtype=np.float32)


class MusicnnExtractor:
    def __init__(self) -> None:
        try:
            from musicnn.extractor import extractor as _extractor  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "musicnn is not installed. Install with: pip install musicnn"
            ) from exc
        self._extractor = _extractor

    def embed(self, audio_path: str) -> np.ndarray:
        _, _, features = self._extractor(audio_path, model="MSD_musicnn", extract_features=True)
        feat = np.asarray(features, dtype=np.float32)
        if feat.ndim == 1:
            return feat
        return feat.mean(axis=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pretrained encoder branch: extract embeddings and train linear multi-label head."
    )
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/manifests/mtat_top50_manifest.csv"))
    parser.add_argument("--tags", type=Path, default=Path("artifacts/manifests/mtat_top50_tags.json"))
    parser.add_argument("--backend", type=str, default="panns", choices=["panns", "musicnn"])
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/runs/mtat_pretrained"))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--on-bad-audio", type=str, default="skip", choices=["skip", "raise"])
    parser.add_argument(
        "--strict-backend",
        action="store_true",
        help="If set, fail when requested backend is unavailable instead of falling back.",
    )
    return parser.parse_args()


def build_extractor(
    backend: str,
    device: str,
    strict_backend: bool = False,
) -> tuple[object, str, str | None]:
    if backend == "panns":
        return PANNExtractor(device=device), "panns", None
    if backend == "musicnn":
        try:
            return MusicnnExtractor(), "musicnn", None
        except Exception as exc:
            if strict_backend:
                raise RuntimeError(
                    "Requested backend 'musicnn' is unavailable and --strict-backend is set."
                ) from exc

            fallback_reason = (
                "musicnn is unavailable in this environment; "
                f"falling back to panns. root_error={type(exc).__name__}: {exc}"
            )
            print(fallback_reason)
            return PANNExtractor(device=device), "panns", fallback_reason
    raise ValueError(backend)


def extract_split_embeddings(
    frame: pd.DataFrame,
    tag_names: list[str],
    extractor,
    max_rows: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    if max_rows is not None:
        frame = frame.head(max_rows).copy()
    frame = frame.reset_index(drop=True)
    labels_matrix = frame[tag_names].to_numpy(dtype=np.float32)

    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    failures: list[dict[str, object]] = []

    for idx, row in enumerate(tqdm(frame.itertuples(index=False), total=len(frame), desc="embed")):
        try:
            x_rows.append(extractor.embed(row.audio_path))
            y_rows.append(labels_matrix[idx])
        except Exception as exc:
            failures.append(
                {
                    "index": idx,
                    "clip_id": int(getattr(row, "clip_id")),
                    "audio_path": str(getattr(row, "audio_path")),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )

    if not x_rows:
        raise RuntimeError("No embeddings were extracted successfully for this split.")

    x = np.stack(x_rows, axis=0)
    y = np.stack(y_rows, axis=0)
    return x, y, failures


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)

    manifest = pd.read_csv(args.manifest)
    with args.tags.open("r", encoding="utf-8") as fp:
        tag_names = json.load(fp)

    extractor, resolved_backend, fallback_reason = build_extractor(
        args.backend,
        args.device,
        strict_backend=args.strict_backend,
    )

    train_df = manifest.loc[manifest["split"] == "train"].copy()
    val_df = manifest.loc[manifest["split"] == "val"].copy()
    test_df = manifest.loc[manifest["split"] == "test"].copy()

    x_train, y_train, fail_train = extract_split_embeddings(
        train_df,
        tag_names,
        extractor,
        max_rows=args.max_train,
    )
    x_val, y_val, fail_val = extract_split_embeddings(
        val_df,
        tag_names,
        extractor,
        max_rows=args.max_val,
    )
    x_test, y_test, fail_test = extract_split_embeddings(
        test_df,
        tag_names,
        extractor,
        max_rows=args.max_test,
    )

    if args.on_bad_audio == "raise" and (fail_train or fail_val or fail_test):
        raise RuntimeError(
            "Encountered unreadable audio files during embedding extraction in strict mode."
        )

    clf = OneVsRestClassifier(
        LogisticRegression(max_iter=1000, solver="liblinear"),
        n_jobs=-1,
    )
    clf.fit(x_train, y_train)

    val_prob = clf.predict_proba(x_val)
    test_prob = clf.predict_proba(x_test)
    val_metric = multilabel_metrics(y_val, val_prob, tag_names=tag_names)
    test_metric = multilabel_metrics(y_test, test_prob, tag_names=tag_names)

    np.savez_compressed(out_dir / "embeddings_train.npz", x=x_train, y=y_train)
    np.savez_compressed(out_dir / "embeddings_val.npz", x=x_val, y=y_val)
    np.savez_compressed(out_dir / "embeddings_test.npz", x=x_test, y=y_test)
    all_failures = fail_train + fail_val + fail_test
    if all_failures:
        pd.DataFrame(all_failures).to_csv(out_dir / "failed_audio.csv", index=False)

    write_json(
        {
            "requested_backend": args.backend,
            "resolved_backend": resolved_backend,
            "backend_fallback_reason": fallback_reason,
            "failed_audio_count": len(all_failures),
            "val": {
                "macro_roc_auc": val_metric["macro_roc_auc"],
                "macro_pr_auc": val_metric["macro_pr_auc"],
                "map": val_metric["map"],
            },
            "test": {
                "macro_roc_auc": test_metric["macro_roc_auc"],
                "macro_pr_auc": test_metric["macro_pr_auc"],
                "map": test_metric["map"],
            },
        },
        out_dir / "summary.json",
    )
    print("Val:", {k: val_metric[k] for k in ("macro_roc_auc", "macro_pr_auc", "map")})
    print("Test:", {k: test_metric[k] for k in ("macro_roc_auc", "macro_pr_auc", "map")})
    print(f"Wrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()
