"""Audio loading and chunking helpers."""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np


def _load_with_torchaudio(path: str | Path, sample_rate: int) -> tuple[np.ndarray, int]:
    import torch
    import torchaudio

    wav, sr = torchaudio.load(str(path))
    if wav.ndim == 2 and wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
        sr = sample_rate
    return wav.squeeze(0).cpu().numpy().astype(np.float32), sr


def _load_with_librosa(path: str | Path, sample_rate: int) -> tuple[np.ndarray, int]:
    import librosa

    # librosa emits warnings when it falls back from PySoundFile to audioread.
    # We intentionally suppress those expected warnings to keep training logs clean.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*PySoundFile failed\. Trying audioread instead\..*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*__audioread_load.*",
            category=FutureWarning,
        )
        wav, sr = librosa.load(str(path), sr=sample_rate, mono=True)
    return wav.astype(np.float32), sr


def load_audio(path: str | Path, sample_rate: int) -> tuple[np.ndarray, int]:
    """Load mono audio and resample to target sample rate."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    torchaudio_error: Exception | None = None
    try:
        return _load_with_torchaudio(path, sample_rate)
    except Exception as exc:
        torchaudio_error = exc

    librosa_error: Exception | None = None
    try:
        return _load_with_librosa(path, sample_rate)
    except Exception as exc:
        librosa_error = exc

    raise RuntimeError(
        "Failed to load audio from "
        f"{path}. torchaudio_error={repr(torchaudio_error)} "
        f"librosa_error={repr(librosa_error)}"
    )


def pad_or_crop(
    audio: np.ndarray,
    length_samples: int,
    random_crop: bool = False,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return a fixed-length 1D signal."""
    if audio.shape[0] == length_samples:
        return audio
    if audio.shape[0] > length_samples:
        if random_crop:
            if rng is None:
                rng = np.random.default_rng()
            max_start = audio.shape[0] - length_samples
            start = int(rng.integers(0, max_start + 1))
        else:
            start = max((audio.shape[0] - length_samples) // 2, 0)
        return audio[start : start + length_samples]

    pad = length_samples - audio.shape[0]
    return np.pad(audio, (0, pad), mode="constant")


def chunk_audio(audio: np.ndarray, chunk_size: int, hop_size: int) -> list[np.ndarray]:
    """Slice a waveform into fixed windows."""
    if audio.shape[0] <= chunk_size:
        return [pad_or_crop(audio, chunk_size, random_crop=False)]

    chunks: list[np.ndarray] = []
    for start in range(0, audio.shape[0] - chunk_size + 1, hop_size):
        chunks.append(audio[start : start + chunk_size])
    if (audio.shape[0] - chunk_size) % hop_size != 0:
        chunks.append(audio[-chunk_size:])
    return chunks
