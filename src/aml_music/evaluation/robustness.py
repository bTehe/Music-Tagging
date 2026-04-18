"""Lightweight audio perturbations for robustness checks."""

from __future__ import annotations

import numpy as np


def add_noise(audio: np.ndarray, snr_db: float = 20.0, rng: np.random.Generator | None = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    signal_power = np.mean(audio**2) + 1e-8
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=audio.shape).astype(np.float32)
    return (audio + noise).astype(np.float32)


def dynamic_range_compression(audio: np.ndarray, gamma: float = 0.6) -> np.ndarray:
    sign = np.sign(audio)
    mag = np.abs(audio)
    comp = sign * (mag**gamma)
    return comp.astype(np.float32)


def time_stretch_resample(audio: np.ndarray, rate: float = 1.1) -> np.ndarray:
    """Approximate time-stretch by resampling and length restoration."""
    if rate <= 0:
        raise ValueError("rate must be > 0")
    import torch
    import torch.nn.functional as F

    x = torch.from_numpy(audio).float().view(1, 1, -1)
    target = int(x.shape[-1] / rate)
    stretched = F.interpolate(x, size=target, mode="linear", align_corners=False)
    restored = F.interpolate(stretched, size=x.shape[-1], mode="linear", align_corners=False)
    return restored.view(-1).cpu().numpy().astype(np.float32)
