"""Track-level pooling for chunk predictions."""

from __future__ import annotations

import numpy as np


def mean_pool(chunk_probs: np.ndarray) -> np.ndarray:
    return chunk_probs.mean(axis=0)


def max_pool(chunk_probs: np.ndarray) -> np.ndarray:
    return chunk_probs.max(axis=0)


def attention_pool(chunk_probs: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Confidence-weighted average pooling over chunk-level probabilities."""
    if chunk_probs.ndim != 2:
        raise ValueError("chunk_probs must be [num_chunks, num_tags]")
    confidence = chunk_probs.max(axis=1) / max(temperature, 1e-6)
    confidence = confidence - confidence.max()
    weights = np.exp(confidence)
    weights = weights / (weights.sum() + 1e-8)
    return (weights[:, None] * chunk_probs).sum(axis=0)


def aggregate_probs(chunk_probs: np.ndarray, mode: str) -> np.ndarray:
    if mode == "mean":
        return mean_pool(chunk_probs)
    if mode == "max":
        return max_pool(chunk_probs)
    if mode == "attention":
        return attention_pool(chunk_probs)
    raise ValueError(f"Unsupported pooling mode: {mode}")
