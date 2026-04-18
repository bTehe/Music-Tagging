"""Training loops for MTAT models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


@dataclass
class EpochOutput:
    loss: float
    y_true: np.ndarray
    y_prob: np.ndarray


def run_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> EpochOutput:
    is_train = optimizer is not None
    model.train(is_train)

    losses: list[float] = []
    true_all: list[np.ndarray] = []
    prob_all: list[np.ndarray] = []

    for batch in dataloader:
        x, y, _ = batch
        x = x.to(device)
        y = y.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        logits = model(x)
        loss = criterion(logits, y)

        if is_train:
            loss.backward()
            optimizer.step()

        losses.append(float(loss.detach().cpu().item()))
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        true_all.append(y.detach().cpu().numpy())
        prob_all.append(probs)

    return EpochOutput(
        loss=float(np.mean(losses)) if losses else float("nan"),
        y_true=np.concatenate(true_all, axis=0) if true_all else np.empty((0, 0)),
        y_prob=np.concatenate(prob_all, axis=0) if prob_all else np.empty((0, 0)),
    )


def save_checkpoint(
    output_dir: str | Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    payload: dict[str, Any] | None = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / f"checkpoint_epoch_{epoch:03d}.pt"
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if payload:
        state.update(payload)
    torch.save(state, ckpt_path)
    return ckpt_path
