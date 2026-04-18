"""1D CNN baseline for raw-waveform comparisons."""

from __future__ import annotations

import torch
from torch import nn


class WaveBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=9, stride=stride, padding=4, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class WaveformCNN(nn.Module):
    def __init__(self, num_tags: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            WaveBlock(1, 32),
            WaveBlock(32, 64),
            WaveBlock(64, 128),
            WaveBlock(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(256, num_tags)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).squeeze(-1)
        return self.dropout(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))
