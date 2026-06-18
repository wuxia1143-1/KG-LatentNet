from __future__ import annotations

import torch
from torch import nn


class ContributionFusion(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, *parts: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat(parts, dim=-1))
