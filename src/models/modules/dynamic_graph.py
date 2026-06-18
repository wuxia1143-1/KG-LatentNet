from __future__ import annotations

import torch
from torch import nn


class DynamicGraphPrior(nn.Module):
    def __init__(self, dynamic_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dynamic_dim, hidden_dim)

    def forward(self, dynamic_features: torch.Tensor, dynamic_mask: torch.Tensor, prior_matrix: torch.Tensor | None = None) -> torch.Tensor:
        masked = dynamic_features * dynamic_mask
        if prior_matrix is not None and prior_matrix.numel() > 0 and prior_matrix.shape[0] == dynamic_features.shape[-1]:
            usable = prior_matrix[: dynamic_features.shape[-1], : dynamic_features.shape[-1]].to(dynamic_features.device, dynamic_features.dtype)
            masked = torch.matmul(masked, usable)
        return torch.tanh(self.proj(masked))
