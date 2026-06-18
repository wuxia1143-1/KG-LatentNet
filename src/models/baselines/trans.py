from __future__ import annotations

import torch
from torch import nn

from src.models.baselines.common import StaticFusionHead, sequence_observed_mask


class TRANSModel(nn.Module):
    """Transformer baseline for masked longitudinal sequences with delta-time encoding."""

    def __init__(self, static_dim: int, dynamic_dim: int, treatment_dim: int, hidden_dim: int = 64, num_heads: int = 4) -> None:
        super().__init__()
        input_dim = dynamic_dim * 2 + treatment_dim + 1
        self.input_projection = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.time_projection = nn.Sequential(nn.Linear(1, hidden_dim), nn.Tanh())
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.fusion_head = StaticFusionHead(static_dim, hidden_dim, hidden_dim)

    def forward(self, batch: dict[str, torch.Tensor], prior_matrix: torch.Tensor | None = None) -> torch.Tensor:
        del prior_matrix
        dynamic = batch["dynamic_features"]
        mask = batch["dynamic_mask"]
        treatment = batch["treatment_features"]
        delta = batch["delta_time"].unsqueeze(-1)
        observed = sequence_observed_mask(batch)

        sequence_input = torch.cat([dynamic, mask, treatment, delta], dim=-1)
        cumulative_time = torch.cumsum(delta, dim=1)
        encoded = self.input_projection(sequence_input) + self.time_projection(cumulative_time)
        cls = self.cls_token.expand(dynamic.shape[0], -1, -1)
        encoded = torch.cat([cls, encoded], dim=1)
        padding_mask = torch.cat(
            [torch.zeros((dynamic.shape[0], 1), dtype=torch.bool, device=dynamic.device), ~observed],
            dim=1,
        )
        hidden = self.encoder(encoded, src_key_padding_mask=padding_mask)
        return self.fusion_head(batch, hidden[:, 0])
