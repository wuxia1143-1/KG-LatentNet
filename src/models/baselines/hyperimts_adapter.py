from __future__ import annotations

import torch
from torch import nn

from src.models.baselines.common import StaticFusionHead, sequence_observed_mask


class HyperIMTSAdapter(nn.Module):
    """Irregular multivariate time-series adapter with time-decayed recurrent states."""

    def __init__(self, static_dim: int, dynamic_dim: int, treatment_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        input_dim = dynamic_dim * 2 + treatment_dim + 1
        self.input_encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.decay = nn.Sequential(nn.Linear(1, hidden_dim), nn.Softplus())
        self.cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.fusion_head = StaticFusionHead(static_dim, hidden_dim, hidden_dim)

    def forward(self, batch: dict[str, torch.Tensor], prior_matrix: torch.Tensor | None = None) -> torch.Tensor:
        del prior_matrix
        dynamic = batch["dynamic_features"]
        mask = batch["dynamic_mask"]
        treatment = batch["treatment_features"]
        delta = batch["delta_time"].unsqueeze(-1)
        observed = sequence_observed_mask(batch)

        batch_size, time_steps, _ = dynamic.shape
        hidden = dynamic.new_zeros((batch_size, self.cell.hidden_size))
        for step in range(time_steps):
            step_input = torch.cat([dynamic[:, step], mask[:, step], treatment[:, step], delta[:, step]], dim=-1)
            encoded = self.input_encoder(step_input)
            decay = torch.exp(-self.decay(delta[:, step]).clamp(max=10.0))
            candidate = self.cell(encoded, hidden * decay)
            update = observed[:, step].to(dynamic.dtype).unsqueeze(-1)
            hidden = candidate * update + hidden * (1.0 - update)
        return self.fusion_head(batch, hidden)
