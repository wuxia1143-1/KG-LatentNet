from __future__ import annotations

import torch
from torch import nn

from src.models.modules.contribution_fusion import ContributionFusion
from src.models.modules.dynamic_graph import DynamicGraphPrior
from src.models.modules.short_delay_update import ShortDelayUpdate


class KGLatentNet(nn.Module):
    def __init__(
        self,
        static_dim: int,
        dynamic_dim: int,
        treatment_dim: int,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.static_encoder = nn.Sequential(nn.Linear(static_dim + 1, hidden_dim), nn.ReLU())
        self.dynamic_graph = DynamicGraphPrior(dynamic_dim, hidden_dim)
        sequence_dim = hidden_dim + treatment_dim + 1
        self.short_delay_update = ShortDelayUpdate(sequence_dim, hidden_dim)
        self.fusion = ContributionFusion(hidden_dim * 2, hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, batch: dict[str, torch.Tensor], prior_matrix: torch.Tensor | None = None) -> torch.Tensor:
        static_context = torch.cat([batch["static_features"], batch["baseline_tbr_b"]], dim=-1)
        static_hidden = self.static_encoder(static_context)

        dynamic_hidden = self.dynamic_graph(batch["dynamic_features"], batch["dynamic_mask"], prior_matrix=prior_matrix)
        delta = batch["delta_time"].unsqueeze(-1)
        sequence = torch.cat([dynamic_hidden, batch["treatment_features"], delta], dim=-1)
        sequence_mask = (batch["dynamic_mask"].sum(dim=-1) > 0) | (batch["treatment_features"].sum(dim=-1) > 0)
        dynamic_summary = self.short_delay_update(sequence, sequence_mask)

        fused = self.fusion(static_hidden, dynamic_summary)
        return self.head(fused)
