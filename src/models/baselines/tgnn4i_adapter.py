from __future__ import annotations

import torch
from torch import nn

from src.models.baselines.common import GraphFeatureEncoder, StaticFusionHead, last_valid_state, sequence_observed_mask


class TGNN4IAdapter(nn.Module):
    """Temporal graph neural adapter for irregular longitudinal feature graphs."""

    def __init__(self, static_dim: int, dynamic_dim: int, treatment_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.graph_encoder = GraphFeatureEncoder(dynamic_dim, hidden_dim)
        self.step_encoder = nn.Sequential(
            nn.Linear(hidden_dim + treatment_dim + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.temporal_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fusion_head = StaticFusionHead(static_dim, hidden_dim, hidden_dim)

    def forward(self, batch: dict[str, torch.Tensor], prior_matrix: torch.Tensor | None = None) -> torch.Tensor:
        dynamic = batch["dynamic_features"]
        mask = batch["dynamic_mask"]
        treatment = batch["treatment_features"]
        delta = batch["delta_time"].unsqueeze(-1)
        observed = sequence_observed_mask(batch)

        graph_steps = []
        for step in range(dynamic.shape[1]):
            graph_summary = self.graph_encoder(dynamic[:, step], mask[:, step], prior_matrix=prior_matrix)
            graph_steps.append(graph_summary)
        graph_sequence = torch.stack(graph_steps, dim=1)
        sequence = self.step_encoder(torch.cat([graph_sequence, treatment, delta], dim=-1))
        output, _ = self.temporal_gru(sequence)
        summary = last_valid_state(output, observed)
        return self.fusion_head(batch, summary)
