from __future__ import annotations

import torch
from torch import nn

from src.models.baselines.common import GraphFeatureEncoder, StaticFusionHead, last_valid_state, sequence_observed_mask


class DHGASAdapter(nn.Module):
    """Dynamic heterogeneous graph-attention adapter with biomarker, treatment, and static node types."""

    def __init__(self, static_dim: int, dynamic_dim: int, treatment_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.dynamic_graph = GraphFeatureEncoder(dynamic_dim, hidden_dim)
        self.treatment_encoder = nn.Sequential(nn.Linear(treatment_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.static_type_encoder = nn.Sequential(nn.Linear(static_dim + 1, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.delta_encoder = nn.Sequential(nn.Linear(1, hidden_dim), nn.Tanh())
        self.type_attention = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        self.temporal_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fusion_head = StaticFusionHead(static_dim, hidden_dim, hidden_dim)

    def forward(self, batch: dict[str, torch.Tensor], prior_matrix: torch.Tensor | None = None) -> torch.Tensor:
        dynamic = batch["dynamic_features"]
        mask = batch["dynamic_mask"]
        treatment = batch["treatment_features"]
        delta = batch["delta_time"].unsqueeze(-1)
        observed = sequence_observed_mask(batch)
        static_context = torch.cat([batch["static_features"], batch["baseline_tbr_b"]], dim=-1)
        static_token = self.static_type_encoder(static_context)

        step_tokens = []
        for step in range(dynamic.shape[1]):
            dynamic_token = self.dynamic_graph(dynamic[:, step], mask[:, step], prior_matrix=prior_matrix)
            treatment_token = self.treatment_encoder(treatment[:, step])
            time_token = self.delta_encoder(delta[:, step])
            typed = torch.stack([dynamic_token, treatment_token, static_token, time_token], dim=1)
            weights = torch.softmax(self.type_attention(typed).squeeze(-1), dim=-1)
            step_tokens.append(torch.sum(typed * weights.unsqueeze(-1), dim=1))
        sequence = torch.stack(step_tokens, dim=1)
        output, _ = self.temporal_gru(sequence)
        summary = last_valid_state(output, observed)
        return self.fusion_head(batch, summary)
