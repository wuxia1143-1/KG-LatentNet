from __future__ import annotations

import torch
from torch import nn

from src.models.baselines.common import GraphFeatureEncoder, StaticFusionHead, masked_mean, sequence_observed_mask


class GraphCareAdapter(nn.Module):
    """Personalized patient-graph adapter with knowledge-guided dynamic node pooling."""

    def __init__(self, static_dim: int, dynamic_dim: int, treatment_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.dynamic_graph = GraphFeatureEncoder(dynamic_dim, hidden_dim)
        self.patient_query = nn.Sequential(nn.Linear(static_dim + 1, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.treatment_encoder = nn.Sequential(nn.Linear(treatment_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.care_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.fusion_head = StaticFusionHead(static_dim, hidden_dim, hidden_dim)

    def forward(self, batch: dict[str, torch.Tensor], prior_matrix: torch.Tensor | None = None) -> torch.Tensor:
        dynamic = batch["dynamic_features"]
        mask = batch["dynamic_mask"]
        treatment = batch["treatment_features"]
        observed = sequence_observed_mask(batch)
        static_context = torch.cat([batch["static_features"], batch["baseline_tbr_b"]], dim=-1)
        query = self.patient_query(static_context)

        graph_steps = []
        for step in range(dynamic.shape[1]):
            graph_steps.append(self.dynamic_graph(dynamic[:, step], mask[:, step], prior_matrix=prior_matrix))
        graph_sequence = torch.stack(graph_steps, dim=1)
        logits = torch.sum(graph_sequence * query.unsqueeze(1), dim=-1) / max(1.0, graph_sequence.shape[-1] ** 0.5)
        logits = logits.masked_fill(~observed, -1e4)
        weights = torch.softmax(logits, dim=-1).masked_fill(~observed, 0.0)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        dynamic_summary = torch.sum(graph_sequence * weights.unsqueeze(-1), dim=1)

        treatment_sequence = self.treatment_encoder(treatment)
        treatment_summary = masked_mean(treatment_sequence, observed)
        care_summary = self.care_fusion(torch.cat([query, dynamic_summary, treatment_summary], dim=-1))
        return self.fusion_head(batch, care_summary)
