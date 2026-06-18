from __future__ import annotations

import torch
from torch import nn


def sequence_observed_mask(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    dynamic_seen = batch["dynamic_mask"].sum(dim=-1) > 0
    treatment_seen = batch["treatment_features"].abs().sum(dim=-1) > 0
    return dynamic_seen | treatment_seen


def masked_mean(sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(sequence.dtype).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (sequence * weights).sum(dim=1) / denom


def last_valid_state(sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    batch_size, time_steps, hidden_dim = sequence.shape
    if time_steps == 0:
        return sequence.new_zeros((batch_size, hidden_dim))
    lengths = mask.long().sum(dim=1).clamp_min(1)
    gather_idx = (lengths - 1).view(batch_size, 1, 1).expand(batch_size, 1, hidden_dim)
    return sequence.gather(dim=1, index=gather_idx).squeeze(1)


def normalized_prior(prior_matrix: torch.Tensor | None, feature_dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if prior_matrix is None or prior_matrix.numel() == 0 or prior_matrix.shape[0] != feature_dim:
        adj = torch.eye(feature_dim, device=device, dtype=dtype)
    else:
        adj = prior_matrix[:feature_dim, :feature_dim].to(device=device, dtype=dtype).abs()
        adj = adj + torch.eye(feature_dim, device=device, dtype=dtype)
    degree = adj.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    return adj / degree


class StaticFusionHead(nn.Module):
    def __init__(self, static_dim: int, sequence_hidden_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.static_encoder = nn.Sequential(
            nn.Linear(static_dim + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(sequence_hidden_dim + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch: dict[str, torch.Tensor], sequence_summary: torch.Tensor) -> torch.Tensor:
        static_context = torch.cat([batch["static_features"], batch["baseline_tbr_b"]], dim=-1)
        static_hidden = self.static_encoder(static_context)
        return self.head(torch.cat([static_hidden, sequence_summary], dim=-1))


class GraphFeatureEncoder(nn.Module):
    def __init__(self, dynamic_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.node_encoder = nn.Sequential(nn.Linear(1, hidden_dim), nn.GELU())
        self.node_attention = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))

    def forward(self, values: torch.Tensor, mask: torch.Tensor, prior_matrix: torch.Tensor | None = None) -> torch.Tensor:
        feature_dim = values.shape[-1]
        adj = normalized_prior(prior_matrix, feature_dim, values.device, values.dtype)
        propagated = torch.matmul(values * mask, adj)
        node_hidden = self.node_encoder(propagated.unsqueeze(-1))
        attn_logits = self.node_attention(node_hidden).squeeze(-1)
        attn_logits = attn_logits.masked_fill(mask <= 0, -1e4)
        all_missing = mask.sum(dim=-1, keepdim=True) <= 0
        attn = torch.softmax(attn_logits, dim=-1).masked_fill(all_missing, 0.0)
        return torch.sum(node_hidden * attn.unsqueeze(-1), dim=-2)
