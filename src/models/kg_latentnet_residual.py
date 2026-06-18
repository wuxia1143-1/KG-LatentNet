from __future__ import annotations

import torch
from torch import nn

from src.models.modules.contribution_fusion import ContributionFusion
from src.models.modules.dynamic_graph import DynamicGraphPrior
from src.models.modules.short_delay_update import ShortDelayUpdate


class KGLatentNetResidual(nn.Module):
    TARGET_MODES = ("endpoint", "delta", "residual_anchor")

    def __init__(
        self,
        static_dim: int,
        dynamic_dim: int,
        treatment_dim: int,
        hidden_dim: int = 64,
        latent_dim: int = 16,
        dropout: float = 0.3,
        target_mode: str = "residual_anchor",
    ) -> None:
        super().__init__()
        if target_mode not in self.TARGET_MODES:
            raise ValueError(f"target_mode must be one of {self.TARGET_MODES}, got {target_mode}")
        self.target_mode = target_mode
        self._hidden_dim = hidden_dim
        self._latent_dim = latent_dim

        self.static_encoder = nn.Sequential(
            nn.Linear(static_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.dynamic_graph = DynamicGraphPrior(dynamic_dim, hidden_dim)
        sequence_dim = hidden_dim + treatment_dim + 1
        self.short_delay_update = ShortDelayUpdate(sequence_dim, hidden_dim)
        self.fusion = ContributionFusion(hidden_dim * 2, hidden_dim)

        self.latent_proj = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.delta_head = nn.Linear(latent_dim, 1)

        if target_mode == "endpoint":
            self.endpoint_head = nn.Linear(latent_dim, 1)

    def encode(
        self,
        batch: dict[str, torch.Tensor],
        prior_matrix: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        static_context = torch.cat(
            [batch["static_features"], batch["baseline_tbr_b"]], dim=-1
        )
        static_hidden = self.static_encoder(static_context)

        dynamic_hidden = self.dynamic_graph(
            batch["dynamic_features"], batch["dynamic_mask"], prior_matrix=prior_matrix
        )
        delta = batch["delta_time"].unsqueeze(-1)
        sequence = torch.cat(
            [dynamic_hidden, batch["treatment_features"], delta], dim=-1
        )
        sequence_mask = (batch["dynamic_mask"].sum(dim=-1) > 0) | (
            batch["treatment_features"].sum(dim=-1) > 0
        )
        dynamic_summary = self.short_delay_update(sequence, sequence_mask)

        fused = self.fusion(static_hidden, dynamic_summary)
        latent = self.latent_proj(fused)
        return latent, fused

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        prior_matrix: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        latent, fused = self.encode(batch, prior_matrix)
        baseline = batch["baseline_tbr_b"].squeeze(-1)

        delta_pred = self.delta_head(latent).squeeze(-1)

        if self.target_mode == "endpoint":
            y_pred = self.endpoint_head(latent).squeeze(-1)
            delta_pred_from_y = y_pred - baseline
        else:
            y_pred = baseline + delta_pred
            delta_pred_from_y = delta_pred

        return {
            "y_pred": y_pred,
            "delta_pred": delta_pred_from_y,
            "baseline_tbr_b": baseline,
            "latent_state": latent,
            "contribution_fused": fused,
        }

    @staticmethod
    def compute_loss(
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        criterion: nn.Module,
        target_mode: str,
        lambda_residual: float = 0.0,
        lambda_anchor: float = 0.0,
        lambda_graph_prior: float = 0.0,
        lambda_smooth: float = 0.0,
        lambda_disentangle: float = 0.0,
        y_range: tuple[float, float] | None = None,
        lambda_range: float = 0.01,
        prior_matrix: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        y_true = batch["endpoint_tbr_y"].squeeze(-1)
        baseline = batch["baseline_tbr_b"].squeeze(-1)
        delta_true = y_true - baseline

        if target_mode == "endpoint":
            main_loss = criterion(outputs["y_pred"], y_true)
        else:
            main_loss = criterion(outputs["delta_pred"], delta_true)

        total = main_loss.clone()

        l_residual = torch.tensor(0.0, device=total.device)
        if lambda_residual > 0:
            l_residual = torch.mean(torch.abs(outputs["delta_pred"]))
            total = total + lambda_residual * l_residual

        l_anchor = torch.tensor(0.0, device=total.device)
        if lambda_anchor > 0:
            l_anchor = torch.mean(torch.abs(outputs["y_pred"] - baseline))
            total = total + lambda_anchor * l_anchor

        l_graph_prior = torch.tensor(0.0, device=total.device)
        if lambda_graph_prior > 0 and prior_matrix is not None:
            latent = outputs["latent_state"]
            latent_norm = latent / (latent.norm(dim=-1, keepdim=True) + 1e-8)
            sim_matrix = torch.matmul(latent_norm, latent_norm.t())
            n = sim_matrix.shape[0]
            if n > 1:
                eye = torch.eye(n, device=sim_matrix.device)
                l_graph_prior = torch.mean((sim_matrix - eye) ** 2)
                total = total + lambda_graph_prior * l_graph_prior

        l_smooth = torch.tensor(0.0, device=total.device)
        if lambda_smooth > 0:
            delta_pred = outputs["delta_pred"]
            if delta_pred.numel() > 1:
                l_smooth = torch.var(delta_pred)
                total = total + lambda_smooth * l_smooth

        l_disentangle = torch.tensor(0.0, device=total.device)
        if lambda_disentangle > 0:
            latent = outputs["latent_state"]
            if latent.shape[0] > 1 and latent.shape[1] > 1:
                latent_centered = latent - latent.mean(dim=0, keepdim=True)
                cov = torch.matmul(latent_centered.t(), latent_centered) / (latent.shape[0] - 1)
                diag = torch.diag(cov)
                off_diag = cov - torch.diag(diag)
                l_disentangle = torch.mean(off_diag ** 2)
                total = total + lambda_disentangle * l_disentangle

        l_range = torch.tensor(0.0, device=total.device)
        if y_range is not None and lambda_range > 0:
            y_min, y_max = y_range
            below = torch.relu(y_min - outputs["y_pred"])
            above = torch.relu(outputs["y_pred"] - y_max)
            l_range = torch.mean(below + above)
            total = total + lambda_range * l_range

        return {
            "total": total,
            "main": main_loss,
            "residual_reg": l_residual,
            "anchor_reg": l_anchor,
            "graph_prior_reg": l_graph_prior,
            "smooth_reg": l_smooth,
            "disentangle_reg": l_disentangle,
            "range_penalty": l_range,
            "delta_true_mean": delta_true.mean().detach(),
            "delta_pred_mean": outputs["delta_pred"].mean().detach(),
        }
