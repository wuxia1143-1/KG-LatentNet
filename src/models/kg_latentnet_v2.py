from __future__ import annotations

import torch
from torch import nn

from src.models.modules.contribution_fusion import ContributionFusion
from src.models.modules.dynamic_graph import DynamicGraphPrior
from src.models.modules.short_delay_update import ShortDelayUpdate


class KGLatentNetV2(nn.Module):
    VARIANTS = (
        "v2_shared_head",
        "v2_horizon_specific_head",
        "v2_summary_only_residual",
        "v2_latent_summary_fusion",
        "v2_strong_anchor",
    )
    READOUT_MODES = ("shared", "horizon_specific")

    def __init__(
        self,
        static_dim: int,
        dynamic_dim: int,
        treatment_dim: int,
        hidden_dim: int = 32,
        latent_dim: int = 16,
        summary_dim: int = 32,
        dropout: float = 0.3,
        model_variant: str = "v2_shared_head",
        readout_head: str = "shared",
        huber_delta: float = 0.1,
    ) -> None:
        super().__init__()
        if model_variant not in self.VARIANTS:
            raise ValueError(f"model_variant must be one of {self.VARIANTS}")
        self.model_variant = model_variant
        self.readout_head = readout_head
        self.huber_delta = huber_delta
        self._hidden_dim = hidden_dim
        self._latent_dim = latent_dim
        self._summary_dim = summary_dim
        self._dynamic_dim = dynamic_dim
        self._treatment_dim = treatment_dim

        self.baseline_encoder = nn.Sequential(
            nn.Linear(static_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.dynamic_graph = DynamicGraphPrior(dynamic_dim, hidden_dim)
        seq_dim = hidden_dim + treatment_dim + 1
        self.short_delay_update = ShortDelayUpdate(seq_dim, hidden_dim)
        self.fusion = ContributionFusion(hidden_dim * 2, hidden_dim)
        self.latent_proj = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        n_dyn_summary = dynamic_dim * 5 + 2
        n_treat_summary = treatment_dim * 2 + 1
        n_summary_total = n_dyn_summary + n_treat_summary
        self._n_summary_total = n_summary_total

        self.summary_encoder = nn.Sequential(
            nn.Linear(n_summary_total, summary_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(summary_dim, summary_dim),
            nn.ReLU(),
        )

        self.delta_latent_head = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.delta_summary_head = nn.Sequential(
            nn.Linear(summary_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.delta_treatment_head = nn.Sequential(
            nn.Linear(treatment_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.gate_net = nn.Sequential(
            nn.Linear(latent_dim + summary_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )

        if model_variant == "v2_strong_anchor":
            self.anchor_scale_gate = nn.Sequential(
                nn.Linear(latent_dim + summary_dim + hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid(),
            )

        if model_variant == "v2_horizon_specific_head" and readout_head == "horizon_specific":
            self.horizon_delta_heads = nn.ModuleDict({
                str(w): nn.Sequential(
                    nn.Linear(latent_dim + summary_dim + hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1),
                )
                for w in [6, 12, 18, 24]
            })

        if model_variant == "v2_summary_only_residual":
            self.summary_only_head = nn.Sequential(
                nn.Linear(summary_dim + hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        if model_variant == "v2_latent_summary_fusion":
            self.latent_summary_fusion = nn.Sequential(
                nn.Linear(latent_dim + summary_dim + hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.fused_delta_head = nn.Sequential(
                nn.Linear(hidden_dim + hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

    def _compute_dynamic_summaries(
        self,
        dynamic_features: torch.Tensor,
        dynamic_mask: torch.Tensor,
        delta_time: torch.Tensor,
        treatment_features: torch.Tensor,
    ) -> torch.Tensor:
        N, T, D = dynamic_features.shape
        if dynamic_mask.dim() == 3:
            mask = (dynamic_mask.float().sum(dim=-1) > 0).float()
        else:
            mask = dynamic_mask.float()
        mask_sum = mask.sum(dim=1).clamp(min=1e-8)

        last_idx = torch.zeros(N, dtype=torch.long, device=dynamic_features.device)
        for i in range(N):
            nz = torch.where(dynamic_mask[i].sum(dim=-1) > 0)[0]
            last_idx[i] = nz[-1] if len(nz) > 0 else 0
        last_val = dynamic_features[torch.arange(N, device=dynamic_features.device), last_idx]

        masked_dyn = dynamic_features * mask.unsqueeze(-1)
        dyn_mean = masked_dyn.sum(dim=1) / mask_sum.unsqueeze(-1)
        dyn_sq_mean = (masked_dyn ** 2).sum(dim=1) / mask_sum.unsqueeze(-1)
        dyn_std = (dyn_sq_mean - dyn_mean ** 2).clamp(min=0).sqrt()
        dyn_max = masked_dyn.max(dim=1)[0]
        dyn_min_raw = masked_dyn.min(dim=1)[0]
        dyn_min = torch.where(mask.sum(dim=-1).unsqueeze(-1) > 0, dyn_min_raw, torch.zeros_like(dyn_min_raw))
        obs_count = mask.sum(dim=1) / T

        dt_cumsum = delta_time.cumsum(dim=1)
        history_len = dt_cumsum[torch.arange(N, device=delta_time.device), last_idx]

        dyn_stats = torch.cat([last_val, dyn_mean, dyn_std, dyn_max, dyn_min, obs_count.unsqueeze(-1),
                               history_len.unsqueeze(-1)], dim=-1)

        N_t, T_t, D_t = treatment_features.shape
        treat_mask = (treatment_features.abs().sum(dim=-1) > 0).float()
        treat_cumsum = treatment_features.cumsum(dim=1)
        treat_cum = treat_cumsum[:, -1, :]

        recent_T = min(T_t, 4)
        treat_recent = treatment_features[:, -recent_T:, :].sum(dim=1)

        last_treat_idx = torch.zeros(N_t, dtype=torch.long, device=treatment_features.device)
        for i in range(N_t):
            nz = torch.where(treat_mask[i] > 0)[0]
            last_treat_idx[i] = nz[-1] if len(nz) > 0 else 0
        time_since_last = delta_time[torch.arange(N_t, device=delta_time.device), last_treat_idx].unsqueeze(-1)

        treat_stats = torch.cat([treat_cum, treat_recent, time_since_last], dim=-1)

        return torch.cat([dyn_stats, treat_stats], dim=-1)

    def encode_temporal(
        self,
        batch: dict[str, torch.Tensor],
        prior_matrix: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        static_context = torch.cat(
            [batch["static_features"], batch["baseline_tbr_b"]], dim=-1
        )
        baseline_hidden = self.baseline_encoder(static_context)

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

        fused = self.fusion(baseline_hidden, dynamic_summary)
        z_final = self.latent_proj(fused)

        return z_final, {
            "baseline_hidden": baseline_hidden,
            "dynamic_summary": dynamic_summary,
            "fused": fused,
        }

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        prior_matrix: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        z_final, temporal_info = self.encode_temporal(batch, prior_matrix)
        baseline_hidden = temporal_info["baseline_hidden"]
        baseline = batch["baseline_tbr_b"].squeeze(-1)

        delta_latent = self.delta_latent_head(
            torch.cat([z_final, baseline_hidden], dim=-1)
        ).squeeze(-1)

        summary_features = self._compute_dynamic_summaries(
            batch["dynamic_features"], batch["dynamic_mask"],
            batch["delta_time"], batch["treatment_features"],
        )
        summary_hidden = self.summary_encoder(summary_features)
        delta_summary = self.delta_summary_head(summary_hidden).squeeze(-1)

        treatment_summary = batch["treatment_features"].mean(dim=1)
        delta_treatment = self.delta_treatment_head(
            torch.cat([treatment_summary, baseline_hidden], dim=-1)
        ).squeeze(-1)

        gate_input = torch.cat([z_final, summary_hidden, baseline_hidden], dim=-1)
        gate_logits = self.gate_net(gate_input)
        gate_weights = torch.softmax(gate_logits, dim=-1)

        if self.model_variant == "v2_strong_anchor":
            scale_gate = self.anchor_scale_gate(gate_input).squeeze(-1)
            delta_raw = (
                gate_weights[:, 0] * delta_latent
                + gate_weights[:, 1] * delta_summary
                + gate_weights[:, 2] * delta_treatment
            )
            delta_pred = scale_gate * delta_raw
        elif self.model_variant == "v2_summary_only_residual":
            delta_pred = self.summary_only_head(
                torch.cat([summary_hidden, baseline_hidden], dim=-1)
            ).squeeze(-1)
        elif self.model_variant == "v2_latent_summary_fusion":
            ls_input = torch.cat([z_final, summary_hidden, baseline_hidden], dim=-1)
            fused_ls = self.latent_summary_fusion(ls_input)
            delta_pred = self.fused_delta_head(
                torch.cat([fused_ls, baseline_hidden], dim=-1)
            ).squeeze(-1)
        elif self.model_variant == "v2_horizon_specific_head" and self.readout_head == "horizon_specific":
            delta_gated = (
                gate_weights[:, 0] * delta_latent
                + gate_weights[:, 1] * delta_summary
                + gate_weights[:, 2] * delta_treatment
            )
            if "endpoint_window" in batch:
                windows = batch["endpoint_window"]
                delta_pred = torch.zeros_like(delta_gated)
                for w_key, head in self.horizon_delta_heads.items():
                    w = int(w_key)
                    mask = (windows == w)
                    if mask.any():
                        delta_pred[mask] = head(gate_input[mask]).squeeze(-1)
                no_horizon = (delta_pred == 0) & (delta_gated != 0)
                delta_pred = torch.where(no_horizon, delta_gated, delta_pred)
            else:
                delta_pred = delta_gated
        else:
            delta_pred = (
                gate_weights[:, 0] * delta_latent
                + gate_weights[:, 1] * delta_summary
                + gate_weights[:, 2] * delta_treatment
            )

        y_pred = baseline + delta_pred

        return {
            "y_pred": y_pred,
            "delta_pred": delta_pred,
            "delta_latent": delta_latent,
            "delta_summary": delta_summary,
            "delta_treatment": delta_treatment,
            "gate_weights": gate_weights,
            "baseline_tbr_b": baseline,
            "latent_state": z_final,
            "summary_hidden": summary_hidden,
            "baseline_hidden": baseline_hidden,
        }

    @staticmethod
    def compute_loss(
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        criterion: nn.Module,
        lambda_delta: float = 0.0,
        lambda_anchor: float = 0.0,
        lambda_prior: float = 0.0,
        lambda_smooth: float = 0.0,
        lambda_disentangle: float = 0.0,
        y_range: tuple[float, float] | None = None,
        lambda_range: float = 0.01,
    ) -> dict[str, torch.Tensor]:
        y_true = batch["endpoint_tbr_y"].squeeze(-1)
        baseline = batch["baseline_tbr_b"].squeeze(-1)
        delta_true = y_true - baseline

        l_endpoint = criterion(outputs["y_pred"], y_true)

        l_delta = torch.tensor(0.0, device=l_endpoint.device)
        if lambda_delta > 0:
            l_delta = torch.mean(torch.abs(outputs["delta_pred"] - delta_true))

        total = l_endpoint + lambda_delta * l_delta

        l_anchor = torch.tensor(0.0, device=total.device)
        if lambda_anchor > 0:
            l_anchor = torch.mean(torch.abs(outputs["delta_pred"]))
            total = total + lambda_anchor * l_anchor

        l_prior = torch.tensor(0.0, device=total.device)
        if lambda_prior > 0:
            latent = outputs["latent_state"]
            latent_norm = latent / (latent.norm(dim=-1, keepdim=True) + 1e-8)
            sim = torch.matmul(latent_norm, latent_norm.t())
            n = sim.shape[0]
            if n > 1:
                eye = torch.eye(n, device=sim.device)
                l_prior = torch.mean((sim - eye) ** 2)
                total = total + lambda_prior * l_prior

        l_smooth = torch.tensor(0.0, device=total.device)
        if lambda_smooth > 0:
            l_smooth = torch.mean(outputs["delta_pred"] ** 2)
            total = total + lambda_smooth * l_smooth

        l_disentangle = torch.tensor(0.0, device=total.device)
        if lambda_disentangle > 0:
            d_lat = outputs["delta_latent"]
            d_sum = outputs["delta_summary"]
            d_trt = outputs["delta_treatment"]
            pairs = [(d_lat, d_sum), (d_lat, d_trt), (d_sum, d_trt)]
            corrs = []
            for a, b in pairs:
                a_c = a - a.mean()
                b_c = b - b.mean()
                denom = (a_c.norm() * b_c.norm() + 1e-8)
                corrs.append((a_c * b_c).sum() / denom)
            l_disentangle = sum(c ** 2 for c in corrs) / len(corrs)
            total = total + lambda_disentangle * l_disentangle

        l_range = torch.tensor(0.0, device=total.device)
        if y_range is not None and lambda_range > 0:
            y_min, y_max = y_range
            below = torch.relu(y_min - outputs["y_pred"])
            above = torch.relu(outputs["y_pred"] - y_max)
            l_range = torch.mean(below + above)
            total = total + lambda_range * l_range

        gate_mean = outputs["gate_weights"].mean(dim=0)

        return {
            "total": total,
            "l_endpoint": l_endpoint,
            "l_delta": l_delta,
            "l_anchor": l_anchor,
            "l_prior": l_prior,
            "l_smooth": l_smooth,
            "l_disentangle": l_disentangle,
            "l_range": l_range,
            "gate_mean_latent": gate_mean[0].detach(),
            "gate_mean_summary": gate_mean[1].detach(),
            "gate_mean_treatment": gate_mean[2].detach(),
            "delta_true_mean": delta_true.mean().detach(),
            "delta_pred_mean": outputs["delta_pred"].mean().detach(),
        }
