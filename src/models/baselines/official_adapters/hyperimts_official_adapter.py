from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from src.models.baselines.official_adapters.common import (
    OfficialAdapterBase,
    add_dummy_observation_for_empty_history,
    prepend_paths,
)


class HyperIMTSOfficialAdapter(OfficialAdapterBase):
    """Official HyperIMTS adapter.

    Official model class imported: `remote_baselines/HyperIMTS/models/HyperIMTS.py::Model`.
    Official input format: padded irregular MTS tensors `x`, `x_mark`, `x_mask`, forecast target placeholders `y`, `y_mark`, `y_mask`.
    Data mapping: KG dynamic features and masks become `x` and `x_mask`; cumulative `delta_time` becomes normalized `x_mark`.
    Output mapping: official one-step multivariate forecast is passed through a small regression head to produce endpoint TBR.
    Official loss: not used, because the current task is scalar endpoint regression; MSE on endpoint_tbr_y is used.
    Official core modules retained: HypergraphEncoder, HypergraphLearner, and Hypergraph decoder inside the official `Model`.
    """

    baseline_name = "hyperimts"
    official_repo_url = "https://github.com/Ladbaby/PyOmniTS.git"
    official_model_class_used = "models.HyperIMTS.Model"
    official_entry_script = "main.py"
    official_dependencies_file = "requirements.txt"

    def __init__(self, project_root: Path, static_dim: int, dynamic_dim: int, treatment_dim: int, hidden_dim: int = 64) -> None:
        super().__init__(project_root)
        del treatment_dim
        repo = self.project_root / "remote_baselines" / "HyperIMTS"
        prepend_paths(repo)
        from models.HyperIMTS import Model as OfficialHyperIMTSModel  # type: ignore

        configs = SimpleNamespace(
            enc_in=dynamic_dim,
            d_model=hidden_dim,
            n_layers=1,
            n_heads=4,
            seq_len=47,
            pred_len=1,
            seq_len_max_irr=47,
            pred_len_max_irr=1,
            task_name="short_term_forecast",
            features="M",
        )
        self.official_model = OfficialHyperIMTSModel(configs)
        self.regression_head = nn.Sequential(
            nn.Linear(dynamic_dim + static_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch: dict[str, torch.Tensor], prior_matrix: torch.Tensor | None = None) -> torch.Tensor:
        del prior_matrix
        x, x_mask = add_dummy_observation_for_empty_history(batch["dynamic_features"], batch["dynamic_mask"])
        cumulative_time = torch.cumsum(batch["delta_time"], dim=1)
        denom = cumulative_time.max(dim=1, keepdim=True).values.clamp_min(1.0)
        x_mark = (cumulative_time / denom).unsqueeze(-1)
        y = torch.zeros((x.shape[0], 1, x.shape[2]), dtype=x.dtype, device=x.device)
        y_mark = torch.ones((x.shape[0], 1, 1), dtype=x.dtype, device=x.device)
        y_mask = torch.ones_like(y)
        official_out = self.official_model(
            x=x,
            x_mark=x_mark,
            x_mask=x_mask,
            y=y,
            y_mark=y_mark,
            y_mask=y_mask,
            exp_stage="test",
        )
        official_pred = official_out["pred"].reshape(x.shape[0], -1)
        if official_pred.shape[1] != x.shape[2]:
            official_pred = official_pred[:, : x.shape[2]]
        features = torch.cat([official_pred, batch["static_features"], batch["baseline_tbr_b"]], dim=-1)
        return self.regression_head(features)
