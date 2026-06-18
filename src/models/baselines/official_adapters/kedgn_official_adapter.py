from __future__ import annotations

from pathlib import Path
import importlib
import sys

import torch
from torch import nn

from src.models.baselines.official_adapters.common import OfficialAdapterBase, prepend_paths, safe_lengths


class KEDGNOfficialAdapter(OfficialAdapterBase):
    """Official KEDGN adapter.

    Official model class imported: `remote_baselines/KEDGN/model.py::KEDGN`.
    Official input format: `P=[observed_data, observed_mask]`, static features, average interval, sequence lengths, time matrix, variable representation tensor.
    Data mapping: KG dynamic features/masks become `P`; train-only preprocessed static+baseline features become `P_static`.
    Output mapping: official KEDGN classifier is instantiated with `n_class=1`, so its forward output is endpoint TBR prediction.
    Official loss: not used, because current endpoint task uses scalar MSE; official core model is unchanged.
    Official core modules retained: Value_Encoder, Time_Encoder, VSDGCRNN, AGCRNCellWithMLP, and official classifier.
    """

    baseline_name = "kedgn"
    official_repo_url = "https://github.com/qianlima-lab/KEDGN.git"
    official_model_class_used = "model.KEDGN"
    official_entry_script = "train.py"
    official_dependencies_file = "requirements.txt"

    def __init__(self, project_root: Path, static_dim: int, dynamic_dim: int, treatment_dim: int, hidden_dim: int = 64) -> None:
        super().__init__(project_root)
        del treatment_dim
        repo = self.project_root / "remote_baselines" / "KEDGN"
        for module_name in ["model", "utils"]:
            if module_name in sys.modules:
                del sys.modules[module_name]
        prepend_paths(repo)
        kedgn_model = importlib.import_module("model")
        kedgn_model.torch = torch
        KEDGN = kedgn_model.KEDGN  # type: ignore[attr-defined]

        self.dynamic_dim = dynamic_dim
        self.official_model = KEDGN(
            DEVICE=torch.device("cpu"),
            hidden_dim=hidden_dim,
            num_of_variables=dynamic_dim,
            num_of_timestamps=47,
            d_static=static_dim + 1,
            n_class=1,
            node_enc_layer=1,
            plm_rep_dim=dynamic_dim,
        )
        self.register_buffer("variable_representations", torch.eye(dynamic_dim, dtype=torch.float32))

    def forward(self, batch: dict[str, torch.Tensor], prior_matrix: torch.Tensor | None = None) -> torch.Tensor:
        del prior_matrix
        values = batch["dynamic_features"]
        mask = batch["dynamic_mask"]
        p_tensor = torch.cat([values, mask], dim=-1)
        observed_any = mask.sum(dim=-1) > 0
        lengths = safe_lengths(observed_any).view(-1, 1).to(values.device)
        cumulative_time = torch.cumsum(batch["delta_time"], dim=1)
        p_time = cumulative_time.unsqueeze(-1).expand(-1, -1, self.dynamic_dim)
        avg_interval = torch.zeros_like(values)
        counts = mask.cumsum(dim=1).clamp_min(1.0)
        avg_interval = p_time / counts
        static = torch.cat([batch["static_features"], batch["baseline_tbr_b"]], dim=-1)
        var_rep = self.variable_representations.to(values.device, values.dtype)
        self.official_model.to(values.device)
        return self.official_model(p_tensor, static, avg_interval, lengths, p_time, var_rep)
