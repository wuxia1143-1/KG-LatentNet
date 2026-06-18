from __future__ import annotations

from pathlib import Path
import sys
import types

import torch
from torch import nn

from src.models.baselines.official_adapters.common import OfficialAdapterBase, prepend_paths


class TGNN4IOfficialAdapter(OfficialAdapterBase):
    """Official TGNN4I adapter.

    Official model class imported: `remote_baselines/TGNN4I/models/gru_graph_model.py::GRUGraphModel`.
    Official input format: PyTorch Geometric `Batch` of temporal graphs with node-wise `y`, `mask`, `delta_t`, `t`, and graph edges.
    Data mapping: each dynamic variable is a graph node; dynamic values/masks become node time series; prior matrix determines edge weights.
    Output mapping: official node forecasts are pooled across variables and combined with static/baseline features for endpoint TBR prediction.
    Official loss: not used; MSE on endpoint_tbr_y is used for the current scalar endpoint task.
    Official core modules retained: GRUGraphModel, GRUGraphCell, DecayCell, and official GNN sequence.
    """

    baseline_name = "tgnn4i"
    official_repo_url = "https://github.com/joeloskarsson/tgnn4i.git"
    official_model_class_used = "models.gru_graph_model.GRUGraphModel"
    official_entry_script = "train.py"
    official_dependencies_file = "requirements.txt"

    def __init__(self, project_root: Path, static_dim: int, dynamic_dim: int, treatment_dim: int, hidden_dim: int = 64) -> None:
        super().__init__(project_root)
        del treatment_dim
        repo = self.project_root / "remote_baselines" / "TGNN4I"
        for module_name in list(sys.modules):
            if module_name == "models" or module_name.startswith("models."):
                del sys.modules[module_name]
        prepend_paths(repo)
        models_dir = repo / "models"
        models_pkg = types.ModuleType("models")
        models_pkg.__path__ = [str(models_dir)]  # type: ignore[attr-defined]
        sys.modules["models"] = models_pkg
        from models.gru_graph_model import GRUGraphModel  # type: ignore

        self.dynamic_dim = dynamic_dim
        config = {
            "model": "gru_graph",
            "device": torch.device("cpu"),
            "num_nodes": dynamic_dim,
            "time_steps": 47,
            "y_dim": 1,
            "feature_dim": 0,
            "has_features": False,
            "time_input": True,
            "mask_input": True,
            "max_pred": 1,
            "param_dim": 1,
            "hidden_dim": hidden_dim,
            "gru_layers": 1,
            "learn_init_state": True,
            "n_fc": 1,
            "decay_type": "none",
            "periodic": False,
            "node_params": False,
            "gru_gnn": 1,
            "pred_gnn": 0,
            "gnn_type": "gcn",
            "state_updates": "obs",
        }
        self.official_model = GRUGraphModel(config)
        self.regression_head = nn.Sequential(nn.Linear(dynamic_dim + static_dim + 1, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))

    def _edge_index_attr(self, prior_matrix: torch.Tensor | None, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        if prior_matrix is None or prior_matrix.shape[0] != self.dynamic_dim:
            adj = torch.eye(self.dynamic_dim, device=device, dtype=dtype)
        else:
            adj = prior_matrix[: self.dynamic_dim, : self.dynamic_dim].to(device=device, dtype=dtype).abs()
            adj = adj + torch.eye(self.dynamic_dim, device=device, dtype=dtype)
        src, dst = torch.where(adj > 0)
        edge_index = torch.stack([src, dst], dim=0).long()
        edge_attr = adj[src, dst].unsqueeze(-1)
        return edge_index, edge_attr

    def forward(self, batch: dict[str, torch.Tensor], prior_matrix: torch.Tensor | None = None) -> torch.Tensor:
        from torch_geometric.data import Batch, Data

        values = batch["dynamic_features"]
        mask = batch["dynamic_mask"]
        times = torch.cumsum(batch["delta_time"], dim=1)
        edge_index, edge_attr = self._edge_index_attr(prior_matrix, values.device, values.dtype)
        graphs = []
        for idx in range(values.shape[0]):
            y = values[idx].transpose(0, 1).unsqueeze(-1)
            node_mask = mask[idx].transpose(0, 1)
            delta_t = torch.cat([times[idx, :1], times[idx, 1:] - times[idx, :-1]]).unsqueeze(0).repeat(self.dynamic_dim, 1)
            graphs.append(
                Data(
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=y,
                    t=times[idx].unsqueeze(0),
                    delta_t=delta_t,
                    update_delta_t=delta_t,
                    mask=node_mask,
                    hop_mask=node_mask,
                    num_nodes=self.dynamic_dim,
                )
            )
        graph_batch = Batch.from_data_list(graphs).to(values.device)
        self.official_model.to(values.device)
        for module in self.official_model.modules():
            for attr in ["decay_target", "decay_weight", "init_decay_weight"]:
                tensor = getattr(module, attr, None)
                if isinstance(tensor, torch.Tensor):
                    setattr(module, attr, tensor.to(values.device))
        pred, _ = self.official_model(graph_batch)
        # Official shape: [time, batch*num_nodes, max_pred, y_dim, param_dim].
        last_step = pred[-1, :, 0, 0, 0].reshape(values.shape[0], self.dynamic_dim)
        features = torch.cat([last_step, batch["static_features"], batch["baseline_tbr_b"]], dim=-1)
        return self.regression_head(features)
