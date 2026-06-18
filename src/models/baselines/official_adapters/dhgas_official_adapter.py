from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from src.models.baselines.official_adapters.common import OfficialAdapterBase, prepend_paths


class DHGASOfficialAdapter(OfficialAdapterBase):
    """Official DHGAS adapter.

    Official model class imported: `remote_baselines/DHGAS/dhgas/models/HTGNN.py::HTGNN`.
    Official input format: a dictionary of time-indexed heterogeneous graphs with node feature dictionaries.
    Data mapping: each batch is represented as dynamic/treatment/static node types over endpoint-truncated time windows.
    Output mapping: official HTGNN encodes dynamic heterogeneous graphs; an adapter head maps encoded dynamic nodes to endpoint TBR.
    Official loss: not used; endpoint TBR MSE is used.
    Official core modules retained: official HTGNN, HTGNNLayer, RelationAgg, TemporalAgg, and GATConv aggregation.
    """

    baseline_name = "dhgas"
    official_repo_url = "https://github.com/wondergo2017/DHGAS.git"
    official_model_class_used = "dhgas.models.HTGNN.HTGNN"
    official_entry_script = "scripts/run/run_model.py"
    official_dependencies_file = "setup.py / readme.md"

    def __init__(self, project_root: Path, static_dim: int, dynamic_dim: int, treatment_dim: int, hidden_dim: int = 64) -> None:
        super().__init__(project_root)
        del static_dim
        repo = self.project_root / "remote_baselines" / "DHGAS"
        prepend_paths(repo)
        from dhgas.models.HTGNN import HTGNN  # type: ignore

        self.dynamic_dim = dynamic_dim
        self.treatment_dim = treatment_dim
        metadata = (
            ["dynamic", "treatment", "static"],
            [("dynamic", "dt", "treatment"), ("treatment", "td", "dynamic"), ("dynamic", "ds", "static"), ("static", "sd", "dynamic")],
        )
        self.official_model = HTGNN(
            n_inp=hidden_dim,
            n_hid=hidden_dim,
            n_layers=1,
            n_heads=2,
            time_window=47,
            norm=False,
            device=torch.device("cpu"),
            metadata=metadata,
            predict_type="dynamic",
            dropout=0.1,
        )
        self.dynamic_encoder = nn.Linear(1, hidden_dim)
        self.treatment_encoder = nn.Linear(1, hidden_dim)
        self.static_encoder = nn.Linear(1, hidden_dim)
        self.regression_head = nn.Sequential(nn.Linear(hidden_dim + 1, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))

    def _graphs(self, batch: dict[str, torch.Tensor]) -> dict[int, object]:
        from torch_geometric.data import HeteroData

        values = batch["dynamic_features"]
        masks = batch["dynamic_mask"]
        treatment = batch["treatment_features"]
        graphs = {}
        batch_size = values.shape[0]
        for step in range(values.shape[1]):
            graph = HeteroData()
            dyn_signal = (values[:, step, :] * masks[:, step, :]).reshape(-1, 1)
            trt_signal = treatment[:, step, :].reshape(-1, 1) if self.treatment_dim else torch.zeros((batch_size, 1), device=values.device)
            static_signal = batch["baseline_tbr_b"].reshape(-1, 1)
            graph["dynamic"].x = self.dynamic_encoder(dyn_signal)
            graph["treatment"].x = self.treatment_encoder(trt_signal)
            graph["static"].x = self.static_encoder(static_signal)
            dt_src, dt_dst, td_src, td_dst, ds_src, ds_dst, sd_src, sd_dst = [], [], [], [], [], [], [], []
            for sample_idx in range(batch_size):
                dyn_base = sample_idx * self.dynamic_dim
                trt_base = sample_idx * max(1, self.treatment_dim)
                for d_idx in range(self.dynamic_dim):
                    dyn_node = dyn_base + d_idx
                    ds_src.append(dyn_node)
                    ds_dst.append(sample_idx)
                    sd_src.append(sample_idx)
                    sd_dst.append(dyn_node)
                    for t_idx in range(max(1, self.treatment_dim)):
                        trt_node = trt_base + t_idx
                        dt_src.append(dyn_node)
                        dt_dst.append(trt_node)
                        td_src.append(trt_node)
                        td_dst.append(dyn_node)
            graph[("dynamic", "dt", "treatment")].edge_index = torch.tensor([dt_src, dt_dst], dtype=torch.long, device=values.device)
            graph[("treatment", "td", "dynamic")].edge_index = torch.tensor([td_src, td_dst], dtype=torch.long, device=values.device)
            graph[("dynamic", "ds", "static")].edge_index = torch.tensor([ds_src, ds_dst], dtype=torch.long, device=values.device)
            graph[("static", "sd", "dynamic")].edge_index = torch.tensor([sd_src, sd_dst], dtype=torch.long, device=values.device)
            graphs[step] = graph
        return graphs

    def forward(self, batch: dict[str, torch.Tensor], prior_matrix: torch.Tensor | None = None) -> torch.Tensor:
        del prior_matrix
        self.official_model.to(batch["dynamic_features"].device)
        z = self.official_model.encode(self._graphs(batch))
        batch_size = batch["dynamic_features"].shape[0]
        z = z.reshape(batch_size, self.dynamic_dim, -1).mean(dim=1)
        return self.regression_head(torch.cat([z, batch["baseline_tbr_b"]], dim=-1))
