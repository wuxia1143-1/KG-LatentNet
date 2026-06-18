from __future__ import annotations

from pathlib import Path
from typing import Any

import importlib.util
import sys
import torch

from src.models.baselines.official_adapters.common import OfficialAdapterBase, prepend_paths


class GraphCareOfficialAdapter(OfficialAdapterBase):
    """Official GraphCare adapter.

    Official model class imported: `remote_baselines/GraphCare/graphcare_/model.py::GraphCare`.
    Official input format: node ids, relation ids, graph edges, node batch vector, visit-node matrix, and direct EHR node matrix.
    Data mapping: dynamic variables, treatment events, and baseline TBR are mapped to patient graph nodes.
    Output mapping: official GraphCare is instantiated with `out_channels=1`, so forward returns endpoint TBR prediction.
    Official loss: not used; current scalar endpoint regression uses MSE.
    Official core modules retained: official GraphCare attention GNN and patient pooling logic.
    """

    baseline_name = "graphcare"
    official_repo_url = "https://github.com/pat-jj/GraphCare.git"
    official_model_class_used = "graphcare_.model.GraphCare"
    official_entry_script = "run.sh / graphcare.py"
    official_dependencies_file = "README PyHealth version note"

    def __init__(self, project_root: Path, static_dim: int, dynamic_dim: int, treatment_dim: int, hidden_dim: int = 64) -> None:
        super().__init__(project_root)
        del static_dim
        repo = self.project_root / "remote_baselines" / "GraphCare"
        prepend_paths(repo)
        model_path = repo / "graphcare_" / "model.py"
        spec = importlib.util.spec_from_file_location("official_graphcare_model", model_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load official GraphCare model from {model_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules["official_graphcare_model"] = module
        spec.loader.exec_module(module)
        GraphCare = module.GraphCare

        self.dynamic_dim = dynamic_dim
        self.treatment_dim = treatment_dim
        self.num_nodes = dynamic_dim + treatment_dim + 1
        self.num_rels = 3
        self.official_model = GraphCare(
            num_nodes=self.num_nodes,
            num_rels=self.num_rels,
            max_visit=47,
            embedding_dim=hidden_dim,
            hidden_dim=hidden_dim,
            out_channels=1,
            layers=1,
            dropout=0.1,
            patient_mode="joint",
            gnn="BAT",
        )

    def input_usage(self) -> dict[str, Any]:
        row = super().input_usage()
        row["static_features"] = False
        row["delta_time"] = False
        row["usage_note"] = "Official GraphCare core uses patient graph nodes from dynamic observations, treatments, and baseline TBR; static features are not part of the official graph interface."
        return row

    def forward(self, batch: dict[str, torch.Tensor], prior_matrix: torch.Tensor | None = None) -> torch.Tensor:
        del prior_matrix
        values = batch["dynamic_features"]
        dynamic_presence = (batch["dynamic_mask"] > 0).float()
        treatment = (batch["treatment_features"] > 0).float()
        batch_size, time_steps, _ = values.shape
        node_ids_list = []
        batch_vec = []
        edge_src = []
        edge_dst = []
        rel_ids = []
        visit_node = torch.zeros((batch_size, time_steps, self.num_nodes), dtype=values.dtype, device=values.device)
        ehr_nodes = []
        for sample_idx in range(batch_size):
            offset = sample_idx * self.num_nodes
            node_ids_list.extend(range(self.num_nodes))
            batch_vec.extend([sample_idx] * self.num_nodes)
            visit_node[sample_idx, :, : self.dynamic_dim] = dynamic_presence[sample_idx]
            visit_node[sample_idx, :, self.dynamic_dim : self.dynamic_dim + self.treatment_dim] = treatment[sample_idx]
            visit_node[sample_idx, :, -1] = batch["baseline_tbr_b"][sample_idx].clamp_min(0.0)
            ehr = visit_node[sample_idx].sum(dim=0).clamp_min(0.0)
            ehr[-1] = batch["baseline_tbr_b"][sample_idx].abs().squeeze() + 1.0
            ehr_nodes.append(ehr)
            active = torch.where(ehr > 0)[0].tolist() or [self.num_nodes - 1]
            for src in active:
                for dst in active:
                    edge_src.append(offset + src)
                    edge_dst.append(offset + dst)
                    rel_ids.append(0 if src < self.dynamic_dim and dst < self.dynamic_dim else 1)
        node_ids = torch.tensor(node_ids_list, dtype=torch.long, device=values.device)
        rel_ids_tensor = torch.tensor(rel_ids or [0], dtype=torch.long, device=values.device)
        edge_index = torch.tensor([edge_src or [0], edge_dst or [0]], dtype=torch.long, device=values.device)
        batch_tensor = torch.tensor(batch_vec, dtype=torch.long, device=values.device)
        self.official_model.to(values.device)
        return self.official_model(node_ids, rel_ids_tensor, edge_index, batch_tensor, visit_node, ehr_nodes)
