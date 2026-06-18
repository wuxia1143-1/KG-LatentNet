from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from src.models.baselines.official_adapters.common import OfficialAdapterBase, prepend_paths


class _SimpleTokenizer:
    def __init__(self, tokens: list[str]) -> None:
        self.token_to_idx = {"<pad>": 0}
        for token in tokens:
            if token not in self.token_to_idx:
                self.token_to_idx[token] = len(self.token_to_idx)

    def get_vocabulary_size(self) -> int:
        return len(self.token_to_idx)

    def get_padding_index(self) -> int:
        return 0

    def batch_encode_3d(self, data: list[list[list[str]]]) -> list[list[list[int]]]:
        max_visits = max(len(patient) for patient in data)
        max_codes = max((len(visit) for patient in data for visit in patient), default=1)
        max_codes = max(1, max_codes)
        encoded = []
        for patient in data:
            patient_rows = []
            for visit in patient:
                row = [self.token_to_idx.get(token, 0) for token in visit[:max_codes]]
                row += [0] * (max_codes - len(row))
                patient_rows.append(row)
            while len(patient_rows) < max_visits:
                patient_rows.append([0] * max_codes)
            encoded.append(patient_rows)
        return encoded


class TRANSOfficialAdapter(OfficialAdapterBase):
    """Official TRANS adapter.

    Official model class imported: `remote_baselines/TRANS/models/Model.py::TRANS`.
    Official input format: `(seqdata, graph_list)`, where `seqdata` contains tokenized EHR sequences and `graph_list` contains PyG HeteroData graphs.
    Data mapping: dynamic variable observations map to `cond_hist`; treatment events map to `procedures` and `drugs`; per-patient graphs connect code nodes to visit nodes.
    Output mapping: official TRANS is instantiated with `output_size=1`, so its forward output is endpoint TBR prediction.
    Official loss: not used; current scalar endpoint regression uses MSE.
    Official core modules retained: official TRANS sequence transformer and HGT graph model.
    """

    baseline_name = "trans"
    official_repo_url = "https://github.com/The-Real-JerryChen/TRANS.git"
    official_model_class_used = "models.Model.TRANS"
    official_entry_script = "train.py --model TRANS --dataset <dataset>"
    official_dependencies_file = "README requirements section"

    def __init__(self, project_root: Path, static_dim: int, dynamic_dim: int, treatment_dim: int, hidden_dim: int = 64) -> None:
        super().__init__(project_root)
        del static_dim
        repo = self.project_root / "remote_baselines" / "TRANS"
        prepend_paths(repo / "models", repo / "layers", repo)
        from Model import TRANS, graph_meta  # type: ignore

        self.dynamic_dim = dynamic_dim
        self.treatment_dim = treatment_dim
        self.cond_tokens = [f"d{i}" for i in range(dynamic_dim)]
        self.proc_tokens = [f"t{i}" for i in range(max(1, treatment_dim))]
        self.drug_tokens = [f"rx{i}" for i in range(max(1, treatment_dim))]
        tokenizers = {
            "cond_hist": _SimpleTokenizer(self.cond_tokens),
            "procedures": _SimpleTokenizer(self.proc_tokens),
            "drugs": _SimpleTokenizer(self.drug_tokens),
        }
        self.official_model = TRANS(
            Tokenizers=tokenizers,
            hidden_size=hidden_dim,
            output_size=1,
            device=torch.device("cpu"),
            graph_meta=graph_meta,
            embedding_dim=hidden_dim,
            dropout=0.1,
            num_heads=2,
            num_layers=1,
            pe=False,
        )

    def _sequence_data(self, batch: dict[str, torch.Tensor]) -> dict[str, list[list[list[str]]]]:
        dynamic_mask = batch["dynamic_mask"].detach().cpu()
        treatment = batch["treatment_features"].detach().cpu()
        seqdata = {"cond_hist": [], "procedures": [], "drugs": []}
        for sample_idx in range(dynamic_mask.shape[0]):
            cond_visits = []
            proc_visits = []
            drug_visits = []
            for step in range(dynamic_mask.shape[1]):
                cond = [self.cond_tokens[i] for i in torch.where(dynamic_mask[sample_idx, step] > 0)[0].tolist()]
                trt_idx = torch.where(treatment[sample_idx, step] > 0)[0].tolist()
                proc = [self.proc_tokens[i] for i in trt_idx] or []
                drug = [self.drug_tokens[i] for i in trt_idx] or []
                cond_visits.append(cond)
                proc_visits.append(proc)
                drug_visits.append(drug)
            seqdata["cond_hist"].append(cond_visits)
            seqdata["procedures"].append(proc_visits)
            seqdata["drugs"].append(drug_visits)
        return seqdata

    def _graph_list(self, batch: dict[str, torch.Tensor]) -> list[object]:
        from torch_geometric.data import HeteroData

        dynamic_mask = batch["dynamic_mask"].detach().cpu()
        treatment = batch["treatment_features"].detach().cpu()
        times = torch.cumsum(batch["delta_time"].detach().cpu(), dim=1)
        graph_list = []
        for sample_idx in range(dynamic_mask.shape[0]):
            data = HeteroData()
            time_steps = dynamic_mask.shape[1]
            data["visit"].x = torch.zeros((time_steps, 1), dtype=torch.float32)
            data["visit"].time = times[sample_idx].float()
            data["co"].x = torch.zeros((len(self.cond_tokens) + 1, 1), dtype=torch.float32)
            data["pr"].x = torch.zeros((len(self.proc_tokens) + 1, 1), dtype=torch.float32)
            data["dh"].x = torch.zeros((len(self.drug_tokens) + 1, 1), dtype=torch.float32)
            edge_payload: dict[tuple[str, str, str], list[list[int]]] = {
                ("co", "in", "visit"): [[], []],
                ("pr", "in", "visit"): [[], []],
                ("dh", "in", "visit"): [[], []],
                ("visit", "connect", "visit"): [[], []],
                ("visit", "has", "co"): [[], []],
                ("visit", "has", "pr"): [[], []],
                ("visit", "has", "dh"): [[], []],
            }
            edge_times: dict[tuple[str, str, str], list[float]] = {key: [] for key in edge_payload}
            for step in range(time_steps):
                for dyn_idx in torch.where(dynamic_mask[sample_idx, step] > 0)[0].tolist() or [0]:
                    code_id = dyn_idx + 1
                    edge_payload[("co", "in", "visit")][0].append(code_id)
                    edge_payload[("co", "in", "visit")][1].append(step)
                    edge_payload[("visit", "has", "co")][0].append(step)
                    edge_payload[("visit", "has", "co")][1].append(code_id)
                    edge_times[("co", "in", "visit")].append(float(times[sample_idx, step]))
                trt_idx = torch.where(treatment[sample_idx, step] > 0)[0].tolist() or [0]
                for t_idx in trt_idx:
                    proc_id = t_idx + 1
                    drug_id = t_idx + 1
                    edge_payload[("pr", "in", "visit")][0].append(proc_id)
                    edge_payload[("pr", "in", "visit")][1].append(step)
                    edge_payload[("dh", "in", "visit")][0].append(drug_id)
                    edge_payload[("dh", "in", "visit")][1].append(step)
                    edge_payload[("visit", "has", "pr")][0].append(step)
                    edge_payload[("visit", "has", "pr")][1].append(proc_id)
                    edge_payload[("visit", "has", "dh")][0].append(step)
                    edge_payload[("visit", "has", "dh")][1].append(drug_id)
                    edge_times[("pr", "in", "visit")].append(float(times[sample_idx, step]))
                    edge_times[("dh", "in", "visit")].append(float(times[sample_idx, step]))
                if step + 1 < time_steps:
                    edge_payload[("visit", "connect", "visit")][0].append(step)
                    edge_payload[("visit", "connect", "visit")][1].append(step + 1)
            data.edge_time_dict = {}
            for edge_type, values in edge_payload.items():
                if values[0]:
                    data[edge_type].edge_index = torch.tensor(values, dtype=torch.long)
                else:
                    data[edge_type].edge_index = torch.zeros((2, 0), dtype=torch.long)
                if edge_type[0] != "visit":
                    data.edge_time_dict[edge_type] = torch.tensor(edge_times[edge_type] or [], dtype=torch.float32)
            graph_list.append(data)
        return graph_list

    def forward(self, batch: dict[str, torch.Tensor], prior_matrix: torch.Tensor | None = None) -> torch.Tensor:
        del prior_matrix
        self.official_model.device = batch["dynamic_features"].device
        self.official_model.to(batch["dynamic_features"].device)
        seqdata = self._sequence_data(batch)
        graph_list = self._graph_list(batch)
        return self.official_model((seqdata, graph_list))
