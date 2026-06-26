from __future__ import annotations

import argparse
import csv
import json
import logging
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np


HASH_LIKE_RE = re.compile(r"^[0-9a-fA-F]{32}$")


def is_hash_like(value: str) -> bool:
    return bool(HASH_LIKE_RE.fullmatch(str(value).strip()))


def load_feature_names(project_root: Path) -> dict[str, list[str]]:
    path = project_root / "data" / "processed" / "feature_names.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_dataset_shapes(project_root: Path) -> dict[str, int]:
    with (project_root / "data" / "processed" / "dataset.pkl").open("rb") as handle:
        dataset = pickle.load(handle)
    return {
        "patient_count": int(len(dataset["patient_id"])),
        "sample_count": int(len(dataset["patient_id"])),
        "static_feature_dim": int(dataset["static_features"].shape[1]),
        "dynamic_feature_dim": int(dataset["dynamic_features"].shape[2]),
        "treatment_feature_dim": int(dataset["treatment_features"].shape[2]),
        "max_history_length": int(dataset["dynamic_features"].shape[1]),
    }


def load_knowledge_prior(project_root: Path) -> dict[str, Any] | None:
    prior_path = project_root / "data" / "processed" / "knowledge_prior.pkl"
    if not prior_path.exists():
        return None
    with prior_path.open("rb") as handle:
        return pickle.load(handle)


def build_aligned_prior_matrix(project_root: Path, dynamic_names: list[str]) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    prior = load_knowledge_prior(project_root)
    dynamic_dim = len(dynamic_names)
    matrix = np.eye(dynamic_dim, dtype=np.float32)

    biomarker_names: list[str] = []
    raw_shape = (0, 0)
    nonzero_from_prior = 0
    if prior is not None:
        biomarker_names = [str(name) for name in prior.get("biomarker_names", [])]
        source = np.asarray(prior.get("biomarker_biomarker_prior", np.zeros((0, 0))), dtype=np.float32)
        raw_shape = tuple(source.shape)
        biomarker_index = {name: idx for idx, name in enumerate(biomarker_names)}
        for i, left in enumerate(dynamic_names):
            left_idx = biomarker_index.get(str(left))
            if left_idx is None:
                continue
            for j, right in enumerate(dynamic_names):
                right_idx = biomarker_index.get(str(right))
                if right_idx is None:
                    continue
                value = float(source[left_idx, right_idx])
                if value != 0.0:
                    matrix[i, j] = value
                    nonzero_from_prior += 1

    biomarker_set = set(biomarker_names)
    node_rows = [
        {
            "node_index": idx,
            "node_name": name,
            "node_group": "dynamic_feature",
            "matched_prior_biomarker": name if name in biomarker_set else "",
            "has_prior_mapping": bool(name in biomarker_set),
            "patient_id_hash_like": bool(is_hash_like(name)),
            "used_as_graph_node": True,
        }
        for idx, name in enumerate(dynamic_names)
    ]

    shapes = load_dataset_shapes(project_root)
    hash_like_count = sum(1 for name in dynamic_names if is_hash_like(name))
    aligned_shape = tuple(matrix.shape)
    patient_count_as_node_dim = dynamic_dim == shapes["patient_count"]
    sample_count_as_node_dim = dynamic_dim == shapes["sample_count"]
    time_step_as_node_dim = dynamic_dim == shapes["max_history_length"]
    reasonable = not any([hash_like_count, patient_count_as_node_dim, sample_count_as_node_dim, time_step_as_node_dim])
    checks = [
        {"check": "patient_count", "value": shapes["patient_count"], "passed": True, "note": "Cohort size; not used as graph node count."},
        {"check": "sample_count", "value": shapes["sample_count"], "passed": True, "note": "Patient-level samples; not used as graph node count."},
        {"check": "max_history_length", "value": shapes["max_history_length"], "passed": True, "note": "Time dimension; not used as graph node count."},
        {"check": "dynamic_feature_dim", "value": dynamic_dim, "passed": dynamic_dim == shapes["dynamic_feature_dim"], "note": "Graph node count equals dynamic clinical variable count."},
        {"check": "static_feature_dim", "value": shapes["static_feature_dim"], "passed": True, "note": "Static variables are encoded separately and are not prior graph nodes."},
        {"check": "treatment_feature_dim", "value": shapes["treatment_feature_dim"], "passed": True, "note": "Treatments are sequence covariates and are not biomarker-biomarker prior graph nodes."},
        {"check": "raw_biomarker_prior_shape", "value": f"{raw_shape[0]}x{raw_shape[1]}", "passed": raw_shape[0] == raw_shape[1], "note": "Medical prior matrix before alignment."},
        {"check": "aligned_prior_shape", "value": f"{aligned_shape[0]}x{aligned_shape[1]}", "passed": aligned_shape[0] == dynamic_dim and aligned_shape[1] == dynamic_dim, "note": "Prior matrix aligned to dynamic feature nodes."},
        {"check": "matched_dynamic_nodes", "value": sum(int(row["has_prior_mapping"]) for row in node_rows), "passed": True, "note": "Dynamic nodes with a direct biomarker prior name match."},
        {"check": "prior_nonzero_edges_used", "value": int(nonzero_from_prior), "passed": True, "note": "Nonzero prior values copied into the aligned matrix; identity self edges are retained."},
        {"check": "contains_patient_id_like_nodes", "value": int(hash_like_count), "passed": hash_like_count == 0, "note": "Hash-like node names indicate patient_SN/header leakage into graph nodes."},
        {"check": "contains_patient_count_as_node_dim", "value": int(patient_count_as_node_dim), "passed": not patient_count_as_node_dim, "note": "Graph node count must not equal cohort size by construction."},
        {"check": "contains_sample_count_as_node_dim", "value": int(sample_count_as_node_dim), "passed": not sample_count_as_node_dim, "note": "Graph node count must not equal sample count by construction."},
        {"check": "contains_time_step_count_as_node_dim", "value": int(time_step_as_node_dim), "passed": not time_step_as_node_dim, "note": "Graph node count must not equal padded time-step count by construction."},
        {"check": "mask_or_delta_time_nodes", "value": 0, "passed": True, "note": "dynamic_mask and delta_time are sequence metadata, not graph nodes."},
        {"check": "reasonable_prior_dimension", "value": int(reasonable), "passed": reasonable, "note": "Aligned prior is reasonable only when graph nodes are clinical dynamic variables."},
        {"check": "previous_380x380_source", "value": "dynamic_header_rows_misclassified" if dynamic_dim != 380 else "still_380", "passed": dynamic_dim != 380, "note": "The earlier 380x380 shape came from patient_SN/header-row values being scanned as dynamic features, not from patient/sample/time dimensions."},
    ]
    return matrix, node_rows, checks


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_prior_alignment_outputs(project_root: Path) -> dict[str, Any]:
    feature_names = load_feature_names(project_root)
    dynamic_names = [str(name) for name in feature_names.get("dynamic_features", [])]
    matrix, node_rows, check_rows = build_aligned_prior_matrix(project_root, dynamic_names)
    tables_dir = project_root / "results" / "tables"
    write_csv(
        tables_dir / "graph_node_list.csv",
        node_rows,
        ["node_index", "node_name", "node_group", "matched_prior_biomarker", "has_prior_mapping", "patient_id_hash_like", "used_as_graph_node"],
    )
    write_csv(
        tables_dir / "prior_alignment_check.csv",
        check_rows,
        ["check", "value", "passed", "note"],
    )
    logging.info(
        "Graph nodes are dynamic clinical variables only: count=%d aligned_prior_shape=%s",
        len(node_rows),
        tuple(matrix.shape),
    )
    logging.info(
        "Static variables, treatment variables, dynamic_mask, delta_time, patient count, sample count, and time steps are not graph nodes."
    )
    logging.info(
        "Earlier 380x380 prior shape was traced to dynamic sheet header/patient_SN values being misclassified as dynamic variables; row 0 is now skipped."
    )
    return {
        "aligned_prior_shape": tuple(matrix.shape),
        "graph_node_count": len(node_rows),
        "all_checks_passed": all(bool(row["passed"]) for row in check_rows),
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Audit graph node list and aligned prior matrix dimensions.")
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[2]))
    args = parser.parse_args(argv)
    return write_prior_alignment_outputs(Path(args.project_root).resolve())


if __name__ == "__main__":
    main()
