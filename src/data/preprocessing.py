from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

STANDARDIZED_FEATURE_CLIP = 10.0


def load_dataset(project_root: Path) -> dict[str, Any]:
    with (project_root / "data" / "processed" / "dataset.pkl").open("rb") as handle:
        return pickle.load(handle)


def load_fold(project_root: Path, fold: int) -> dict[str, Any]:
    return json.loads((project_root / "data" / "splits" / f"fold_{fold}.json").read_text(encoding="utf-8"))


def index_by_patient(dataset: dict[str, Any]) -> dict[str, int]:
    return {str(pid): idx for idx, pid in enumerate(dataset["patient_id"])}


def ids_to_indices(dataset: dict[str, Any], ids: list[str]) -> np.ndarray:
    lookup = index_by_patient(dataset)
    return np.asarray([lookup[str(pid)] for pid in ids], dtype=np.int64)


def nanmean_std(values: np.ndarray, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(values, axis=axis)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.nanstd(values, axis=axis)
    std = np.where((np.isfinite(std)) & (std > 1e-6), std, 1.0)
    return mean.astype(np.float32), std.astype(np.float32)


def fit_preprocess(dataset: dict[str, Any], train_indices: np.ndarray) -> dict[str, Any]:
    static_train = dataset["static_features"][train_indices]
    baseline_train = dataset["baseline_tbr_b"][train_indices]
    dynamic_train = dataset["dynamic_features"][train_indices]
    dynamic_mask_train = dataset["dynamic_mask"][train_indices]

    static_mean, static_std = nanmean_std(static_train, axis=0)
    baseline_mean, baseline_std = nanmean_std(baseline_train, axis=0)

    dynamic_dim = dynamic_train.shape[-1]
    dynamic_mean = np.zeros(dynamic_dim, dtype=np.float32)
    dynamic_std = np.ones(dynamic_dim, dtype=np.float32)
    for feature_idx in range(dynamic_dim):
        observed = dynamic_mask_train[:, :, feature_idx] > 0
        values = dynamic_train[:, :, feature_idx][observed]
        if len(values):
            dynamic_mean[feature_idx] = float(np.nanmean(values))
            std = float(np.nanstd(values))
            dynamic_std[feature_idx] = std if std > 1e-6 and np.isfinite(std) else 1.0

    return {
        "static_mean": static_mean,
        "static_std": static_std,
        "baseline_mean": baseline_mean,
        "baseline_std": baseline_std,
        "dynamic_mean": dynamic_mean,
        "dynamic_std": dynamic_std,
        "fit_on": "train_only",
        "endpoint_tbr_y_used_for_fit": False,
    }


def apply_preprocess(dataset: dict[str, Any], params: dict[str, Any], indices: np.ndarray) -> dict[str, np.ndarray]:
    static = dataset["static_features"][indices].astype(np.float32)
    static = np.where(np.isfinite(static), static, params["static_mean"])
    static = (static - params["static_mean"]) / params["static_std"]
    static = np.clip(static, -STANDARDIZED_FEATURE_CLIP, STANDARDIZED_FEATURE_CLIP)

    baseline = dataset["baseline_tbr_b"][indices].astype(np.float32)
    baseline = np.where(np.isfinite(baseline), baseline, params["baseline_mean"])
    baseline = (baseline - params["baseline_mean"]) / params["baseline_std"]
    baseline = np.clip(baseline, -STANDARDIZED_FEATURE_CLIP, STANDARDIZED_FEATURE_CLIP)

    dynamic = dataset["dynamic_features"][indices].astype(np.float32)
    dynamic_mask = dataset["dynamic_mask"][indices].astype(np.float32)
    dynamic = np.where(dynamic_mask > 0, dynamic, params["dynamic_mean"].reshape(1, 1, -1))
    dynamic = (dynamic - params["dynamic_mean"].reshape(1, 1, -1)) / params["dynamic_std"].reshape(1, 1, -1)
    dynamic = np.clip(dynamic, -STANDARDIZED_FEATURE_CLIP, STANDARDIZED_FEATURE_CLIP)
    dynamic = dynamic * dynamic_mask

    return {
        "static_features": static.astype(np.float32),
        "dynamic_features": dynamic.astype(np.float32),
        "dynamic_mask": dynamic_mask.astype(np.float32),
        "delta_time": dataset["delta_time"][indices].astype(np.float32),
        "treatment_features": dataset["treatment_features"][indices].astype(np.float32),
        "baseline_tbr_b": baseline.astype(np.float32),
        "endpoint_tbr_y": dataset["endpoint_tbr_y"][indices].astype(np.float32),
        "endpoint_window": dataset["endpoint_window"][indices],
        "patient_id": [dataset["patient_id"][idx] for idx in indices],
        "endpoint_time": [dataset["endpoint_time"][idx] for idx in indices],
    }


def create_fold_preprocess(project_root: Path, fold: int) -> Path:
    dataset = load_dataset(project_root)
    fold_payload = load_fold(project_root, fold)
    train_indices = ids_to_indices(dataset, fold_payload["train_patient_ids"])
    params = fit_preprocess(dataset, train_indices)
    output = project_root / "data" / "processed" / f"fold_{fold}_preprocess.pkl"
    with output.open("wb") as handle:
        pickle.dump(params, handle)
    return output


def create_all_fold_preprocess(project_root: Path) -> list[Path]:
    return [create_fold_preprocess(project_root, fold) for fold in range(5)]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Create train-only preprocessing objects for every fold.")
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[2]))
    args = parser.parse_args(argv)
    create_all_fold_preprocess(Path(args.project_root).resolve())


if __name__ == "__main__":
    main()
