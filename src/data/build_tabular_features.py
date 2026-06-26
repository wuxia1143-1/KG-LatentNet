from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.impute import SimpleImputer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import ids_to_indices, load_dataset, load_fold  # noqa: E402


LEAKAGE_TOKENS = [
    "endpoint_tbr_y",
    "endpoint_window",
    "endpoint_time",
    "endpoint_month",
    "y_true",
    "y_pred",
    "absolute_error",
    "胸主动脉tbr值",
    "胸主动脉tbr",
    "tbr值",
    "endpoint",
    "label",
    "目标",
    "随访tbr",
]
BASELINE_TBR_ALLOWED = {"胸主动脉.2", "baseline_tbr", "baseline_tbr_b"}
STATIC_TIME_EXCLUSION_NAMES = {"治疗前后间隔时间"}
STATIC_TIME_EXCLUSION_TOKENS = ["随访时间", "复查时间", "endpoint", "followup"]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def safe_name(name: Any) -> str:
    text = str(name).strip()
    return text if text else "unnamed"


def leakage_name_hit(feature_name: str) -> str:
    lower = feature_name.lower()
    if any(allowed in feature_name for allowed in BASELINE_TBR_ALLOWED):
        return ""
    for token in LEAKAGE_TOKENS:
        if token and token in lower:
            return token
    return ""


def static_exclusion_hit(name: str) -> str:
    if name in STATIC_TIME_EXCLUSION_NAMES:
        return "explicit_endpoint_interval_like_static_feature"
    lower = name.lower()
    for token in STATIC_TIME_EXCLUSION_TOKENS:
        if token in lower:
            return f"static_time_token:{token}"
    return ""


def assert_no_leakage(feature_names: list[str]) -> None:
    hits = [(name, leakage_name_hit(name)) for name in feature_names]
    hits = [(name, token) for name, token in hits if token]
    if hits:
        preview = "; ".join(f"{name} -> {token}" for name, token in hits[:20])
        raise ValueError(f"Leakage-like feature names detected in tabular X: {preview}")


def finite_or_nan(value: float) -> float:
    value = float(value)
    return value if math.isfinite(value) else float("nan")


def summarize_observed(values: np.ndarray, obs_mask: np.ndarray, obs_time: np.ndarray) -> dict[str, float]:
    observed = np.asarray(values[obs_mask], dtype=np.float64)
    times = np.asarray(obs_time[obs_mask], dtype=np.float64)
    out: dict[str, float] = {
        "last": float("nan"),
        "mean": float("nan"),
        "std": float("nan"),
        "min": float("nan"),
        "max": float("nan"),
        "first": float("nan"),
        "change": float("nan"),
        "slope": float("nan"),
    }
    if observed.size == 0:
        return out
    out["first"] = finite_or_nan(observed[0])
    out["last"] = finite_or_nan(observed[-1])
    out["mean"] = finite_or_nan(np.mean(observed))
    out["std"] = finite_or_nan(np.std(observed))
    out["min"] = finite_or_nan(np.min(observed))
    out["max"] = finite_or_nan(np.max(observed))
    out["change"] = finite_or_nan(observed[-1] - observed[0])
    if observed.size >= 2 and np.nanmax(times) > np.nanmin(times):
        centered = times - np.nanmean(times)
        denom = float(np.nansum(centered**2))
        if denom > 1e-12:
            out["slope"] = finite_or_nan(np.nansum(centered * (observed - np.nanmean(observed))) / denom)
    return out


def add_feature(rows: list[list[float]], names: list[str], row_idx: int, name: str, value: float) -> None:
    if row_idx == 0:
        names.append(name)
    rows[row_idx].append(float(value) if math.isfinite(float(value)) else float("nan"))


def raw_tabular_matrix(dataset: dict[str, Any]) -> tuple[np.ndarray, list[str], list[dict[str, Any]]]:
    patient_count = len(dataset["patient_id"])
    rows: list[list[float]] = [[] for _ in range(patient_count)]
    names: list[str] = []
    exclusion_rows: list[dict[str, Any]] = []

    static_names = [safe_name(name) for name in dataset["feature_names"].get("static_features", [])]
    baseline_names = {safe_name(name) for name in dataset["feature_names"].get("baseline_tbr_b", [])}
    for col_idx, source_name in enumerate(static_names):
        reason = static_exclusion_hit(source_name)
        if reason:
            exclusion_rows.append({"source_group": "static", "feature_name": source_name, "reason": reason})
            continue
        if source_name in baseline_names or source_name in BASELINE_TBR_ALLOWED:
            exclusion_rows.append({"source_group": "static", "feature_name": source_name, "reason": "deduplicated_to_baseline_tbr_b"})
            continue
        candidate_name = f"static::{source_name}"
        if leakage_name_hit(candidate_name):
            exclusion_rows.append({"source_group": "static", "feature_name": source_name, "reason": "leakage_blacklist"})
            continue
        values = dataset["static_features"][:, col_idx]
        for row_idx, value in enumerate(values):
            add_feature(rows, names, row_idx, candidate_name, finite_or_nan(value))

    for row_idx, value in enumerate(dataset["baseline_tbr_b"].reshape(-1)):
        add_feature(rows, names, row_idx, "baseline_tbr_b", finite_or_nan(value))

    dynamic = np.asarray(dataset["dynamic_features"], dtype=np.float32)
    dynamic_mask = np.asarray(dataset["dynamic_mask"], dtype=np.float32) > 0
    delta_time = np.asarray(dataset["delta_time"], dtype=np.float32)
    treatment = np.asarray(dataset["treatment_features"], dtype=np.float32)
    endpoint_month = np.asarray(dataset.get("endpoint_month", np.zeros(patient_count)), dtype=np.float32).reshape(-1)
    dynamic_names = [safe_name(name) for name in dataset["feature_names"].get("dynamic_features", [])]
    treatment_names = [safe_name(name) for name in dataset["feature_names"].get("treatment_features", [])]
    max_steps = int(dynamic.shape[1])
    cumulative_time = np.cumsum(np.clip(delta_time, a_min=0.0, a_max=None), axis=1)

    any_history_mask = (dynamic_mask.any(axis=2)) | (np.abs(treatment).sum(axis=2) > 0)
    dynamic_observed_cells = dynamic_mask.sum(axis=(1, 2)).astype(np.float64)
    total_dynamic_cells = float(max_steps * max(dynamic.shape[2], 1))
    for row_idx in range(patient_count):
        observed_steps = np.where(any_history_mask[row_idx])[0]
        num_history = float(observed_steps.size)
        add_feature(rows, names, row_idx, "history::num_history_records", num_history)
        add_feature(rows, names, row_idx, "history::has_dynamic_history", float(num_history > 0))
        add_feature(rows, names, row_idx, "history::overall_missing_rate", 1.0 - dynamic_observed_cells[row_idx] / total_dynamic_cells)
        if observed_steps.size:
            min_obs = float(cumulative_time[row_idx, observed_steps[0]])
            max_obs = float(cumulative_time[row_idx, observed_steps[-1]])
            gap = float(endpoint_month[row_idx] - max_obs) if math.isfinite(float(endpoint_month[row_idx])) else float("nan")
        else:
            min_obs = float("nan")
            max_obs = float("nan")
            gap = float("nan")
        add_feature(rows, names, row_idx, "history::min_observation_month_before_assessment", min_obs)
        add_feature(rows, names, row_idx, "history::max_observation_month_before_assessment", max_obs)
        add_feature(rows, names, row_idx, "history::gap_last_obs_to_assessment_month", gap)

    stats = ["last", "mean", "std", "min", "max", "first", "change", "slope"]
    for feature_idx, source_name in enumerate(dynamic_names):
        if leakage_name_hit(source_name):
            exclusion_rows.append({"source_group": "dynamic", "feature_name": source_name, "reason": "leakage_blacklist"})
            continue
        for row_idx in range(patient_count):
            obs_mask = dynamic_mask[row_idx, :, feature_idx]
            summary = summarize_observed(dynamic[row_idx, :, feature_idx], obs_mask, cumulative_time[row_idx])
            for stat in stats:
                add_feature(rows, names, row_idx, f"dynamic::{source_name}::{stat}", summary[stat])
            count = float(obs_mask.sum())
            add_feature(rows, names, row_idx, f"dynamic::{source_name}::observed_count", count)
            add_feature(rows, names, row_idx, f"dynamic::{source_name}::missing_rate", 1.0 - count / max_steps)
            add_feature(rows, names, row_idx, f"dynamic::{source_name}::missing_indicator", float(count == 0))
            if count > 0:
                last_idx = int(np.where(obs_mask)[0][-1])
                gap = float(endpoint_month[row_idx] - cumulative_time[row_idx, last_idx]) if math.isfinite(float(endpoint_month[row_idx])) else float("nan")
            else:
                gap = float("nan")
            add_feature(rows, names, row_idx, f"dynamic::{source_name}::gap_last_obs_to_assessment_month", gap)

    for feature_idx, source_name in enumerate(treatment_names):
        if leakage_name_hit(source_name):
            exclusion_rows.append({"source_group": "treatment", "feature_name": source_name, "reason": "leakage_blacklist"})
            continue
        values = treatment[:, :, feature_idx]
        active = values > 0
        for row_idx in range(patient_count):
            row_values = values[row_idx]
            row_active = active[row_idx]
            add_feature(rows, names, row_idx, f"treatment::{source_name}::any", float(np.any(row_active)))
            add_feature(rows, names, row_idx, f"treatment::{source_name}::count", float(np.sum(row_active)))
            add_feature(rows, names, row_idx, f"treatment::{source_name}::sum", finite_or_nan(np.sum(row_values)))
            add_feature(rows, names, row_idx, f"treatment::{source_name}::mean", finite_or_nan(np.mean(row_values)))
            active_idx = np.where(row_active)[0]
            last = float(row_values[active_idx[-1]]) if active_idx.size else 0.0
            add_feature(rows, names, row_idx, f"treatment::{source_name}::last", last)

    assert_no_leakage(names)
    return np.asarray(rows, dtype=np.float32), names, exclusion_rows


def metadata_for_indices(dataset: dict[str, Any], indices: np.ndarray) -> dict[str, Any]:
    return {
        "patient_id": [str(dataset["patient_id"][idx]) for idx in indices],
        "endpoint_window": np.asarray(dataset["endpoint_window"][indices], dtype=np.int64),
        "endpoint_time": [str(dataset["endpoint_time"][idx]) for idx in indices],
        "endpoint_tbr_y": np.asarray(dataset["endpoint_tbr_y"][indices], dtype=np.float32).reshape(-1),
    }


def save_tabular_split(
    path: Path,
    x_raw: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    meta: dict[str, Any],
    imputer: SimpleImputer,
) -> None:
    x = imputer.transform(x_raw).astype(np.float32)
    payload = {
        "X": x,
        "X_raw": x_raw.astype(np.float32),
        "y": y.astype(np.float32).reshape(-1),
        "patient_id": meta["patient_id"],
        "endpoint_window": meta["endpoint_window"],
        "endpoint_time": meta["endpoint_time"],
        "feature_names": feature_names,
        "preprocessing": {
            "imputer": imputer,
            "imputer_strategy": "median",
            "imputer_fit_on": "train_only",
            "scaler_used": False,
            "endpoint_tbr_y_used_for_fit": False,
            "endpoint_window_used_as_feature": False,
            "endpoint_time_used_as_feature": False,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def split_overlap(fold: dict[str, Any]) -> dict[str, int]:
    train = set(map(str, fold["train_patient_ids"]))
    val = set(map(str, fold["val_patient_ids"]))
    test = set(map(str, fold["test_patient_ids"]))
    return {
        "train_val_overlap": len(train & val),
        "train_test_overlap": len(train & test),
        "val_test_overlap": len(val & test),
    }


def write_summaries(
    project_root: Path,
    x_raw: np.ndarray,
    feature_names: list[str],
    exclusion_rows: list[dict[str, Any]],
    fold_rows: list[dict[str, Any]],
) -> None:
    group_counts = Counter(name.split("::", 1)[0] if "::" in name else name for name in feature_names)
    summary_rows = [
        {"feature_group": group, "feature_count": count}
        for group, count in sorted(group_counts.items())
    ]
    summary_rows.append({"feature_group": "total", "feature_count": len(feature_names)})
    write_csv(project_root / "results" / "tables" / "tabular_feature_summary.csv", summary_rows, ["feature_group", "feature_count"])

    missing_rows = []
    for idx, name in enumerate(feature_names):
        column = x_raw[:, idx]
        missing = ~np.isfinite(column)
        missing_rows.append(
            {
                "feature_name": name,
                "missing_rate_before_impute": float(np.mean(missing)),
                "observed_count_before_impute": int(np.sum(~missing)),
            }
        )
    write_csv(project_root / "results" / "tables" / "tabular_missing_summary.csv", missing_rows, ["feature_name", "missing_rate_before_impute", "observed_count_before_impute"])
    write_csv(project_root / "results" / "tables" / "tabular_feature_exclusions.csv", exclusion_rows, ["source_group", "feature_name", "reason"])
    write_csv(project_root / "results" / "tables" / "tabular_fold_preprocess_summary.csv", fold_rows, ["fold", "train_n", "val_n", "test_n", "feature_count", "train_val_overlap", "train_test_overlap", "val_test_overlap", "imputer_fit_on", "scaler_used"])


def build_tabular_for_all_folds(project_root: Path, folds: int = 5) -> dict[str, Any]:
    dataset = load_dataset(project_root)
    if len(dataset["patient_id"]) != 417:
        raise ValueError(f"Expected 417 patient-level samples, got {len(dataset['patient_id'])}")
    x_raw, feature_names, exclusion_rows = raw_tabular_matrix(dataset)
    assert_no_leakage(feature_names)

    output_dir = project_root / "data" / "processed" / "tabular"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tabular_feature_names.json").write_text(json.dumps(feature_names, ensure_ascii=False, indent=2), encoding="utf-8")

    all_y = np.asarray(dataset["endpoint_tbr_y"], dtype=np.float32).reshape(-1)
    fold_rows: list[dict[str, Any]] = []
    for fold_idx in range(folds):
        fold = load_fold(project_root, fold_idx)
        train_idx = ids_to_indices(dataset, fold["train_patient_ids"])
        val_idx = ids_to_indices(dataset, fold["val_patient_ids"])
        test_idx = ids_to_indices(dataset, fold["test_patient_ids"])
        overlap = split_overlap(fold)
        if any(overlap.values()):
            raise ValueError(f"Patient-level leakage in fold {fold_idx}: {overlap}")
        imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        imputer.fit(x_raw[train_idx])
        for split_name, indices in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
            save_tabular_split(
                output_dir / f"fold_{fold_idx}_tabular_{split_name}.pkl",
                x_raw[indices],
                all_y[indices],
                feature_names,
                metadata_for_indices(dataset, indices),
                imputer,
            )
        fold_rows.append(
            {
                "fold": fold_idx,
                "train_n": len(train_idx),
                "val_n": len(val_idx),
                "test_n": len(test_idx),
                "feature_count": len(feature_names),
                **overlap,
                "imputer_fit_on": "train_only",
                "scaler_used": False,
            }
        )
    write_summaries(project_root, x_raw, feature_names, exclusion_rows, fold_rows)
    return {"feature_count": len(feature_names), "folds": folds, "patient_count": len(dataset["patient_id"])}


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Build fold-specific tabular features for classical ML baselines.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args(argv)
    result = build_tabular_for_all_folds(Path(args.project_root).resolve(), folds=args.folds)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()
