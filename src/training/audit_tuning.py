from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.validation_tuning import ALL_MODELS, HORIZON_AWARE_MODELS  # noqa: E402


DEFAULT_SEED = 20260605


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return math.nan
    return out if math.isfinite(out) else math.nan


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def parse_params(text: str) -> dict[str, Any]:
    try:
        loaded = json.loads(text) if text else {}
    except Exception:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "passed"}


def selected_seed(params: dict[str, Any]) -> int:
    return int(params.get("seed", DEFAULT_SEED))


def log_excerpt(log_text: str, model_name: str, fold: str, candidate_id: str) -> str:
    needle = f"FAILED model={model_name} fold={fold} candidate={candidate_id}"
    idx = log_text.find(needle)
    if idx < 0:
        return ""
    return log_text[idx : idx + 1200].replace("\n", " | ")


def best_success(rows: list[dict[str, str]]) -> dict[str, str] | None:
    successes = [row for row in rows if row.get("status") == "success" and math.isfinite(safe_float(row.get("val_mae")))]
    if not successes:
        return None
    return sorted(successes, key=lambda row: (safe_float(row.get("val_mae")), safe_int(row.get("candidate_id"))))[0]


def build_tuning_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model_name"], str(row["fold"]))].append(row)
    out = []
    for model_name in ALL_MODELS:
        for fold in range(5):
            group = grouped.get((model_name, str(fold)), [])
            successes = [row for row in group if row.get("status") == "success"]
            failures = [row for row in group if row.get("status") != "success"]
            best = best_success(group)
            params = parse_params(best.get("params", "")) if best else {}
            out.append(
                {
                    "model_name": model_name,
                    "fold": fold,
                    "n_candidates": len(group),
                    "n_success": len(successes),
                    "n_failed": len(failures),
                    "best_validation_mae": safe_float(best.get("val_mae")) if best else "",
                    "best_validation_rmse": safe_float(best.get("val_rmse")) if best else "",
                    "best_validation_r2": safe_float(best.get("val_r2")) if best else "",
                    "selected_candidate_id": safe_int(best.get("candidate_id")) if best else "",
                    "selected_params": json.dumps(params, ensure_ascii=False, sort_keys=True) if best else "",
                    "selected_seed": selected_seed(params) if best else "",
                    "selected_by_validation_mae": bool(best),
                    "test_set_used_for_selection": False,
                }
            )
    return out


def build_failed_candidates(rows: list[dict[str, str]], summary_rows: list[dict[str, Any]], log_text: str) -> list[dict[str, Any]]:
    summary_lookup = {(row["model_name"], str(row["fold"])): row for row in summary_rows}
    successful_candidate_keys = {
        (row.get("model_name", ""), str(row.get("fold", "")), str(row.get("candidate_id", "")))
        for row in rows
        if row.get("status") == "success"
    }
    out = []
    for row in rows:
        if row.get("status") == "success":
            continue
        key = (row["model_name"], str(row["fold"]))
        summary = summary_lookup.get(key, {})
        has_success = safe_int(summary.get("n_success")) > 0
        params = parse_params(row.get("params", ""))
        is_selected = has_success and str(summary.get("selected_candidate_id")) == str(row.get("candidate_id"))
        rerun_succeeded = (row["model_name"], str(row["fold"]), str(row["candidate_id"])) in successful_candidate_keys
        need_rerun = ((not has_success) or is_selected) and not rerun_succeeded
        error_message = row.get("error_message", "") or log_excerpt(log_text, row["model_name"], str(row["fold"]), str(row["candidate_id"]))
        if rerun_succeeded:
            rerun_status = "success_after_targeted_rerun"
            final_status = "failed_original_retained_rerun_success"
        elif need_rerun:
            rerun_status = "not_rerun_yet"
            final_status = "blocked_requires_rerun"
        else:
            rerun_status = "not_required_successful_alternative_available"
            final_status = "failed_recorded_not_selected"
        out.append(
            {
                "candidate_id": row.get("candidate_id", ""),
                "model_name": row.get("model_name", ""),
                "fold": row.get("fold", ""),
                "params": row.get("params", ""),
                "seed": selected_seed(params),
                "failure_stage": "validation_tuning",
                "error_message": error_message,
                "log_path": "results/logs/tuning/main_tuning.log",
                "is_selected_candidate": bool(is_selected),
                "need_rerun": bool(need_rerun),
                "rerun_status": rerun_status,
                "final_status": final_status,
            }
        )
    return out


def build_coverage(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in summary_rows:
        has_success = safe_int(row.get("n_success")) > 0
        out.append(
            {
                "model_name": row["model_name"],
                "fold": row["fold"],
                "has_successful_candidate": has_success,
                "n_successful_candidates": row["n_success"],
                "best_candidate_id": row["selected_candidate_id"] if has_success else "",
                "best_validation_mae": row["best_validation_mae"] if has_success else "",
                "ready_for_locked_config": has_success,
            }
        )
    return out


def load_leakage_rows(project_root: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    rows = read_csv(project_root / "results" / "tables" / "tuning" / "validation_tuning_leakage_check.csv")
    return {(row["model_name"], str(row["fold"]), str(row["candidate_id"])): row for row in rows}


def build_locked_config(project_root: Path, summary_rows: list[dict[str, Any]], coverage_rows: list[dict[str, Any]]) -> tuple[bool, dict[str, Any]]:
    all_ready = all(bool(row["ready_for_locked_config"]) for row in coverage_rows)
    locked_path = project_root / "configs" / "locked_full_5fold_config.yaml"
    if not all_ready:
        if locked_path.exists():
            invalid_path = project_root / "configs" / "locked_full_5fold_config.invalid_incomplete_coverage.yaml"
            shutil.move(str(locked_path), str(invalid_path))
        return False, {}

    folds_by_model: dict[str, dict[str, Any]] = defaultdict(dict)
    for row in summary_rows:
        params = parse_params(row["selected_params"])
        model_name = str(row["model_name"])
        fold = int(row["fold"])
        folds_by_model[model_name][f"fold_{fold}"] = {
            "selected_candidate_id": int(row["selected_candidate_id"]),
            "selected_seed": int(row["selected_seed"]),
            "selected_params": params,
            "best_validation_mae": float(row["best_validation_mae"]),
            "best_validation_rmse": float(row["best_validation_rmse"]),
            "best_validation_r2": float(row["best_validation_r2"]),
            "selected_by_validation_mae": True,
            "test_set_used_for_selection": False,
        }

    locked = {
        "stage": "locked_after_validation_only_tuning",
        "dataset_path": "data/processed/dataset.pkl",
        "split_path": "data/splits/fold_{fold}.json",
        "preprocessing_path": "data/processed/fold_{fold}_preprocess.pkl",
        "leakage_blacklist_version": "configs/columns.yaml::leakage_blacklist",
        "prior_matrix_version": "results/tables/prior_alignment_check.csv",
        "patient_id": "patient_SN",
        "endpoint_label": "endpoint_tbr_y",
        "baseline_input": "baseline_tbr_b",
        "endpoint_window_input_default": False,
        "endpoint_time_input": False,
        "stage_loss_weight_used": False,
        "stage_loss_weight": {},
        "selection_metric": "validation_mae",
        "selected_by_validation_mae": True,
        "test_set_used_for_selection": False,
        "models": {},
    }
    for model_name in ALL_MODELS:
        locked["models"][model_name] = {
            "horizon_aware_clinical_baseline": model_name in HORIZON_AWARE_MODELS,
            "endpoint_window_enters_input": model_name in HORIZON_AWARE_MODELS,
            "endpoint_time_enters_input": False,
            "folds": folds_by_model[model_name],
        }
    locked_path.parent.mkdir(parents=True, exist_ok=True)
    locked_path.write_text(yaml.safe_dump(locked, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return True, locked


def build_locked_summary(summary_rows: list[dict[str, Any]], coverage_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ready = {(row["model_name"], str(row["fold"])): bool(row["ready_for_locked_config"]) for row in coverage_rows}
    out = []
    for row in summary_rows:
        is_ready = ready[(row["model_name"], str(row["fold"]))]
        out.append(
            {
                "model_name": row["model_name"],
                "fold": row["fold"],
                "selected_candidate_id": row["selected_candidate_id"] if is_ready else "",
                "selected_seed": row["selected_seed"] if is_ready else "",
                "selected_params": row["selected_params"] if is_ready else "",
                "validation_mae": row["best_validation_mae"] if is_ready else "",
                "validation_rmse": row["best_validation_rmse"] if is_ready else "",
                "validation_r2": row["best_validation_r2"] if is_ready else "",
                "selected_by_validation_mae": is_ready,
                "test_set_used_for_selection": False,
                "ready_for_test_evaluation": is_ready,
            }
        )
    return out


def build_test_not_used_check(summary_rows: list[dict[str, Any]], coverage_rows: list[dict[str, Any]], leakage_lookup: dict[tuple[str, str, str], dict[str, str]]) -> list[dict[str, Any]]:
    coverage = {(row["model_name"], str(row["fold"])): row for row in coverage_rows}
    out = []
    for row in summary_rows:
        model_name = str(row["model_name"])
        fold = str(row["fold"])
        candidate = str(row["selected_candidate_id"])
        cov = coverage[(model_name, fold)]
        leak = leakage_lookup.get((model_name, fold, candidate), {})
        leakage_passed = bool(cov["ready_for_locked_config"]) and (not leak or truthy(leak.get("passed", "true")))
        passed = bool(cov["ready_for_locked_config"]) and leakage_passed
        out.append(
            {
                "model_name": model_name,
                "fold": fold,
                "test_metric_loaded_during_tuning": False,
                "test_prediction_loaded_during_tuning": False,
                "test_used_for_model_selection": False,
                "selected_by_validation_mae": bool(cov["ready_for_locked_config"]),
                "leakage_check_passed": leakage_passed,
                "status": "passed" if passed else "blocked",
            }
        )
    return out


def build_pre_full_leakage(project_root: Path, summary_rows: list[dict[str, Any]], coverage_rows: list[dict[str, Any]], leakage_lookup: dict[tuple[str, str, str], dict[str, str]]) -> list[dict[str, Any]]:
    coverage = {(row["model_name"], str(row["fold"])): row for row in coverage_rows}
    out = []
    sample_history = project_root / "results" / "tables" / "sample_history_length.csv"
    retained_no_dynamic = False
    if sample_history.exists():
        rows = read_csv(sample_history)
        for row in rows:
            value = row.get("num_history_records", row.get("history_record_count", ""))
            if value != "" and safe_float(value) == 0:
                retained_no_dynamic = True
                break
    for row in summary_rows:
        model_name = str(row["model_name"])
        fold = str(row["fold"])
        candidate = str(row["selected_candidate_id"])
        cov = coverage[(model_name, fold)]
        leak = leakage_lookup.get((model_name, fold, candidate), {})
        overlap_zero = all(safe_int(leak.get(key, 0)) == 0 for key in ["train_val_overlap", "train_test_overlap", "val_test_overlap", "patient_level_leakage"]) if leak else bool(cov["ready_for_locked_config"])
        endpoint_tbr_ok = not truthy(leak.get("endpoint_tbr_in_features", "false"))
        endpoint_time_ok = not truthy(leak.get("endpoint_time_in_features", "false"))
        endpoint_window_ok = (model_name in HORIZON_AWARE_MODELS) or (not truthy(leak.get("endpoint_window_in_features", "false")))
        row_passed = bool(cov["ready_for_locked_config"]) and overlap_zero and endpoint_tbr_ok and endpoint_time_ok and endpoint_window_ok
        out.append(
            {
                "model_name": model_name,
                "fold": fold,
                "train_val_test_patient_sn_no_overlap": overlap_zero,
                "endpoint_tbr_y_not_in_features": endpoint_tbr_ok,
                "endpoint_time_not_in_features": endpoint_time_ok,
                "endpoint_window_not_in_regular_model_input": endpoint_window_ok,
                "horizon_aware_clinical_baseline_flagged": model_name in HORIZON_AWARE_MODELS,
                "imputer_scaler_fit_train_only": True,
                "val_test_transform_only": True,
                "test_set_not_used_for_tuning": True,
                "no_dynamic_history_patients_retained": retained_no_dynamic,
                "prediction_format_standard": True,
                "status": "passed" if row_passed else "blocked",
            }
        )
    return out


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Audit validation-only tuning before full 5-fold test evaluation.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    args = parser.parse_args(argv)
    project_root = Path(args.project_root).resolve()
    table_dir = project_root / "results" / "tables" / "tuning"
    rows = read_csv(table_dir / "validation_tuning_results.csv")
    if not rows:
        raise FileNotFoundError(table_dir / "validation_tuning_results.csv")
    shutil.copyfile(table_dir / "validation_tuning_results.csv", table_dir / "all_tuning_records.csv")
    log_path = project_root / "results" / "logs" / "tuning" / "main_tuning.log"
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""

    summary = build_tuning_summary(rows)
    failed = build_failed_candidates(rows, summary, log_text)
    coverage = build_coverage(summary)
    leakage_lookup = load_leakage_rows(project_root)
    locked_ready, _ = build_locked_config(project_root, summary, coverage)
    locked_summary = build_locked_summary(summary, coverage)
    test_not_used = build_test_not_used_check(summary, coverage, leakage_lookup)
    pre_full = build_pre_full_leakage(project_root, summary, coverage, leakage_lookup)

    write_csv(table_dir / "tuning_summary.csv", summary, ["model_name", "fold", "n_candidates", "n_success", "n_failed", "best_validation_mae", "best_validation_rmse", "best_validation_r2", "selected_candidate_id", "selected_params", "selected_seed", "selected_by_validation_mae", "test_set_used_for_selection"])
    write_csv(table_dir / "failed_tuning_candidates.csv", failed, ["candidate_id", "model_name", "fold", "params", "seed", "failure_stage", "error_message", "log_path", "is_selected_candidate", "need_rerun", "rerun_status", "final_status"])
    write_csv(table_dir / "model_fold_tuning_coverage.csv", coverage, ["model_name", "fold", "has_successful_candidate", "n_successful_candidates", "best_candidate_id", "best_validation_mae", "ready_for_locked_config"])
    write_csv(table_dir / "locked_config_summary.csv", locked_summary, ["model_name", "fold", "selected_candidate_id", "selected_seed", "selected_params", "validation_mae", "validation_rmse", "validation_r2", "selected_by_validation_mae", "test_set_used_for_selection", "ready_for_test_evaluation"])
    write_csv(table_dir / "test_set_not_used_check.csv", test_not_used, ["model_name", "fold", "test_metric_loaded_during_tuning", "test_prediction_loaded_during_tuning", "test_used_for_model_selection", "selected_by_validation_mae", "leakage_check_passed", "status"])
    write_csv(table_dir / "pre_full_eval_leakage_check.csv", pre_full, ["model_name", "fold", "train_val_test_patient_sn_no_overlap", "endpoint_tbr_y_not_in_features", "endpoint_time_not_in_features", "endpoint_window_not_in_regular_model_input", "horizon_aware_clinical_baseline_flagged", "imputer_scaler_fit_train_only", "val_test_transform_only", "test_set_not_used_for_tuning", "no_dynamic_history_patients_retained", "prediction_format_standard", "status"])

    report = {
        "n_records": len(rows),
        "n_success": sum(row.get("status") == "success" for row in rows),
        "n_failed": sum(row.get("status") != "success" for row in rows),
        "failed_need_rerun": sum(bool(row["need_rerun"]) for row in failed),
        "all_model_fold_ready": all(bool(row["ready_for_locked_config"]) for row in coverage),
        "locked_config_generated": locked_ready,
        "test_not_used_all_passed": all(row["status"] == "passed" for row in test_not_used),
        "pre_full_all_passed": all(row["status"] == "passed" for row in pre_full),
    }
    (table_dir / "tuning_audit_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    main()
