#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/KG_LatentNet_Project}"
ENV_DIR="${ENV_DIR:-/root/kg_latentnet_env}"
PYTHON="${ENV_DIR}/bin/python"
if [ ! -x "${PYTHON}" ]; then
  PYTHON="$(command -v python3)"
fi

cd "${PROJECT_ROOT}"
"${PYTHON}" - <<'PY'
from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path

root = Path.cwd()
out_dir = root / "results" / "tables" / "full_5fold"
status_path = out_dir / "all_models_training_status.csv"
if not status_path.exists():
    raise SystemExit(f"missing status table: {status_path}")

rows = list(csv.DictReader(status_path.open(encoding="utf-8-sig")))
latest: dict[tuple[str, int], dict[str, str]] = {}
for source_idx, row in enumerate(rows, start=2):
    model = row.get("model_name", "")
    fold_text = row.get("fold", "")
    if not model or fold_text == "":
        continue
    try:
        fold = int(fold_text)
    except ValueError:
        continue
    row = dict(row)
    row["source_row"] = str(source_idx)
    latest[(model, fold)] = row

latest_rows = sorted(latest.values(), key=lambda r: (r.get("model_name", ""), int(r.get("fold", 0))))

def to_float(value: str | None) -> float:
    if value is None or value == "":
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan

def fmt(value: float) -> str:
    if math.isnan(value):
        return ""
    return f"{value:.10g}"

def mean(values: list[float]) -> float:
    values = [v for v in values if not math.isnan(v)]
    return statistics.mean(values) if values else math.nan

def stdev(values: list[float]) -> float:
    values = [v for v in values if not math.isnan(v)]
    return statistics.stdev(values) if len(values) > 1 else 0.0 if values else math.nan

fold_fields = [
    "model_name", "fold", "selected_params", "selected_seed", "train_patient_count", "val_patient_count",
    "test_patient_count", "best_validation_mae", "test_mae", "test_mse", "test_rmse", "test_r2",
    "model_class", "checkpoint_path", "prediction_path", "runtime_sec", "status", "error_message",
    "source_row", "prediction_exists", "checkpoint_exists",
]
fold_out = []
completion_out = []
for row in latest_rows:
    pred_path = row.get("prediction_path", "")
    ckpt_path = row.get("checkpoint_path", "")
    pred_exists = bool(pred_path) and Path(pred_path).exists()
    ckpt_exists = bool(ckpt_path) and Path(ckpt_path).exists()
    enriched = {field: row.get(field, "") for field in fold_fields}
    enriched["prediction_exists"] = str(pred_exists)
    enriched["checkpoint_exists"] = str(ckpt_exists)
    fold_out.append(enriched)
    completion_out.append({
        "model_name": row.get("model_name", ""),
        "fold": row.get("fold", ""),
        "latest_status": row.get("status", ""),
        "prediction_exists": str(pred_exists),
        "checkpoint_exists": str(ckpt_exists),
        "source_row": row.get("source_row", ""),
        "ok_for_final_results": str(row.get("status") == "success" and pred_exists),
    })

models = sorted({row.get("model_name", "") for row in latest_rows})
summary_fields = [
    "model_name", "n_folds", "n_success", "folds", "mean_test_mae", "std_test_mae", "mean_test_rmse",
    "std_test_rmse", "mean_test_r2", "std_test_r2", "mean_runtime_sec", "total_runtime_sec",
    "prediction_files_present", "checkpoint_files_present", "status",
]
summary_out = []
for model in models:
    model_rows = [r for r in latest_rows if r.get("model_name") == model]
    success_rows = [r for r in model_rows if r.get("status") == "success" and r.get("prediction_path") and Path(r.get("prediction_path", "")).exists()]
    maes = [to_float(r.get("test_mae")) for r in success_rows]
    rmses = [to_float(r.get("test_rmse")) for r in success_rows]
    r2s = [to_float(r.get("test_r2")) for r in success_rows]
    runtimes = [to_float(r.get("runtime_sec")) for r in success_rows]
    pred_present = sum(1 for r in model_rows if r.get("prediction_path") and Path(r.get("prediction_path", "")).exists())
    ckpt_present = sum(1 for r in model_rows if r.get("checkpoint_path") and Path(r.get("checkpoint_path", "")).exists())
    summary_out.append({
        "model_name": model,
        "n_folds": len(model_rows),
        "n_success": len(success_rows),
        "folds": ";".join(str(int(r.get("fold", 0))) for r in model_rows),
        "mean_test_mae": fmt(mean(maes)),
        "std_test_mae": fmt(stdev(maes)),
        "mean_test_rmse": fmt(mean(rmses)),
        "std_test_rmse": fmt(stdev(rmses)),
        "mean_test_r2": fmt(mean(r2s)),
        "std_test_r2": fmt(stdev(r2s)),
        "mean_runtime_sec": fmt(mean(runtimes)),
        "total_runtime_sec": fmt(sum(v for v in runtimes if not math.isnan(v))),
        "prediction_files_present": pred_present,
        "checkpoint_files_present": ckpt_present,
        "status": "success" if len(model_rows) == 5 and len(success_rows) == 5 else "incomplete",
    })

def write_csv(path: Path, records: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

write_csv(out_dir / "all_models_5fold_fold_results.csv", fold_out, fold_fields)
write_csv(out_dir / "all_models_5fold_results.csv", summary_out, summary_fields)
write_csv(out_dir / "full_evaluation_completion_check.csv", completion_out, ["model_name", "fold", "latest_status", "prediction_exists", "checkpoint_exists", "source_row", "ok_for_final_results"])

n_latest_success = sum(1 for r in fold_out if r["status"] == "success" and r["prediction_exists"] == "True")
status = {
    "stage": "full_5fold_results_collection",
    "status": "completed" if n_latest_success == len(fold_out) else "incomplete",
    "n_latest_rows": len(fold_out),
    "n_latest_success_with_predictions": n_latest_success,
    "model_count": len(summary_out),
    "output": "results/tables/full_5fold/all_models_5fold_results.csv",
}
(root / "results" / "logs" / "full_5fold" / "current_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(status, ensure_ascii=False, indent=2))
PY