from __future__ import annotations

import argparse
import csv
import itertools
import json
import logging
import math
import pickle
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.build_tabular_features import build_tabular_for_all_folds, leakage_name_hit  # noqa: E402
from src.data.preprocessing import load_fold  # noqa: E402
from src.models.baselines.random_forest import build_random_forest_regressor  # noqa: E402
from src.models.baselines.xgboost_regressor import build_xgb_regressor  # noqa: E402
from src.training.evaluate import evaluate_tabular_regression  # noqa: E402
from src.training.metrics import regression_metrics  # noqa: E402


MODEL_ORDER = ["random_forest", "xgboost"]
PREDICTION_NAMES = {"random_forest": "random_forest", "xgboost": "xgboost"}
DISPLAY_NAMES = {"random_forest": "RandomForestRegressor", "xgboost": "XGBRegressor"}


DEFAULT_CONFIG: dict[str, Any] = {
    "smoke_test": {
        "fold": 0,
        "random_forest": {
            "n_estimators": 20,
            "max_depth": 3,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
        },
        "xgboost": {
            "n_estimators": 20,
            "max_depth": 2,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_lambda": 1.0,
            "reg_alpha": 0.0,
        },
    },
    "full_training": {
        "folds": [0, 1, 2, 3, 4],
        "selection_metric": "val_mae",
        "random_forest": {
            "n_estimators": [100, 300, 500],
            "max_depth": [3, 5, 8, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 3, 5],
            "max_features": ["sqrt", 0.5, 1.0],
        },
        "xgboost": {
            "n_estimators": [100, 300, 500],
            "max_depth": [2, 3, 5],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "subsample": [0.7, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.9, 1.0],
            "reg_lambda": [1.0, 3.0, 5.0],
            "reg_alpha": [0.0, 0.1, 1.0],
        },
    },
}


def deep_update(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    out = dict(left)
    for key, value in right.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_config(project_root: Path) -> dict[str, Any]:
    path = project_root / "configs" / "classical_ml.yaml"
    if not path.exists():
        return DEFAULT_CONFIG
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return deep_update(DEFAULT_CONFIG, loaded)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str], append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    mode = "a" if append else "w"
    with path.open(mode, newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if not append or not exists:
            writer.writeheader()
        writer.writerows(rows)


def setup_logger(project_root: Path, model_name: str, suffix: str) -> logging.Logger:
    log_path = project_root / "results" / "logs" / f"{model_name}_{suffix}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"classical.{model_name}.{suffix}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def reset_smoke_outputs(project_root: Path) -> None:
    for path in [
        project_root / "results" / "tables" / "classical_ml_smoke_test_metrics.csv",
        project_root / "results" / "tables" / "classical_ml_input_usage.csv",
        project_root / "results" / "tables" / "classical_ml_leakage_check.csv",
        project_root / "results" / "tables" / "classical_ml_smoke_feature_importance.csv",
    ]:
        if path.exists():
            path.unlink()
    for model_name in MODEL_ORDER:
        path = project_root / "results" / "predictions" / f"{PREDICTION_NAMES[model_name]}_fold0_smoke_predictions.csv"
        if path.exists():
            path.unlink()


def load_tabular(project_root: Path, fold: int, split: str) -> dict[str, Any]:
    path = project_root / "data" / "processed" / "tabular" / f"fold_{fold}_tabular_{split}.pkl"
    with path.open("rb") as handle:
        return pickle.load(handle)


def model_builder(model_name: str, params: dict[str, Any]):
    if model_name == "random_forest":
        return build_random_forest_regressor(params)
    if model_name == "xgboost":
        return build_xgb_regressor(params)
    raise KeyError(model_name)


def split_leakage(project_root: Path, fold: int) -> dict[str, int]:
    payload = load_fold(project_root, fold)
    train = set(map(str, payload["train_patient_ids"]))
    val = set(map(str, payload["val_patient_ids"]))
    test = set(map(str, payload["test_patient_ids"]))
    return {
        "train_val_overlap": len(train & val),
        "train_test_overlap": len(train & test),
        "val_test_overlap": len(val & test),
    }


def feature_leakage_flags(feature_names: list[str]) -> dict[str, Any]:
    hits = [(name, leakage_name_hit(name)) for name in feature_names]
    hits = [(name, token) for name, token in hits if token]
    endpoint_window = any("endpoint_window" in name.lower() for name in feature_names)
    endpoint_time = any("endpoint_time" in name.lower() for name in feature_names)
    endpoint_tbr = any(token in name.lower() for name in feature_names for token in ["endpoint_tbr_y", "胸主动脉tbr值", "胸主动脉tbr", "tbr值"])
    return {
        "endpoint_tbr_in_features": bool(endpoint_tbr),
        "endpoint_window_in_features": bool(endpoint_window),
        "endpoint_time_in_features": bool(endpoint_time),
        "leakage_blacklist_passed": len(hits) == 0,
        "leakage_hits": "; ".join(f"{name}->{token}" for name, token in hits[:20]),
    }


def prediction_nan(rows: list[dict[str, Any]]) -> bool:
    return any(not math.isfinite(float(row["y_pred"])) for row in rows)


def target_nan(payload: dict[str, Any]) -> bool:
    y = np.asarray(payload["y"], dtype=np.float64).reshape(-1)
    return bool(np.any(~np.isfinite(y)))


def save_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    write_csv(path, rows, ["patient_id", "endpoint_window", "y_true", "y_pred", "absolute_error"])


def top_feature_importance(model: Any, feature_names: list[str], top_k: int = 20) -> list[dict[str, Any]]:
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return []
    values = np.asarray(importances, dtype=np.float64).reshape(-1)
    order = np.argsort(values)[::-1][:top_k]
    return [
        {
            "rank": int(rank + 1),
            "feature_name": feature_names[int(idx)],
            "importance": float(values[int(idx)]),
        }
        for rank, idx in enumerate(order)
    ]


def log_shapes(logger: logging.Logger, train: dict[str, Any], val: dict[str, Any], test: dict[str, Any]) -> None:
    logger.info("train_patient_count=%d val_patient_count=%d test_patient_count=%d", len(train["patient_id"]), len(val["patient_id"]), len(test["patient_id"]))
    logger.info("X_train_shape=%s", tuple(train["X"].shape))
    logger.info("X_val_shape=%s", tuple(val["X"].shape))
    logger.info("X_test_shape=%s", tuple(test["X"].shape))
    logger.info("y_train_shape=%s y_val_shape=%s y_test_shape=%s", tuple(train["y"].shape), tuple(val["y"].shape), tuple(test["y"].shape))
    logger.info("feature_count=%d", len(train["feature_names"]))


def train_and_evaluate(
    project_root: Path,
    model_name: str,
    fold: int,
    params: dict[str, Any],
    mode: str,
    logger: logging.Logger,
) -> dict[str, Any]:
    train = load_tabular(project_root, fold, "train")
    val = load_tabular(project_root, fold, "val")
    test = load_tabular(project_root, fold, "test")
    feature_names = train["feature_names"]
    leak_flags = feature_leakage_flags(feature_names)
    overlap = split_leakage(project_root, fold)
    patient_leakage = int(any(value > 0 for value in overlap.values()))
    if not leak_flags["leakage_blacklist_passed"] or patient_leakage:
        raise RuntimeError(f"Leakage check failed: {leak_flags}, overlap={overlap}")

    model = model_builder(model_name, params)
    logger.info("model_name=%s", model_name)
    logger.info("model_class=%s", DISPLAY_NAMES[model_name])
    logger.info("mode=%s fold=%d", mode, fold)
    logger.info("params=%s", json.dumps(params, ensure_ascii=False, sort_keys=True))
    log_shapes(logger, train, val, test)
    logger.info("endpoint_tbr_in_features=%s", str(leak_flags["endpoint_tbr_in_features"]).lower())
    logger.info("endpoint_window_in_features=%s", str(leak_flags["endpoint_window_in_features"]).lower())
    logger.info("endpoint_time_in_features=%s", str(leak_flags["endpoint_time_in_features"]).lower())
    logger.info("patient_level_leakage=%d", patient_leakage)
    logger.info("imputer_fit_on=%s scaler_used=%s", train["preprocessing"]["imputer_fit_on"], train["preprocessing"]["scaler_used"])

    model.fit(train["X"], train["y"])
    train_rows = evaluate_tabular_regression(model, train)
    val_rows = evaluate_tabular_regression(model, val)
    train_metrics = regression_metrics(train_rows)
    val_metrics = regression_metrics(val_rows)
    train_loss_is_nan = not math.isfinite(float(train_metrics["mse"]))
    pred_is_nan = prediction_nan(val_rows)
    y_nan = target_nan(train) or target_nan(val) or target_nan(test)
    logger.info("train_mse=%.8f val_mse=%.8f val_mae=%.8f val_rmse=%.8f", train_metrics["mse"], val_metrics["mse"], val_metrics["mae"], val_metrics["rmse"])
    logger.info("loss_is_nan=%s", str(train_loss_is_nan).lower())
    logger.info("prediction_is_nan=%s", str(pred_is_nan).lower())
    logger.info("target_is_nan=%s", str(y_nan).lower())

    importance_rows = top_feature_importance(model, feature_names, top_k=20)
    for row in importance_rows:
        logger.info("top_feature_importance rank=%d feature=%s importance=%.10f", row["rank"], row["feature_name"], row["importance"])
    return {
        "model": model,
        "train": train,
        "val": val,
        "test": test,
        "val_rows": val_rows,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "feature_names": feature_names,
        "leak_flags": leak_flags,
        "overlap": overlap,
        "patient_leakage": patient_leakage,
        "train_loss_is_nan": train_loss_is_nan,
        "prediction_is_nan": pred_is_nan,
        "importance_rows": importance_rows,
    }


def smoke_one(project_root: Path, model_name: str, fold: int, config: dict[str, Any]) -> dict[str, Any]:
    logger = setup_logger(project_root, model_name, "smoke_test")
    try:
        params = dict(config["smoke_test"][model_name])
        result = train_and_evaluate(project_root, model_name, fold, params, "smoke", logger)
        pred_path = project_root / "results" / "predictions" / f"{PREDICTION_NAMES[model_name]}_fold{fold}_smoke_predictions.csv"
        save_predictions(pred_path, result["val_rows"])
        logger.info("prediction_file_saved_path=%s", pred_path)
        logger.info("y_true/y_pred/absolute_error columns present=true")

        metrics = result["val_metrics"]
        write_csv(
            project_root / "results" / "tables" / "classical_ml_smoke_test_metrics.csv",
            [
                {
                    "model_name": model_name,
                    "fold": fold,
                    "split": "val",
                    "n": int(metrics["n"]),
                    "mae": metrics["mae"],
                    "mse": metrics["mse"],
                    "rmse": metrics["rmse"],
                    "loss_is_nan": result["train_loss_is_nan"],
                    "prediction_is_nan": result["prediction_is_nan"],
                    "prediction_path": str(pred_path),
                    "status": "success",
                    "error_message": "",
                }
            ],
            ["model_name", "fold", "split", "n", "mae", "mse", "rmse", "loss_is_nan", "prediction_is_nan", "prediction_path", "status", "error_message"],
            append=True,
        )
        write_csv(
            project_root / "results" / "tables" / "classical_ml_leakage_check.csv",
            [
                {
                    "model_name": model_name,
                    "fold": fold,
                    **result["leak_flags"],
                    **result["overlap"],
                    "patient_level_leakage": result["patient_leakage"],
                    "passed": (not result["leak_flags"]["endpoint_tbr_in_features"]) and result["leak_flags"]["leakage_blacklist_passed"] and result["patient_leakage"] == 0,
                }
            ],
            ["model_name", "fold", "endpoint_tbr_in_features", "endpoint_window_in_features", "endpoint_time_in_features", "leakage_blacklist_passed", "leakage_hits", "train_val_overlap", "train_test_overlap", "val_test_overlap", "patient_level_leakage", "passed"],
            append=True,
        )
        feature_names = result["feature_names"]
        group_counts = {group: sum(1 for name in feature_names if name.startswith(f"{group}::")) for group in ["static", "dynamic", "treatment", "history"]}
        write_csv(
            project_root / "results" / "tables" / "classical_ml_input_usage.csv",
            [
                {
                    "model_name": model_name,
                    "fold": fold,
                    "static_features": group_counts["static"],
                    "baseline_tbr_b": "used",
                    "treatment_features": group_counts["treatment"],
                    "dynamic_summary_features": group_counts["dynamic"],
                    "history_length_features": group_counts["history"],
                    "missingness_features": sum(1 for name in feature_names if "missing" in name.lower() or "observed_count" in name.lower()),
                    "endpoint_tbr_y": "target_only",
                    "endpoint_window": "metadata_only_not_X",
                    "endpoint_time": "metadata_only_not_X",
                    "feature_count": len(feature_names),
                    "prediction_path": str(pred_path),
                    "top20_feature_importance_path": str(project_root / "results" / "tables" / "classical_ml_smoke_feature_importance.csv"),
                }
            ],
            ["model_name", "fold", "static_features", "baseline_tbr_b", "treatment_features", "dynamic_summary_features", "history_length_features", "missingness_features", "endpoint_tbr_y", "endpoint_window", "endpoint_time", "feature_count", "prediction_path", "top20_feature_importance_path"],
            append=True,
        )
        importance_rows = [{"model_name": model_name, "fold": fold, **row} for row in result["importance_rows"]]
        write_csv(
            project_root / "results" / "tables" / "classical_ml_smoke_feature_importance.csv",
            importance_rows,
            ["model_name", "fold", "rank", "feature_name", "importance"],
            append=True,
        )
        return {"model_name": model_name, "status": "success", "prediction_path": str(pred_path)}
    except Exception as exc:
        logger.error("classical smoke failed: %s", exc)
        logger.error(traceback.format_exc())
        write_csv(
            project_root / "results" / "tables" / "classical_ml_smoke_test_metrics.csv",
            [
                {
                    "model_name": model_name,
                    "fold": fold,
                    "split": "val",
                    "n": 0,
                    "mae": math.nan,
                    "mse": math.nan,
                    "rmse": math.nan,
                    "loss_is_nan": True,
                    "prediction_is_nan": True,
                    "prediction_path": "",
                    "status": "failed",
                    "error_message": str(exc),
                }
            ],
            ["model_name", "fold", "split", "n", "mae", "mse", "rmse", "loss_is_nan", "prediction_is_nan", "prediction_path", "status", "error_message"],
            append=True,
        )
        return {"model_name": model_name, "status": "failed", "error_message": str(exc)}


def param_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(grid.keys())
    values = [grid[key] for key in keys]
    return [dict(zip(keys, combo, strict=False)) for combo in itertools.product(*values)]


def full_one(project_root: Path, model_name: str, fold: int, config: dict[str, Any]) -> dict[str, Any]:
    logger = setup_logger(project_root, model_name, f"fold{fold}_full")
    best: dict[str, Any] | None = None
    for params in param_grid(config["full_training"][model_name]):
        result = train_and_evaluate(project_root, model_name, fold, params, "full_grid_search", logger)
        metric = float(result["val_metrics"]["mae"])
        if best is None or metric < best["val_mae"]:
            best = {"params": params, "val_mae": metric}
    if best is None:
        raise RuntimeError(f"No parameter set evaluated for {model_name} fold {fold}")
    final = train_and_evaluate(project_root, model_name, fold, best["params"], "full_selected", logger)
    test_rows = evaluate_tabular_regression(final["model"], final["test"])
    test_metrics = regression_metrics(test_rows)
    pred_path = project_root / "results" / "predictions" / f"{PREDICTION_NAMES[model_name]}_fold{fold}_test_predictions.csv"
    save_predictions(pred_path, test_rows)
    write_csv(
        project_root / "results" / "tables" / "classical_ml_5fold_metrics.csv",
        [{"model_name": model_name, "fold": fold, "split": "test", "n": int(test_metrics["n"]), "mae": test_metrics["mae"], "mse": test_metrics["mse"], "rmse": test_metrics["rmse"], "prediction_path": str(pred_path)}],
        ["model_name", "fold", "split", "n", "mae", "mse", "rmse", "prediction_path"],
        append=True,
    )
    write_csv(
        project_root / "results" / "tables" / "classical_ml_best_params.csv",
        [{"model_name": model_name, "fold": fold, "selection_metric": "val_mae", "best_val_mae": best["val_mae"], "best_params": json.dumps(best["params"], ensure_ascii=False, sort_keys=True)}],
        ["model_name", "fold", "selection_metric", "best_val_mae", "best_params"],
        append=True,
    )
    importance_rows = [{"model_name": model_name, "fold": fold, **row} for row in final["importance_rows"]]
    write_csv(
        project_root / "results" / "tables" / "classical_ml_5fold_feature_importance.csv",
        importance_rows,
        ["model_name", "fold", "rank", "feature_name", "importance"],
        append=True,
    )
    return {"model_name": model_name, "fold": fold, "status": "success", "prediction_path": str(pred_path)}


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Run classical ML baseline smoke tests or full 5-fold training.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--mode", choices=["smoke", "full"], required=True)
    parser.add_argument("--baseline", choices=MODEL_ORDER + ["all"], default="all")
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--build-tabular", action="store_true")
    parser.add_argument("--reset-smoke-outputs", action="store_true")
    args = parser.parse_args(argv)

    project_root = Path(args.project_root).resolve()
    config = load_config(project_root)
    if args.build_tabular:
        build_tabular_for_all_folds(project_root, folds=5)
    if args.mode == "smoke" and args.reset_smoke_outputs:
        reset_smoke_outputs(project_root)
    models = MODEL_ORDER if args.baseline == "all" else [args.baseline]
    results: list[dict[str, Any]] = []
    if args.mode == "smoke":
        fold = int(args.fold if args.fold is not None else config["smoke_test"]["fold"])
        for model_name in models:
            results.append(smoke_one(project_root, model_name, fold, config))
    else:
        folds = [int(args.fold)] if args.fold is not None else [int(fold) for fold in config["full_training"]["folds"]]
        for fold in folds:
            for model_name in models:
                results.append(full_one(project_root, model_name, fold, config))
    print(json.dumps({"results": results}, ensure_ascii=False, indent=2))
    return {"results": results}


if __name__ == "__main__":
    main()
