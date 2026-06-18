from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from statsmodels.regression.mixed_linear_model import MixedLM
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import apply_preprocess, ids_to_indices, load_dataset, load_fold  # noqa: E402
from src.data.prior_alignment import build_aligned_prior_matrix  # noqa: E402
from src.models.baselines.random_forest import build_random_forest_regressor  # noqa: E402
from src.models.baselines.xgboost_regressor import build_xgb_regressor  # noqa: E402
from src.training.validation_tuning import (  # noqa: E402
    ALL_MODELS,
    HORIZON_AWARE_MODELS,
    TensorFoldDataset,
    build_torch_model,
    clinical_feature_indices,
    collate,
    load_tabular,
    set_seed,
)
from src.models.modules.losses import EndpointMSELoss  # noqa: E402
from src.training.metrics import regression_metrics  # noqa: E402


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str], append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a" if append else "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if not append or not exists:
            writer.writeheader()
        writer.writerows(rows)


def safe_metric_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    metrics = regression_metrics(rows)
    y_true = np.asarray([float(row["y_true"]) for row in rows], dtype=np.float64)
    y_pred = np.asarray([float(row["y_pred"]) for row in rows], dtype=np.float64)
    metrics["r2"] = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else math.nan
    return metrics


def update_status(project_root: Path, payload: dict[str, Any]) -> None:
    path = project_root / "results" / "logs" / "full_5fold" / "current_status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"updated_at": time.strftime("%Y-%m-%d %H:%M:%S"), **payload}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def split_overlap(project_root: Path, fold: int) -> dict[str, int]:
    payload = load_fold(project_root, fold)
    train = set(map(str, payload["train_patient_ids"]))
    val = set(map(str, payload["val_patient_ids"]))
    test = set(map(str, payload["test_patient_ids"]))
    return {
        "train_val_overlap": len(train & val),
        "train_test_overlap": len(train & test),
        "val_test_overlap": len(val & test),
        "patient_level_leakage": int(bool((train & val) or (train & test) or (val & test))),
    }


def feature_flags(model_name: str, feature_names: list[str]) -> dict[str, bool]:
    lowered = [str(name).lower() for name in feature_names]
    endpoint_tbr = any(("endpoint_tbr_y" in name) or ("胸主动脉tbr值" in name) or ("tbr值" in name) for name in lowered)
    endpoint_time = any("endpoint_time" in name for name in lowered)
    endpoint_window = any("endpoint_window" in name for name in lowered) and model_name not in HORIZON_AWARE_MODELS
    return {
        "endpoint_tbr_in_features": bool(endpoint_tbr),
        "endpoint_time_in_features": bool(endpoint_time),
        "endpoint_window_in_features": bool(endpoint_window),
    }


def prediction_rows(payload: dict[str, Any], y_pred: np.ndarray, fold: int) -> list[dict[str, Any]]:
    y_true = np.asarray(payload["y"], dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    return [
        {
            "patient_id": str(pid),
            "fold": fold,
            "endpoint_window": int(window),
            "y_true": float(true),
            "y_pred": float(pred),
            "absolute_error": float(abs(true - pred)),
        }
        for pid, window, true, pred in zip(payload["patient_id"], payload["endpoint_window"], y_true, y_pred, strict=False)
    ]


def add_endpoint_window_onehot(train: dict[str, Any], test: dict[str, Any], x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    tr = np.asarray(train["endpoint_window"]).reshape(-1, 1)
    te = np.asarray(test["endpoint_window"]).reshape(-1, 1)
    return np.concatenate([x_train, enc.fit_transform(tr)], axis=1), np.concatenate([x_test, enc.transform(te)], axis=1)


def fit_predict_tabular(project_root: Path, model_name: str, fold: int, params: dict[str, Any]) -> tuple[list[dict[str, Any]], str, list[str]]:
    train = load_tabular(project_root, fold, "train")
    test = load_tabular(project_root, fold, "test")
    feature_names = [str(x) for x in train["feature_names"]]
    if model_name in {"baseline_tbr_only", "clinical_core", "clinical_horizon_aware"}:
        idx = clinical_feature_indices(feature_names, model_name)
        x_train = train["X"][:, idx]
        x_test = test["X"][:, idx]
        used_names = [feature_names[i] for i in idx]
        if model_name == "clinical_horizon_aware":
            x_train, x_test = add_endpoint_window_onehot(train, test, x_train, x_test)
            used_names += ["endpoint_window_onehot_6", "endpoint_window_onehot_12", "endpoint_window_onehot_18", "endpoint_window_onehot_24"]
        estimator = LinearRegression()
    elif model_name == "linear_regression":
        x_train, x_test, used_names = train["X"], test["X"], feature_names
        estimator = LinearRegression()
    elif model_name == "ridge":
        x_train, x_test, used_names = train["X"], test["X"], feature_names
        estimator = Ridge(**params)
    elif model_name == "elasticnet":
        x_train, x_test, used_names = train["X"], test["X"], feature_names
        estimator = ElasticNet(max_iter=10000, random_state=int(params.get("seed", 20260605)), **{k: v for k, v in params.items() if k != "seed"})
    elif model_name == "random_forest":
        x_train, x_test, used_names = train["X"], test["X"], feature_names
        estimator = build_random_forest_regressor(params)
    elif model_name == "xgboost":
        x_train, x_test, used_names = train["X"], test["X"], feature_names
        estimator = build_xgb_regressor(params)
    elif model_name == "linear_mixed_effects":
        idx = clinical_feature_indices(feature_names, "clinical_core")[:6]
        x_train = train["X"][:, idx]
        x_test = test["X"][:, idx]
        cols = [f"x{i}" for i in range(x_train.shape[1])]
        df = pd.DataFrame(x_train, columns=cols)
        df["y"] = train["y"]
        df["group"] = np.asarray(train["endpoint_window"]).astype(str)
        result = None
        last_error = None
        for formula in (["y ~ " + " + ".join(cols), "y ~ x0"] if cols else ["y ~ 1"]):
            model = MixedLM.from_formula(formula, groups="group", data=df)
            for method in ["lbfgs", "powell", "nm"]:
                try:
                    result = model.fit(reml=False, method=method, maxiter=300, disp=False)
                    break
                except Exception as exc:
                    last_error = exc
            if result is not None:
                break
            try:
                result = model.fit_regularized(alpha=0.1)
                break
            except Exception as exc:
                last_error = exc
        if result is None:
            raise RuntimeError(f"MixedLM failed in full evaluation: {last_error}")
        y_pred = np.asarray(result.predict(pd.DataFrame(x_test, columns=cols)), dtype=np.float64)
        return prediction_rows(test, y_pred, fold), "statsmodels.MixedLM", [feature_names[i] for i in idx]
    else:
        raise KeyError(model_name)
    estimator.fit(x_train, train["y"])
    y_pred = np.asarray(estimator.predict(x_test), dtype=np.float64)
    return prediction_rows(test, y_pred, fold), type(estimator).__name__, used_names


@torch.no_grad()
def evaluate_torch_rows(model, loader, device, prior_matrix, fold: int) -> list[dict[str, Any]]:
    model.eval()
    rows = []
    for batch in loader:
        tb = {key: value.to(device) for key, value in batch["tensors"].items()}
        pred = model(tb, prior_matrix=prior_matrix)
        y_true = tb["endpoint_tbr_y"].detach().cpu().numpy().reshape(-1)
        y_pred = pred.detach().cpu().numpy().reshape(-1)
        for pid, window, true, out in zip(batch["patient_id"], batch["endpoint_window"].cpu().numpy().tolist(), y_true, y_pred, strict=False):
            rows.append({"patient_id": pid, "fold": fold, "endpoint_window": int(window), "y_true": float(true), "y_pred": float(out), "absolute_error": float(abs(true - out))})
    return rows


@torch.no_grad()
def extract_kg_latent_outputs(model, loader, device, prior_matrix, fold: int) -> dict[str, Any]:
    model.eval()
    rows = []
    for batch in loader:
        tb = {key: value.to(device) for key, value in batch["tensors"].items()}
        static_context = torch.cat([tb["static_features"], tb["baseline_tbr_b"]], dim=-1)
        static_hidden = model.static_encoder(static_context)
        dynamic_hidden = model.dynamic_graph(tb["dynamic_features"], tb["dynamic_mask"], prior_matrix=prior_matrix)
        sequence = torch.cat([dynamic_hidden, tb["treatment_features"], tb["delta_time"].unsqueeze(-1)], dim=-1)
        sequence_mask = (tb["dynamic_mask"].sum(dim=-1) > 0) | (tb["treatment_features"].sum(dim=-1) > 0)
        dynamic_summary = model.short_delay_update(sequence, sequence_mask)
        fused = model.fusion(static_hidden, dynamic_summary)
        pred = model.head(fused).detach().cpu().numpy().reshape(-1)
        latent_score = fused.mean(dim=1).detach().cpu().numpy().reshape(-1)
        short_score = static_hidden.norm(dim=1).detach().cpu().numpy().reshape(-1)
        delayed_score = dynamic_summary.norm(dim=1).detach().cpu().numpy().reshape(-1)
        for pid, window, p, ls, ss, ds in zip(batch["patient_id"], batch["endpoint_window"].cpu().numpy().tolist(), pred, latent_score, short_score, delayed_score, strict=False):
            rows.append({"patient_id": pid, "fold": fold, "endpoint_window": int(window), "y_pred": float(p), "latent_state_score": float(ls), "short_contribution_score": float(ss), "delayed_contribution_score": float(ds)})
    return {
        "patient_latent_rows": rows,
        "prior_matrix_used": prior_matrix.detach().cpu().numpy(),
        "dynamic_graph_projection_weight": model.dynamic_graph.proj.weight.detach().cpu().numpy(),
        "learned_relation_weights_available": False,
        "note": "Current KGLatentNet implementation uses prior-guided dynamic projection and does not expose a separate learned adjacency matrix.",
    }


def fit_predict_torch(project_root: Path, model_name: str, fold: int, params: dict[str, Any], seed: int) -> tuple[list[dict[str, Any]], str, str]:
    set_seed(seed)
    dataset = load_dataset(project_root)
    fold_payload = load_fold(project_root, fold)
    with (project_root / "data" / "processed" / f"fold_{fold}_preprocess.pkl").open("rb") as handle:
        preprocess = pickle.load(handle)
    train_arrays = apply_preprocess(dataset, preprocess, ids_to_indices(dataset, fold_payload["train_patient_ids"]))
    val_arrays = apply_preprocess(dataset, preprocess, ids_to_indices(dataset, fold_payload["val_patient_ids"]))
    test_arrays = apply_preprocess(dataset, preprocess, ids_to_indices(dataset, fold_payload["test_patient_ids"]))
    train_loader = DataLoader(TensorFoldDataset(train_arrays), batch_size=int(params.get("batch_size", 32)), shuffle=True, collate_fn=collate)
    val_loader = DataLoader(TensorFoldDataset(val_arrays), batch_size=64, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(TensorFoldDataset(test_arrays), batch_size=64, shuffle=False, collate_fn=collate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    static_dim = int(dataset["static_features"].shape[1])
    dynamic_dim = int(dataset["dynamic_features"].shape[2])
    treatment_dim = int(dataset["treatment_features"].shape[2])
    model = build_torch_model(model_name, project_root, static_dim, dynamic_dim, treatment_dim, params).to(device)
    prior_np, _, prior_checks = build_aligned_prior_matrix(project_root, dataset["feature_names"]["dynamic_features"])
    if not all(bool(row["passed"]) for row in prior_checks):
        raise RuntimeError("Prior alignment failed in full evaluation.")
    prior_matrix = torch.tensor(prior_np, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(params.get("learning_rate", 1e-3)), weight_decay=float(params.get("weight_decay", 0.0)))
    criterion = EndpointMSELoss()
    max_epochs = int(params.get("max_epochs", 20 if model_name == "kg_latentnet" or model_name in {"grud", "time_aware_lstm", "retain"} else 8))
    patience = int(params.get("patience", 5 if max_epochs > 8 else 3))
    best_state = None
    best_val_mae = float("inf")
    bad_epochs = 0
    for _epoch in range(1, max_epochs + 1):
        model.train()
        for batch in train_loader:
            tb = {key: value.to(device) for key, value in batch["tensors"].items()}
            optimizer.zero_grad(set_to_none=True)
            pred = model(tb, prior_matrix=prior_matrix)
            loss = criterion(pred, tb["endpoint_tbr_y"])
            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError(f"NaN/Inf loss for {model_name} fold {fold}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(params.get("gradient_clip", 5.0)))
            optimizer.step()
        val_rows = evaluate_torch_rows(model, val_loader, device, prior_matrix, fold)
        val_mae = safe_metric_rows(val_rows)["mae"]
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    ckpt_path = project_root / "results" / "checkpoints" / "full_5fold" / f"{model_name}_fold{fold}.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_name": model_name, "fold": fold, "params": params, "seed": seed, "model_state_dict": best_state or model.state_dict()}, ckpt_path)
    if model_name == "kg_latentnet":
        latent = extract_kg_latent_outputs(model, test_loader, device, prior_matrix, fold)
        latent_path = project_root / "results" / "latent" / "full_5fold" / f"kg_latentnet_fold{fold}_latent_states.pkl"
        latent_path.parent.mkdir(parents=True, exist_ok=True)
        with latent_path.open("wb") as handle:
            pickle.dump(latent, handle)
        contributions_path = project_root / "results" / "latent" / "full_5fold" / f"kg_latentnet_fold{fold}_contributions.pkl"
        with contributions_path.open("wb") as handle:
            pickle.dump(
                {
                    "fold": fold,
                    "patient_contribution_rows": latent["patient_latent_rows"],
                    "contribution_fields": ["short_contribution_score", "delayed_contribution_score"],
                    "note": "Scores are model-derived summary norms from the static and dynamic branches.",
                },
                handle,
            )
        relation_path = project_root / "results" / "latent" / "full_5fold" / f"kg_latentnet_fold{fold}_relation_weights.pkl"
        with relation_path.open("wb") as handle:
            pickle.dump(
                {
                    "fold": fold,
                    "learned_relation_weights_available": False,
                    "prior_matrix_used": latent["prior_matrix_used"],
                    "dynamic_graph_projection_weight": latent["dynamic_graph_projection_weight"],
                    "note": "Current KGLatentNet uses a prior-guided projection and does not expose a separate learned adjacency/attention matrix.",
                },
                handle,
            )
    rows = evaluate_torch_rows(model, test_loader, device, prior_matrix, fold)
    return rows, type(model).__name__, str(ckpt_path)


def load_locked(project_root: Path, config_path: str) -> dict[str, Any]:
    path = project_root / config_path
    if not path.exists():
        raise FileNotFoundError(path)
    locked = yaml.safe_load(path.read_text(encoding="utf-8"))
    if locked.get("test_set_used_for_selection") is not False:
        raise RuntimeError("Refusing full evaluation: locked config does not assert test_set_used_for_selection=false.")
    return locked


def checks_pass(project_root: Path) -> None:
    for rel in ["results/tables/tuning/locked_config_summary.csv", "results/tables/tuning/test_set_not_used_check.csv", "results/tables/tuning/pre_full_eval_leakage_check.csv"]:
        path = project_root / rel
        if not path.exists():
            raise FileNotFoundError(path)
        rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
        if rel.endswith("test_set_not_used_check.csv") or rel.endswith("pre_full_eval_leakage_check.csv"):
            bad = [row for row in rows if row.get("status") != "passed"]
            if bad:
                raise RuntimeError(f"Pre-full check failed: {rel} has {len(bad)} blocked rows.")
        if rel.endswith("locked_config_summary.csv"):
            bad = [row for row in rows if str(row.get("ready_for_test_evaluation", "")).lower() not in {"true", "1"}]
            if bad:
                raise RuntimeError(f"Locked config summary has {len(bad)} rows not ready.")


def run_one(project_root: Path, locked: dict[str, Any], model_name: str, fold: int) -> dict[str, Any]:
    update_status(project_root, {"stage": "full_5fold_test_evaluation", "model": model_name, "fold": fold, "status": "running"})
    info = locked["models"][model_name]["folds"][f"fold_{fold}"]
    params = dict(info["selected_params"] or {})
    seed = int(info["selected_seed"])
    start = time.time()
    if model_name in {"baseline_tbr_only", "clinical_core", "clinical_horizon_aware", "linear_regression", "ridge", "elasticnet", "linear_mixed_effects", "random_forest", "xgboost"}:
        rows, model_class, feature_names = fit_predict_tabular(project_root, model_name, fold, params)
        ckpt_path = ""
    else:
        rows, model_class, ckpt_path = fit_predict_torch(project_root, model_name, fold, params, seed)
        feature_names = ["static_features", "dynamic_features", "dynamic_mask", "delta_time", "treatment_features", "baseline_tbr_b"]
    metrics = safe_metric_rows(rows)
    pred_nan = any(not math.isfinite(float(row["y_pred"])) for row in rows)
    pred_path = project_root / "results" / "predictions" / "full_5fold" / f"{model_name}_fold{fold}_predictions.csv"
    write_csv(pred_path, rows, ["patient_id", "fold", "endpoint_window", "y_true", "y_pred", "absolute_error"])
    overlap = split_overlap(project_root, fold)
    flags = feature_flags(model_name, feature_names)
    loss_is_nan = not math.isfinite(float(metrics["mse"]))
    passed = (not pred_nan) and (not loss_is_nan) and overlap["patient_level_leakage"] == 0 and not any(flags.values())
    return {
        "model_name": model_name,
        "fold": fold,
        "selected_params": json.dumps(params, ensure_ascii=False, sort_keys=True),
        "selected_seed": seed,
        "train_patient_count": len(load_fold(project_root, fold)["train_patient_ids"]),
        "val_patient_count": len(load_fold(project_root, fold)["val_patient_ids"]),
        "test_patient_count": len(load_fold(project_root, fold)["test_patient_ids"]),
        "best_validation_mae": info["best_validation_mae"],
        "test_mae": metrics["mae"],
        "test_mse": metrics["mse"],
        "test_rmse": metrics["rmse"],
        "test_r2": metrics["r2"],
        **flags,
        **overlap,
        "loss_is_nan": loss_is_nan,
        "prediction_is_nan": pred_nan,
        "model_class": model_class,
        "checkpoint_path": ckpt_path,
        "prediction_path": str(pred_path),
        "runtime_sec": round(time.time() - start, 3),
        "status": "success" if passed else "failed",
        "error_message": "" if passed else "full evaluation checks failed",
    }


def reset_outputs(project_root: Path) -> None:
    for rel in [
        "results/tables/full_5fold/all_models_training_status.csv",
        "results/tables/full_5fold/all_models_leakage_check.csv",
        "results/tables/full_5fold/all_models_test_metrics.csv",
    ]:
        path = project_root / rel
        if path.exists():
            path.unlink()


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Run locked full 5-fold test evaluation.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--config", default="configs/locked_full_5fold_config.yaml")
    parser.add_argument("--models", default="all")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args(argv)
    project_root = Path(args.project_root).resolve()
    locked = load_locked(project_root, args.config)
    checks_pass(project_root)
    if args.reset:
        reset_outputs(project_root)
    model_list = ALL_MODELS if args.models == "all" else [x.strip() for x in args.models.split(",") if x.strip()]
    metric_fields = ["model_name", "fold", "selected_params", "selected_seed", "train_patient_count", "val_patient_count", "test_patient_count", "best_validation_mae", "test_mae", "test_mse", "test_rmse", "test_r2", "model_class", "checkpoint_path", "prediction_path", "runtime_sec", "status", "error_message"]
    leak_fields = ["model_name", "fold", "endpoint_tbr_in_features", "endpoint_time_in_features", "endpoint_window_in_features", "train_val_overlap", "train_test_overlap", "val_test_overlap", "patient_level_leakage", "loss_is_nan", "prediction_is_nan", "status"]
    rows = []
    for model_name in model_list:
        for fold in locked["folds"] if "folds" in locked else [0, 1, 2, 3, 4]:
            try:
                result = run_one(project_root, locked, model_name, int(fold))
            except Exception as exc:
                result = {"model_name": model_name, "fold": int(fold), "selected_params": "", "selected_seed": "", "train_patient_count": "", "val_patient_count": "", "test_patient_count": "", "best_validation_mae": "", "test_mae": math.nan, "test_mse": math.nan, "test_rmse": math.nan, "test_r2": math.nan, "endpoint_tbr_in_features": True, "endpoint_time_in_features": True, "endpoint_window_in_features": True, "train_val_overlap": "", "train_test_overlap": "", "val_test_overlap": "", "patient_level_leakage": "", "loss_is_nan": True, "prediction_is_nan": True, "model_class": "", "checkpoint_path": "", "prediction_path": "", "runtime_sec": 0, "status": "failed", "error_message": str(exc) + "\n" + traceback.format_exc()}
            rows.append(result)
            write_csv(project_root / "results" / "tables" / "full_5fold" / "all_models_training_status.csv", [result], metric_fields, append=True)
            write_csv(project_root / "results" / "tables" / "full_5fold" / "all_models_leakage_check.csv", [result], leak_fields, append=True)
            write_csv(project_root / "results" / "tables" / "full_5fold" / "all_models_test_metrics.csv", [result], metric_fields, append=True)
    update_status(project_root, {"stage": "full_5fold_test_evaluation", "status": "completed", "n_rows": len(rows)})
    print(json.dumps({"n_rows": len(rows), "n_success": sum(row["status"] == "success" for row in rows)}, ensure_ascii=False, indent=2))
    return {"rows": rows}


if __name__ == "__main__":
    main()
