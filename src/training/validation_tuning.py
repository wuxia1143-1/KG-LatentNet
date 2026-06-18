from __future__ import annotations

import argparse
import csv
import itertools
import json
import logging
import math
import pickle
import random
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
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import apply_preprocess, ids_to_indices, load_dataset, load_fold  # noqa: E402
from src.data.prior_alignment import build_aligned_prior_matrix  # noqa: E402
from src.models.baselines.official_adapters import OFFICIAL_BASELINE_REGISTRY  # noqa: E402
from src.models.baselines.random_forest import build_random_forest_regressor  # noqa: E402
from src.models.baselines.xgboost_regressor import build_xgb_regressor  # noqa: E402
from src.models.kg_latentnet import KGLatentNet  # noqa: E402
from src.models.modules.losses import EndpointMSELoss  # noqa: E402


CLINICAL_MODELS = [
    "baseline_tbr_only",
    "clinical_core",
    "clinical_horizon_aware",
    "linear_regression",
    "ridge",
    "elasticnet",
    "linear_mixed_effects",
]
CLASSICAL_MODELS = ["random_forest", "xgboost"]
TIME_SERIES_MODELS = ["grud", "time_aware_lstm", "retain"]
OFFICIAL_MODELS = ["hyperimts", "trans", "tgnn4i", "dhgas", "kedgn", "graphcare"]
PROPOSED_MODELS = ["kg_latentnet"]
ALL_MODELS = CLINICAL_MODELS + CLASSICAL_MODELS + TIME_SERIES_MODELS + OFFICIAL_MODELS + PROPOSED_MODELS
HORIZON_AWARE_MODELS = {"clinical_horizon_aware", "linear_mixed_effects"}


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str], append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a" if append else "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if not append or not exists:
            writer.writeheader()
        writer.writerows(rows)


def setup_logger(project_root: Path) -> logging.Logger:
    path = project_root / "results" / "logs" / "tuning" / "main_tuning.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("validation_tuning")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_tabular(project_root: Path, fold: int, split: str) -> dict[str, Any]:
    path = project_root / "data" / "processed" / "tabular" / f"fold_{fold}_tabular_{split}.pkl"
    with path.open("rb") as handle:
        return pickle.load(handle)


def params_product(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(grid)
    return [dict(zip(keys, values, strict=False)) for values in itertools.product(*[grid[key] for key in keys])]


def mae_rmse_r2(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    error = y_pred - y_true
    mse = float(np.mean(error**2))
    return {
        "mae": float(np.mean(np.abs(error))),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
    }


def prediction_rows(payload: dict[str, Any], y_pred: np.ndarray) -> list[dict[str, Any]]:
    y_true = np.asarray(payload["y"], dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    return [
        {
            "patient_id": str(pid),
            "endpoint_window": int(window),
            "y_true": float(true),
            "y_pred": float(pred),
            "absolute_error": float(abs(true - pred)),
        }
        for pid, window, true, pred in zip(payload["patient_id"], payload["endpoint_window"], y_true, y_pred, strict=False)
    ]


def split_overlap(project_root: Path, fold: int) -> dict[str, int]:
    payload = load_fold(project_root, fold)
    train = set(map(str, payload["train_patient_ids"]))
    val = set(map(str, payload["val_patient_ids"]))
    test = set(map(str, payload["test_patient_ids"]))
    return {
        "train_val_overlap": len(train & val),
        "train_test_overlap": len(train & test),
        "val_test_overlap": len(val & test),
    }


def check_feature_names(model_name: str, feature_names: list[str]) -> dict[str, Any]:
    lower = [str(name).lower() for name in feature_names]
    endpoint_tbr = any(("endpoint_tbr_y" in name) or ("胸主动脉tbr值" in name) or ("tbr值" in name) for name in lower)
    endpoint_time = any("endpoint_time" in name for name in lower)
    endpoint_window = any("endpoint_window" in name for name in lower)
    allowed_window = model_name in HORIZON_AWARE_MODELS
    return {
        "endpoint_tbr_in_features": bool(endpoint_tbr),
        "endpoint_time_in_features": bool(endpoint_time),
        "endpoint_window_in_features": bool(endpoint_window and not allowed_window),
        "horizon_aware_clinical_baseline": bool(model_name in HORIZON_AWARE_MODELS),
    }


def status_update(project_root: Path, payload: dict[str, Any]) -> None:
    path = project_root / "results" / "logs" / "tuning" / "current_status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"updated_at": time.strftime("%Y-%m-%d %H:%M:%S"), **payload}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def clinical_feature_indices(feature_names: list[str], mode: str) -> list[int]:
    if mode == "baseline_tbr_only":
        wanted = ["baseline_tbr_b"]
    else:
        tokens = ["baseline_tbr_b", "年龄", "age", "性别", "sex", "stage", "分期", "tnm"]
        wanted = tokens
    selected = []
    for idx, name in enumerate(feature_names):
        lower = name.lower()
        if any(token.lower() in lower for token in wanted):
            selected.append(idx)
    if not selected:
        selected = [idx for idx, name in enumerate(feature_names) if "baseline_tbr_b" in str(name)]
    return selected


def add_endpoint_window_onehot(train: dict[str, Any], val: dict[str, Any], x_train: np.ndarray, x_val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    tr = np.asarray(train["endpoint_window"]).reshape(-1, 1)
    va = np.asarray(val["endpoint_window"]).reshape(-1, 1)
    return np.concatenate([x_train, enc.fit_transform(tr)], axis=1), np.concatenate([x_val, enc.transform(va)], axis=1)


def fit_predict_tabular(model_name: str, fold: int, params: dict[str, Any], project_root: Path) -> dict[str, Any]:
    train = load_tabular(project_root, fold, "train")
    val = load_tabular(project_root, fold, "val")
    feature_names = [str(x) for x in train["feature_names"]]
    if model_name in {"baseline_tbr_only", "clinical_core", "clinical_horizon_aware"}:
        idx = clinical_feature_indices(feature_names, model_name)
        x_train = train["X"][:, idx]
        x_val = val["X"][:, idx]
        names = [feature_names[i] for i in idx]
        if model_name == "clinical_horizon_aware":
            x_train, x_val = add_endpoint_window_onehot(train, val, x_train, x_val)
            names = names + ["endpoint_window_onehot_6", "endpoint_window_onehot_12", "endpoint_window_onehot_18", "endpoint_window_onehot_24"]
        estimator = LinearRegression()
    elif model_name == "linear_regression":
        x_train, x_val, names = train["X"], val["X"], feature_names
        estimator = LinearRegression()
    elif model_name == "ridge":
        x_train, x_val, names = train["X"], val["X"], feature_names
        estimator = Ridge(**params)
    elif model_name == "elasticnet":
        x_train, x_val, names = train["X"], val["X"], feature_names
        estimator = ElasticNet(max_iter=10000, random_state=20260605, **params)
    elif model_name == "random_forest":
        x_train, x_val, names = train["X"], val["X"], feature_names
        estimator = build_random_forest_regressor(params)
    elif model_name == "xgboost":
        x_train, x_val, names = train["X"], val["X"], feature_names
        estimator = build_xgb_regressor(params)
    elif model_name == "linear_mixed_effects":
        idx = clinical_feature_indices(feature_names, "clinical_core")[:6]
        x_train = train["X"][:, idx]
        x_val = val["X"][:, idx]
        cols = [f"x{i}" for i in range(x_train.shape[1])]
        df = pd.DataFrame(x_train, columns=cols)
        df["y"] = train["y"]
        df["group"] = np.asarray(train["endpoint_window"]).astype(str)
        formulas = ["y ~ " + " + ".join(cols), "y ~ x0"] if cols else ["y ~ 1"]
        result = None
        last_error: Exception | None = None
        for formula in formulas:
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
            raise RuntimeError(f"MixedLM failed after fallback optimizers: {last_error}")
        val_df = pd.DataFrame(x_val, columns=cols)
        y_pred = np.asarray(result.predict(val_df), dtype=np.float64)
        return {"y_pred": y_pred, "feature_names": [feature_names[i] for i in idx], "model_class": "statsmodels.MixedLM"}
    else:
        raise KeyError(model_name)
    estimator.fit(x_train, train["y"])
    return {"y_pred": np.asarray(estimator.predict(x_val), dtype=np.float64), "feature_names": names, "model_class": type(estimator).__name__}


class TensorFoldDataset(Dataset):
    def __init__(self, arrays: dict[str, Any]) -> None:
        self.arrays = arrays
        self.n = len(arrays["patient_id"])

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict[str, Any]:
        keys = ["static_features", "dynamic_features", "dynamic_mask", "delta_time", "treatment_features", "baseline_tbr_b", "endpoint_tbr_y"]
        return {
            "tensors": {key: torch.tensor(self.arrays[key][idx], dtype=torch.float32) for key in keys},
            "patient_id": str(self.arrays["patient_id"][idx]),
            "endpoint_window": int(self.arrays["endpoint_window"][idx]),
        }


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    keys = batch[0]["tensors"]
    return {
        "tensors": {key: torch.stack([item["tensors"][key] for item in batch], dim=0) for key in keys},
        "patient_id": [item["patient_id"] for item in batch],
        "endpoint_window": torch.tensor([item["endpoint_window"] for item in batch], dtype=torch.long),
    }


class GRUDRegressor(nn.Module):
    def __init__(self, static_dim: int, dynamic_dim: int, treatment_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.gru = nn.GRU(dynamic_dim * 2 + treatment_dim + 1, hidden_dim, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_dim + static_dim + 1, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, batch: dict[str, torch.Tensor], prior_matrix: torch.Tensor | None = None) -> torch.Tensor:
        del prior_matrix
        delta = batch["delta_time"].unsqueeze(-1)
        seq = torch.cat([batch["dynamic_features"] * batch["dynamic_mask"], batch["dynamic_mask"], batch["treatment_features"], delta], dim=-1)
        out, _ = self.gru(seq)
        mask = ((batch["dynamic_mask"].sum(-1) > 0) | (batch["treatment_features"].abs().sum(-1) > 0)).long()
        lengths = mask.sum(1).clamp_min(1)
        idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, out.shape[-1])
        last = out.gather(1, idx).squeeze(1)
        return self.head(torch.cat([last, batch["static_features"], batch["baseline_tbr_b"]], dim=-1))


class TimeAwareLSTMRegressor(nn.Module):
    def __init__(self, static_dim: int, dynamic_dim: int, treatment_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.lstm = nn.LSTM(dynamic_dim + treatment_dim + 1, hidden_dim, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_dim + static_dim + 1, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, batch: dict[str, torch.Tensor], prior_matrix: torch.Tensor | None = None) -> torch.Tensor:
        del prior_matrix
        seq = torch.cat([batch["dynamic_features"] * batch["dynamic_mask"], batch["treatment_features"], batch["delta_time"].unsqueeze(-1)], dim=-1)
        out, _ = self.lstm(seq)
        weights = torch.softmax(-torch.cumsum(batch["delta_time"], dim=1), dim=1).unsqueeze(-1)
        summary = (out * weights).sum(dim=1)
        return self.head(torch.cat([summary, batch["static_features"], batch["baseline_tbr_b"]], dim=-1))


class RetainRegressor(nn.Module):
    def __init__(self, static_dim: int, dynamic_dim: int, treatment_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.embed = nn.Linear(dynamic_dim + treatment_dim + 1, hidden_dim)
        self.alpha = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.beta = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.alpha_head = nn.Linear(hidden_dim, 1)
        self.beta_head = nn.Linear(hidden_dim, hidden_dim)
        self.head = nn.Sequential(nn.Linear(hidden_dim + static_dim + 1, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, batch: dict[str, torch.Tensor], prior_matrix: torch.Tensor | None = None) -> torch.Tensor:
        del prior_matrix
        seq = torch.cat([batch["dynamic_features"] * batch["dynamic_mask"], batch["treatment_features"], batch["delta_time"].unsqueeze(-1)], dim=-1)
        emb = torch.tanh(self.embed(seq))
        a, _ = self.alpha(emb)
        b, _ = self.beta(emb)
        weights = torch.softmax(self.alpha_head(a).squeeze(-1), dim=1).unsqueeze(-1)
        context = (weights * torch.tanh(self.beta_head(b)) * emb).sum(dim=1)
        return self.head(torch.cat([context, batch["static_features"], batch["baseline_tbr_b"]], dim=-1))


def build_torch_model(model_name: str, project_root: Path, static_dim: int, dynamic_dim: int, treatment_dim: int, params: dict[str, Any]) -> nn.Module:
    hidden_dim = int(params.get("hidden_dim", 64))
    if model_name == "kg_latentnet":
        return KGLatentNet(static_dim, dynamic_dim, treatment_dim, hidden_dim=hidden_dim)
    if model_name == "grud":
        return GRUDRegressor(static_dim, dynamic_dim, treatment_dim, hidden_dim=hidden_dim)
    if model_name == "time_aware_lstm":
        return TimeAwareLSTMRegressor(static_dim, dynamic_dim, treatment_dim, hidden_dim=hidden_dim)
    if model_name == "retain":
        return RetainRegressor(static_dim, dynamic_dim, treatment_dim, hidden_dim=hidden_dim)
    if model_name in OFFICIAL_BASELINE_REGISTRY:
        return OFFICIAL_BASELINE_REGISTRY[model_name](project_root, static_dim, dynamic_dim, treatment_dim, hidden_dim=hidden_dim)
    raise KeyError(model_name)


@torch.no_grad()
def eval_torch(model: nn.Module, loader: DataLoader, device: torch.device, prior_matrix: torch.Tensor | None) -> tuple[float, list[dict[str, Any]], np.ndarray]:
    criterion = EndpointMSELoss()
    model.eval()
    losses = []
    rows = []
    preds = []
    for batch in loader:
        tb = {key: value.to(device) for key, value in batch["tensors"].items()}
        pred = model(tb, prior_matrix=prior_matrix)
        loss = criterion(pred, tb["endpoint_tbr_y"])
        losses.append(float(loss.item()))
        y_true = tb["endpoint_tbr_y"].detach().cpu().numpy().reshape(-1)
        y_pred = pred.detach().cpu().numpy().reshape(-1)
        preds.extend(y_pred.tolist())
        for pid, window, true, out in zip(batch["patient_id"], batch["endpoint_window"].numpy().tolist(), y_true, y_pred, strict=False):
            rows.append({"patient_id": pid, "endpoint_window": int(window), "y_true": float(true), "y_pred": float(out), "absolute_error": float(abs(true - out))})
    return float(np.mean(losses)) if losses else float("nan"), rows, np.asarray(preds, dtype=np.float64)


def fit_predict_torch(model_name: str, fold: int, params: dict[str, Any], project_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    dataset = load_dataset(project_root)
    fold_payload = load_fold(project_root, fold)
    with (project_root / "data" / "processed" / f"fold_{fold}_preprocess.pkl").open("rb") as handle:
        preprocess = pickle.load(handle)
    train_arrays = apply_preprocess(dataset, preprocess, ids_to_indices(dataset, fold_payload["train_patient_ids"]))
    val_arrays = apply_preprocess(dataset, preprocess, ids_to_indices(dataset, fold_payload["val_patient_ids"]))
    train_loader = DataLoader(TensorFoldDataset(train_arrays), batch_size=int(config["batch_size"]), shuffle=True, collate_fn=collate)
    val_loader = DataLoader(TensorFoldDataset(val_arrays), batch_size=int(config["eval_batch_size"]), shuffle=False, collate_fn=collate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    static_dim = int(dataset["static_features"].shape[1])
    dynamic_dim = int(dataset["dynamic_features"].shape[2])
    treatment_dim = int(dataset["treatment_features"].shape[2])
    model = build_torch_model(model_name, project_root, static_dim, dynamic_dim, treatment_dim, params).to(device)
    prior_np, _, prior_checks = build_aligned_prior_matrix(project_root, dataset["feature_names"]["dynamic_features"])
    if not all(bool(row["passed"]) for row in prior_checks):
        raise RuntimeError("Prior alignment failed during tuning.")
    prior_matrix = torch.tensor(prior_np, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(params.get("learning_rate", 1e-3)), weight_decay=float(params.get("weight_decay", 0.0)))
    criterion = EndpointMSELoss()
    max_epochs = int(config["max_epochs"]["official_models" if model_name in OFFICIAL_MODELS else "torch_models"])
    patience = int(config["patience"]["official_models" if model_name in OFFICIAL_MODELS else "torch_models"])
    best_rows: list[dict[str, Any]] = []
    best_pred = np.zeros(len(val_arrays["patient_id"]), dtype=np.float64)
    best_mae = float("inf")
    best_epoch = 0
    bad_epochs = 0
    best_state = None
    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch in train_loader:
            tb = {key: value.to(device) for key, value in batch["tensors"].items()}
            optimizer.zero_grad(set_to_none=True)
            pred = model(tb, prior_matrix=prior_matrix)
            loss = criterion(pred, tb["endpoint_tbr_y"])
            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError(f"NaN/Inf loss for {model_name} fold {fold} epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(params.get("gradient_clip", 5.0)))
            optimizer.step()
        _, val_rows, val_pred = eval_torch(model, val_loader, device, prior_matrix)
        metric = mae_rmse_r2(np.asarray([r["y_true"] for r in val_rows]), val_pred)
        if metric["mae"] < best_mae:
            best_mae = metric["mae"]
            best_epoch = epoch
            best_rows = val_rows
            best_pred = val_pred
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= patience:
            break
    ckpt_path = project_root / "results" / "checkpoints" / "tuning" / f"{model_name}_fold{fold}_best_validation.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_name": model_name, "fold": fold, "params": params, "best_epoch": best_epoch, "model_state_dict": best_state}, ckpt_path)
    return {"y_pred": best_pred, "prediction_rows": best_rows, "best_epoch": best_epoch, "model_class": type(model).__name__, "checkpoint_path": str(ckpt_path)}


def candidate_grid(model_name: str) -> list[dict[str, Any]]:
    grids: dict[str, dict[str, list[Any]]] = {
        "baseline_tbr_only": {"candidate": ["fixed"]},
        "clinical_core": {"candidate": ["fixed"]},
        "clinical_horizon_aware": {"candidate": ["fixed"]},
        "linear_regression": {"candidate": ["fixed"]},
        "ridge": {"alpha": [0.1, 1.0, 10.0]},
        "elasticnet": {"alpha": [0.001, 0.01, 0.1], "l1_ratio": [0.2, 0.5, 0.8]},
        "linear_mixed_effects": {"candidate": ["fixed"]},
        "random_forest": {"n_estimators": [100, 300], "max_depth": [3, 5, None], "min_samples_leaf": [1, 3], "max_features": ["sqrt", 0.5]},
        "xgboost": {"n_estimators": [100, 300], "max_depth": [2, 3], "learning_rate": [0.03, 0.05], "subsample": [0.9], "colsample_bytree": [0.9], "reg_lambda": [1.0, 3.0], "reg_alpha": [0.0, 0.1]},
        "grud": {"hidden_dim": [32, 64], "learning_rate": [0.001]},
        "time_aware_lstm": {"hidden_dim": [32, 64], "learning_rate": [0.001]},
        "retain": {"hidden_dim": [32, 64], "learning_rate": [0.001]},
        "kg_latentnet": {"hidden_dim": [32, 64], "learning_rate": [0.001, 0.0003], "weight_decay": [0.0, 0.0001]},
    }
    if model_name in OFFICIAL_MODELS:
        return params_product({"hidden_dim": [32], "learning_rate": [0.001]})
    return params_product(grids[model_name])


def run_candidate(project_root: Path, model_name: str, fold: int, candidate_id: int, params: dict[str, Any], config: dict[str, Any], logger: logging.Logger) -> dict[str, Any]:
    status_update(project_root, {"stage": "validation_only_tuning", "model": model_name, "fold": fold, "candidate_id": candidate_id, "status": "running"})
    overlap = split_overlap(project_root, fold)
    patient_leakage = int(any(overlap.values()))
    if patient_leakage:
        raise RuntimeError(f"Patient-level leakage in fold {fold}: {overlap}")
    start = time.time()
    if model_name in CLINICAL_MODELS + CLASSICAL_MODELS:
        result = fit_predict_tabular(model_name, fold, params, project_root)
        train = load_tabular(project_root, fold, "train")
        val = load_tabular(project_root, fold, "val")
        y_true = np.asarray(val["y"], dtype=np.float64)
        rows = prediction_rows(val, result["y_pred"])
        feature_names = result["feature_names"]
    else:
        result = fit_predict_torch(model_name, fold, params, project_root, config)
        rows = result["prediction_rows"]
        y_true = np.asarray([row["y_true"] for row in rows], dtype=np.float64)
        feature_names = ["static_features", "dynamic_features", "dynamic_mask", "delta_time", "treatment_features", "baseline_tbr_b"]
        train = {"patient_id": load_fold(project_root, fold)["train_patient_ids"]}
        val = {"patient_id": load_fold(project_root, fold)["val_patient_ids"]}
    metric = mae_rmse_r2(y_true, np.asarray([row["y_pred"] for row in rows], dtype=np.float64))
    leak = check_feature_names(model_name, feature_names)
    prediction_is_nan = any(not math.isfinite(float(row["y_pred"])) for row in rows)
    pred_path = project_root / "results" / "predictions" / "tuning" / f"{model_name}_fold{fold}_candidate{candidate_id}_val_predictions.csv"
    write_csv(pred_path, rows, ["patient_id", "endpoint_window", "y_true", "y_pred", "absolute_error"])
    row = {
        "model_name": model_name,
        "fold": fold,
        "candidate_id": candidate_id,
        "params": json.dumps(params, ensure_ascii=False, sort_keys=True),
        "status": "success",
        "val_n": len(rows),
        "train_n": len(train["patient_id"]),
        "val_mae": metric["mae"],
        "val_mse": metric["mse"],
        "val_rmse": metric["rmse"],
        "val_r2": metric["r2"],
        "best_epoch": result.get("best_epoch", ""),
        "model_class": result.get("model_class", ""),
        "prediction_path": str(pred_path),
        "checkpoint_path": result.get("checkpoint_path", ""),
        "runtime_sec": round(time.time() - start, 3),
        "loss_is_nan": False,
        "prediction_is_nan": prediction_is_nan,
        "test_set_used_for_selection": False,
        "error_message": "",
    }
    leak_row = {"model_name": model_name, "fold": fold, "candidate_id": candidate_id, **overlap, "patient_level_leakage": patient_leakage, **leak, "passed": (not patient_leakage) and (not leak["endpoint_tbr_in_features"]) and (not leak["endpoint_time_in_features"]) and (not leak["endpoint_window_in_features"]) and (not prediction_is_nan)}
    logger.info("SUCCESS model=%s fold=%s candidate=%s val_mae=%.6f val_rmse=%.6f", model_name, fold, candidate_id, metric["mae"], metric["rmse"])
    return {"result": row, "leak": leak_row}


def reset_outputs(project_root: Path) -> None:
    for rel in [
        "results/tables/tuning/validation_tuning_results.csv",
        "results/tables/tuning/validation_tuning_leakage_check.csv",
        "results/tables/tuning/validation_tuning_best_by_model.csv",
        "results/tables/tuning/validation_tuning_status.csv",
    ]:
        path = project_root / rel
        if path.exists():
            path.unlink()
    locked = project_root / "configs" / "locked_full_5fold_config.yaml"
    if locked.exists():
        locked.unlink()


def write_locked_config(project_root: Path, result_rows: list[dict[str, Any]], config: dict[str, Any]) -> None:
    successful = [row for row in result_rows if row["status"] == "success"]
    locked: dict[str, Any] = {
        "locked_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stage": "locked_after_validation_only_tuning",
        "selection_metric": "mean_validation_mae_across_available_folds",
        "test_set_used_for_selection": False,
        "folds": config["folds"],
        "models": {},
    }
    best_rows = []
    for model_name in ALL_MODELS:
        rows = [row for row in successful if row["model_name"] == model_name]
        if not rows:
            locked["models"][model_name] = {"status": "failed", "best_params": None}
            continue
        by_candidate: dict[int, list[dict[str, Any]]] = {}
        for row in rows:
            by_candidate.setdefault(int(row["candidate_id"]), []).append(row)
        ranked = sorted(by_candidate.items(), key=lambda item: (np.mean([float(r["val_mae"]) for r in item[1]]), -len(item[1])))
        candidate_id, chosen_rows = ranked[0]
        params = json.loads(chosen_rows[0]["params"])
        mean_mae = float(np.mean([float(r["val_mae"]) for r in chosen_rows]))
        locked["models"][model_name] = {
            "status": "locked",
            "candidate_id": int(candidate_id),
            "best_params": params,
            "mean_val_mae": mean_mae,
            "folds_completed": sorted(int(r["fold"]) for r in chosen_rows),
            "horizon_aware_clinical_baseline": model_name in HORIZON_AWARE_MODELS,
        }
        best_rows.append({"model_name": model_name, "candidate_id": candidate_id, "best_params": json.dumps(params, ensure_ascii=False, sort_keys=True), "mean_val_mae": mean_mae, "folds_completed": ";".join(str(int(r["fold"])) for r in chosen_rows), "test_set_used_for_selection": False})
    (project_root / "configs").mkdir(parents=True, exist_ok=True)
    (project_root / "configs" / "locked_full_5fold_config.yaml").write_text(yaml.safe_dump(locked, allow_unicode=True, sort_keys=False), encoding="utf-8")
    write_csv(project_root / "results" / "tables" / "tuning" / "validation_tuning_best_by_model.csv", best_rows, ["model_name", "candidate_id", "best_params", "mean_val_mae", "folds_completed", "test_set_used_for_selection"])


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Validation-only tuning; no test evaluation is performed.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--config", default="configs/validation_tuning.yaml")
    parser.add_argument("--models", default="all")
    parser.add_argument("--folds", default="")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--skip-locked-config", action="store_true")
    args = parser.parse_args(argv)
    project_root = Path(args.project_root).resolve()
    logger = setup_logger(project_root)
    config = yaml.safe_load((project_root / args.config).read_text(encoding="utf-8"))
    if args.folds:
        config["folds"] = [int(x) for x in args.folds.split(",") if x.strip()]
    models = ALL_MODELS if args.models == "all" else [x.strip() for x in args.models.split(",") if x.strip()]
    if args.reset:
        reset_outputs(project_root)
    set_seed(int(config["seed"]))
    logger.info("Starting validation-only tuning. models=%s folds=%s test_set_used_for_selection=false", models, config["folds"])
    result_rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    metric_fields = ["model_name", "fold", "candidate_id", "params", "status", "val_n", "train_n", "val_mae", "val_mse", "val_rmse", "val_r2", "best_epoch", "model_class", "prediction_path", "checkpoint_path", "runtime_sec", "loss_is_nan", "prediction_is_nan", "test_set_used_for_selection", "error_message"]
    leak_fields = ["model_name", "fold", "candidate_id", "train_val_overlap", "train_test_overlap", "val_test_overlap", "patient_level_leakage", "endpoint_tbr_in_features", "endpoint_time_in_features", "endpoint_window_in_features", "horizon_aware_clinical_baseline", "passed"]
    for model_name in models:
        candidates = candidate_grid(model_name)
        for fold in config["folds"]:
            for candidate_id, params in enumerate(candidates):
                try:
                    out = run_candidate(project_root, model_name, int(fold), candidate_id, params, config, logger)
                    result_rows.append(out["result"])
                    write_csv(project_root / "results" / "tables" / "tuning" / "validation_tuning_results.csv", [out["result"]], metric_fields, append=True)
                    write_csv(project_root / "results" / "tables" / "tuning" / "validation_tuning_leakage_check.csv", [out["leak"]], leak_fields, append=True)
                except Exception as exc:
                    logger.error("FAILED model=%s fold=%s candidate=%s error=%s", model_name, fold, candidate_id, exc)
                    logger.error(traceback.format_exc())
                    fail = {"model_name": model_name, "fold": int(fold), "candidate_id": candidate_id, "params": json.dumps(params, ensure_ascii=False, sort_keys=True), "status": "failed", "val_n": 0, "train_n": 0, "val_mae": math.nan, "val_mse": math.nan, "val_rmse": math.nan, "val_r2": math.nan, "best_epoch": "", "model_class": "", "prediction_path": "", "checkpoint_path": "", "runtime_sec": 0, "loss_is_nan": True, "prediction_is_nan": True, "test_set_used_for_selection": False, "error_message": str(exc)}
                    result_rows.append(fail)
                    write_csv(project_root / "results" / "tables" / "tuning" / "validation_tuning_results.csv", [fail], metric_fields, append=True)
                status_rows.append({"model_name": model_name, "fold": int(fold), "candidate_id": candidate_id, "status": result_rows[-1]["status"], "test_set_used_for_selection": False})
                write_csv(project_root / "results" / "tables" / "tuning" / "validation_tuning_status.csv", [status_rows[-1]], ["model_name", "fold", "candidate_id", "status", "test_set_used_for_selection"], append=True)
    if not args.skip_locked_config:
        write_locked_config(project_root, result_rows, config)
        status_update(project_root, {"stage": "validation_only_tuning", "status": "completed", "locked_config": "configs/locked_full_5fold_config.yaml", "test_set_used_for_selection": False})
        logger.info("Validation-only tuning completed. locked_config=configs/locked_full_5fold_config.yaml")
    else:
        status_update(project_root, {"stage": "validation_only_tuning_targeted_rerun", "status": "completed", "locked_config": "not_updated_by_targeted_rerun", "test_set_used_for_selection": False})
        logger.info("Targeted validation-only rerun completed. locked config intentionally not updated.")
    return {"models": models, "locked_config": str(project_root / "configs" / "locked_full_5fold_config.yaml")}


if __name__ == "__main__":
    main()
