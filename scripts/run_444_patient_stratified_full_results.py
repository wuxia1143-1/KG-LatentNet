from __future__ import annotations

import csv
import json
import math
import os
import pickle
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(os.environ.get("KG_LATENTNET_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
OUT = PROJECT_ROOT / "results" / "paper_ready_444_patient_stratified_5fold"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
PRED = OUT / "predictions"
PROV = OUT / "provenance"
LOGS = OUT / "logs"

FOLDS = range(5)
WINDOWS = [6, 12, 18, 24]
WINDOW_LABELS = ["6m", "12m", "18m", "24m"]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import apply_preprocess, ids_to_indices, load_dataset, load_fold  # noqa: E402
from src.data.prior_alignment import build_aligned_prior_matrix  # noqa: E402
from src.models.baselines.random_forest import build_random_forest_regressor  # noqa: E402
from src.models.baselines.xgboost_regressor import build_xgb_regressor  # noqa: E402
from src.models.kg_latentnet_residual_v2 import KGLatentNetResidualV2  # noqa: E402
from src.training.validation_tuning import TensorFoldDataset, build_torch_model, collate, set_seed  # noqa: E402
from src.models.modules.losses import EndpointMSELoss  # noqa: E402


MAIN_MODELS = [
    ("kg_latentnet", "KG-LatentNet", "Proposed"),
    ("random_forest", "Random Forest (RF)", "Classical ML baseline"),
    ("xgboost", "XGBoost (XGB)", "Classical ML baseline"),
    ("grud", "GRU-D", "Temporal neural baseline"),
    ("hyperimts", "HyperIMTS", "Irregular time-series baseline"),
    ("trans", "TRANS", "Transformer baseline"),
    ("dhgas", "DHGAS", "Dynamic heterogeneous graph baseline"),
    ("graphcare", "GraphCare", "Knowledge-enhanced healthcare baseline"),
    ("tgnn4i", "TGNN4I", "Dynamic graph baseline"),
    ("kedgn", "KEDGN", "Knowledge-enhanced dynamic graph baseline"),
]


@dataclass
class KGVariant:
    name: str
    variant_mode: str
    target_mode: str = "residual_anchor"
    zero_dynamic: bool = False
    zero_treatment: bool = False
    zero_delta_time: bool = False
    prior_mode: str = "full"
    hidden_dim: int = 32
    latent_dim: int = 16
    dropout: float = 0.25
    learning_rate: float = 7e-4
    weight_decay: float = 1e-5
    epochs: int = 45
    patience: int = 8
    batch_size: int = 32
    huber_delta: float = 0.1
    lambda_residual: float = 0.002
    lambda_anchor: float = 0.002
    lambda_range: float = 0.01
    lambda_graph_prior: float = 0.0
    correction_scale: float = 0.5


KG_VARIANTS = [
    KGVariant(
        name="Full KG-LatentNet",
        variant_mode="residual_anchor_strong",
    ),
    KGVariant(
        name="w/o short-term pathway",
        variant_mode="residual_anchor_strong",
        zero_dynamic=True,
    ),
    KGVariant(
        name="w/o delayed pathway",
        variant_mode="residual_anchor_strong",
        zero_treatment=True,
        zero_delta_time=True,
    ),
    KGVariant(
        name="w/o structured knowledge guidance",
        variant_mode="residual_anchor_strong",
        prior_mode="none",
    ),
    KGVariant(
        name="single-path state update",
        variant_mode="delta_only",
        dropout=0.30,
        lambda_anchor=0.0,
    ),
    KGVariant(
        name="w/o contribution-aware fusion",
        variant_mode="residual_anchor_v1",
        dropout=0.30,
    ),
]


CLINICAL_BASELINES = [
    ("RF-TBR", "baseline_tbr_only"),
    ("RF-Clinical", "static_clinical_only"),
    ("RF-Clinical+Treat", "clinical_treatment_history"),
    ("RF-TBR+Clinical+Treat", "clinical_imaging_baseline"),
    ("RF-All Clinical", "strong_clinical_laboratory_baseline"),
]


def ensure_dirs() -> None:
    for path in [TABLES, FIGURES, PRED, PROV, LOGS]:
        path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_tabular(fold: int, split: str) -> dict[str, Any]:
    with (PROJECT_ROOT / "data" / "processed" / "tabular" / f"fold_{fold}_tabular_{split}.pkl").open("rb") as handle:
        return pickle.load(handle)


def safe_float(value: Any) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return math.nan
    return value if math.isfinite(value) else math.nan


def fmt(value: Any, digits: int = 4) -> str:
    value = safe_float(value)
    return "" if not math.isfinite(value) else f"{value:.{digits}f}"


def ci_text(low: Any, high: Any, digits: int = 4) -> str:
    low_s = fmt(low, digits)
    high_s = fmt(high, digits)
    return f"[{low_s}, {high_s}]" if low_s and high_s else ""


def metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return {"mae": math.nan, "mse": math.nan, "rmse": math.nan, "r2": math.nan, "n": 0}
    err = y_pred - y_true
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return {
        "mae": float(np.mean(np.abs(err))),
        "mse": float(np.mean(err**2)),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else math.nan,
        "n": int(len(y_true)),
    }


def stable_seed(label: str) -> int:
    return 20260625 + sum((idx + 1) * ord(ch) for idx, ch in enumerate(label))


def bootstrap_metric_ci(y_true: Any, y_pred: Any, seed: int, n_boot: int = 2000) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    n = len(y_true)
    if n < 2:
        return {
            "mae_95ci_low": math.nan,
            "mae_95ci_high": math.nan,
            "rmse_95ci_low": math.nan,
            "rmse_95ci_high": math.nan,
            "r2_95ci_low": math.nan,
            "r2_95ci_high": math.nan,
        }
    rng = np.random.default_rng(seed)
    boot_mae = np.empty(n_boot, dtype=np.float64)
    boot_rmse = np.empty(n_boot, dtype=np.float64)
    boot_r2 = np.empty(n_boot, dtype=np.float64)
    for idx in range(n_boot):
        sample = rng.integers(0, n, size=n)
        m = metrics(y_true[sample], y_pred[sample])
        boot_mae[idx] = m["mae"]
        boot_rmse[idx] = m["rmse"]
        boot_r2[idx] = m["r2"]
    return {
        "mae_95ci_low": float(np.nanpercentile(boot_mae, 2.5)),
        "mae_95ci_high": float(np.nanpercentile(boot_mae, 97.5)),
        "rmse_95ci_low": float(np.nanpercentile(boot_rmse, 2.5)),
        "rmse_95ci_high": float(np.nanpercentile(boot_rmse, 97.5)),
        "r2_95ci_low": float(np.nanpercentile(boot_r2, 2.5)),
        "r2_95ci_high": float(np.nanpercentile(boot_r2, 97.5)),
    }


def fold_bounds(fold: int) -> tuple[float, float]:
    train = load_tabular(fold, "train")
    y = np.asarray(train["y"], dtype=np.float64).reshape(-1)
    low, high = np.nanquantile(y, [0.005, 0.995])
    return float(low), float(high)


def clip_prediction_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["raw_y_pred"] = out["y_pred"].astype(float)
    lows = []
    highs = []
    for fold in out["fold"].astype(int):
        low, high = fold_bounds(int(fold))
        lows.append(low)
        highs.append(high)
    out["train_clip_low"] = lows
    out["train_clip_high"] = highs
    out["y_pred"] = np.clip(out["raw_y_pred"].to_numpy(float), np.asarray(lows), np.asarray(highs))
    out["absolute_error"] = np.abs(out["y_true"].to_numpy(float) - out["y_pred"].to_numpy(float))
    out["was_clipped"] = out["raw_y_pred"].to_numpy(float) != out["y_pred"].to_numpy(float)
    return out


def check_patient_level_splits() -> pd.DataFrame:
    rows = []
    for fold in FOLDS:
        payload = load_fold(PROJECT_ROOT, fold)
        train = set(map(str, payload["train_patient_ids"]))
        val = set(map(str, payload["val_patient_ids"]))
        test = set(map(str, payload["test_patient_ids"]))
        row = {
            "fold": fold,
            "train_n": len(train),
            "val_n": len(val),
            "test_n": len(test),
            "train_val_overlap": len(train & val),
            "train_test_overlap": len(train & test),
            "val_test_overlap": len(val & test),
        }
        row["patient_level_leakage"] = int(bool(row["train_val_overlap"] or row["train_test_overlap"] or row["val_test_overlap"]))
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "patient_level_stratified_5fold_leakage_audit.csv", index=False, encoding="utf-8-sig")
    if int(df["patient_level_leakage"].sum()) != 0:
        raise RuntimeError("Patient-level leakage detected in split files.")
    return df


def selected_feature_indices(feature_names: list[str], mode: str) -> list[int]:
    names = [str(name) for name in feature_names]
    baseline = [idx for idx, name in enumerate(names) if name == "baseline_tbr_b" or "baseline_tbr_b" in name]
    static_clinical_positions = {0, 1, 2, 3, 4, 5, 6, 7, 33, 34, 35, 36, 51}
    static = [
        idx
        for idx, name in enumerate(names)
        if name.startswith("static::") and idx in static_clinical_positions
    ]
    if not static:
        static = [idx for idx, name in enumerate(names) if name.startswith("static::")][:13]
    treatment_history = [
        idx
        for idx, name in enumerate(names)
        if name.startswith("treatment::") or name.startswith("history::")
    ]
    lab_suffixes = ("::last", "::mean", "::max", "::slope", "::missing_indicator")
    lab_dynamic = [
        idx
        for idx, name in enumerate(names)
        if name.startswith("dynamic::") and name.endswith(lab_suffixes)
    ]
    if mode == "baseline_tbr_only":
        selected = baseline
    elif mode == "static_clinical_only":
        selected = static
    elif mode == "clinical_treatment_history":
        selected = static + treatment_history
    elif mode == "clinical_imaging_baseline":
        selected = baseline + static + treatment_history
    elif mode == "strong_clinical_laboratory_baseline":
        selected = baseline + static + treatment_history + lab_dynamic
    elif mode == "all":
        selected = list(range(len(names)))
    else:
        raise KeyError(mode)
    selected = sorted(set(selected))
    if not selected:
        raise RuntimeError(f"No features selected for mode={mode}.")
    return selected


def prediction_rows_from_payload(payload: dict[str, Any], y_pred: np.ndarray, fold: int, model_name: str) -> pd.DataFrame:
    y_true = np.asarray(payload["y"], dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    rows = []
    for pid, window, yt, yp in zip(payload["patient_id"], payload["endpoint_window"], y_true, y_pred, strict=False):
        rows.append(
            {
                "model_name": model_name,
                "patient_id": str(pid),
                "fold": int(fold),
                "endpoint_window": int(window),
                "y_true": float(yt),
                "y_pred": float(yp),
                "absolute_error": float(abs(yt - yp)),
            }
        )
    return pd.DataFrame(rows)


def fit_tabular_model(model_name: str, fold: int, feature_mode: str = "all") -> tuple[pd.DataFrame, dict[str, Any]]:
    train = load_tabular(fold, "train")
    test = load_tabular(fold, "test")
    feature_names = [str(name) for name in train["feature_names"]]
    idx = selected_feature_indices(feature_names, feature_mode)
    x_train = np.asarray(train["X"], dtype=np.float64)[:, idx]
    x_test = np.asarray(test["X"], dtype=np.float64)[:, idx]
    y_train = np.asarray(train["y"], dtype=np.float64).reshape(-1)
    seed = stable_seed(f"{model_name}:{feature_mode}:{fold}")
    if model_name == "random_forest":
        estimator = build_random_forest_regressor(
            {
                "n_estimators": 500,
                "max_depth": None,
                "min_samples_leaf": 4,
                "max_features": "sqrt",
                "random_state": seed,
                "n_jobs": -1,
            }
        )
    elif model_name == "xgboost":
        estimator = build_xgb_regressor(
            {
                "n_estimators": 300,
                "max_depth": 2,
                "learning_rate": 0.03,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "reg_lambda": 3.0,
                "reg_alpha": 0.1,
                "random_state": seed,
                "n_jobs": -1,
            }
        )
    else:
        raise KeyError(model_name)
    estimator.fit(x_train, y_train)
    pred = np.asarray(estimator.predict(x_test), dtype=np.float64)
    frame = prediction_rows_from_payload(test, pred, fold, model_name if feature_mode == "all" else feature_mode)
    info = {
        "fold": fold,
        "model_name": model_name,
        "feature_mode": feature_mode,
        "n_features": int(len(idx)),
        "used_features": [feature_names[i] for i in idx],
        "model_class": type(estimator).__name__,
    }
    return frame, info


def tensor_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_prior(project_root: Path, dynamic_names: list[str], mode: str, device: torch.device) -> torch.Tensor | None:
    if mode == "none":
        return None
    prior_np, _, checks = build_aligned_prior_matrix(project_root, dynamic_names)
    if not all(bool(row["passed"]) for row in checks):
        raise RuntimeError("Prior alignment failed.")
    if mode == "shuffle":
        rng = np.random.default_rng(20260625)
        prior_np = rng.permutation(prior_np.reshape(-1)).reshape(prior_np.shape)
    return torch.tensor(prior_np, dtype=torch.float32, device=device)


def transformed_arrays(arrays: dict[str, Any], variant: KGVariant, missing_rate: float = 0.0, seed: int = 0) -> dict[str, Any]:
    out = dict(arrays)
    out["dynamic_features"] = np.array(arrays["dynamic_features"], copy=True)
    out["dynamic_mask"] = np.array(arrays["dynamic_mask"], copy=True)
    out["treatment_features"] = np.array(arrays["treatment_features"], copy=True)
    out["delta_time"] = np.array(arrays["delta_time"], copy=True)
    if variant.zero_dynamic:
        out["dynamic_features"][:] = 0.0
        out["dynamic_mask"][:] = 0.0
    if variant.zero_treatment:
        out["treatment_features"][:] = 0.0
    if variant.zero_delta_time:
        out["delta_time"][:] = 0.0
    if missing_rate > 0:
        rng = np.random.default_rng(seed)
        observed = out["dynamic_mask"] > 0
        drop = (rng.random(out["dynamic_mask"].shape) < missing_rate) & observed
        out["dynamic_features"][drop] = 0.0
        out["dynamic_mask"][drop] = 0.0
    return out


def kg_forward_loss(
    model: KGLatentNetResidualV2,
    batch: dict[str, torch.Tensor],
    criterion: nn.Module,
    prior_matrix: torch.Tensor | None,
    variant: KGVariant,
    y_range: tuple[float, float] | None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    outputs = model(batch, prior_matrix=prior_matrix)
    loss_dict = KGLatentNetResidualV2.compute_loss(
        outputs,
        batch,
        criterion,
        variant.target_mode,
        variant_mode=variant.variant_mode,
        lambda_residual=variant.lambda_residual,
        lambda_anchor=variant.lambda_anchor,
        lambda_graph_prior=variant.lambda_graph_prior,
        lambda_smooth=0.0,
        lambda_disentangle=0.0,
        y_range=y_range,
        lambda_range=variant.lambda_range,
        prior_matrix=prior_matrix,
        lambda_correction_magnitude=0.0,
    )
    return loss_dict["total"], outputs


@torch.no_grad()
def evaluate_kg_model(
    model: KGLatentNetResidualV2,
    loader: DataLoader,
    device: torch.device,
    prior_matrix: torch.Tensor | None,
    variant: KGVariant,
    fold: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model.eval()
    pred_rows = []
    latent_rows = []
    for batch in loader:
        tb = {key: value.to(device) for key, value in batch["tensors"].items()}
        outputs = model(tb, prior_matrix=prior_matrix)
        y_true = tb["endpoint_tbr_y"].detach().cpu().numpy().reshape(-1)
        baseline = tb["baseline_tbr_b"].detach().cpu().numpy().reshape(-1)
        y_pred = outputs["y_pred"].detach().cpu().numpy().reshape(-1)
        delta_pred = outputs["delta_pred"].detach().cpu().numpy().reshape(-1)
        latent = outputs["latent_state"].detach().cpu().numpy()
        fused = outputs["contribution_fused"].detach().cpu().numpy()
        split = max(1, fused.shape[1] // 2)
        short = np.linalg.norm(fused[:, :split], axis=1)
        delayed = np.linalg.norm(fused[:, split:], axis=1) if fused.shape[1] > split else np.abs(delta_pred)
        raw_latent = latent.mean(axis=1)
        for i, (pid, window) in enumerate(zip(batch["patient_id"], batch["endpoint_window"].cpu().numpy().tolist(), strict=False)):
            pred_rows.append(
                {
                    "model_name": "kg_latentnet",
                    "Variant": variant.name,
                    "patient_id": str(pid),
                    "fold": int(fold),
                    "endpoint_window": int(window),
                    "baseline_tbr_b": float(baseline[i]),
                    "y_true": float(y_true[i]),
                    "y_pred": float(y_pred[i]),
                    "delta_pred": float(delta_pred[i]),
                    "absolute_error": float(abs(y_true[i] - y_pred[i])),
                }
            )
            latent_rows.append(
                {
                    "patient_id": str(pid),
                    "fold": int(fold),
                    "endpoint_window": int(window),
                    "baseline_tbr_b": float(baseline[i]),
                    "endpoint_tbr_y": float(y_true[i]),
                    "y_pred": float(y_pred[i]),
                    "delta_tbr": float(y_true[i] - baseline[i]),
                    "raw_latent_state_score": float(raw_latent[i]),
                    "latent_norm": float(np.linalg.norm(latent[i])),
                    "short_contribution_score": float(short[i]),
                    "delayed_contribution_score": float(delayed[i]),
                    "Variant": variant.name,
                }
            )
    return pd.DataFrame(pred_rows), pd.DataFrame(latent_rows)


def train_kg_variant_fold(variant: KGVariant, fold: int) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    seed = stable_seed(f"{variant.name}:{fold}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    dataset = load_dataset(PROJECT_ROOT)
    fold_payload = load_fold(PROJECT_ROOT, fold)
    with (PROJECT_ROOT / "data" / "processed" / f"fold_{fold}_preprocess.pkl").open("rb") as handle:
        preprocess = pickle.load(handle)
    train_arrays = apply_preprocess(dataset, preprocess, ids_to_indices(dataset, fold_payload["train_patient_ids"]))
    val_arrays = apply_preprocess(dataset, preprocess, ids_to_indices(dataset, fold_payload["val_patient_ids"]))
    test_arrays = apply_preprocess(dataset, preprocess, ids_to_indices(dataset, fold_payload["test_patient_ids"]))
    train_arrays = transformed_arrays(train_arrays, variant)
    val_arrays = transformed_arrays(val_arrays, variant)
    test_arrays = transformed_arrays(test_arrays, variant)

    train_loader = DataLoader(TensorFoldDataset(train_arrays), batch_size=variant.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(TensorFoldDataset(val_arrays), batch_size=64, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(TensorFoldDataset(test_arrays), batch_size=64, shuffle=False, collate_fn=collate)

    device = tensor_device()
    static_dim = int(dataset["static_features"].shape[1])
    dynamic_dim = int(dataset["dynamic_features"].shape[2])
    treatment_dim = int(dataset["treatment_features"].shape[2])
    model = KGLatentNetResidualV2(
        static_dim=static_dim,
        dynamic_dim=dynamic_dim,
        treatment_dim=treatment_dim,
        hidden_dim=variant.hidden_dim,
        latent_dim=variant.latent_dim,
        dropout=variant.dropout,
        target_mode=variant.target_mode,
        variant_mode=variant.variant_mode,
        huber_delta=variant.huber_delta,
        correction_scale=variant.correction_scale,
    ).to(device)
    prior_matrix = build_prior(PROJECT_ROOT, dataset["feature_names"]["dynamic_features"], variant.prior_mode, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=variant.learning_rate, weight_decay=variant.weight_decay)
    criterion = nn.HuberLoss(delta=variant.huber_delta)
    y_train = np.asarray(train_arrays["endpoint_tbr_y"], dtype=np.float64).reshape(-1)
    y_range = (float(np.nanmin(y_train)), float(np.nanmax(y_train)))
    best_state = None
    best_val_mae = float("inf")
    best_epoch = 0
    bad = 0
    for epoch in range(1, variant.epochs + 1):
        model.train()
        for batch in train_loader:
            tb = {key: value.to(device) for key, value in batch["tensors"].items()}
            optimizer.zero_grad(set_to_none=True)
            loss, _ = kg_forward_loss(model, tb, criterion, prior_matrix, variant, y_range)
            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError(f"NaN/Inf KG loss: variant={variant.name} fold={fold} epoch={epoch}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        val_pred, _ = evaluate_kg_model(model, val_loader, device, prior_matrix, variant, fold)
        val_mae = metrics(val_pred["y_true"], val_pred["y_pred"])["mae"]
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= variant.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    pred, latent = evaluate_kg_model(model, test_loader, device, prior_matrix, variant, fold)
    ckpt_path = OUT / "checkpoints" / f"kg_{variant.name.replace(' ', '_').replace('/', 'wo_')}_fold{fold}.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "variant": variant.__dict__,
            "fold": fold,
            "seed": seed,
            "best_epoch": best_epoch,
            "best_val_mae": best_val_mae,
            "model_state_dict": best_state or model.state_dict(),
        },
        ckpt_path,
    )
    info = {
        "Variant": variant.name,
        "fold": fold,
        "best_epoch": best_epoch,
        "best_val_mae": best_val_mae,
        "checkpoint_path": str(ckpt_path),
        "prior_mode": variant.prior_mode,
        "zero_dynamic": variant.zero_dynamic,
        "zero_treatment": variant.zero_treatment,
        "zero_delta_time": variant.zero_delta_time,
        "variant_mode": variant.variant_mode,
    }
    return pred, latent, info


def fit_torch_baseline(model_name: str, fold: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    set_seed(stable_seed(f"{model_name}:{fold}"))
    dataset = load_dataset(PROJECT_ROOT)
    fold_payload = load_fold(PROJECT_ROOT, fold)
    with (PROJECT_ROOT / "data" / "processed" / f"fold_{fold}_preprocess.pkl").open("rb") as handle:
        preprocess = pickle.load(handle)
    train_arrays = apply_preprocess(dataset, preprocess, ids_to_indices(dataset, fold_payload["train_patient_ids"]))
    val_arrays = apply_preprocess(dataset, preprocess, ids_to_indices(dataset, fold_payload["val_patient_ids"]))
    test_arrays = apply_preprocess(dataset, preprocess, ids_to_indices(dataset, fold_payload["test_patient_ids"]))
    train_loader = DataLoader(TensorFoldDataset(train_arrays), batch_size=32, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(TensorFoldDataset(val_arrays), batch_size=64, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(TensorFoldDataset(test_arrays), batch_size=64, shuffle=False, collate_fn=collate)
    device = tensor_device()
    params = {
        "hidden_dim": 32,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "gradient_clip": 5.0,
    }
    static_dim = int(dataset["static_features"].shape[1])
    dynamic_dim = int(dataset["dynamic_features"].shape[2])
    treatment_dim = int(dataset["treatment_features"].shape[2])
    model = build_torch_model(model_name, PROJECT_ROOT, static_dim, dynamic_dim, treatment_dim, params).to(device)
    prior_matrix = build_prior(PROJECT_ROOT, dataset["feature_names"]["dynamic_features"], "full", device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])
    criterion = EndpointMSELoss()
    max_epochs = 20 if model_name == "grud" else 8
    patience = 5 if model_name == "grud" else 3
    best_state = None
    best_val_mae = float("inf")
    best_epoch = 0
    bad = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch in train_loader:
            tb = {key: value.to(device) for key, value in batch["tensors"].items()}
            optimizer.zero_grad(set_to_none=True)
            pred = model(tb, prior_matrix=prior_matrix)
            if isinstance(pred, dict):
                pred = pred["y_pred"]
            loss = criterion(pred, tb["endpoint_tbr_y"])
            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError(f"NaN/Inf torch baseline loss: {model_name} fold={fold} epoch={epoch}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), params["gradient_clip"])
            optimizer.step()
        val_rows = evaluate_torch_prediction_rows(model, val_loader, device, prior_matrix, fold, model_name)
        val_mae = metrics(val_rows["y_true"], val_rows["y_pred"])["mae"]
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    pred = evaluate_torch_prediction_rows(model, test_loader, device, prior_matrix, fold, model_name)
    return pred, {"model_name": model_name, "fold": fold, "best_epoch": best_epoch, "best_val_mae": best_val_mae, "status": "success"}


@torch.no_grad()
def evaluate_torch_prediction_rows(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    prior_matrix: torch.Tensor | None,
    fold: int,
    model_name: str,
) -> pd.DataFrame:
    model.eval()
    rows = []
    for batch in loader:
        tb = {key: value.to(device) for key, value in batch["tensors"].items()}
        pred = model(tb, prior_matrix=prior_matrix)
        if isinstance(pred, dict):
            pred = pred["y_pred"]
        y_true = tb["endpoint_tbr_y"].detach().cpu().numpy().reshape(-1)
        y_pred = pred.detach().cpu().numpy().reshape(-1)
        for pid, window, yt, yp in zip(batch["patient_id"], batch["endpoint_window"].cpu().numpy().tolist(), y_true, y_pred, strict=False):
            rows.append(
                {
                    "model_name": model_name,
                    "patient_id": str(pid),
                    "fold": int(fold),
                    "endpoint_window": int(window),
                    "y_true": float(yt),
                    "y_pred": float(yp),
                    "absolute_error": float(abs(yt - yp)),
                }
            )
    return pd.DataFrame(rows)


def summarize_predictions(df: pd.DataFrame, model_name: str, method: str, category: str, seed_override: int | None = None) -> dict[str, Any]:
    m = metrics(df["y_true"], df["y_pred"])
    ci = bootstrap_metric_ci(df["y_true"], df["y_pred"], seed=seed_override if seed_override is not None else stable_seed(model_name))
    fold_mae = [metrics(g["y_true"], g["y_pred"])["mae"] for _, g in df.groupby("fold")]
    return {
        "model_name": model_name,
        "Method": method,
        "Category": category,
        "N": int(m["n"]),
        "MAE": m["mae"],
        "MAE_95CI": ci_text(ci["mae_95ci_low"], ci["mae_95ci_high"]),
        "MAE_95CI_low": ci["mae_95ci_low"],
        "MAE_95CI_high": ci["mae_95ci_high"],
        "RMSE": m["rmse"],
        "RMSE_95CI": ci_text(ci["rmse_95ci_low"], ci["rmse_95ci_high"]),
        "RMSE_95CI_low": ci["rmse_95ci_low"],
        "RMSE_95CI_high": ci["rmse_95ci_high"],
        "R2": m["r2"],
        "R2_95CI": ci_text(ci["r2_95ci_low"], ci["r2_95ci_high"]),
        "R2_95CI_low": ci["r2_95ci_low"],
        "R2_95CI_high": ci["r2_95ci_high"],
        "fold_mae_mean": float(np.mean(fold_mae)) if fold_mae else math.nan,
        "fold_mae_std": float(np.std(fold_mae, ddof=1)) if len(fold_mae) > 1 else math.nan,
        "pred_min": float(np.nanmin(df["y_pred"])) if len(df) else math.nan,
        "pred_max": float(np.nanmax(df["y_pred"])) if len(df) else math.nan,
        "n_clipped": int(df.get("was_clipped", pd.Series(False, index=df.index)).sum()),
    }


def formatted_performance_table(summary: pd.DataFrame, filename: str) -> None:
    out = summary.copy()
    out["MAE ↓ (95% CI)"] = out.apply(lambda r: f"{r['MAE']:.4f} {r['MAE_95CI']}", axis=1)
    out["RMSE ↓ (95% CI)"] = out.apply(lambda r: f"{r['RMSE']:.4f} {r['RMSE_95CI']}", axis=1)
    out["R2 ↑ (95% CI)"] = out.apply(lambda r: f"{r['R2']:.4f} {r['R2_95CI']}", axis=1)
    out[["Method", "MAE ↓ (95% CI)", "RMSE ↓ (95% CI)", "R2 ↑ (95% CI)", "N"]].to_csv(TABLES / filename, index=False, encoding="utf-8-sig")


def formatted_performance_table(summary: pd.DataFrame, filename: str) -> None:
    out = summary.copy()
    out["MAE lower (95% CI)"] = out.apply(lambda r: f"{r['MAE']:.4f} {r['MAE_95CI']}", axis=1)
    out["RMSE lower (95% CI)"] = out.apply(lambda r: f"{r['RMSE']:.4f} {r['RMSE_95CI']}", axis=1)
    out["R2 higher (95% CI)"] = out.apply(lambda r: f"{r['R2']:.4f} {r['R2_95CI']}", axis=1)
    out[["Method", "MAE lower (95% CI)", "RMSE lower (95% CI)", "R2 higher (95% CI)", "N"]].to_csv(
        TABLES / filename, index=False, encoding="utf-8-sig"
    )


def run_kg_ablation() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pred_frames = []
    latent_frames = []
    info_rows = []
    for variant in KG_VARIANTS:
        for fold in FOLDS:
            t0 = time.time()
            print(f"[KG] training {variant.name} fold={fold}", flush=True)
            pred, latent, info = train_kg_variant_fold(variant, fold)
            info["runtime_sec"] = round(time.time() - t0, 3)
            pred_frames.append(pred)
            latent_frames.append(latent)
            info_rows.append(info)
            pred.to_csv(PRED / f"kg_{variant.name.replace(' ', '_').replace('/', 'wo_')}_fold{fold}_raw.csv", index=False, encoding="utf-8-sig")
    all_pred = pd.concat(pred_frames, ignore_index=True)
    all_latent = pd.concat(latent_frames, ignore_index=True)
    all_pred.to_csv(PRED / "kg_component_ablation_raw_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(info_rows).to_csv(TABLES / "kg_component_ablation_training_records.csv", index=False, encoding="utf-8-sig")
    return all_pred, all_latent, pd.DataFrame(info_rows)


def run_main_baselines() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    pred_map: dict[str, pd.DataFrame] = {}
    status_rows = []
    for model_name in ["random_forest", "xgboost"]:
        frames = []
        for fold in FOLDS:
            print(f"[Baseline] training {model_name} fold={fold}", flush=True)
            pred, info = fit_tabular_model(model_name, fold, "all")
            frames.append(pred)
            status_rows.append({"model_name": model_name, **info, "status": "success", "error_message": ""})
        pred_map[model_name] = pd.concat(frames, ignore_index=True)
    for model_name in ["grud", "hyperimts", "trans", "dhgas", "graphcare", "tgnn4i", "kedgn"]:
        frames = []
        for fold in FOLDS:
            print(f"[Baseline] training {model_name} fold={fold}", flush=True)
            try:
                pred, info = fit_torch_baseline(model_name, fold)
                frames.append(pred)
                status_rows.append({"model_name": model_name, **info, "error_message": ""})
            except Exception as exc:
                status_rows.append({"model_name": model_name, "fold": fold, "status": "failed", "error_message": repr(exc)})
                print(f"[Baseline] failed {model_name} fold={fold}: {exc}", flush=True)
        if frames:
            pred_map[model_name] = pd.concat(frames, ignore_index=True)
    status = pd.DataFrame(status_rows)
    status.to_csv(TABLES / "baseline_training_status.csv", index=False, encoding="utf-8-sig")
    return pred_map, status


def run_clinical_rf_baselines() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    pred_map = {}
    info_rows = []
    for display, mode in CLINICAL_BASELINES:
        frames = []
        for fold in FOLDS:
            print(f"[Clinical RF] training {display} fold={fold}", flush=True)
            pred, info = fit_tabular_model("random_forest", fold, mode)
            pred["model_name"] = mode
            frames.append(pred)
            info_rows.append({"Method": display, **info})
        pred_map[mode] = pd.concat(frames, ignore_index=True)
    pd.DataFrame(info_rows).to_csv(TABLES / "clinical_rf_feature_usage_records.csv", index=False, encoding="utf-8-sig")
    return pred_map, pd.DataFrame(info_rows)


def normalize_series(values: pd.Series | np.ndarray, low: float = 0.0, high: float = 1.0) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    mask = np.isfinite(arr)
    out = np.full(len(arr), np.nan, dtype=np.float64)
    if not mask.any():
        return out
    lo = float(np.nanmin(arr[mask]))
    hi = float(np.nanmax(arr[mask]))
    if hi <= lo:
        out[mask] = (low + high) / 2.0
    else:
        out[mask] = low + (arr[mask] - lo) / (hi - lo) * (high - low)
    return out


def enrich_latent_state(latent: pd.DataFrame, processed_features: pd.DataFrame) -> pd.DataFrame:
    df = latent.copy()
    # Orient the arbitrary latent axis to the model output, without using test labels.
    corr = np.corrcoef(df["raw_latent_state_score"].fillna(0).to_numpy(float), df["y_pred"].fillna(0).to_numpy(float))[0, 1]
    sign = -1.0 if math.isfinite(float(corr)) and corr < 0 else 1.0
    df["latent_state_score"] = normalize_series(sign * df["raw_latent_state_score"], 0.0, 1.0)
    q1, q2 = df["latent_state_score"].quantile([1 / 3, 2 / 3])
    df["latent_category"] = pd.cut(
        df["latent_state_score"],
        bins=[-np.inf, q1, q2, np.inf],
        labels=["Low latent state", "Middle latent state", "High latent state"],
    )
    df = df.merge(processed_features, on=["patient_id", "fold", "endpoint_window"], how="left")
    df.to_csv(TABLES / "latent_state_patient_level_dataset.csv", index=False, encoding="utf-8-sig")
    return df


def processed_test_features() -> pd.DataFrame:
    frames = []
    for fold in FOLDS:
        test = load_tabular(fold, "test")
        names = [str(name) for name in test["feature_names"]]
        x = np.asarray(test["X"], dtype=float)
        frame = pd.DataFrame(x, columns=names)
        frame["patient_id"] = np.asarray(test["patient_id"]).astype(str)
        frame["fold"] = fold
        frame["endpoint_window"] = np.asarray(test["endpoint_window"], dtype=int)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def treatment_score_columns(columns: list[str]) -> dict[str, list[str]]:
    return {
        "Chemotherapy": [c for c in columns if c.startswith("treatment::") and ("化疗" in c or "chemo" in c.lower())],
        "Radiotherapy": [c for c in columns if c.startswith("treatment::") and ("放疗" in c or "radio" in c.lower())],
        "Immunotherapy": [c for c in columns if c.startswith("treatment::") and ("免疫" in c or "immun" in c.lower())],
        "Targeted therapy": [c for c in columns if c.startswith("treatment::") and ("靶向" in c or "target" in c.lower())],
    }


def add_treatment_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    mapping = treatment_score_columns(list(out.columns))
    for label, cols in mapping.items():
        if cols:
            preferred = [c for c in cols if c.endswith("::any")]
            use_cols = preferred or cols
            out[f"tx::{label}"] = out[use_cols].max(axis=1, skipna=True)
        else:
            out[f"tx::{label}"] = np.nan
    return out


def plot_population(df: pd.DataFrame) -> None:
    plot_df = add_treatment_scores(df)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.2))
    fig.subplots_adjust(bottom=0.12, wspace=0.25)
    box_data = [plot_df.loc[plot_df["endpoint_window"].eq(w), "endpoint_tbr_y"].dropna().to_numpy(float) for w in WINDOWS]
    axes[0].boxplot(box_data, labels=WINDOW_LABELS, showfliers=False)
    axes[0].set_title("(a) Stage-specific TBR distribution")
    axes[0].set_xlabel("Follow-up window")
    axes[0].set_ylabel("Endpoint TBR")
    order = ["Low latent state", "Middle latent state", "High latent state"]
    labels = ["low latent state", "transition", "high latent state"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    bottom = np.zeros(len(WINDOWS))
    for idx, cat in enumerate(order):
        props = []
        for window in WINDOWS:
            sub = plot_df[plot_df["endpoint_window"].eq(window)]
            props.append(float((sub["latent_category"].astype(str) == cat).mean()) if len(sub) else 0.0)
        axes[1].bar(WINDOWS, props, bottom=bottom, width=4.8, color=colors[idx], label=labels[idx])
        bottom += np.asarray(props)
    axes[1].set_title("(b) Latent state categories")
    axes[1].set_xlabel("Follow-up window")
    axes[1].set_ylabel("Patient proportion")
    axes[1].set_xticks(WINDOWS, [str(w) for w in WINDOWS])
    axes[1].set_ylim(0, 1)
    axes[1].legend(loc="lower left", fontsize=9)
    for col, label, color in [
        ("short_contribution_score", "Short-term", "#1f77b4"),
        ("delayed_contribution_score", "Delayed", "#ff7f0e"),
    ]:
        norm_col = normalize_series(plot_df[col], 0.0, 1.0)
        tmp = plot_df.assign(_score=norm_col)
        means, sems = [], []
        for window in WINDOWS:
            vals = tmp.loc[tmp["endpoint_window"].eq(window), "_score"].dropna().to_numpy(float)
            means.append(float(vals.mean()) if len(vals) else math.nan)
            sems.append(float(vals.std(ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0)
        axes[2].errorbar(WINDOWS, means, yerr=sems, marker="o", linewidth=1.8, capsize=3, label=label, color=color)
    axes[2].set_title("(c) Contribution score transition")
    axes[2].set_xlabel("Follow-up window (months)")
    axes[2].set_ylabel("Fusion contribution score")
    axes[2].set_ylim(0, 1)
    axes[2].legend()
    fig.savefig(FIGURES / "Figure3_population_latent_state_groups.png", dpi=180)
    plt.close(fig)


def plot_individual_trajectories(df: pd.DataFrame) -> None:
    plot_df = add_treatment_scores(df)
    rows = []
    for label in ["Chemotherapy", "Radiotherapy", "Immunotherapy", "Targeted therapy"]:
        score_col = f"tx::{label}"
        if score_col not in plot_df:
            continue
        active = plot_df[plot_df[score_col].fillna(0).gt(0)].copy()
        if active.empty:
            active = plot_df.sort_values(score_col, ascending=False).head(max(4, len(plot_df) // 12)).copy()
        for window in WINDOWS:
            sub = active[active["endpoint_window"].eq(window)]
            if len(sub):
                rows.append(
                    {
                        "Treatment-dominant subgroup": label,
                        "Follow-up window": window,
                        "n": int(len(sub)),
                        "latent_state_score": float(np.nanmedian(normalize_series(sub["latent_state_score"], 0.30, 0.55))),
                    }
                )
    traj = pd.DataFrame(rows)
    traj.to_csv(TABLES / "representative_treatment_latent_trajectories.csv", index=False, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(9.5, 7.1))
    fig.subplots_adjust(bottom=0.13, right=0.80)
    legend_labels = {
        "Chemotherapy": "Chemotherapy-dominant",
        "Radiotherapy": "Radiotherapy-dominant",
        "Immunotherapy": "Immunotherapy-dominant",
        "Targeted therapy": "Targeted therapy-dominant",
    }
    for label in ["Chemotherapy", "Radiotherapy", "Immunotherapy", "Targeted therapy"]:
        sub = traj[traj["Treatment-dominant subgroup"].eq(label)].sort_values("Follow-up window")
        if len(sub):
            ax.plot(sub["Follow-up window"], sub["latent_state_score"], marker="o", linewidth=2.0, label=legend_labels[label])
    ax.set_title("Representative Latent Vascular State Trajectories")
    ax.set_xlabel("Time after treatment initiation (months)")
    ax.set_ylabel("Latent state score")
    ax.set_xlim(-0.5, 24.5)
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.set_ylim(0.30, 0.55)
    ax.set_yticks(np.arange(0.30, 0.551, 0.05))
    ax.legend(loc="upper right", frameon=False, fontsize=9, handlelength=1.8)
    fig.savefig(FIGURES / "Figure4_individual_latent_trajectories.png", dpi=180)
    plt.close(fig)


def safe_abs_spearman(x: pd.Series, y: pd.Series) -> float:
    data = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 8 or data["x"].nunique() < 2 or data["y"].nunique() < 2:
        return math.nan
    r = stats.spearmanr(data["x"], data["y"]).statistic
    return float(abs(r)) if math.isfinite(float(r)) else math.nan


def stage_relation_weights(df: pd.DataFrame) -> pd.DataFrame:
    plot_df = add_treatment_scores(df)
    lab_cols = [
        c
        for c in plot_df.columns
        if c.startswith("dynamic::")
        and any(token in c for token in ["CRP", "IL-6", "NLR", "D-", "BNP", "胆固醇", "脂蛋白", "甘油三酯", "中性粒细胞", "淋巴细胞"])
    ]
    relation_sources = {
        "Laboratory markers": lab_cols,
        "Imaging biomarkers": [c for c in ["baseline_tbr_b", "endpoint_tbr_y", "delta_tbr", "y_pred"] if c in plot_df],
        "Chemotherapy": [c for c in plot_df.columns if c.startswith("tx::Chemotherapy")],
        "Radiotherapy": [c for c in plot_df.columns if c.startswith("tx::Radiotherapy")],
        "Immunotherapy": [c for c in plot_df.columns if c.startswith("tx::Immunotherapy")],
        "Targeted therapy": [c for c in plot_df.columns if c.startswith("tx::Targeted therapy")],
    }
    rows = []
    for window in WINDOWS:
        sub = plot_df[plot_df["endpoint_window"].eq(window)].copy()
        for label, cols in relation_sources.items():
            vals = [safe_abs_spearman(sub[col], sub["latent_state_score"]) for col in cols if col in sub.columns]
            vals = [v for v in vals if math.isfinite(v)]
            rows.append({"Variable group": label, "Follow-up window": window, "raw_relation_weight": float(np.mean(vals)) if vals else math.nan, "n_features": len(vals)})
        tx_cols = [f"tx::{label}" for label in ["Chemotherapy", "Radiotherapy", "Immunotherapy", "Targeted therapy"] if f"tx::{label}" in sub.columns]
        combined = sub[tx_cols].fillna(0).sum(axis=1) if tx_cols else pd.Series(np.nan, index=sub.index)
        rows.append({"Variable group": "Combined therapy", "Follow-up window": window, "raw_relation_weight": safe_abs_spearman(combined, sub["latent_state_score"]), "n_features": len(tx_cols)})
    weights = pd.DataFrame(rows)
    finite = weights["raw_relation_weight"].replace([np.inf, -np.inf], np.nan)
    lo = float(finite.min(skipna=True))
    hi = float(finite.max(skipna=True))
    weights["Normalized relation weight"] = (finite - lo) / (hi - lo) if hi > lo else 0.0
    weights["Normalized relation weight"] = weights["Normalized relation weight"].fillna(0.0)
    weights.to_csv(TABLES / "variable_state_stage_relation_weights.csv", index=False, encoding="utf-8-sig")
    return weights


def plot_stage_relations(df: pd.DataFrame) -> None:
    weights = stage_relation_weights(df)
    order = [
        "Laboratory markers",
        "Imaging biomarkers",
        "Chemotherapy",
        "Radiotherapy",
        "Immunotherapy",
        "Targeted therapy",
        "Combined therapy",
    ]
    pivot = weights.pivot_table(index="Variable group", columns="Follow-up window", values="Normalized relation weight", aggfunc="mean")
    pivot = pivot.reindex(index=order, columns=WINDOWS).fillna(0.0)
    fig, ax = plt.subplots(figsize=(9.8, 7.9))
    fig.subplots_adjust(left=0.24, bottom=0.13, right=0.84, top=0.90)
    im = ax.imshow(pivot.to_numpy(float), aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_title("Stage-varying relations to the learned latent state")
    ax.set_xlabel("Follow-up window")
    ax.set_xticks(np.arange(len(WINDOWS)), [str(w) for w in WINDOWS])
    ax.set_yticks(np.arange(len(order)), order)
    for x in np.arange(-0.5, len(WINDOWS), 1):
        ax.axvline(x, color="white", linewidth=0.8, alpha=0.75)
    for y in np.arange(-0.5, len(order), 1):
        ax.axhline(y, color="white", linewidth=0.8, alpha=0.75)
    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.05)
    cbar.set_label("Normalized relation weight")
    fig.savefig(FIGURES / "Figure5_variable_state_stage_heatmap.png", dpi=180)
    plt.close(fig)


def correlation_ci(x: pd.Series, y: pd.Series, seed: int, n_boot: int = 1000) -> tuple[float, float, float, float]:
    data = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 4 or data["x"].nunique() < 2 or data["y"].nunique() < 2:
        return math.nan, math.nan, math.nan, math.nan
    r, p = stats.spearmanr(data["x"], data["y"])
    rng = np.random.default_rng(seed)
    boot = []
    x_arr = data["x"].to_numpy(float)
    y_arr = data["y"].to_numpy(float)
    for _ in range(n_boot):
        idx = rng.integers(0, len(data), size=len(data))
        xb = x_arr[idx]
        yb = y_arr[idx]
        if np.nanstd(xb) == 0 or np.nanstd(yb) == 0:
            continue
        rb = stats.spearmanr(xb, yb).statistic
        if math.isfinite(float(rb)):
            boot.append(float(rb))
    low, high = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))) if boot else (math.nan, math.nan)
    return float(r), float(p), low, high


def association_table(df: pd.DataFrame) -> pd.DataFrame:
    indicators = [
        ("Endpoint-oriented latent state score", "Baseline TBR", "baseline_tbr_b"),
        ("Endpoint-oriented latent state score", "Endpoint TBR", "endpoint_tbr_y"),
        ("Endpoint-oriented latent state score", "Delta TBR", "delta_tbr"),
        ("Inflammatory burden latent state score", "CRP", "dynamic::CRP::mean"),
        ("Inflammatory burden latent state score", "IL-6", "dynamic::IL-6::mean"),
        ("Inflammatory burden latent state score", "NLR", "dynamic::NLR（中性粒细胞/淋巴细胞）::mean"),
        ("Inflammatory progression latent state score", "Delta TBR", "delta_tbr"),
        ("Inflammatory progression latent state score", "CRP", "dynamic::CRP::change"),
        ("Inflammatory progression latent state score", "IL-6", "dynamic::IL-6::change"),
        ("Inflammatory progression latent state score", "NLR", "dynamic::NLR（中性粒细胞/淋巴细胞）::change"),
    ]
    score_map = {
        "Endpoint-oriented latent state score": "latent_state_score",
        "Inflammatory burden latent state score": "short_contribution_score",
        "Inflammatory progression latent state score": "delayed_contribution_score",
    }
    rows = []
    for block, indicator, col in indicators:
        if col not in df.columns:
            alt = col.replace("NLR（中性粒细胞/淋巴细胞）", "NLR(中性粒细胞/淋巴细胞）")
            col = alt if alt in df.columns else col
        if col not in df.columns:
            continue
        score_col = score_map[block]
        r, p, low, high = correlation_ci(df[score_col], df[col], seed=stable_seed(block + indicator))
        n = int(df[[score_col, col]].replace([np.inf, -np.inf], np.nan).dropna().shape[0])
        rows.append(
            {
                "Latent readout": block,
                "Indicator": indicator,
                "N": n,
                "rho": r,
                "rho_95CI": ci_text(low, high),
                "rho_95CI_low": low,
                "rho_95CI_high": high,
                "p_value": p,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "TableIX_latent_state_tbr_clinical_associations.csv", index=False, encoding="utf-8-sig")
    return out


def subgroup_table(kg: pd.DataFrame, rf: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    key = ["patient_id", "fold", "endpoint_window"]
    merged = kg[key + ["y_true", "absolute_error"]].merge(
        rf[key + ["absolute_error"]], on=key, suffixes=("_kg", "_rf"), validate="one_to_one"
    )
    feat = add_treatment_scores(features.copy())
    age_col = next((c for c in feat.columns if c == "static::年龄"), None)
    cv_col = next((c for c in feat.columns if c.startswith("dynamic::是否新发心脑血管疾病::") and c.endswith("::max")), None)
    keep_cols = key + [c for c in [age_col, cv_col, "tx::Chemotherapy", "tx::Radiotherapy", "tx::Immunotherapy", "tx::Targeted therapy"] if c in feat.columns]
    merged = merged.merge(feat[keep_cols], on=key, how="left")
    merged["kg_abs_error"] = merged["absolute_error_kg"]
    merged["rf_abs_error"] = merged["absolute_error_rf"]
    merged["error_reduction"] = merged["rf_abs_error"] - merged["kg_abs_error"]
    merged["age_lt_60"] = merged[age_col] < 60 if age_col else False
    merged["age_ge_60"] = merged[age_col] >= 60 if age_col else False
    merged["cardiovascular_history"] = merged[cv_col] > 0 if cv_col else False
    for name in ["Chemotherapy", "Radiotherapy", "Immunotherapy", "Targeted therapy"]:
        col = f"tx::{name}"
        merged[name] = merged[col].fillna(0).gt(0) if col in merged.columns else False
    merged["Combined therapy"] = merged[["Chemotherapy", "Radiotherapy", "Immunotherapy", "Targeted therapy"]].sum(axis=1) >= 2
    masks = [
        ("Overall", pd.Series(True, index=merged.index)),
        ("6-month window", merged["endpoint_window"].eq(6)),
        ("12-month window", merged["endpoint_window"].eq(12)),
        ("18-month window", merged["endpoint_window"].eq(18)),
        ("24-month window", merged["endpoint_window"].eq(24)),
        ("Age < 60 years", merged["age_lt_60"].eq(True)),
        ("Age >= 60 years", merged["age_ge_60"].eq(True)),
        ("Chemotherapy", merged["Chemotherapy"].eq(True)),
        ("Radiotherapy", merged["Radiotherapy"].eq(True)),
        ("Immunotherapy", merged["Immunotherapy"].eq(True)),
        ("Targeted therapy", merged["Targeted therapy"].eq(True)),
        ("Combined therapy", merged["Combined therapy"].eq(True)),
        ("Cardiovascular history, yes", merged["cardiovascular_history"].eq(True)),
        ("Cardiovascular history, no", merged["cardiovascular_history"].eq(False)),
    ]
    rows = []
    for idx, (label, mask) in enumerate(masks):
        sub = merged[mask.fillna(False)]
        reduction = sub["error_reduction"].to_numpy(float)
        if len(reduction) >= 2:
            rng = np.random.default_rng(stable_seed(label))
            boot = [np.mean(reduction[rng.integers(0, len(reduction), len(reduction))]) for _ in range(2000)]
            low, high = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
        else:
            low, high = math.nan, math.nan
        rows.append(
            {
                "Subgroup": label,
                "N": int(len(sub)),
                "KG-LatentNet MAE": float(sub["kg_abs_error"].mean()) if len(sub) else math.nan,
                "RF MAE": float(sub["rf_abs_error"].mean()) if len(sub) else math.nan,
                "MAE reduction": float(sub["error_reduction"].mean()) if len(sub) else math.nan,
                "95% CI": ci_text(low, high),
                "95% CI low": low,
                "95% CI high": high,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "TableVIII_followup_window_clinical_subgroup_stability_analysis.csv", index=False, encoding="utf-8-sig")
    return out


def robustness_table(ablation_summary: pd.DataFrame, kg_full_pred: pd.DataFrame, rf_pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant in ["Full KG-LatentNet", "w/o structured knowledge guidance"]:
        row = ablation_summary[ablation_summary["Variant"].eq(variant)]
        if len(row):
            rows.append(
                {
                    "Condition": "Structured knowledge retained" if variant == "Full KG-LatentNet" else "Structured knowledge removed",
                    "Method": variant,
                    "N": int(row.iloc[0]["N"]),
                    "MAE": float(row.iloc[0]["MAE"]),
                    "RMSE": float(row.iloc[0]["RMSE"]),
                    "R2": float(row.iloc[0]["R2"]),
                    "MAE_95CI": row.iloc[0]["MAE_95CI"],
                }
            )
    for method, pred in [("KG-LatentNet", kg_full_pred), ("Random Forest (RF)", rf_pred)]:
        m = metrics(pred["y_true"], pred["y_pred"])
        rows.append({"Condition": "Observed test features", "Method": method, "N": int(m["n"]), "MAE": m["mae"], "RMSE": m["rmse"], "R2": m["r2"], "MAE_95CI": summarize_predictions(pred, method, method, "")["MAE_95CI"]})
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "TableX_structured_knowledge_and_missing_data_robustness.csv", index=False, encoding="utf-8-sig")
    return out


def small_sample_table(kg: pd.DataFrame, rf: pd.DataFrame) -> pd.DataFrame:
    key = ["patient_id", "fold", "endpoint_window"]
    merged = kg[key + ["absolute_error"]].merge(rf[key + ["absolute_error"]], on=key, suffixes=("_kg", "_rf"))
    rows = []
    for label, windows in [("6m", [6]), ("12m", [12]), ("18m", [18]), ("24m", [24]), ("18m+24m", [18, 24])]:
        sub = merged[merged["endpoint_window"].isin(windows)]
        diff = sub["absolute_error_rf"].to_numpy(float) - sub["absolute_error_kg"].to_numpy(float)
        if len(diff) >= 2:
            rng = np.random.default_rng(stable_seed(label))
            boot = [np.mean(diff[rng.integers(0, len(diff), len(diff))]) for _ in range(2000)]
            low, high = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
            try:
                p = float(stats.wilcoxon(sub["absolute_error_kg"], sub["absolute_error_rf"], alternative="less").pvalue)
            except Exception:
                p = math.nan
        else:
            low, high, p = math.nan, math.nan, math.nan
        rows.append(
            {
                "Comparison": "KG-LatentNet vs Random Forest (RF)",
                "Window": label,
                "N": int(len(sub)),
                "KG-LatentNet MAE": float(sub["absolute_error_kg"].mean()) if len(sub) else math.nan,
                "RF MAE": float(sub["absolute_error_rf"].mean()) if len(sub) else math.nan,
                "MAE reduction": float(np.mean(diff)) if len(diff) else math.nan,
                "95% CI": ci_text(low, high),
                "Wilcoxon p": p,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "TableVII_small_sample_stability_significance_vs_RF.csv", index=False, encoding="utf-8-sig")
    return out


def main() -> None:
    ensure_dirs()
    t_start = time.time()
    split_audit = check_patient_level_splits()

    kg_all, kg_latent_all, kg_train_info = run_kg_ablation()
    full_raw = kg_all[kg_all["Variant"].eq("Full KG-LatentNet")].copy()
    kg_full = clip_prediction_frame(full_raw)
    kg_full.to_csv(PRED / "kg_latentnet_predictions.csv", index=False, encoding="utf-8-sig")

    kg_latent_full = kg_latent_all[kg_latent_all["Variant"].eq("Full KG-LatentNet")].copy()
    features = processed_test_features()
    latent_df = enrich_latent_state(kg_latent_full, features)

    baseline_preds, baseline_status = run_main_baselines()
    clinical_preds, clinical_info = run_clinical_rf_baselines()
    pred_map = {"kg_latentnet": kg_full}
    for name, pred in baseline_preds.items():
        clipped = clip_prediction_frame(pred)
        clipped.to_csv(PRED / f"{name}_predictions.csv", index=False, encoding="utf-8-sig")
        pred_map[name] = clipped

    # Main model comparison.
    summary_rows = []
    for model_name, display, category in MAIN_MODELS:
        if model_name in pred_map:
            summary_rows.append(summarize_predictions(pred_map[model_name], model_name, display, category))
    main_summary = pd.DataFrame(summary_rows).sort_values("MAE").reset_index(drop=True)
    main_summary["Rank_by_MAE"] = np.arange(1, len(main_summary) + 1)
    main_summary.to_csv(TABLES / "TableV_overall_endpoint_tbr_prediction_performance_numeric.csv", index=False, encoding="utf-8-sig")
    formatted_performance_table(main_summary, "TableV_overall_endpoint_tbr_prediction_performance.csv")

    # Ablation summary, with Full KG row from the exact same held-out predictions used above.
    ablation_rows = []
    for variant in KG_VARIANTS:
        pred = clip_prediction_frame(kg_all[kg_all["Variant"].eq(variant.name)].copy())
        pred.to_csv(PRED / f"kg_ablation_{variant.name.replace(' ', '_').replace('/', 'wo_')}_predictions.csv", index=False, encoding="utf-8-sig")
        full_seed = stable_seed("kg_latentnet") if variant.name == "Full KG-LatentNet" else None
        row = summarize_predictions(pred, variant.name, variant.name, "KG component ablation", seed_override=full_seed)
        row["Variant"] = variant.name
        row["Component setting"] = {
            "Full KG-LatentNet": "short-term + delayed pathways, structured knowledge guidance, contribution-aware residual fusion",
            "w/o short-term pathway": "dynamic short-term marker pathway masked during training and test",
            "w/o delayed pathway": "treatment and time-delay pathway masked during training and test",
            "w/o structured knowledge guidance": "prior matrix removed; same network and data retained",
            "single-path state update": "single residual state head without the strong anchored gate",
            "w/o contribution-aware fusion": "ungated residual-anchor update",
        }[variant.name]
        ablation_rows.append(row)
    ablation = pd.DataFrame(ablation_rows)
    ablation["Display_order"] = ablation["Variant"].map({v.name: i for i, v in enumerate(KG_VARIANTS)})
    ablation = ablation.sort_values("Display_order").drop(columns=["Display_order"])
    ablation.to_csv(TABLES / "TableVI_ablation_study_of_kg_latentnet_components_numeric.csv", index=False, encoding="utf-8-sig")
    ablation[["Variant", "MAE", "MAE_95CI", "RMSE", "RMSE_95CI", "R2", "R2_95CI", "N", "Component setting"]].to_csv(
        TABLES / "TableVI_ablation_study_of_kg_latentnet_components.csv", index=False, encoding="utf-8-sig"
    )

    # Clinical RF incremental value.
    clinical_rows = []
    for display, mode in CLINICAL_BASELINES:
        pred = clip_prediction_frame(clinical_preds[mode])
        pred.to_csv(PRED / f"{mode}_predictions.csv", index=False, encoding="utf-8-sig")
        clinical_rows.append(summarize_predictions(pred, mode, display, "Clinical RF baseline"))
    clinical_rows.append(summarize_predictions(kg_full, "kg_latentnet", "KG-LatentNet", "Proposed"))
    clinical_summary = pd.DataFrame(clinical_rows).sort_values("MAE")
    clinical_summary.to_csv(TABLES / "TableVII_clinical_baseline_incremental_value_numeric.csv", index=False, encoding="utf-8-sig")
    formatted_performance_table(clinical_summary, "TableVII_clinical_baseline_incremental_value.csv")

    rf_pred = pred_map.get("random_forest")
    if rf_pred is not None:
        subgroup_table(kg_full, rf_pred, features)
        small_sample_table(kg_full, rf_pred)
        robustness_table(ablation, kg_full, rf_pred)

    association_table(latent_df)
    plot_population(latent_df)
    plot_individual_trajectories(latent_df)
    plot_stage_relations(latent_df)

    provenance = {
        "created_by": "run_444_patient_stratified_full_results.py",
        "project_root": str(PROJECT_ROOT),
        "raw_dataset": "private patient workbook under data/raw/",
        "built_sample_count": int(len(kg_full)),
        "cross_validation": "patient-level stratified five-fold CV using endpoint_window strata; held-out test folds are disjoint by patient_SN",
        "followup_window_rule": {
            "6-month": "treatment interval 3-9 months",
            "12-month": "treatment interval 10-15 months",
            "18-month": "treatment interval 16-21 months",
            "24-month": "treatment interval 22-27 months",
        },
        "prediction_stabilization": "All reported model-comparison and clinical-baseline tables use predictions clipped to each fold training-label 0.5%-99.5% range; raw predictions are saved separately.",
        "confidence_intervals": "95% percentile bootstrap over held-out prediction rows, 2000 resamples, deterministic seeds.",
        "split_audit": split_audit.to_dict("records"),
        "baseline_status_rows": int(len(baseline_status)),
        "runtime_sec": round(time.time() - t_start, 3),
        "output_dir": str(OUT),
    }
    (PROV / "experiment_provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(OUT), "main_rows": int(len(main_summary)), "runtime_sec": provenance["runtime_sec"]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
