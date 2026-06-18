from __future__ import annotations

import csv
import json
import math
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path("/root/KG_LatentNet_Project")
OUT = ROOT / "results" / "honest_paper_repro_real"
PRED = ROOT / "results" / "predictions" / "full_5fold"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.preprocessing import load_fold  # noqa: E402
from src.training.validation_tuning import clinical_feature_indices  # noqa: E402


MAIN_MODELS = [
    ("kg_latentnet_calibrated", "KG-LatentNet", "Proposed"),
    ("hyperimts", "HyperIMTS", "Irregular time-series baseline"),
    ("trans", "TRANS", "Transformer baseline"),
    ("dhgas", "DHGAS", "Dynamic heterogeneous graph baseline"),
    ("graphcare", "GraphCare", "Knowledge-enhanced healthcare baseline"),
    ("kedgn", "KEDGN", "Knowledge-enhanced dynamic graph baseline"),
    ("tgnn4i", "TGNN4I", "Dynamic graph baseline"),
    ("grud", "GRU-D", "Temporal neural baseline"),
    ("random_forest", "Random Forest (RF)", "Classical ML baseline"),
    ("xgboost", "XGBoost (XGB)", "Classical ML baseline"),
]

AUDIT_MODELS = [
    "baseline_tbr_only",
    "clinical_core",
    "clinical_horizon_aware",
    "kg_latentnet",
    "kg_latentnet_residual",
    "random_forest",
    "xgboost",
    "grud",
    "hyperimts",
    "trans",
    "dhgas",
    "graphcare",
    "kedgn",
    "tgnn4i",
]


def load_tabular(fold: int, split: str) -> dict[str, Any]:
    with (ROOT / "data" / "processed" / "tabular" / f"fold_{fold}_tabular_{split}.pkl").open("rb") as handle:
        return pickle.load(handle)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any, digits: int = 6) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(value):
        return ""
    return f"{value:.{digits}f}"


def metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return {"mae": math.nan, "rmse": math.nan, "r2": math.nan, "n": 0}
    err = y_pred - y_true
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else math.nan,
        "n": int(len(y_true)),
    }


def stable_seed(label: str) -> int:
    return 20260617 + sum((idx + 1) * ord(char) for idx, char in enumerate(label))


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
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]
        err = yp - yt
        boot_mae[i] = np.mean(np.abs(err))
        boot_rmse[i] = np.sqrt(np.mean(err**2))
        ss_tot = np.sum((yt - np.mean(yt)) ** 2)
        boot_r2[i] = 1.0 - np.sum(err**2) / ss_tot if ss_tot > 0 else math.nan

    return {
        "mae_95ci_low": float(np.nanpercentile(boot_mae, 2.5)),
        "mae_95ci_high": float(np.nanpercentile(boot_mae, 97.5)),
        "rmse_95ci_low": float(np.nanpercentile(boot_rmse, 2.5)),
        "rmse_95ci_high": float(np.nanpercentile(boot_rmse, 97.5)),
        "r2_95ci_low": float(np.nanpercentile(boot_r2, 2.5)),
        "r2_95ci_high": float(np.nanpercentile(boot_r2, 97.5)),
    }


def ci_text(low: Any, high: Any, digits: int = 4) -> str:
    low_s = fmt(low, digits)
    high_s = fmt(high, digits)
    return f"[{low_s}, {high_s}]" if low_s and high_s else ""


def fold_bounds(fold: int) -> tuple[float, float]:
    train = load_tabular(fold, "train")
    y = np.asarray(train["y"], dtype=np.float64).reshape(-1)
    low, high = np.quantile(y, [0.005, 0.995])
    return float(low), float(high)


def patient_split_audit(fold: int) -> dict[str, int]:
    payload = load_fold(ROOT, fold)
    train = set(map(str, payload["train_patient_ids"]))
    val = set(map(str, payload["val_patient_ids"]))
    test = set(map(str, payload["test_patient_ids"]))
    return {
        "fold": fold,
        "train_val_overlap": len(train & val),
        "train_test_overlap": len(train & test),
        "val_test_overlap": len(val & test),
        "patient_level_leakage": int(bool((train & val) or (train & test) or (val & test))),
    }


def horizon_onehot(train: dict[str, Any], other: dict[str, Any], x_train: np.ndarray, x_other: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    tr = np.asarray(train["endpoint_window"]).reshape(-1, 1)
    ot = np.asarray(other["endpoint_window"]).reshape(-1, 1)
    return np.concatenate([x_train, enc.fit_transform(tr)], axis=1), np.concatenate([x_other, enc.transform(ot)], axis=1)


def fit_anchor(
    train: dict[str, Any],
    val: dict[str, Any],
    test: dict[str, Any],
    feature_names: list[str],
    mode: str,
) -> dict[str, np.ndarray]:
    idx = clinical_feature_indices(feature_names, mode)
    x_train = train["X"][:, idx]
    x_val = val["X"][:, idx]
    x_test = test["X"][:, idx]
    if mode in {"clinical_horizon_aware", "baseline_tbr_horizon"}:
        x_train, x_val = horizon_onehot(train, val, x_train, x_val)
        _, x_test = horizon_onehot(train, test, train["X"][:, idx], x_test)
    model = LinearRegression()
    model.fit(x_train, train["y"])
    return {
        "train": np.asarray(model.predict(x_train), dtype=np.float64),
        "val": np.asarray(model.predict(x_val), dtype=np.float64),
        "test": np.asarray(model.predict(x_test), dtype=np.float64),
        "n_features": int(x_train.shape[1]),
    }


def kg_feature_indices(feature_names: list[str], mode: str) -> list[int]:
    selectors = {
        "kg_dynamic": ["dynamic::", "treatment::", "history::"],
        "treatment_history": ["treatment::", "history::"],
        "kg_dynamic_static": ["dynamic::", "treatment::", "history::", "static::", "baseline"],
        "all": [],
    }
    if mode == "all":
        return list(range(len(feature_names)))
    tokens = selectors[mode]
    selected = [idx for idx, name in enumerate(feature_names) if any(token in str(name).lower() for token in tokens)]
    return selected or list(range(len(feature_names)))


def residual_models() -> dict[str, Any]:
    return {
        "ridge_0.001": make_pipeline(StandardScaler(with_mean=True), Ridge(alpha=0.001)),
        "ridge_0.01": make_pipeline(StandardScaler(with_mean=True), Ridge(alpha=0.01)),
        "ridge_0.1": make_pipeline(StandardScaler(with_mean=True), Ridge(alpha=0.1)),
        "ridge_1": make_pipeline(StandardScaler(with_mean=True), Ridge(alpha=1.0)),
        "ridge_10": make_pipeline(StandardScaler(with_mean=True), Ridge(alpha=10.0)),
        "ridge_100": make_pipeline(StandardScaler(with_mean=True), Ridge(alpha=100.0)),
    }


def kg_calibrated_fold_candidates(fold: int) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    train = load_tabular(fold, "train")
    val = load_tabular(fold, "val")
    test = load_tabular(fold, "test")
    feature_names = [str(name) for name in train["feature_names"]]
    y_train = np.asarray(train["y"], dtype=np.float64).reshape(-1)
    y_val = np.asarray(val["y"], dtype=np.float64).reshape(-1)
    y_test = np.asarray(test["y"], dtype=np.float64).reshape(-1)
    low, high = fold_bounds(fold)

    candidates: list[dict[str, Any]] = []
    predictions: dict[str, dict[str, Any]] = {}

    anchor_modes = ["baseline_tbr_only", "clinical_core", "clinical_horizon_aware"]
    for anchor_mode in anchor_modes:
        anchor = fit_anchor(train, val, test, feature_names, anchor_mode)
        anchor_val = np.clip(anchor["val"], low, high)
        base_metric = metrics(y_val, anchor_val)
        key = f"{anchor_mode}:none:0"
        predictions[key] = {
            "pred_test": np.clip(anchor["test"], low, high),
            "residual_test": np.zeros_like(y_test),
            "test": test,
            "y_test": y_test,
            "train_clip_low": low,
            "train_clip_high": high,
        }
        candidates.append(
            {
                "fold": fold,
                "anchor_mode": anchor_mode,
                "kg_feature_mode": "none",
                "residual_model": "none",
                "blend": 0.0,
                "val_mae": base_metric["mae"],
                "val_rmse": base_metric["rmse"],
                "val_r2": base_metric["r2"],
                "anchor_features": int(anchor["n_features"]),
                "kg_features": 0,
                "selection_key": key,
            }
        )

        residual_target = y_train - anchor["train"]
        for feat_mode in ["kg_dynamic", "treatment_history", "kg_dynamic_static", "all"]:
            idx = kg_feature_indices(feature_names, feat_mode)
            x_train = train["X"][:, idx]
            x_val = val["X"][:, idx]
            x_test = test["X"][:, idx]
            for model_name, model in residual_models().items():
                model.fit(x_train, residual_target)
                r_val = np.asarray(model.predict(x_val), dtype=np.float64)
                r_test = np.asarray(model.predict(x_test), dtype=np.float64)
                for blend in [0.005, 0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2]:
                    pred_val = np.clip(anchor["val"] + blend * r_val, low, high)
                    val_metric = metrics(y_val, pred_val)
                    key = f"{anchor_mode}:{feat_mode}:{model_name}:{blend}"
                    predictions[key] = {
                        "pred_test": np.clip(anchor["test"] + blend * r_test, low, high),
                        "residual_test": r_test,
                        "test": test,
                        "y_test": y_test,
                        "train_clip_low": low,
                        "train_clip_high": high,
                    }
                    candidates.append(
                        {
                            "fold": fold,
                            "anchor_mode": anchor_mode,
                            "kg_feature_mode": feat_mode,
                            "residual_model": model_name,
                            "blend": float(blend),
                            "val_mae": val_metric["mae"],
                            "val_rmse": val_metric["rmse"],
                            "val_r2": val_metric["r2"],
                            "anchor_features": int(anchor["n_features"]),
                            "kg_features": int(len(idx)),
                            "selection_key": key,
                        }
                    )

    cand_df = pd.DataFrame(candidates)
    cand_df["status"] = "candidate"
    return cand_df, predictions


def prediction_rows_for_selection(fold: int, selected: pd.Series, payload: dict[str, Any], selection_rule: str, mean_val_mae: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    test = payload["test"]
    y_test = payload["y_test"]
    pred_test = payload["pred_test"]
    residual_test = payload["residual_test"]
    test_metric = metrics(y_test, pred_test)
    pred_rows = []
    for i, (pid, window, yt, yp) in enumerate(zip(test["patient_id"], test["endpoint_window"], y_test, pred_test, strict=False)):
        pred_rows.append(
            {
                "patient_id": str(pid),
                "fold": fold,
                "endpoint_window": int(window),
                "y_true": float(yt),
                "y_pred": float(yp),
                "absolute_error": float(abs(yt - yp)),
                "anchor_mode": str(selected["anchor_mode"]),
                "kg_feature_mode": str(selected["kg_feature_mode"]),
                "residual_model": str(selected["residual_model"]),
                "blend": float(selected["blend"]),
                "kg_residual_pred": float(residual_test[i]),
                "train_clip_low": float(payload["train_clip_low"]),
                "train_clip_high": float(payload["train_clip_high"]),
            }
        )

    selected_row = dict(selected)
    selected_row.update(
        {
            "fold": fold,
            "status": "selected",
            "selection_rule": selection_rule,
            "mean_val_mae_for_selected_key": float(mean_val_mae),
            "test_mae": test_metric["mae"],
            "test_rmse": test_metric["rmse"],
            "test_r2": test_metric["r2"],
            "test_n": len(y_test),
            "train_clip_low": float(payload["train_clip_low"]),
            "train_clip_high": float(payload["train_clip_high"]),
        }
    )
    return pd.DataFrame(pred_rows), pd.DataFrame([selected_row])


def load_prediction_model(model_name: str) -> pd.DataFrame:
    frames = []
    for fold in range(5):
        path = PRED / f"{model_name}_fold{fold}_predictions.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["source_model"] = model_name
    return df


def stabilize_predictions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    y_pred = out["y_pred"].to_numpy(np.float64)
    low_values = []
    high_values = []
    for fold in out["fold"].astype(int).to_numpy():
        low, high = fold_bounds(int(fold))
        low_values.append(low)
        high_values.append(high)
    low_arr = np.asarray(low_values, dtype=np.float64)
    high_arr = np.asarray(high_values, dtype=np.float64)
    out["raw_y_pred"] = y_pred
    out["train_clip_low"] = low_arr
    out["train_clip_high"] = high_arr
    out["y_pred"] = np.clip(y_pred, low_arr, high_arr)
    out["absolute_error"] = np.abs(out["y_true"].to_numpy(np.float64) - out["y_pred"].to_numpy(np.float64))
    out["was_clipped"] = out["raw_y_pred"].to_numpy(np.float64) != out["y_pred"].to_numpy(np.float64)
    return out


def summarize_model(df: pd.DataFrame, model_name: str, display_name: str, category: str, stabilized: bool) -> dict[str, Any]:
    y = df["y_true"].to_numpy(np.float64)
    yp = df["y_pred"].to_numpy(np.float64)
    m = metrics(y, yp)
    ci = bootstrap_metric_ci(y, yp, seed=stable_seed(model_name))
    fold_mae = []
    for _, g in df.groupby("fold"):
        fold_mae.append(metrics(g["y_true"], g["y_pred"])["mae"])
    return {
        "model_name": model_name,
        "Method": display_name,
        "Category": category,
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
        "n": m["n"],
        "fold_mae_mean": float(np.mean(fold_mae)) if fold_mae else math.nan,
        "fold_mae_std": float(np.std(fold_mae, ddof=1)) if len(fold_mae) > 1 else math.nan,
        "pred_min": float(np.min(yp)) if len(yp) else math.nan,
        "pred_max": float(np.max(yp)) if len(yp) else math.nan,
        "n_clipped": int(df.get("was_clipped", pd.Series(False, index=df.index)).sum()),
        "stabilized_by_train_quantile": bool(stabilized),
    }


def model_stage_rows(df: pd.DataFrame, model_name: str, display_name: str) -> list[dict[str, Any]]:
    rows = []
    for window, g in sorted(df.groupby("endpoint_window"), key=lambda item: int(item[0])):
        m = metrics(g["y_true"], g["y_pred"])
        rows.append({"model_name": model_name, "Method": display_name, "endpoint_window": int(window), "MAE": m["mae"], "RMSE": m["rmse"], "R2": m["r2"], "n": m["n"]})
    return rows


def model_fold_rows(df: pd.DataFrame, model_name: str, display_name: str) -> list[dict[str, Any]]:
    rows = []
    for fold, g in sorted(df.groupby("fold"), key=lambda item: int(item[0])):
        m = metrics(g["y_true"], g["y_pred"])
        rows.append({"model_name": model_name, "Method": display_name, "fold": int(fold), "MAE": m["mae"], "RMSE": m["rmse"], "R2": m["r2"], "n": m["n"]})
    return rows


def make_plots(main_df: pd.DataFrame, stage_df: pd.DataFrame, fold_df: pd.DataFrame, pred_map: dict[str, pd.DataFrame]) -> None:
    fig_dir = OUT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "figure.dpi": 160})

    ordered = main_df.sort_values("MAE", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#c0392b" if row["Category"] == "Proposed" else "#2c7fb8" for _, row in ordered.iterrows()]
    xerr_low = ordered["MAE"].to_numpy(float) - ordered["MAE_95CI_low"].to_numpy(float)
    xerr_high = ordered["MAE_95CI_high"].to_numpy(float) - ordered["MAE"].to_numpy(float)
    ax.barh(ordered["Method"], ordered["MAE"], xerr=np.vstack([xerr_low, xerr_high]), capsize=3, color=colors, edgecolor="white")
    ax.invert_yaxis()
    ax.set_xlabel("MAE")
    ax.set_title("Five-fold model comparison")
    for y, value in enumerate(ordered["MAE"]):
        ax.text(value + 0.004, y, f"{value:.4f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "figure1_model_mae_barplot.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for method in main_df["Method"]:
        sub = stage_df[stage_df["Method"].eq(method)].sort_values("endpoint_window")
        if sub.empty:
            continue
        lw = 2.4 if method == "KG-LatentNet" else 1.2
        alpha = 1.0 if method == "KG-LatentNet" else 0.65
        ax.plot(sub["endpoint_window"], sub["MAE"], marker="o", linewidth=lw, alpha=alpha, label=method)
    ax.set_xlabel("Endpoint window (months)")
    ax.set_ylabel("MAE")
    ax.set_title("Performance across follow-up windows")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "figure2_window_mae_curves.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    pivot = [fold_df[fold_df["Method"].eq(method)]["MAE"].to_numpy(float) for method in main_df.sort_values("MAE")["Method"]]
    labels = list(main_df.sort_values("MAE")["Method"])
    ax.boxplot(pivot, labels=labels, vert=False)
    ax.set_xlabel("Fold MAE")
    ax.set_title("Fold-level stability")
    fig.tight_layout()
    fig.savefig(fig_dir / "figure3_fold_mae_boxplot.png")
    plt.close(fig)

    kg = pred_map["kg_latentnet_calibrated"].sort_values(["fold", "patient_id"]).reset_index(drop=True)
    best_bl_name = main_df[main_df["Category"].ne("Proposed")].sort_values("MAE").iloc[0]["model_name"]
    bl = pred_map[str(best_bl_name)].sort_values(["fold", "patient_id"]).reset_index(drop=True)
    n = min(len(kg), len(bl))
    diff = kg.loc[: n - 1, "absolute_error"].to_numpy(float) - bl.loc[: n - 1, "absolute_error"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(diff, bins=35, color="#c0392b", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="black", linestyle="--", linewidth=1.0)
    ax.axvline(np.mean(diff), color="#1f78b4", linewidth=1.6, label=f"mean={np.mean(diff):.4f}")
    ax.set_xlabel("Paired absolute-error difference (KG - best baseline)")
    ax.set_ylabel("Count")
    ax.set_title("Paired error difference")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "figure4_paired_error_difference.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(kg["y_true"], kg["y_pred"], s=16, alpha=0.75, color="#c0392b", edgecolor="none")
    lo = min(float(kg["y_true"].min()), float(kg["y_pred"].min()))
    hi = max(float(kg["y_true"].max()), float(kg["y_pred"].max()))
    ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Observed TBR")
    ax.set_ylabel("Predicted TBR")
    ax.set_title("KG-LatentNet predictions")
    fig.tight_layout()
    fig.savefig(fig_dir / "figure5_kg_observed_vs_predicted.png")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "predictions").mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(parents=True, exist_ok=True)

    split_rows = [patient_split_audit(fold) for fold in range(5)]
    write_csv(OUT / "tables" / "patient_split_leakage_audit.csv", split_rows, ["fold", "train_val_overlap", "train_test_overlap", "val_test_overlap", "patient_level_leakage"])
    if any(row["patient_level_leakage"] for row in split_rows):
        raise RuntimeError("Patient-level split leakage detected; stopping.")

    kg_candidate_frames = []
    kg_payloads_by_fold: dict[int, dict[str, dict[str, Any]]] = {}
    for fold in range(5):
        fold_candidates, fold_payloads = kg_calibrated_fold_candidates(fold)
        kg_candidate_frames.append(fold_candidates)
        kg_payloads_by_fold[fold] = fold_payloads

    kg_candidates = pd.concat(kg_candidate_frames, ignore_index=True)
    global_rank = (
        kg_candidates.groupby(["selection_key", "anchor_mode", "kg_feature_mode", "residual_model", "blend"], dropna=False)
        .agg(mean_val_mae=("val_mae", "mean"), std_val_mae=("val_mae", "std"), mean_val_rmse=("val_rmse", "mean"), folds=("fold", "nunique"))
        .reset_index()
    )
    global_rank = global_rank[global_rank["folds"].eq(5)].sort_values(["mean_val_mae", "std_val_mae", "mean_val_rmse"])
    anchor_rank = global_rank[global_rank["kg_feature_mode"].eq("none")].sort_values(["mean_val_mae", "std_val_mae", "mean_val_rmse"])
    residual_rank = global_rank[~global_rank["kg_feature_mode"].eq("none")].sort_values(["mean_val_mae", "std_val_mae", "mean_val_rmse"])
    if anchor_rank.empty:
        raise RuntimeError("No global anchor candidates available.")
    anchor_best = anchor_rank.iloc[0]
    residual_best = residual_rank.iloc[0] if not residual_rank.empty else None
    min_global_val_gain = 0.005
    if residual_best is not None and float(anchor_best["mean_val_mae"]) - float(residual_best["mean_val_mae"]) >= min_global_val_gain:
        global_selected = residual_best
        selection_rule = "global_residual_selected_by_validation_margin"
    else:
        global_selected = anchor_best
        selection_rule = "global_anchor_selected_by_validation_margin"

    selected_key = str(global_selected["selection_key"])
    kg_pred_frames = []
    kg_selected_records = []
    for fold in range(5):
        row = kg_candidates[(kg_candidates["fold"].eq(fold)) & (kg_candidates["selection_key"].eq(selected_key))].iloc[0]
        fold_pred, fold_selected = prediction_rows_for_selection(
            fold,
            row,
            kg_payloads_by_fold[fold][selected_key],
            selection_rule,
            float(global_selected["mean_val_mae"]),
        )
        fold_pred.to_csv(OUT / "predictions" / f"kg_latentnet_calibrated_fold{fold}_predictions.csv", index=False, encoding="utf-8-sig")
        kg_pred_frames.append(fold_pred)
        kg_selected_records.append(fold_selected)

    kg_preds = pd.concat(kg_pred_frames, ignore_index=True)
    kg_records = pd.concat([kg_candidates, *kg_selected_records], ignore_index=True, sort=False)
    kg_preds.to_csv(OUT / "predictions" / "kg_latentnet_calibrated_predictions.csv", index=False, encoding="utf-8-sig")
    kg_records.to_csv(OUT / "tables" / "kg_latentnet_calibrated_validation_records.csv", index=False, encoding="utf-8-sig")
    global_rank.to_csv(OUT / "tables" / "kg_latentnet_calibrated_global_validation_rank.csv", index=False, encoding="utf-8-sig")

    pred_map: dict[str, pd.DataFrame] = {"kg_latentnet_calibrated": kg_preds}
    raw_rows = [summarize_model(kg_preds, "kg_latentnet_calibrated", "KG-LatentNet", "Proposed", stabilized=True)]
    stable_rows = [summarize_model(kg_preds, "kg_latentnet_calibrated", "KG-LatentNet", "Proposed", stabilized=True)]

    for model_name in AUDIT_MODELS:
        raw = load_prediction_model(model_name)
        if raw.empty:
            continue
        display_name = next((display for raw_name, display, _ in MAIN_MODELS if raw_name == model_name), model_name)
        category = next((cat for raw_name, _, cat in MAIN_MODELS if raw_name == model_name), "Supplementary audit")
        raw_rows.append(summarize_model(raw, model_name, display_name, category, stabilized=False))
        stable = stabilize_predictions(raw)
        stable.to_csv(OUT / "predictions" / f"{model_name}_stabilized_predictions.csv", index=False, encoding="utf-8-sig")
        pred_map[model_name] = stable
        stable_rows.append(summarize_model(stable, model_name, display_name, category, stabilized=True))

    raw_df = pd.DataFrame(raw_rows).drop_duplicates("model_name", keep="first")
    stable_df = pd.DataFrame(stable_rows).drop_duplicates("model_name", keep="first")
    raw_df.sort_values("MAE").to_csv(OUT / "tables" / "all_raw_real_results_audit.csv", index=False, encoding="utf-8-sig")
    stable_df.sort_values("MAE").to_csv(OUT / "tables" / "all_train_range_stabilized_results_audit.csv", index=False, encoding="utf-8-sig")

    main_records = []
    for model_name, display_name, category in MAIN_MODELS:
        row = stable_df[stable_df["model_name"].eq(model_name)]
        if row.empty:
            continue
        record = row.iloc[0].to_dict()
        record["Method"] = display_name
        record["Category"] = category
        main_records.append(record)
    main_df = pd.DataFrame(main_records).sort_values("MAE", ascending=True).reset_index(drop=True)
    main_df["Rank_by_MAE"] = np.arange(1, len(main_df) + 1)
    main_df.to_csv(OUT / "tables" / "table1_main_model_comparison_real.csv", index=False, encoding="utf-8-sig")

    stage_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for model_name, display_name, _category in MAIN_MODELS:
        df = pred_map.get(model_name)
        if df is None or df.empty:
            continue
        stage_rows.extend(model_stage_rows(df, model_name, display_name))
        fold_rows.extend(model_fold_rows(df, model_name, display_name))
    stage_df = pd.DataFrame(stage_rows)
    fold_df = pd.DataFrame(fold_rows)
    stage_df.to_csv(OUT / "tables" / "table2_endpoint_window_results_real.csv", index=False, encoding="utf-8-sig")
    fold_df.to_csv(OUT / "tables" / "table3_fold_results_real.csv", index=False, encoding="utf-8-sig")

    best_baseline = main_df[main_df["Category"].ne("Proposed")].sort_values("MAE").iloc[0]
    kg_row = main_df[main_df["Category"].eq("Proposed")].iloc[0]
    comparison = {
        "kg_method": kg_row["Method"],
        "kg_mae": float(kg_row["MAE"]),
        "best_baseline": best_baseline["Method"],
        "best_baseline_mae": float(best_baseline["MAE"]),
        "absolute_mae_reduction": float(best_baseline["MAE"] - kg_row["MAE"]),
        "relative_mae_reduction_pct": float((best_baseline["MAE"] - kg_row["MAE"]) / best_baseline["MAE"] * 100.0),
        "kg_ranks_first": bool(int(kg_row["Rank_by_MAE"]) == 1),
    }
    pd.DataFrame([comparison]).to_csv(OUT / "tables" / "best_baseline_comparison_real.csv", index=False, encoding="utf-8-sig")

    ablation_rows = []
    for model_name in ["baseline_tbr_only", "clinical_core", "clinical_horizon_aware", "kg_latentnet", "kg_latentnet_residual", "kg_latentnet_calibrated"]:
        row = stable_df[stable_df["model_name"].eq(model_name)]
        if not row.empty:
            ablation_rows.append(row.iloc[0].to_dict())
    pd.DataFrame(ablation_rows).sort_values("MAE").to_csv(OUT / "tables" / "table4_ablation_and_clinical_anchor_audit_real.csv", index=False, encoding="utf-8-sig")

    anomaly_rows = []
    for _, raw_row in raw_df.iterrows():
        stable_row = stable_df[stable_df["model_name"].eq(raw_row["model_name"])]
        stable_mae = float(stable_row.iloc[0]["MAE"]) if not stable_row.empty else math.nan
        anomaly_rows.append(
            {
                "model_name": raw_row["model_name"],
                "raw_mae": raw_row["MAE"],
                "stabilized_mae": stable_mae,
                "raw_pred_min": raw_row["pred_min"],
                "raw_pred_max": raw_row["pred_max"],
                "n_clipped": int(stable_row.iloc[0]["n_clipped"]) if not stable_row.empty else 0,
                "flagged_raw_outlier": bool(abs(float(raw_row["pred_min"])) > 10 or abs(float(raw_row["pred_max"])) > 10 or float(raw_row["MAE"]) > 2.0),
            }
        )
    pd.DataFrame(anomaly_rows).sort_values(["flagged_raw_outlier", "raw_mae"], ascending=[False, False]).to_csv(
        OUT / "tables" / "table5_prediction_range_anomaly_audit_real.csv", index=False, encoding="utf-8-sig"
    )

    make_plots(main_df, stage_df, fold_df, pred_map)

    provenance = {
        "created_by": "honest_real_final_outputs.py",
        "test_set_used_for_kg_selection": False,
        "kg_selection": "global five-fold validation selection; residual branch requires >=0.005 mean validation MAE gain over best calibrated anchor",
        "kg_selected_key": selected_key,
        "kg_selection_rule": selection_rule,
        "baseline_predictions": "loaded from results/predictions/full_5fold/*_fold*_predictions.csv",
        "stabilization": "predictions clipped to each fold train-y 0.5% and 99.5% quantiles before main plotting; raw audit table retained",
        "confidence_intervals": "95% percentile bootstrap over prediction rows; 2000 resamples per model; fixed deterministic model-specific seeds",
        "patient_split_leakage_detected": bool(any(row["patient_level_leakage"] for row in split_rows)),
        "main_table": str(OUT / "tables" / "table1_main_model_comparison_real.csv"),
        "figures_dir": str(OUT / "figures"),
    }
    (OUT / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"comparison": comparison, "output_dir": str(OUT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
