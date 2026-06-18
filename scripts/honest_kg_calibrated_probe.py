from __future__ import annotations

import json
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path("/root/KG_LatentNet_Project")
OUT = ROOT / "results" / "honest_paper_repro_probe"


def load_tabular(fold: int, split: str) -> dict[str, Any]:
    with (ROOT / "data" / "processed" / "tabular" / f"fold_{fold}_tabular_{split}.pkl").open("rb") as handle:
        return pickle.load(handle)


def metric(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    err = y_pred - y_true
    mse = float(np.mean(err**2))
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else math.nan,
    }


def clinical_indices(feature_names: list[str]) -> list[int]:
    tokens = ["baseline_tbr_b", "年龄", "age", "性别", "sex", "stage", "分期", "tnm"]
    idx = [i for i, name in enumerate(feature_names) if any(t.lower() in str(name).lower() for t in tokens)]
    if not idx:
        idx = [i for i, name in enumerate(feature_names) if "baseline_tbr_b" in str(name)]
    return idx


def select_indices(feature_names: list[str], mode: str) -> list[int]:
    if mode == "all":
        return list(range(len(feature_names)))
    prefixes = {
        "kg_dynamic": ["dynamic::", "treatment::", "history::"],
        "kg_dynamic_static": ["dynamic::", "treatment::", "history::", "static::", "baseline"],
        "treatment_history": ["treatment::", "history::"],
    }[mode]
    idx = [i for i, name in enumerate(feature_names) if any(str(name).lower().startswith(p) or p in str(name).lower() for p in prefixes)]
    return idx or list(range(len(feature_names)))


def horizon_design(train: dict[str, Any], other: dict[str, Any], idx: list[int]) -> tuple[np.ndarray, np.ndarray]:
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    x_train = train["X"][:, idx]
    x_other = other["X"][:, idx]
    tr_w = np.asarray(train["endpoint_window"]).reshape(-1, 1)
    ot_w = np.asarray(other["endpoint_window"]).reshape(-1, 1)
    return (
        np.concatenate([x_train, enc.fit_transform(tr_w)], axis=1),
        np.concatenate([x_other, enc.transform(ot_w)], axis=1),
    )


def fit_anchor(train: dict[str, Any], val: dict[str, Any], test: dict[str, Any], feature_names: list[str]) -> dict[str, np.ndarray]:
    idx = clinical_indices(feature_names)
    x_tr, x_val = horizon_design(train, val, idx)
    _, x_te = horizon_design(train, test, idx)
    model = LinearRegression()
    model.fit(x_tr, train["y"])
    return {
        "train": model.predict(x_tr),
        "val": model.predict(x_val),
        "test": model.predict(x_te),
    }


def residual_candidates(seed: int) -> dict[str, Any]:
    return {
        "ridge_0.001": make_pipeline(StandardScaler(with_mean=True), Ridge(alpha=0.001)),
        "ridge_0.01": make_pipeline(StandardScaler(with_mean=True), Ridge(alpha=0.01)),
        "ridge_0.1": make_pipeline(StandardScaler(with_mean=True), Ridge(alpha=0.1)),
        "ridge_1.0": make_pipeline(StandardScaler(with_mean=True), Ridge(alpha=1.0)),
        "ridge_10": make_pipeline(StandardScaler(with_mean=True), Ridge(alpha=10.0)),
        "ridge_100": make_pipeline(StandardScaler(with_mean=True), Ridge(alpha=100.0)),
    }


def train_fold(fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = load_tabular(fold, "train")
    val = load_tabular(fold, "val")
    test = load_tabular(fold, "test")
    feature_names = [str(x) for x in train["feature_names"]]
    anchor = fit_anchor(train, val, test, feature_names)
    train_resid = np.asarray(train["y"], dtype=float) - anchor["train"]
    y_val = np.asarray(val["y"], dtype=float)
    y_test = np.asarray(test["y"], dtype=float)
    bounds = np.quantile(np.asarray(train["y"], dtype=float), [0.005, 0.995])

    candidates = []
    for feat_mode in ["kg_dynamic", "treatment_history", "kg_dynamic_static", "all"]:
        idx = select_indices(feature_names, feat_mode)
        x_train = train["X"][:, idx]
        x_val = val["X"][:, idx]
        x_test = test["X"][:, idx]
        for name, model in residual_candidates(20260617 + fold).items():
            try:
                model.fit(x_train, train_resid)
                r_val = np.asarray(model.predict(x_val), dtype=float)
                r_test = np.asarray(model.predict(x_test), dtype=float)
            except Exception as exc:
                candidates.append({"fold": fold, "status": "error", "candidate": f"{feat_mode}:{name}", "error": str(exc)})
                continue
            for blend in [0.0, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5]:
                pred_val = np.clip(anchor["val"] + blend * r_val, bounds[0], bounds[1])
                val_m = metric(y_val, pred_val)
                candidates.append({
                    "fold": fold,
                    "status": "ok",
                    "candidate": f"{feat_mode}:{name}",
                    "blend": blend,
                    "val_mae": val_m["mae"],
                    "val_rmse": val_m["rmse"],
                    "val_r2": val_m["r2"],
                    "feature_count": len(idx),
                })

    cand_df = pd.DataFrame(candidates)
    ok = cand_df[cand_df["status"].eq("ok")].copy()
    best = ok.sort_values(["val_mae", "val_rmse"]).iloc[0]
    feat_mode, model_name = str(best["candidate"]).split(":", 1)
    idx = select_indices(feature_names, feat_mode)
    model = residual_candidates(20260617 + fold)[model_name]
    model.fit(train["X"][:, idx], train_resid)
    r_test = np.asarray(model.predict(test["X"][:, idx]), dtype=float)
    pred_test = np.clip(anchor["test"] + float(best["blend"]) * r_test, bounds[0], bounds[1])
    rows = []
    for pid, window, yt, yp in zip(test["patient_id"], test["endpoint_window"], y_test, pred_test, strict=False):
        rows.append({
            "patient_id": str(pid),
            "fold": fold,
            "endpoint_window": int(window),
            "y_true": float(yt),
            "y_pred": float(yp),
            "absolute_error": float(abs(yt - yp)),
            "anchor_pred": float(anchor["test"][len(rows)]),
            "kg_residual_pred": float(r_test[len(rows)]),
            "blend": float(best["blend"]),
            "candidate": str(best["candidate"]),
        })
    test_m = metric(y_test, pred_test)
    selected = pd.DataFrame([{
        "fold": fold,
        "candidate": str(best["candidate"]),
        "blend": float(best["blend"]),
        "feature_count": int(best["feature_count"]),
        "best_val_mae": float(best["val_mae"]),
        "best_val_rmse": float(best["val_rmse"]),
        "best_val_r2": float(best["val_r2"]),
        "test_mae": test_m["mae"],
        "test_rmse": test_m["rmse"],
        "test_r2": test_m["r2"],
        "train_y_clip_low": float(bounds[0]),
        "train_y_clip_high": float(bounds[1]),
        "status": "success",
    }])
    return pd.DataFrame(rows), pd.concat([cand_df, selected], ignore_index=True, sort=False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pred_frames = []
    record_frames = []
    for fold in range(5):
        preds, records = train_fold(fold)
        pred_frames.append(preds)
        record_frames.append(records)
        preds.to_csv(OUT / f"kg_latentnet_calibrated_fold{fold}_predictions.csv", index=False, encoding="utf-8-sig")
    all_preds = pd.concat(pred_frames, ignore_index=True)
    all_records = pd.concat(record_frames, ignore_index=True)
    all_preds.to_csv(OUT / "kg_latentnet_calibrated_predictions.csv", index=False, encoding="utf-8-sig")
    all_records.to_csv(OUT / "kg_latentnet_calibrated_tuning_records.csv", index=False, encoding="utf-8-sig")
    fold_rows = []
    for fold, frame in all_preds.groupby("fold"):
        m = metric(frame["y_true"], frame["y_pred"])
        fold_rows.append({"model_name": "kg_latentnet_calibrated", "fold": int(fold), **m, "n": len(frame)})
    folds = pd.DataFrame(fold_rows)
    summary = folds[["mae", "rmse", "r2"]].mean().to_dict()
    summary.update({"model_name": "kg_latentnet_calibrated", "n_folds": 5, "n": len(all_preds)})
    folds.to_csv(OUT / "kg_latentnet_calibrated_fold_metrics.csv", index=False, encoding="utf-8-sig")
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
