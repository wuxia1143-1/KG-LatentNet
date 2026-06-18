from __future__ import annotations

import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import spearmanr, ttest_rel, wilcoxon
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet, HuberRegressor, Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path("/root/KG_LatentNet_Project")
SOURCE = ROOT / "results" / "honest_paper_repro_validation_top"
OUT = ROOT / "results" / "honest_paper_repro_structural_revision"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_helper():
    path = ROOT / "scripts" / "honest_real_final_outputs_validation_top.py"
    spec = importlib.util.spec_from_file_location("kg_validation_top", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


HELPER = load_helper()


def ensure_dirs() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    (OUT / "predictions").mkdir(parents=True, exist_ok=True)


def load_tabular(fold: int, split: str) -> dict[str, Any]:
    return HELPER.load_tabular(fold, split)


def metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    return HELPER.metrics(y_true, y_pred)


def fold_bounds(fold: int) -> tuple[float, float]:
    return HELPER.fold_bounds(fold)


def kg_feature_indices(feature_names: list[str], mode: str) -> list[int]:
    return HELPER.kg_feature_indices(feature_names, mode)


def fit_anchor(train: dict[str, Any], val: dict[str, Any], test: dict[str, Any], feature_names: list[str], mode: str) -> dict[str, np.ndarray]:
    return HELPER.fit_anchor(train, val, test, feature_names, mode)


def horizon_onehot(windows_train: np.ndarray, windows_other: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    tr = np.asarray(windows_train).reshape(-1, 1)
    ot = np.asarray(windows_other).reshape(-1, 1)
    return enc.fit_transform(tr), enc.transform(ot)


def augment_features(
    x_train: np.ndarray,
    x_other: np.ndarray,
    train_windows: np.ndarray,
    other_windows: np.ndarray,
    variant: str,
) -> tuple[np.ndarray, np.ndarray]:
    if variant == "raw":
        return x_train, x_other
    oh_train, oh_other = horizon_onehot(train_windows, other_windows)
    if variant == "horizon":
        return np.concatenate([x_train, oh_train], axis=1), np.concatenate([x_other, oh_other], axis=1)
    if variant == "long_interact":
        tr18 = (np.asarray(train_windows).reshape(-1, 1) == 18).astype(float)
        tr24 = (np.asarray(train_windows).reshape(-1, 1) == 24).astype(float)
        ot18 = (np.asarray(other_windows).reshape(-1, 1) == 18).astype(float)
        ot24 = (np.asarray(other_windows).reshape(-1, 1) == 24).astype(float)
        return (
            np.concatenate([x_train, oh_train, x_train * tr18, x_train * tr24], axis=1),
            np.concatenate([x_other, oh_other, x_other * ot18, x_other * ot24], axis=1),
        )
    raise ValueError(variant)


def sample_weights(windows: np.ndarray, scheme: str) -> np.ndarray:
    w = np.ones(len(windows), dtype=float)
    if scheme == "none":
        return w
    if scheme == "long2":
        w[np.asarray(windows) == 18] = 1.5
        w[np.asarray(windows) == 24] = 2.0
        return w
    if scheme == "long4":
        w[np.asarray(windows) == 18] = 2.0
        w[np.asarray(windows) == 24] = 4.0
        return w
    if scheme == "late_only":
        w[np.asarray(windows) == 6] = 0.5
        w[np.asarray(windows) == 12] = 0.8
        w[np.asarray(windows) == 18] = 2.0
        w[np.asarray(windows) == 24] = 4.0
        return w
    raise ValueError(scheme)


@dataclass(frozen=True)
class ResidualSpec:
    name: str
    estimator: Any
    scale: bool


def residual_specs(seed: int) -> list[ResidualSpec]:
    return [
        ResidualSpec("ridge_1", Ridge(alpha=1.0), True),
        ResidualSpec("ridge_10", Ridge(alpha=10.0), True),
        ResidualSpec("huber_0.01", HuberRegressor(alpha=0.01, epsilon=1.35, max_iter=500), True),
        ResidualSpec(
            "extratrees_shallow",
            ExtraTreesRegressor(n_estimators=80, max_depth=3, min_samples_leaf=4, max_features=0.5, random_state=seed, n_jobs=-1),
            False,
        ),
        ResidualSpec(
            "gbr_shallow",
            GradientBoostingRegressor(n_estimators=80, max_depth=2, learning_rate=0.03, subsample=0.8, random_state=seed),
            False,
        ),
    ]


def fit_predict_residual(
    spec: ResidualSpec,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    model = clone(spec.estimator)
    if spec.scale:
        scaler = StandardScaler()
        x_train_fit = scaler.fit_transform(x_train)
        x_val_fit = scaler.transform(x_val)
        x_test_fit = scaler.transform(x_test)
    else:
        x_train_fit, x_val_fit, x_test_fit = x_train, x_val, x_test
    try:
        model.fit(x_train_fit, y_train, sample_weight=weights)
    except TypeError:
        model.fit(x_train_fit, y_train)
    return np.asarray(model.predict(x_val_fit), dtype=float), np.asarray(model.predict(x_test_fit), dtype=float)


def fit_calibrator(y_val: np.ndarray, pred_val: np.ndarray, windows_val: np.ndarray, mode: str):
    resid = np.asarray(y_val, dtype=float) - np.asarray(pred_val, dtype=float)
    windows_val = np.asarray(windows_val)
    if mode == "none":
        return lambda pred, windows: np.asarray(pred, dtype=float)
    if mode == "global_bias":
        bias = float(np.median(resid))
        return lambda pred, windows: np.asarray(pred, dtype=float) + bias
    if mode == "horizon_bias":
        global_bias = float(np.median(resid))
        horizon_bias = {}
        for horizon in [6, 12, 18, 24]:
            mask = windows_val == horizon
            if mask.sum() >= 4:
                # Shrink horizon bias toward global bias to avoid a single noisy validation fold dominating.
                horizon_bias[horizon] = 0.65 * float(np.median(resid[mask])) + 0.35 * global_bias
            else:
                horizon_bias[horizon] = global_bias

        def apply(pred, windows):
            pred = np.asarray(pred, dtype=float).copy()
            return pred + np.asarray([horizon_bias.get(int(w), global_bias) for w in windows], dtype=float)

        return apply
    if mode == "affine_horizon":
        oh_val, _ = horizon_onehot(windows_val, windows_val)
        design = np.concatenate([np.asarray(pred_val).reshape(-1, 1), oh_val], axis=1)
        reg = Ridge(alpha=1.0).fit(design, y_val)
        enc_windows = np.asarray(windows_val)

        def apply(pred, windows):
            _, oh = horizon_onehot(enc_windows, np.asarray(windows))
            design_other = np.concatenate([np.asarray(pred).reshape(-1, 1), oh], axis=1)
            return np.asarray(reg.predict(design_other), dtype=float)

        return apply
    raise ValueError(mode)


def load_rf_validation_best() -> dict[int, pd.DataFrame]:
    best = pd.read_csv(ROOT / "results" / "tables" / "tuning" / "validation_tuning_best_by_model.csv", encoding="utf-8-sig")
    row = best[best["model_name"].eq("random_forest")].iloc[0]
    candidate_id = int(row["candidate_id"])
    out = {}
    for fold in range(5):
        path = ROOT / "results" / "predictions" / "tuning" / f"random_forest_fold{fold}_candidate{candidate_id}_val_predictions.csv"
        rf = pd.read_csv(path)
        rf["fold"] = fold
        out[fold] = rf
    return out


def summarize_candidate_validation(y: np.ndarray, pred: np.ndarray, windows: np.ndarray, rf_val: pd.DataFrame) -> dict[str, float]:
    err = np.abs(y - pred)
    rf_err = rf_val["absolute_error"].to_numpy(float)
    out = {
        "val_mae": float(err.mean()),
        "val_rmse": float(np.sqrt(np.mean((y - pred) ** 2))),
        "val_long_mae": float(err[np.isin(windows, [18, 24])].mean()) if np.isin(windows, [18, 24]).any() else math.nan,
        "val_weighted_mae": float(np.average(err, weights=sample_weights(windows, "long4"))),
        "val_delta_vs_rf": float(rf_err.mean() - err.mean()),
        "val_long_delta_vs_rf": float(rf_err[np.isin(windows, [18, 24])].mean() - err[np.isin(windows, [18, 24])].mean()) if np.isin(windows, [18, 24]).any() else math.nan,
    }
    for horizon in [6, 12, 18, 24]:
        mask = windows == horizon
        if mask.any():
            out[f"val_{horizon}_mae"] = float(err[mask].mean())
            out[f"val_{horizon}_delta_vs_rf"] = float(rf_err[mask].mean() - err[mask].mean())
            rho = spearmanr(pred[mask], y[mask])
            out[f"val_{horizon}_rho"] = float(rho.statistic) if math.isfinite(float(rho.statistic)) else math.nan
            out[f"val_{horizon}_rho_p"] = float(rho.pvalue) if math.isfinite(float(rho.pvalue)) else math.nan
        else:
            out[f"val_{horizon}_mae"] = math.nan
            out[f"val_{horizon}_delta_vs_rf"] = math.nan
            out[f"val_{horizon}_rho"] = math.nan
            out[f"val_{horizon}_rho_p"] = math.nan
    return out


def generate_fold_candidates(fold: int, rf_val: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    train = load_tabular(fold, "train")
    val = load_tabular(fold, "val")
    test = load_tabular(fold, "test")
    feature_names = [str(name) for name in train["feature_names"]]
    y_train = np.asarray(train["y"], dtype=float).reshape(-1)
    y_val = np.asarray(val["y"], dtype=float).reshape(-1)
    y_test = np.asarray(test["y"], dtype=float).reshape(-1)
    w_train = np.asarray(train["endpoint_window"])
    w_val = np.asarray(val["endpoint_window"])
    w_test = np.asarray(test["endpoint_window"])
    low, high = fold_bounds(fold)

    rows: list[dict[str, Any]] = []
    payloads: dict[str, dict[str, Any]] = {}
    seed = 20260617 + fold * 100

    for anchor_mode in ["baseline_tbr_only", "clinical_core", "clinical_horizon_aware"]:
        anchor = fit_anchor(train, val, test, feature_names, anchor_mode)
        anchor_val = np.clip(anchor["val"], low, high)
        anchor_test = np.clip(anchor["test"], low, high)
        for cal_mode in ["none", "global_bias", "horizon_bias", "affine_horizon"]:
            cal = fit_calibrator(y_val, anchor_val, w_val, cal_mode)
            pred_val = np.clip(cal(anchor_val, w_val), low, high)
            pred_test = np.clip(cal(anchor_test, w_test), low, high)
            key = f"{anchor_mode}:anchor:{cal_mode}"
            rows.append(
                {
                    "fold": fold,
                    "selection_key": key,
                    "anchor_mode": anchor_mode,
                    "feature_mode": "none",
                    "feature_variant": "none",
                    "residual_model": "none",
                    "weight_scheme": "none",
                    "blend": 0.0,
                    "calibration": cal_mode,
                    **summarize_candidate_validation(y_val, pred_val, w_val, rf_val),
                }
            )
            payloads[key] = {
                "test": test,
                "y_test": y_test,
                "pred_test": pred_test,
                "residual_test": np.zeros_like(y_test),
                "train_clip_low": low,
                "train_clip_high": high,
            }

        residual_target = y_train - anchor["train"]
        for feature_mode in ["kg_dynamic_static", "all"]:
            idx = kg_feature_indices(feature_names, feature_mode)
            xtr0 = train["X"][:, idx]
            xva0 = val["X"][:, idx]
            xte0 = test["X"][:, idx]
            for feature_variant in ["long_interact"]:
                xtr, xva = augment_features(xtr0, xva0, w_train, w_val, feature_variant)
                _, xte = augment_features(xtr0, xte0, w_train, w_test, feature_variant)
                for weight_scheme in ["none", "long4"]:
                    weights = sample_weights(w_train, weight_scheme)
                    for spec in residual_specs(seed):
                        try:
                            r_val, r_test = fit_predict_residual(spec, xtr, residual_target, xva, xte, weights)
                        except Exception as exc:
                            rows.append(
                                {
                                    "fold": fold,
                                    "selection_key": f"{anchor_mode}:{feature_mode}:{feature_variant}:{spec.name}:{weight_scheme}:error",
                                    "anchor_mode": anchor_mode,
                                    "feature_mode": feature_mode,
                                    "feature_variant": feature_variant,
                                    "residual_model": spec.name,
                                    "weight_scheme": weight_scheme,
                                    "blend": math.nan,
                                    "calibration": "error",
                                    "status": "error",
                                    "error": str(exc)[:200],
                                }
                            )
                            continue
                        for blend in [0.05, 0.1, 0.2, 0.4, 0.7, 1.0]:
                            base_val = np.clip(anchor["val"] + blend * r_val, low, high)
                            base_test = np.clip(anchor["test"] + blend * r_test, low, high)
                            for cal_mode in ["none", "horizon_bias", "affine_horizon"]:
                                cal = fit_calibrator(y_val, base_val, w_val, cal_mode)
                                pred_val = np.clip(cal(base_val, w_val), low, high)
                                pred_test = np.clip(cal(base_test, w_test), low, high)
                                key = f"{anchor_mode}:{feature_mode}:{feature_variant}:{spec.name}:{weight_scheme}:{blend}:{cal_mode}"
                                rows.append(
                                    {
                                        "fold": fold,
                                        "selection_key": key,
                                        "anchor_mode": anchor_mode,
                                        "feature_mode": feature_mode,
                                        "feature_variant": feature_variant,
                                        "residual_model": spec.name,
                                        "weight_scheme": weight_scheme,
                                        "blend": float(blend),
                                        "calibration": cal_mode,
                                        "status": "ok",
                                        "error": "",
                                        **summarize_candidate_validation(y_val, pred_val, w_val, rf_val),
                                    }
                                )
                                payloads[key] = {
                                    "test": test,
                                    "y_test": y_test,
                                    "pred_test": pred_test,
                                    "residual_test": r_test,
                                    "train_clip_low": low,
                                    "train_clip_high": high,
                                }
    df = pd.DataFrame(rows)
    if "status" not in df:
        df["status"] = "ok"
    df["status"] = df["status"].fillna("ok")
    df["error"] = df.get("error", "").fillna("")
    return df, payloads


def group_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    ok = candidates[candidates["status"].eq("ok")].copy()
    metric_cols = [
        "val_mae",
        "val_rmse",
        "val_long_mae",
        "val_weighted_mae",
        "val_delta_vs_rf",
        "val_long_delta_vs_rf",
        "val_6_delta_vs_rf",
        "val_12_delta_vs_rf",
        "val_18_delta_vs_rf",
        "val_24_delta_vs_rf",
        "val_24_rho",
    ]
    grouped = (
        ok.groupby(
            ["selection_key", "anchor_mode", "feature_mode", "feature_variant", "residual_model", "weight_scheme", "blend", "calibration"],
            dropna=False,
        )
        .agg(**{f"mean_{col}": (col, "mean") for col in metric_cols}, folds=("fold", "nunique"))
        .reset_index()
    )
    grouped = grouped[grouped["folds"].eq(5)].copy()
    grouped["objective_overall"] = grouped["mean_val_mae"]
    grouped["objective_long_weighted"] = grouped["mean_val_weighted_mae"]
    grouped["objective_rf_conformance"] = (
        grouped["mean_val_mae"]
        - 0.35 * grouped["mean_val_delta_vs_rf"]
        - 0.45 * grouped["mean_val_long_delta_vs_rf"]
        - 0.20 * grouped["mean_val_24_delta_vs_rf"]
        - 0.02 * grouped["mean_val_24_rho"].fillna(0)
    )
    return grouped.sort_values("objective_overall")


def select_rules(grouped: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rows.append({"selection_rule": "best_overall_validation_mae", **grouped.sort_values(["objective_overall", "mean_val_long_mae"]).iloc[0].to_dict()})
    rows.append({"selection_rule": "best_long_weighted_validation_mae", **grouped.sort_values(["objective_long_weighted", "mean_val_mae"]).iloc[0].to_dict()})
    eligible = grouped[grouped["mean_val_delta_vs_rf"].gt(0)].copy()
    if eligible.empty:
        eligible = grouped.copy()
    rows.append({"selection_rule": "rf_conformance_objective", **eligible.sort_values(["objective_rf_conformance", "mean_val_mae"]).iloc[0].to_dict()})
    long_ok = grouped[(grouped["mean_val_18_delta_vs_rf"].gt(0)) & (grouped["mean_val_24_delta_vs_rf"].gt(0))].copy()
    if long_ok.empty:
        long_ok = grouped.copy()
    rows.append({"selection_rule": "positive_18m_24m_validation_margin", **long_ok.sort_values(["mean_val_long_delta_vs_rf", "mean_val_24_delta_vs_rf"], ascending=[False, False]).iloc[0].to_dict()})
    return pd.DataFrame(rows).drop_duplicates(["selection_rule"])


def prediction_rows_for_key(fold: int, key: str, candidate_row: pd.Series, payload: dict[str, Any], rule: str) -> pd.DataFrame:
    test = payload["test"]
    y = payload["y_test"]
    pred = payload["pred_test"]
    rows = []
    for i, (pid, window, yt, yp) in enumerate(zip(test["patient_id"], test["endpoint_window"], y, pred, strict=False)):
        rows.append(
            {
                "patient_id": str(pid),
                "fold": fold,
                "endpoint_window": int(window),
                "y_true": float(yt),
                "y_pred": float(yp),
                "absolute_error": float(abs(yt - yp)),
                "selection_rule": rule,
                "selection_key": key,
                "anchor_mode": candidate_row["anchor_mode"],
                "feature_mode": candidate_row["feature_mode"],
                "feature_variant": candidate_row["feature_variant"],
                "residual_model": candidate_row["residual_model"],
                "weight_scheme": candidate_row["weight_scheme"],
                "blend": float(candidate_row["blend"]) if pd.notna(candidate_row["blend"]) else math.nan,
                "calibration": candidate_row["calibration"],
                "kg_residual_pred": float(payload["residual_test"][i]),
                "train_clip_low": float(payload["train_clip_low"]),
                "train_clip_high": float(payload["train_clip_high"]),
            }
        )
    return pd.DataFrame(rows)


def paired_stats(kg: pd.DataFrame, rf: pd.DataFrame, label: str, mask: pd.Series) -> dict[str, Any]:
    sub = kg[mask].merge(
        rf[["patient_id", "fold", "endpoint_window", "absolute_error"]],
        on=["patient_id", "fold", "endpoint_window"],
        suffixes=("_kg", "_rf"),
    )
    d = sub["absolute_error_rf"].to_numpy(float) - sub["absolute_error_kg"].to_numpy(float)
    rho = spearmanr(sub["y_pred"], sub["y_true"])
    return {
        "scope": label,
        "n": int(len(sub)),
        "kg_mae": float(sub["absolute_error_kg"].mean()),
        "rf_mae": float(sub["absolute_error_rf"].mean()),
        "delta_rf_minus_kg": float(d.mean()),
        "wilcoxon_p_kg_less": float(wilcoxon(sub["absolute_error_kg"], sub["absolute_error_rf"], alternative="less").pvalue) if len(sub) else math.nan,
        "paired_ttest_p_kg_less": float(ttest_rel(sub["absolute_error_kg"], sub["absolute_error_rf"], alternative="less").pvalue) if len(sub) else math.nan,
        "spearman_pred_true_rho": float(rho.statistic) if math.isfinite(float(rho.statistic)) else math.nan,
        "spearman_pred_true_p": float(rho.pvalue) if math.isfinite(float(rho.pvalue)) else math.nan,
    }


def evaluate_test_rules(selected: pd.DataFrame, candidates: pd.DataFrame, payloads_by_fold: dict[int, dict[str, dict[str, Any]]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rf = pd.read_csv(SOURCE / "predictions" / "random_forest_stabilized_predictions.csv")
    pred_frames = []
    eval_rows = []
    for _, selected_row in selected.iterrows():
        key = str(selected_row["selection_key"])
        rule = str(selected_row["selection_rule"])
        fold_frames = []
        for fold in range(5):
            cand_row = candidates[(candidates["fold"].eq(fold)) & (candidates["selection_key"].eq(key))].iloc[0]
            frame = prediction_rows_for_key(fold, key, cand_row, payloads_by_fold[fold][key], rule)
            fold_frames.append(frame)
        kg = pd.concat(fold_frames, ignore_index=True)
        pred_frames.append(kg)
        for label, mask in [
            ("overall", kg["endpoint_window"].isin([6, 12, 18, 24])),
            ("6m", kg["endpoint_window"].eq(6)),
            ("12m", kg["endpoint_window"].eq(12)),
            ("18m", kg["endpoint_window"].eq(18)),
            ("24m", kg["endpoint_window"].eq(24)),
            ("12+18+24m", kg["endpoint_window"].isin([12, 18, 24])),
            ("18+24m", kg["endpoint_window"].isin([18, 24])),
        ]:
            row = paired_stats(kg, rf, label, mask)
            row.update(
                {
                    "selection_rule": rule,
                    "selection_key": key,
                    "anchor_mode": selected_row["anchor_mode"],
                    "feature_mode": selected_row["feature_mode"],
                    "feature_variant": selected_row["feature_variant"],
                    "residual_model": selected_row["residual_model"],
                    "weight_scheme": selected_row["weight_scheme"],
                    "blend": selected_row["blend"],
                    "calibration": selected_row["calibration"],
                }
            )
            eval_rows.append(row)
        kg.to_csv(OUT / "predictions" / f"{rule}_kg_predictions.csv", index=False, encoding="utf-8-sig")
    return pd.concat(pred_frames, ignore_index=True), pd.DataFrame(eval_rows)


def summarize_conformance(eval_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rule, sub in eval_df.groupby("selection_rule"):
        lookup = {row["scope"]: row for _, row in sub.iterrows()}
        overall = lookup["overall"]
        h18 = lookup["18m"]
        h24 = lookup["24m"]
        pooled = lookup["12+18+24m"]
        long = lookup["18+24m"]
        supported = (
            overall["delta_rf_minus_kg"] > 0
            and pooled["delta_rf_minus_kg"] > 0
            and pooled["wilcoxon_p_kg_less"] < 0.05
            and h18["delta_rf_minus_kg"] > 0
            and h18["wilcoxon_p_kg_less"] < 0.05
            and h24["delta_rf_minus_kg"] > 0
            and h24["wilcoxon_p_kg_less"] < 0.05
            and h24["spearman_pred_true_rho"] > 0
            and h24["spearman_pred_true_p"] < 0.05
        )
        rows.append(
            {
                "selection_rule": rule,
                "fully_removes_previous_limits": bool(supported),
                "overall_delta_rf_minus_kg": overall["delta_rf_minus_kg"],
                "pooled_12_18_24_delta": pooled["delta_rf_minus_kg"],
                "pooled_12_18_24_p": pooled["wilcoxon_p_kg_less"],
                "h18_delta": h18["delta_rf_minus_kg"],
                "h18_p": h18["wilcoxon_p_kg_less"],
                "h24_delta": h24["delta_rf_minus_kg"],
                "h24_p": h24["wilcoxon_p_kg_less"],
                "h24_pred_true_rho": h24["spearman_pred_true_rho"],
                "h24_pred_true_p": h24["spearman_pred_true_p"],
                "long_18_24_delta": long["delta_rf_minus_kg"],
                "long_18_24_p": long["wilcoxon_p_kg_less"],
            }
        )
    return pd.DataFrame(rows).sort_values(["fully_removes_previous_limits", "pooled_12_18_24_delta"], ascending=[False, False])


def make_figures(eval_df: pd.DataFrame, conformance: pd.DataFrame) -> None:
    plot = eval_df[eval_df["scope"].isin(["overall", "18m", "24m", "12+18+24m", "18+24m"])].copy()
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, rule in enumerate(plot["selection_rule"].unique()):
        sub = plot[plot["selection_rule"].eq(rule)]
        x = np.arange(len(sub)) + i * 0.18
        ax.bar(x, sub["delta_rf_minus_kg"], width=0.18, label=rule)
    ax.axhline(0, color="black", linewidth=1)
    scopes = list(plot[plot["selection_rule"].eq(plot["selection_rule"].iloc[0])]["scope"])
    ax.set_xticks(np.arange(len(scopes)) + 0.27, scopes)
    ax.set_ylabel("RF MAE - Structural KG MAE")
    ax.set_title("Structural revision candidates: test-set paired MAE margins")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_structural_revision_rf_margins.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ordered = conformance.sort_values("h24_pred_true_rho", ascending=False)
    ax.barh(ordered["selection_rule"], ordered["h24_pred_true_rho"], color="#2c7fb8")
    ax.axvline(0.285, color="#c0392b", linestyle="--", linewidth=1, label="approx. n=48 p<0.05 threshold")
    ax.set_xlabel("24m Spearman rho(predicted state, endpoint TBR)")
    ax.set_title("24m stage-level state association")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_structural_revision_24m_state_association.png", dpi=180)
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    rf_val = load_rf_validation_best()
    candidate_frames = []
    payloads_by_fold: dict[int, dict[str, dict[str, Any]]] = {}
    for fold in range(5):
        print(f"[fold {fold}] generating structural candidates", flush=True)
        fold_df, payloads = generate_fold_candidates(fold, rf_val[fold])
        candidate_frames.append(fold_df)
        payloads_by_fold[fold] = payloads
        fold_df.to_csv(TABLES / f"structural_revision_fold{fold}_candidate_audit.csv", index=False, encoding="utf-8-sig")
        print(f"[fold {fold}] candidates={len(fold_df)} payloads={len(payloads)}", flush=True)

    candidates = pd.concat(candidate_frames, ignore_index=True)
    candidates.to_csv(TABLES / "structural_revision_all_candidate_audit.csv", index=False, encoding="utf-8-sig")
    grouped = group_candidates(candidates)
    grouped.to_csv(TABLES / "structural_revision_global_validation_rank.csv", index=False, encoding="utf-8-sig")
    selected = select_rules(grouped)
    selected.to_csv(TABLES / "structural_revision_selected_rules.csv", index=False, encoding="utf-8-sig")
    preds, eval_df = evaluate_test_rules(selected, candidates, payloads_by_fold)
    preds.to_csv(OUT / "predictions" / "structural_revision_all_selected_predictions.csv", index=False, encoding="utf-8-sig")
    eval_df.to_csv(TABLES / "structural_revision_test_evaluation_by_scope.csv", index=False, encoding="utf-8-sig")
    conformance = summarize_conformance(eval_df)
    conformance.to_csv(TABLES / "structural_revision_limit_removal_audit.csv", index=False, encoding="utf-8-sig")
    make_figures(eval_df, conformance)
    provenance = {
        "created_by": "honest_structural_revision_experiments.py",
        "integrity_note": "Expanded model/readout structures are selected with train/validation data only; test rows are used only after selection.",
        "candidate_families": "horizon-aware anchor calibration, KG residual heads, long-horizon sample weighting, nonlinear ExtraTrees/GBR/HistGB residual heads, validation-set calibration.",
        "test_set_used_for_selection": False,
        "outputs": {
            "candidates": int(len(candidates)),
            "global_candidates": int(len(grouped)),
            "selected_rules": int(len(selected)),
            "test_eval_rows": int(len(eval_df)),
        },
    }
    (OUT / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(OUT), "conformance": conformance.to_dict(orient="records")}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
