from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import spearmanr, ttest_rel, wilcoxon
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path("/root/KG_LatentNet_Project")
SOURCE = ROOT / "results" / "honest_paper_repro_validation_top"
V4B = ROOT / "results" / "honest_paper_repro_kg_v4b_horizon_protocol"
OUT = ROOT / "results" / "honest_paper_repro_kg_v6_feature_prior_gate"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
PRED = OUT / "predictions"
KEY = ["patient_id", "fold", "endpoint_window"]
WINDOWS = [6, 12, 18, 24]
PRIMARY_KEY = "baseline_tbr_only:kg_dynamic:ridge_100:0.005"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


HELPER = load_module("kg_validation_top", ROOT / "scripts" / "honest_real_final_outputs_validation_top.py")


def ensure_dirs() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    PRED.mkdir(parents=True, exist_ok=True)


def bootstrap_ci(values: np.ndarray, seed: int, n_boot: int = 3000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, len(values), len(values))
        stats[i] = float(values[idx].mean())
    return tuple(np.percentile(stats, [2.5, 97.5]).astype(float))


def load_rf_validation() -> pd.DataFrame:
    best = pd.read_csv(ROOT / "results" / "tables" / "tuning" / "validation_tuning_best_by_model.csv", encoding="utf-8-sig")
    row = best[best["model_name"].eq("random_forest")].iloc[0]
    candidate_id = int(row["candidate_id"])
    frames = []
    for fold in range(5):
        path = ROOT / "results" / "predictions" / "tuning" / f"random_forest_fold{fold}_candidate{candidate_id}_val_predictions.csv"
        frame = pd.read_csv(path)
        frame["fold"] = fold
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    out["patient_id"] = out["patient_id"].astype(str)
    out["fold"] = out["fold"].astype(int)
    out["endpoint_window"] = out["endpoint_window"].astype(int)
    return out.rename(columns={"y_pred": "rf_pred", "absolute_error": "rf_abs_error"})


def selected_v4b_keys() -> dict[int, str]:
    selected = pd.read_csv(V4B / "tables" / "kg_v4b_horizon_selected_keys.csv")
    sub = selected[selected["policy"].eq("horizon_rho_within_1se")]
    return {int(row["endpoint_window"]): str(row["selection_key"]) for _, row in sub.iterrows()}


def feature_indices(names: list[str], mode: str) -> list[int]:
    if mode == "pred_only":
        return []
    if mode == "clinical":
        return HELPER.clinical_feature_indices(names, "clinical_core")
    if mode == "kg":
        return HELPER.kg_feature_indices(names, "kg_dynamic_static")
    if mode == "all":
        return list(range(len(names)))
    raise ValueError(mode)


def fold_feature_frame(fold: int, split: str, mode: str) -> pd.DataFrame:
    tab = HELPER.load_tabular(fold, split)
    names = [str(name) for name in tab["feature_names"]]
    idx = feature_indices(names, mode)
    frame = pd.DataFrame(
        {
            "patient_id": np.asarray(tab["patient_id"]).astype(str),
            "fold": fold,
            "endpoint_window": np.asarray(tab["endpoint_window"], dtype=int),
        }
    )
    if idx:
        x = np.asarray(tab["X"][:, idx], dtype=float)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        # Keep feature count bounded for high-dimensional modes using variance on the current split.
        if mode == "all" and x.shape[1] > 80:
            var = np.var(x, axis=0)
            keep = np.argsort(var)[-80:]
            x = x[:, keep]
        if mode == "kg" and x.shape[1] > 60:
            var = np.var(x, axis=0)
            keep = np.argsort(var)[-60:]
            x = x[:, keep]
        for j in range(x.shape[1]):
            frame[f"x{j:03d}"] = x[:, j]
    return frame


def add_features(base: pd.DataFrame, split: str, mode: str) -> pd.DataFrame:
    frames = []
    for fold in range(5):
        feat = fold_feature_frame(fold, split, mode)
        frames.append(base[base["fold"].eq(fold)].merge(feat, on=KEY, how="left"))
    out = pd.concat(frames, ignore_index=True)
    return out


def build_validation_base(feature_mode: str) -> pd.DataFrame:
    rows = pd.read_csv(V4B / "tables" / "kg_v4b_validation_prediction_rows.csv")
    rows["patient_id"] = rows["patient_id"].astype(str)
    rows["fold"] = rows["fold"].astype(int)
    rows["endpoint_window"] = rows["endpoint_window"].astype(int)
    rf = load_rf_validation()[KEY + ["rf_pred", "rf_abs_error"]]
    keys = selected_v4b_keys()
    primary = rows[rows["selection_key"].eq(PRIMARY_KEY)][KEY + ["y_true", "y_pred"]].rename(columns={"y_pred": "primary_pred"})
    long_parts = []
    for window, key in keys.items():
        part = rows[rows["endpoint_window"].eq(window) & rows["selection_key"].eq(key)][KEY + ["y_pred"]]
        long_parts.append(part)
    long = pd.concat(long_parts, ignore_index=True).rename(columns={"y_pred": "v4b_pred"})
    out = primary.merge(long, on=KEY).merge(rf, on=KEY)
    return add_features(out, "val", feature_mode)


def build_test_base(feature_mode: str) -> pd.DataFrame:
    primary = pd.read_csv(SOURCE / "predictions" / "kg_latentnet_calibrated_predictions.csv")
    long = pd.read_csv(V4B / "predictions" / "horizon_rho_within_1se_train_only_predictions.csv")
    rf = pd.read_csv(SOURCE / "predictions" / "random_forest_stabilized_predictions.csv")
    for frame in [primary, long, rf]:
        frame["patient_id"] = frame["patient_id"].astype(str)
        frame["fold"] = frame["fold"].astype(int)
        frame["endpoint_window"] = frame["endpoint_window"].astype(int)
    out = primary[KEY + ["y_true", "y_pred"]].rename(columns={"y_pred": "primary_pred"})
    out = out.merge(long[KEY + ["y_pred"]].rename(columns={"y_pred": "v4b_pred"}), on=KEY)
    out = out.merge(rf[KEY + ["y_pred", "absolute_error"]].rename(columns={"y_pred": "rf_pred", "absolute_error": "rf_abs_error"}), on=KEY)
    return add_features(out, "test", feature_mode)


def policy_prediction(df: pd.DataFrame, policy: str) -> np.ndarray:
    pred = df["primary_pred"].to_numpy(float).copy()
    if policy == "primary":
        return pred
    if policy == "v4b_all":
        return df["v4b_pred"].to_numpy(float)
    if policy == "hybrid24":
        mask = df["endpoint_window"].eq(24).to_numpy()
    elif policy == "hybrid18_24":
        mask = df["endpoint_window"].isin([18, 24]).to_numpy()
    else:
        raise ValueError(policy)
    pred[mask] = df.loc[mask, "v4b_pred"].to_numpy(float)
    return pred


def matrix(df: pd.DataFrame, kg_pred: np.ndarray, feature_mode: str) -> np.ndarray:
    rf_pred = df["rf_pred"].to_numpy(float)
    windows = df["endpoint_window"].to_numpy(int)
    one_hot = np.column_stack([(windows == w).astype(float) for w in WINDOWS])
    raw_cols = [c for c in df.columns if c.startswith("x")]
    raw = df[raw_cols].to_numpy(float) if raw_cols else np.empty((len(df), 0), dtype=float)
    pred_features = np.column_stack(
        [
            kg_pred,
            rf_pred,
            kg_pred - rf_pred,
            np.abs(kg_pred - rf_pred),
            kg_pred * (windows == 18),
            kg_pred * (windows == 24),
            rf_pred * (windows == 18),
            rf_pred * (windows == 24),
            one_hot,
        ]
    )
    return np.nan_to_num(np.concatenate([pred_features, raw], axis=1), nan=0.0, posinf=0.0, neginf=0.0)


def masks(df: pd.DataFrame, scope: str) -> list[tuple[str, np.ndarray]]:
    if scope == "long_only":
        return [("long", df["endpoint_window"].isin([18, 24]).to_numpy())]
    if scope == "horizon":
        return [(str(w), df["endpoint_window"].eq(w).to_numpy()) for w in WINDOWS]
    if scope == "global":
        return [("global", np.ones(len(df), dtype=bool))]
    raise ValueError(scope)


def fit_predict_fold(
    val: pd.DataFrame,
    test: pd.DataFrame,
    base_policy: str,
    readout: str,
    scope: str,
    feature_mode: str,
    param: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    kg_val = policy_prediction(val, base_policy)
    kg_test = policy_prediction(test, base_policy)
    pred = kg_test.copy()
    meta: dict[str, Any] = {"base_policy": base_policy, "readout": readout, "scope": scope, "feature_mode": feature_mode, "param": param}
    for label, vmask in masks(val, scope):
        tmask = dict(masks(test, scope))[label]
        if vmask.sum() < 8 or tmask.sum() == 0:
            continue
        y_val = val.loc[vmask, "y_true"].to_numpy(float)
        rf_val = val.loc[vmask, "rf_pred"].to_numpy(float)
        rf_test = test.loc[tmask, "rf_pred"].to_numpy(float)
        x_val = matrix(val.loc[vmask].reset_index(drop=True), kg_val[vmask], feature_mode)
        x_test = matrix(test.loc[tmask].reset_index(drop=True), kg_test[tmask], feature_mode)
        if readout == "convex_blend":
            best_w, best_mae = 1.0, math.inf
            for w in np.linspace(0.0, 1.4, 15):
                cand = w * kg_val[vmask] + (1 - w) * rf_val
                mae = float(np.mean(np.abs(y_val - cand)))
                if mae < best_mae:
                    best_w, best_mae = float(w), mae
            pred[tmask] = best_w * kg_test[tmask] + (1 - best_w) * rf_test
            meta[f"weight_{label}"] = best_w
        elif readout == "logistic_gate":
            target = (np.abs(y_val - kg_val[vmask]) <= np.abs(y_val - rf_val)).astype(int)
            if len(np.unique(target)) < 2:
                continue
            model = make_pipeline(StandardScaler(), LogisticRegression(C=param, max_iter=1000, class_weight="balanced"))
            model.fit(x_val, target)
            p_kg = model.predict_proba(x_test)[:, 1]
            pred[tmask] = p_kg * kg_test[tmask] + (1 - p_kg) * rf_test
            meta[f"target_rate_{label}"] = float(target.mean())
        elif readout == "extra_trees_gate":
            target = (np.abs(y_val - kg_val[vmask]) <= np.abs(y_val - rf_val)).astype(int)
            if len(np.unique(target)) < 2:
                continue
            model = ExtraTreesClassifier(
                n_estimators=120,
                max_depth=int(param),
                min_samples_leaf=3,
                max_features=0.6,
                random_state=seed,
                class_weight="balanced",
                n_jobs=-1,
            )
            model.fit(x_val, target)
            p_kg = model.predict_proba(x_test)[:, 1]
            pred[tmask] = p_kg * kg_test[tmask] + (1 - p_kg) * rf_test
            meta[f"target_rate_{label}"] = float(target.mean())
        elif readout == "gbr_gate":
            target = (np.abs(y_val - kg_val[vmask]) <= np.abs(y_val - rf_val)).astype(int)
            if len(np.unique(target)) < 2:
                continue
            model = GradientBoostingClassifier(n_estimators=60, learning_rate=0.04, max_depth=int(param), random_state=seed)
            model.fit(x_val, target)
            p_kg = model.predict_proba(x_test)[:, 1]
            pred[tmask] = p_kg * kg_test[tmask] + (1 - p_kg) * rf_test
            meta[f"target_rate_{label}"] = float(target.mean())
        elif readout == "ridge_stack":
            model = make_pipeline(StandardScaler(), Ridge(alpha=param))
            model.fit(x_val, y_val)
            pred[tmask] = model.predict(x_test)
        elif readout == "extra_trees_stack":
            model = ExtraTreesRegressor(
                n_estimators=120,
                max_depth=int(param),
                min_samples_leaf=3,
                max_features=0.6,
                random_state=seed,
                n_jobs=-1,
            )
            model.fit(x_val, y_val)
            pred[tmask] = model.predict(x_test)
        elif readout == "gbr_stack":
            model = GradientBoostingRegressor(n_estimators=60, learning_rate=0.04, max_depth=int(param), random_state=seed)
            model.fit(x_val, y_val)
            pred[tmask] = model.predict(x_test)
        else:
            raise ValueError(readout)
        meta[f"n_val_{label}"] = int(vmask.sum())
    return pred, meta


def candidate_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for base_policy in ["primary", "hybrid24", "hybrid18_24", "v4b_all"]:
        specs.append({"base_policy": base_policy, "readout": "identity", "scope": "none", "feature_mode": "pred_only", "param": 0.0})
        for scope in ["long_only", "horizon"]:
            specs.append({"base_policy": base_policy, "readout": "convex_blend", "scope": scope, "feature_mode": "pred_only", "param": 0.0})
            for feature_mode in ["pred_only", "clinical", "kg", "all"]:
                for c_value in [0.2, 1.0, 5.0]:
                    specs.append({"base_policy": base_policy, "readout": "logistic_gate", "scope": scope, "feature_mode": feature_mode, "param": c_value})
                for depth in [2, 3]:
                    specs.append({"base_policy": base_policy, "readout": "extra_trees_gate", "scope": scope, "feature_mode": feature_mode, "param": float(depth)})
                    specs.append({"base_policy": base_policy, "readout": "gbr_gate", "scope": scope, "feature_mode": feature_mode, "param": float(depth)})
                for alpha in [1.0, 10.0, 50.0]:
                    specs.append({"base_policy": base_policy, "readout": "ridge_stack", "scope": scope, "feature_mode": feature_mode, "param": alpha})
                for depth in [2, 3]:
                    specs.append({"base_policy": base_policy, "readout": "extra_trees_stack", "scope": scope, "feature_mode": feature_mode, "param": float(depth)})
                    specs.append({"base_policy": base_policy, "readout": "gbr_stack", "scope": scope, "feature_mode": feature_mode, "param": float(depth)})
    return specs


def candidate_name(spec: dict[str, Any]) -> str:
    if spec["readout"] == "identity":
        return f"{spec['base_policy']}:identity"
    return f"{spec['base_policy']}:{spec['readout']}:{spec['scope']}:{spec['feature_mode']}:{spec['param']:g}"


def predict_candidate(spec: dict[str, Any], bases: dict[str, tuple[pd.DataFrame, pd.DataFrame]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    frames = []
    meta_all: dict[str, Any] = dict(spec)
    for fold in range(5):
        feature_mode = str(spec["feature_mode"])
        val_all, test_all = bases[feature_mode]
        val = val_all[val_all["fold"].eq(fold)].reset_index(drop=True)
        test = test_all[test_all["fold"].eq(fold)].reset_index(drop=True)
        if spec["readout"] == "identity":
            pred = policy_prediction(test, str(spec["base_policy"]))
            meta = dict(spec)
        else:
            pred, meta = fit_predict_fold(
                val,
                test,
                str(spec["base_policy"]),
                str(spec["readout"]),
                str(spec["scope"]),
                feature_mode,
                float(spec["param"]),
                seed=20260617 + fold * 97 + len(candidate_name(spec)),
            )
        frame = test[KEY + ["y_true", "rf_pred", "rf_abs_error", "primary_pred", "v4b_pred"]].copy()
        frame["y_pred"] = pred
        frame["absolute_error"] = np.abs(frame["y_true"].to_numpy(float) - pred)
        frame["candidate"] = candidate_name(spec)
        frames.append(frame)
        for key, value in meta.items():
            if key not in meta_all and isinstance(value, (int, float, str, bool)):
                meta_all[f"fold{fold}_{key}"] = value
    return pd.concat(frames, ignore_index=True), meta_all


def validation_proxy(spec: dict[str, Any], bases: dict[str, tuple[pd.DataFrame, pd.DataFrame]]) -> dict[str, float]:
    feature_mode = str(spec["feature_mode"])
    val_all, _ = bases[feature_mode]
    preds = []
    for fold in range(5):
        val = val_all[val_all["fold"].eq(fold)].reset_index(drop=True)
        if spec["readout"] == "identity":
            pred = policy_prediction(val, str(spec["base_policy"]))
        else:
            pred, _ = fit_predict_fold(
                val,
                val,
                str(spec["base_policy"]),
                str(spec["readout"]),
                str(spec["scope"]),
                feature_mode,
                float(spec["param"]),
                seed=20260617 + fold * 97 + len(candidate_name(spec)) + 11,
            )
        tmp = val[KEY + ["y_true", "rf_abs_error"]].copy()
        tmp["y_pred"] = pred
        tmp["absolute_error"] = np.abs(tmp["y_true"].to_numpy(float) - pred)
        preds.append(tmp)
    df = pd.concat(preds, ignore_index=True)
    err = df["absolute_error"].to_numpy(float)
    rferr = df["rf_abs_error"].to_numpy(float)
    long = df["endpoint_window"].isin([18, 24]).to_numpy()
    h18 = df["endpoint_window"].eq(18).to_numpy()
    h24 = df["endpoint_window"].eq(24).to_numpy()
    rho24 = spearmanr(df.loc[h24, "y_pred"], df.loc[h24, "y_true"]) if h24.any() else None
    return {
        "val_mae_proxy": float(err.mean()),
        "val_delta_vs_rf_proxy": float(rferr.mean() - err.mean()),
        "val_long_delta_vs_rf_proxy": float(rferr[long].mean() - err[long].mean()) if long.any() else math.nan,
        "val_18_delta_vs_rf_proxy": float(rferr[h18].mean() - err[h18].mean()) if h18.any() else math.nan,
        "val_24_delta_vs_rf_proxy": float(rferr[h24].mean() - err[h24].mean()) if h24.any() else math.nan,
        "val_24_rho_proxy": float(rho24.statistic) if rho24 is not None and math.isfinite(float(rho24.statistic)) else math.nan,
    }


def paired_stats(df: pd.DataFrame, scope: str, mask: pd.Series) -> dict[str, Any]:
    sub = df[mask].copy()
    d = sub["rf_abs_error"].to_numpy(float) - sub["absolute_error"].to_numpy(float)
    low, high = bootstrap_ci(d, 20260618 + len(scope) * 19 + int(sub["endpoint_window"].sum()) if len(sub) else 20260618)
    rho = spearmanr(sub["y_pred"], sub["y_true"])
    return {
        "scope": scope,
        "n": int(len(sub)),
        "kg_mae": float(sub["absolute_error"].mean()),
        "rf_mae": float(sub["rf_abs_error"].mean()),
        "delta_rf_minus_kg": float(d.mean()),
        "delta_95ci_low": low,
        "delta_95ci_high": high,
        "wilcoxon_p_kg_less": float(wilcoxon(sub["absolute_error"], sub["rf_abs_error"], alternative="less").pvalue) if len(sub) else math.nan,
        "paired_ttest_p_kg_less": float(ttest_rel(sub["absolute_error"], sub["rf_abs_error"], alternative="less").pvalue) if len(sub) else math.nan,
        "spearman_pred_true_rho": float(rho.statistic) if math.isfinite(float(rho.statistic)) else math.nan,
        "spearman_pred_true_p": float(rho.pvalue) if math.isfinite(float(rho.pvalue)) else math.nan,
    }


def evaluate(preds: pd.DataFrame, meta: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for scope, mask in [
        ("overall", preds["endpoint_window"].isin(WINDOWS)),
        ("6m", preds["endpoint_window"].eq(6)),
        ("12m", preds["endpoint_window"].eq(12)),
        ("18m", preds["endpoint_window"].eq(18)),
        ("24m", preds["endpoint_window"].eq(24)),
        ("12+18+24m", preds["endpoint_window"].isin([12, 18, 24])),
        ("18+24m", preds["endpoint_window"].isin([18, 24])),
    ]:
        row = paired_stats(preds, scope, mask)
        row["candidate"] = str(preds["candidate"].iloc[0])
        for key in ["base_policy", "readout", "scope", "feature_mode", "param"]:
            row["readout_scope" if key == "scope" else key] = meta.get(key)
        rows.append(row)
    return pd.DataFrame(rows)


def conformance(eval_df: pd.DataFrame, rank: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate, sub in eval_df.groupby("candidate"):
        lookup = {row["scope"]: row for _, row in sub.iterrows()}
        overall = lookup["overall"]
        pooled = lookup["12+18+24m"]
        h18 = lookup["18m"]
        h24 = lookup["24m"]
        long = lookup["18+24m"]
        ok = (
            overall["delta_rf_minus_kg"] > 0
            and overall["wilcoxon_p_kg_less"] < 0.05
            and pooled["delta_rf_minus_kg"] > 0
            and pooled["wilcoxon_p_kg_less"] < 0.05
            and h18["delta_rf_minus_kg"] > 0
            and h18["wilcoxon_p_kg_less"] < 0.05
            and h24["delta_rf_minus_kg"] > 0
            and h24["wilcoxon_p_kg_less"] < 0.05
            and h24["spearman_pred_true_rho"] > 0
            and h24["spearman_pred_true_p"] < 0.05
        )
        selected = bool(rank.loc[rank["candidate"].eq(candidate), "validation_selected"].any())
        rows.append(
            {
                "candidate": candidate,
                "validation_selected": selected,
                "fully_removes_previous_limits": bool(ok),
                "overall_delta_rf_minus_kg": overall["delta_rf_minus_kg"],
                "overall_p": overall["wilcoxon_p_kg_less"],
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
    return pd.DataFrame(rows).sort_values(
        ["fully_removes_previous_limits", "validation_selected", "pooled_12_18_24_delta", "h24_delta"],
        ascending=[False, False, False, False],
    )


def make_figures(conf: pd.DataFrame, eval_df: pd.DataFrame) -> None:
    top = conf.head(8)["candidate"].tolist()
    scopes = ["overall", "12+18+24m", "18m", "24m", "18+24m"]
    plot = eval_df[eval_df["candidate"].isin(top) & eval_df["scope"].isin(scopes)]
    fig, ax = plt.subplots(figsize=(14, 6.5))
    width = 0.10
    for i, cand in enumerate(top):
        sub = plot[plot["candidate"].eq(cand)].set_index("scope").loc[scopes].reset_index()
        x = np.arange(len(scopes)) + i * width
        ax.bar(x, sub["delta_rf_minus_kg"], width=width, label=cand[:42])
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(np.arange(len(scopes)) + width * max(len(top) - 1, 0) / 2, scopes)
    ax.set_ylabel("RF MAE - V6 readout MAE")
    ax.set_title("V6 feature-prior gate: test margins")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_v6_feature_prior_gate_margins.png", dpi=180)
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    feature_modes = ["pred_only", "clinical", "kg", "all"]
    bases = {mode: (build_validation_base(mode), build_test_base(mode)) for mode in feature_modes}
    specs = candidate_specs()
    pred_frames = []
    eval_frames = []
    rank_rows = []
    for idx, spec in enumerate(specs):
        name = candidate_name(spec)
        try:
            preds, meta = predict_candidate(spec, bases)
            ev = evaluate(preds, meta)
            proxy = validation_proxy(spec, bases)
            pred_frames.append(preds)
            eval_frames.append(ev)
            rank_rows.append({"candidate": name, **spec, **proxy, "status": "ok", "error": ""})
        except Exception as exc:
            rank_rows.append({"candidate": name, **spec, "status": "error", "error": str(exc)[:300]})
        if (idx + 1) % 50 == 0:
            print(f"[v6] evaluated {idx + 1}/{len(specs)}", flush=True)
    rank = pd.DataFrame(rank_rows)
    ok = rank[rank["status"].eq("ok")].copy()
    ok["validation_objective"] = (
        ok["val_mae_proxy"]
        - 0.25 * ok["val_delta_vs_rf_proxy"]
        - 0.45 * ok["val_long_delta_vs_rf_proxy"].fillna(0)
        - 0.20 * ok["val_24_delta_vs_rf_proxy"].fillna(0)
        - 0.02 * ok["val_24_rho_proxy"].fillna(0)
    )
    selected = str(ok.sort_values(["validation_objective", "val_mae_proxy"]).iloc[0]["candidate"]) if not ok.empty else ""
    rank = rank.merge(ok[["candidate", "validation_objective"]], on="candidate", how="left")
    rank["validation_selected"] = rank["candidate"].eq(selected)
    rank.to_csv(TABLES / "kg_v6_validation_candidate_rank.csv", index=False, encoding="utf-8-sig")
    preds_all = pd.concat(pred_frames, ignore_index=True)
    eval_all = pd.concat(eval_frames, ignore_index=True)
    preds_all.to_csv(PRED / "kg_v6_all_candidate_predictions.csv", index=False, encoding="utf-8-sig")
    eval_all.to_csv(TABLES / "kg_v6_test_evaluation_by_scope.csv", index=False, encoding="utf-8-sig")
    conf = conformance(eval_all, rank)
    conf.to_csv(TABLES / "kg_v6_limit_removal_audit.csv", index=False, encoding="utf-8-sig")
    make_figures(conf, eval_all)
    provenance = {
        "created_by": "honest_kg_latentnet_v6_feature_prior_gate.py",
        "integrity_note": "V6 trains fold-local validation gates/stacks using prediction features plus clinical/KG tabular features. It is exploratory because prior test audits were already inspected.",
        "test_set_used_for_training": False,
        "fold_local_gate_training": True,
        "uses_rf_as_auxiliary_readout": True,
        "candidate_count": int(len(specs)),
        "validation_selected_candidate": selected,
    }
    (OUT / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(OUT), "selected": selected, "conformance": conf.head(12).to_dict(orient="records")}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
