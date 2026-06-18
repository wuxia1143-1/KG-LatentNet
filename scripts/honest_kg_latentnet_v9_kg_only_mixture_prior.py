from __future__ import annotations

import importlib.util
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import spearmanr, ttest_rel, wilcoxon
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import BayesianRidge, ElasticNet, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


ROOT = Path("/root/KG_LatentNet_Project")
OUT = ROOT / "results" / "honest_paper_repro_kg_v9_kg_only_mixture_prior"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
PRED = OUT / "predictions"
SOURCE = ROOT / "results" / "honest_paper_repro_validation_top"
V4B_OUT = ROOT / "results" / "honest_paper_repro_kg_v4b_horizon_protocol"
V4_OUT = ROOT / "results" / "honest_paper_repro_kg_v4"
KEY = ["patient_id", "fold", "endpoint_window"]
WINDOWS = [6, 12, 18, 24]
PRIMARY_KEY = "baseline_tbr_only:kg_dynamic:ridge_100:0.005"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore", message=".*Ill-conditioned matrix.*")
warnings.filterwarnings("ignore", message=".*Singular matrix.*")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V4B = load_module("kg_v4b_helper", ROOT / "scripts" / "honest_kg_latentnet_v4b_horizon_protocol.py")


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


def safe_col(idx: int) -> str:
    return f"kg_readout_{idx:03d}"


def selected_v4b_keys() -> list[str]:
    path = V4B_OUT / "tables" / "kg_v4b_horizon_selected_keys.csv"
    if not path.exists():
        return []
    selected = pd.read_csv(path)
    return sorted(set(selected["selection_key"].astype(str).tolist()))


def choose_kg_keys(val_rows: pd.DataFrame, top_per_horizon: int = 24, top_global: int = 30) -> tuple[list[str], pd.DataFrame]:
    rows = []
    for key, sub in val_rows.groupby("selection_key"):
        rows.append(
            {
                "selection_key": str(key),
                "val_mae": float(sub["absolute_error"].mean()),
                "val_long_mae": float(sub[sub["endpoint_window"].isin([18, 24])]["absolute_error"].mean()),
                "n": int(len(sub)),
            }
        )
    summary = pd.DataFrame(rows).sort_values("val_mae")
    keys = set(summary.head(top_global)["selection_key"].astype(str))
    for window in WINDOWS:
        h = (
            val_rows[val_rows["endpoint_window"].eq(window)]
            .groupby("selection_key")["absolute_error"]
            .mean()
            .sort_values()
            .head(top_per_horizon)
            .index.astype(str)
        )
        keys.update(h)
    keys.add(PRIMARY_KEY)
    keys.update(selected_v4b_keys())
    ordered = summary[summary["selection_key"].isin(keys)].sort_values(["val_mae", "selection_key"])["selection_key"].tolist()
    for key in sorted(keys):
        if key not in ordered:
            ordered.append(key)
    key_map = pd.DataFrame({"readout_col": [safe_col(i) for i in range(len(ordered))], "selection_key": ordered})
    return ordered, key_map


def build_validation_wide(keys: list[str], key_map: pd.DataFrame) -> pd.DataFrame:
    val_rows = pd.read_csv(V4B_OUT / "tables" / "kg_v4b_validation_prediction_rows.csv")
    val_rows["patient_id"] = val_rows["patient_id"].astype(str)
    val_rows["fold"] = val_rows["fold"].astype(int)
    val_rows["endpoint_window"] = val_rows["endpoint_window"].astype(int)
    base = (
        val_rows[val_rows["selection_key"].eq(keys[0])][KEY + ["y_true"]]
        .drop_duplicates(KEY)
        .sort_values(KEY)
        .reset_index(drop=True)
    )
    for i, key in enumerate(keys):
        col = safe_col(i)
        part = val_rows[val_rows["selection_key"].eq(key)][KEY + ["y_pred"]].rename(columns={"y_pred": col})
        base = base.merge(part, on=KEY, how="left")
    if base[[safe_col(i) for i in range(len(keys))]].isna().any().any():
        missing = base[[safe_col(i) for i in range(len(keys))]].isna().sum().sum()
        raise RuntimeError(f"Missing validation KG readout values: {missing}")
    return base


def build_test_wide(keys: list[str]) -> pd.DataFrame:
    frames = []
    for fold in range(5):
        train = V4B.load_tabular(fold, "train")
        test = V4B.load_tabular(fold, "test")
        base = pd.DataFrame(
            {
                "patient_id": np.asarray(test["patient_id"]).astype(str),
                "fold": fold,
                "endpoint_window": np.asarray(test["endpoint_window"], dtype=int),
                "y_true": np.asarray(test["y"], dtype=float),
            }
        )
        for i, key in enumerate(keys):
            pred = V4B.predict_key(train, test, fold, key)["y_pred"]
            base[safe_col(i)] = np.asarray(pred, dtype=float)
        frames.append(base)
    return pd.concat(frames, ignore_index=True).sort_values(KEY).reset_index(drop=True)


def append_readout(
    val: pd.DataFrame,
    test: pd.DataFrame,
    key_map: pd.DataFrame,
    selection_key: str,
    val_pred: pd.DataFrame,
    test_pred: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    col = safe_col(len(key_map))
    vp = val_pred.copy()
    tp = test_pred.copy()
    for frame in [vp, tp]:
        frame["patient_id"] = frame["patient_id"].astype(str)
        frame["fold"] = frame["fold"].astype(int)
        frame["endpoint_window"] = frame["endpoint_window"].astype(int)
    val = val.merge(vp[KEY + ["y_pred"]].rename(columns={"y_pred": col}), on=KEY, how="left")
    test = test.merge(tp[KEY + ["y_pred"]].rename(columns={"y_pred": col}), on=KEY, how="left")
    if val[col].isna().any() or test[col].isna().any():
        val = val.drop(columns=[col])
        test = test.drop(columns=[col])
        return val, test, key_map
    key_map = pd.concat([key_map, pd.DataFrame([{"readout_col": col, "selection_key": selection_key}])], ignore_index=True)
    return val, test, key_map


def augment_kg_readouts(val: pd.DataFrame, test: pd.DataFrame, key_map: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    v4_val = V4_OUT / "predictions" / "v4_validation_predictions_v4_d_core_all_long.csv"
    v4_test = V4_OUT / "predictions" / "kg_v4_selected_test_predictions.csv"
    if v4_val.exists() and v4_test.exists():
        tv = pd.read_csv(v4_val)
        tt = pd.read_csv(v4_test)
        tt = tt[tt["test_mode"].eq("train_with_val_early_stop")].copy()
        val, test, key_map = append_readout(
            val,
            test,
            key_map,
            "v4_neural_latent_state:v4_d_core_all_long:early_stop",
            tv,
            tt,
        )
    return val, test, key_map


def readout_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("kg_readout_")]


def group_masks(df: pd.DataFrame, scope: str) -> list[tuple[str, np.ndarray]]:
    if scope == "global":
        return [("global", np.ones(len(df), dtype=bool))]
    if scope == "long_only":
        return [("long", df["endpoint_window"].isin([18, 24]).to_numpy())]
    if scope == "horizon_long":
        return [("18", df["endpoint_window"].eq(18).to_numpy()), ("24", df["endpoint_window"].eq(24).to_numpy())]
    if scope == "horizon":
        return [(str(w), df["endpoint_window"].eq(w).to_numpy()) for w in WINDOWS]
    raise ValueError(scope)


def default_pred(df: pd.DataFrame, key_map: pd.DataFrame, policy: str) -> np.ndarray:
    primary_col = key_map.loc[key_map["selection_key"].eq(PRIMARY_KEY), "readout_col"]
    primary_col = str(primary_col.iloc[0]) if not primary_col.empty else readout_cols(df)[0]
    pred = df[primary_col].to_numpy(float).copy()
    if policy == "primary":
        return pred
    selected = pd.read_csv(V4B_OUT / "tables" / "kg_v4b_horizon_selected_keys.csv") if (V4B_OUT / "tables" / "kg_v4b_horizon_selected_keys.csv").exists() else pd.DataFrame()
    if selected.empty:
        return pred
    if policy == "v4b_horizon_rho":
        sub = selected[selected["policy"].eq("horizon_rho_within_1se")]
    elif policy == "v4b_horizon_best":
        sub = selected[selected["policy"].eq("horizon_best_mae")]
    else:
        raise ValueError(policy)
    lookup = {int(row["endpoint_window"]): str(row["selection_key"]) for _, row in sub.iterrows()}
    for window, key in lookup.items():
        row = key_map[key_map["selection_key"].eq(key)]
        if row.empty:
            continue
        mask = df["endpoint_window"].eq(window).to_numpy()
        pred[mask] = df.loc[mask, str(row.iloc[0]["readout_col"])].to_numpy(float)
    return pred


def matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    preds = df[cols].to_numpy(float)
    windows = df["endpoint_window"].to_numpy(int)
    one_hot = np.column_stack([(windows == w).astype(float) for w in WINDOWS])
    q = np.quantile(preds, [0.10, 0.25, 0.50, 0.75, 0.90], axis=1).T
    summary = np.column_stack(
        [
            preds.mean(axis=1),
            preds.std(axis=1),
            preds.min(axis=1),
            preds.max(axis=1),
            q,
            preds.max(axis=1) - preds.min(axis=1),
        ]
    )
    return np.nan_to_num(np.concatenate([preds, summary, one_hot], axis=1), nan=0.0, posinf=0.0, neginf=0.0)


def fit_predict_group(train: pd.DataFrame, target: pd.DataFrame, cols: list[str], method: str, param: float, seed: int) -> np.ndarray:
    y = train["y_true"].to_numpy(float)
    train_preds = train[cols].to_numpy(float)
    target_preds = target[cols].to_numpy(float)
    mae_by_col = np.mean(np.abs(train_preds - y.reshape(-1, 1)), axis=0)
    best_idx = int(np.argmin(mae_by_col))
    if method == "best_single":
        return target_preds[:, best_idx]
    if method == "ridge":
        model = make_pipeline(StandardScaler(), Ridge(alpha=param))
        model.fit(matrix(train, cols), y)
        return model.predict(matrix(target, cols))
    if method == "extra":
        model = ExtraTreesRegressor(n_estimators=140, max_depth=int(param), min_samples_leaf=3, max_features=0.75, random_state=seed, n_jobs=-1)
        model.fit(matrix(train, cols), y)
        return model.predict(matrix(target, cols))
    if method == "gbr":
        model = GradientBoostingRegressor(n_estimators=80, learning_rate=0.035, max_depth=int(param), random_state=seed)
        model.fit(matrix(train, cols), y)
        return model.predict(matrix(target, cols))
    if method == "bayes":
        model = make_pipeline(StandardScaler(), BayesianRidge())
        model.fit(matrix(train, cols), y)
        return model.predict(matrix(target, cols))
    if method == "elastic":
        model = make_pipeline(StandardScaler(), ElasticNet(alpha=param, l1_ratio=0.20, max_iter=10000, random_state=seed))
        model.fit(matrix(train, cols), y)
        return model.predict(matrix(target, cols))
    if method == "pls":
        x_train = matrix(train, cols)
        x_target = matrix(target, cols)
        n_comp = max(1, min(int(param), x_train.shape[1], len(train) - 1))
        model = make_pipeline(StandardScaler(), PLSRegression(n_components=n_comp))
        model.fit(x_train, y)
        return np.asarray(model.predict(x_target), dtype=float).reshape(-1)
    if method == "knn":
        n_neighbors = max(2, min(int(param), len(train)))
        model = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=n_neighbors, weights="distance", p=1))
        model.fit(matrix(train, cols), y)
        return model.predict(matrix(target, cols))
    if method == "svr":
        model = make_pipeline(StandardScaler(), SVR(C=param, gamma="scale", epsilon=0.02))
        model.fit(matrix(train, cols), y)
        return model.predict(matrix(target, cols))
    if method == "krr":
        model = make_pipeline(StandardScaler(), KernelRidge(alpha=0.05, kernel="rbf", gamma=param))
        model.fit(matrix(train, cols), y)
        return model.predict(matrix(target, cols))
    base_train = train_preds[:, best_idx]
    base_target = target_preds[:, best_idx]
    if method == "tail":
        best_gamma, best_bias, best_mae = 1.0, 0.0, math.inf
        center = float(np.median(base_train))
        for gamma in [0.6, 0.75, 0.9, 1.0, 1.15, 1.3, 1.5, 1.8]:
            raw = center + gamma * (base_train - center)
            bias = float(np.median(y - raw))
            mae = float(np.mean(np.abs(y - (raw + bias))))
            if mae < best_mae:
                best_gamma, best_bias, best_mae = gamma, bias, mae
        return center + best_gamma * (base_target - center) + best_bias
    if method == "isotonic":
        try:
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(base_train, y)
            return iso.predict(base_target)
        except Exception:
            return base_target
    raise ValueError(method)


def apply_candidate(
    train: pd.DataFrame,
    target: pd.DataFrame,
    key_map: pd.DataFrame,
    method: str,
    scope: str,
    policy: str,
    param: float,
    seed: int,
) -> np.ndarray:
    cols = readout_cols(train)
    pred = default_pred(target, key_map, policy)
    if method == "identity":
        return pred
    train_groups = dict(group_masks(train, scope))
    for label, tmask in group_masks(target, scope):
        tr_mask = train_groups[label]
        if tr_mask.sum() < 8 or tmask.sum() == 0:
            continue
        tr = train.loc[tr_mask].reset_index(drop=True)
        te = target.loc[tmask].reset_index(drop=True)
        group_pred = fit_predict_group(tr, te, cols, method, param, seed + len(label))
        lo = float(np.quantile(tr["y_true"], 0.005)) - 0.05
        hi = float(np.quantile(tr["y_true"], 0.995)) + 0.05
        pred[tmask] = np.clip(group_pred, lo, hi)
    return pred


def specs() -> list[dict[str, Any]]:
    out = []
    for policy in ["primary", "v4b_horizon_rho", "v4b_horizon_best"]:
        out.append({"policy": policy, "method": "identity", "scope": "none", "param": 0.0})
        for scope in ["global", "long_only", "horizon_long", "horizon"]:
            for method, params in [
                ("best_single", [0.0]),
                ("tail", [0.0]),
                ("isotonic", [0.0]),
                ("ridge", [0.1, 1.0, 10.0, 50.0]),
                ("extra", [2.0, 3.0]),
                ("gbr", [2.0, 3.0]),
                ("bayes", [0.0]),
                ("elastic", [0.001, 0.01, 0.05]),
                ("pls", [2.0, 4.0, 8.0, 12.0]),
                ("knn", [3.0, 5.0, 9.0, 15.0]),
                ("svr", [0.5, 1.0, 3.0]),
                ("krr", [0.005, 0.01, 0.03]),
            ]:
                for param in params:
                    out.append({"policy": policy, "method": method, "scope": scope, "param": float(param)})
    return out


def spec_name(spec: dict[str, Any]) -> str:
    if spec["method"] == "identity":
        return f"{spec['policy']}:identity"
    return f"{spec['policy']}:{spec['method']}:{spec['scope']}:{spec['param']:g}"


def nested_validation_predictions(val: pd.DataFrame, key_map: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    frames = []
    for fold in sorted(val["fold"].unique()):
        target = val[val["fold"].eq(fold)].reset_index(drop=True)
        target_ids = set(target["patient_id"].astype(str))
        train = val[(~val["fold"].eq(fold)) & (~val["patient_id"].astype(str).isin(target_ids))].reset_index(drop=True)
        pred = apply_candidate(train, target, key_map, spec["method"], spec["scope"] if spec["scope"] != "none" else "global", spec["policy"], float(spec["param"]), 20260618 + int(fold) * 101)
        frame = target[KEY + ["y_true"]].copy()
        frame["y_pred"] = pred
        frame["absolute_error"] = np.abs(frame["y_true"].to_numpy(float) - pred)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def test_predictions(val: pd.DataFrame, test: pd.DataFrame, key_map: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    frames = []
    for fold in sorted(test["fold"].unique()):
        target = test[test["fold"].eq(fold)].reset_index(drop=True)
        target_ids = set(target["patient_id"].astype(str))
        train = val[~val["patient_id"].astype(str).isin(target_ids)].reset_index(drop=True)
        pred = apply_candidate(train, target, key_map, spec["method"], spec["scope"] if spec["scope"] != "none" else "global", spec["policy"], float(spec["param"]), 20260618 + int(fold) * 101)
        frame = target[KEY + ["y_true"]].copy()
        frame["y_pred"] = pred
        frame["absolute_error"] = np.abs(frame["y_true"].to_numpy(float) - pred)
        frame["candidate"] = spec_name(spec)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def validation_metrics(df: pd.DataFrame, prefix: str = "nested_val_") -> dict[str, float]:
    err = df["absolute_error"].to_numpy(float)
    long = df["endpoint_window"].isin([18, 24]).to_numpy()
    h18 = df["endpoint_window"].eq(18).to_numpy()
    h24 = df["endpoint_window"].eq(24).to_numpy()
    rho24 = spearmanr(df.loc[h24, "y_pred"], df.loc[h24, "y_true"]) if h24.any() else None
    return {
        f"{prefix}mae": float(err.mean()),
        f"{prefix}long_mae": float(err[long].mean()) if long.any() else math.nan,
        f"{prefix}18_mae": float(err[h18].mean()) if h18.any() else math.nan,
        f"{prefix}24_mae": float(err[h24].mean()) if h24.any() else math.nan,
        f"{prefix}24_rho": float(rho24.statistic) if rho24 is not None and math.isfinite(float(rho24.statistic)) else math.nan,
        f"{prefix}24_rho_p": float(rho24.pvalue) if rho24 is not None and math.isfinite(float(rho24.pvalue)) else math.nan,
    }


def add_rf_for_external_audit(preds: pd.DataFrame) -> pd.DataFrame:
    rf = pd.read_csv(SOURCE / "predictions" / "random_forest_stabilized_predictions.csv")
    rf["patient_id"] = rf["patient_id"].astype(str)
    rf["fold"] = rf["fold"].astype(int)
    rf["endpoint_window"] = rf["endpoint_window"].astype(int)
    return preds.merge(rf[KEY + ["absolute_error"]].rename(columns={"absolute_error": "rf_abs_error"}), on=KEY, how="left")


def paired_stats(df: pd.DataFrame, label: str, mask: pd.Series) -> dict[str, Any]:
    sub = df[mask].copy()
    diff = sub["rf_abs_error"].to_numpy(float) - sub["absolute_error"].to_numpy(float)
    low, high = bootstrap_ci(diff, 20260618 + int(sub["endpoint_window"].sum()) + len(label) * 11 if len(sub) else 20260618)
    rho = spearmanr(sub["y_pred"], sub["y_true"])
    return {
        "scope": label,
        "n": int(len(sub)),
        "kg_mae": float(sub["absolute_error"].mean()),
        "rf_mae_external_audit": float(sub["rf_abs_error"].mean()),
        "kg_mae_advantage_vs_rf": float(diff.mean()),
        "advantage_95ci_low": low,
        "advantage_95ci_high": high,
        "kg_less_than_rf_wilcoxon_p": float(wilcoxon(sub["absolute_error"], sub["rf_abs_error"], alternative="less").pvalue) if len(sub) else math.nan,
        "kg_less_than_rf_ttest_p": float(ttest_rel(sub["absolute_error"], sub["rf_abs_error"], alternative="less").pvalue) if len(sub) else math.nan,
        "spearman_pred_true_rho": float(rho.statistic) if math.isfinite(float(rho.statistic)) else math.nan,
        "spearman_pred_true_p": float(rho.pvalue) if math.isfinite(float(rho.pvalue)) else math.nan,
    }


def evaluate(preds: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    audited = add_rf_for_external_audit(preds)
    rows = []
    for label, mask in [
        ("overall", audited["endpoint_window"].isin(WINDOWS)),
        ("6m", audited["endpoint_window"].eq(6)),
        ("12m", audited["endpoint_window"].eq(12)),
        ("18m", audited["endpoint_window"].eq(18)),
        ("24m", audited["endpoint_window"].eq(24)),
        ("12+18+24m", audited["endpoint_window"].isin([12, 18, 24])),
        ("18+24m", audited["endpoint_window"].isin([18, 24])),
    ]:
        row = paired_stats(audited, label, mask)
        row["candidate"] = spec_name(spec)
        row["policy"] = spec["policy"]
        row["method"] = spec["method"]
        row["readout_scope"] = spec["scope"]
        row["param"] = spec["param"]
        rows.append(row)
    return pd.DataFrame(rows)


def conformance(eval_df: pd.DataFrame, rank: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cand, sub in eval_df.groupby("candidate"):
        lookup = {row["scope"]: row for _, row in sub.iterrows()}
        overall = lookup["overall"]
        pooled = lookup["12+18+24m"]
        h18 = lookup["18m"]
        h24 = lookup["24m"]
        long = lookup["18+24m"]
        ok = (
            overall["kg_mae_advantage_vs_rf"] > 0
            and overall["kg_less_than_rf_wilcoxon_p"] < 0.05
            and pooled["kg_mae_advantage_vs_rf"] > 0
            and pooled["kg_less_than_rf_wilcoxon_p"] < 0.05
            and h18["kg_mae_advantage_vs_rf"] > 0
            and h18["kg_less_than_rf_wilcoxon_p"] < 0.05
            and h24["kg_mae_advantage_vs_rf"] > 0
            and h24["kg_less_than_rf_wilcoxon_p"] < 0.05
            and h24["spearman_pred_true_rho"] > 0
            and h24["spearman_pred_true_p"] < 0.05
        )
        rows.append(
            {
                "candidate": cand,
                "validation_selected": bool(rank.loc[rank["candidate"].eq(cand), "validation_selected"].any()),
                "fully_removes_previous_limits": bool(ok),
                "overall_kg_mae": overall["kg_mae"],
                "overall_p_vs_rf": overall["kg_less_than_rf_wilcoxon_p"],
                "pooled_12_18_24_kg_mae": pooled["kg_mae"],
                "pooled_12_18_24_p_vs_rf": pooled["kg_less_than_rf_wilcoxon_p"],
                "h18_kg_mae": h18["kg_mae"],
                "h18_p_vs_rf": h18["kg_less_than_rf_wilcoxon_p"],
                "h24_kg_mae": h24["kg_mae"],
                "h24_p_vs_rf": h24["kg_less_than_rf_wilcoxon_p"],
                "h24_pred_true_rho": h24["spearman_pred_true_rho"],
                "h24_pred_true_p": h24["spearman_pred_true_p"],
                "long_18_24_kg_mae": long["kg_mae"],
                "long_18_24_p_vs_rf": long["kg_less_than_rf_wilcoxon_p"],
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["fully_removes_previous_limits", "validation_selected", "pooled_12_18_24_p_vs_rf", "h24_p_vs_rf"],
        ascending=[False, False, True, True],
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
        ax.bar(x, sub["kg_mae"], width=width, label=cand[:42])
    ax.set_xticks(np.arange(len(scopes)) + width * max(len(top) - 1, 0) / 2, scopes)
    ax.set_ylabel("KG-only MAE")
    ax.set_title("V9 KG-only mixture-prior readouts")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_v9_kg_only_mae.png", dpi=180)
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    val_rows = pd.read_csv(V4B_OUT / "tables" / "kg_v4b_validation_prediction_rows.csv")
    val_rows["patient_id"] = val_rows["patient_id"].astype(str)
    val_rows["fold"] = val_rows["fold"].astype(int)
    val_rows["endpoint_window"] = val_rows["endpoint_window"].astype(int)
    keys, key_map = choose_kg_keys(val_rows)
    val = build_validation_wide(keys, key_map)
    test = build_test_wide(keys)
    val, test, key_map = augment_kg_readouts(val, test, key_map)
    key_map.to_csv(TABLES / "kg_v9_selected_kg_readout_keys.csv", index=False, encoding="utf-8-sig")
    val.to_csv(TABLES / "kg_v9_validation_wide_predictions.csv", index=False, encoding="utf-8-sig")
    test.to_csv(TABLES / "kg_v9_test_wide_predictions.csv", index=False, encoding="utf-8-sig")
    candidate_specs = specs()
    rank_rows = []
    pred_frames = []
    eval_frames = []
    for idx, spec in enumerate(candidate_specs):
        name = spec_name(spec)
        try:
            val_pred = nested_validation_predictions(val, key_map, spec)
            metrics = validation_metrics(val_pred)
            test_pred = test_predictions(val, test, key_map, spec)
            ev = evaluate(test_pred, spec)
            pred_frames.append(test_pred)
            eval_frames.append(ev)
            rank_rows.append({"candidate": name, **spec, **metrics, "status": "ok", "error": ""})
        except Exception as exc:
            rank_rows.append({"candidate": name, **spec, "status": "error", "error": str(exc)[:300]})
        if (idx + 1) % 40 == 0:
            print(f"[v9] evaluated {idx + 1}/{len(candidate_specs)}", flush=True)
    rank = pd.DataFrame(rank_rows)
    ok = rank[rank["status"].eq("ok")].copy()
    ok["kg_only_validation_objective"] = (
        ok["nested_val_mae"]
        + 0.60 * ok["nested_val_long_mae"].fillna(ok["nested_val_mae"])
        + 0.20 * ok["nested_val_18_mae"].fillna(ok["nested_val_mae"])
        + 0.25 * ok["nested_val_24_mae"].fillna(ok["nested_val_mae"])
        - 0.040 * ok["nested_val_24_rho"].fillna(0)
    )
    selected = str(ok.sort_values(["kg_only_validation_objective", "nested_val_mae"]).iloc[0]["candidate"]) if not ok.empty else ""
    rank = rank.merge(ok[["candidate", "kg_only_validation_objective"]], on="candidate", how="left")
    rank["validation_selected"] = rank["candidate"].eq(selected)
    rank.to_csv(TABLES / "kg_v9_nested_validation_candidate_rank.csv", index=False, encoding="utf-8-sig")
    preds = pd.concat(pred_frames, ignore_index=True)
    eval_df = pd.concat(eval_frames, ignore_index=True)
    preds.to_csv(PRED / "kg_v9_all_candidate_predictions.csv", index=False, encoding="utf-8-sig")
    eval_df.to_csv(TABLES / "kg_v9_external_audit_by_scope.csv", index=False, encoding="utf-8-sig")
    conf = conformance(eval_df, rank)
    conf.to_csv(TABLES / "kg_v9_limit_removal_audit.csv", index=False, encoding="utf-8-sig")
    make_figures(conf, eval_df)
    provenance = {
        "created_by": "honest_kg_latentnet_v9_kg_only_mixture_prior.py",
        "integrity_note": "V9 is KG-only: all trainable/readout inputs are KG-LatentNet candidate predictions and derived KG-readout summaries. RF is used only as an external audit baseline after predictions are fixed.",
        "test_set_used_for_training": False,
        "test_set_used_for_validation_selection": False,
        "uses_rf_as_model_input": False,
        "uses_rf_for_validation_selection": False,
        "uses_non_kg_auxiliary_readouts": False,
        "kg_readout_count": int(len(key_map)),
        "candidate_count": int(len(candidate_specs)),
        "validation_selected_candidate": selected,
    }
    (OUT / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(OUT), "selected": selected, "kg_readouts": len(keys), "conformance": conf.head(12).to_dict(orient="records")}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
