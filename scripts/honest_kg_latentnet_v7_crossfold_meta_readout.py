from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import spearmanr, ttest_rel, wilcoxon
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path("/root/KG_LatentNet_Project")
SOURCE = ROOT / "results" / "honest_paper_repro_validation_top"
V4B = ROOT / "results" / "honest_paper_repro_kg_v4b_horizon_protocol"
OUT = ROOT / "results" / "honest_paper_repro_kg_v7_crossfold_meta_readout"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
PRED = OUT / "predictions"
KEY = ["patient_id", "fold", "endpoint_window"]
WINDOWS = [6, 12, 18, 24]
PRIMARY_KEY = "baseline_tbr_only:kg_dynamic:ridge_100:0.005"
OPTIONAL_AUX = ["xgboost", "hyperimts"]


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


def best_candidate_id(model_name: str) -> int | None:
    path = ROOT / "results" / "tables" / "tuning" / "validation_tuning_best_by_model.csv"
    if not path.exists():
        return None
    best = pd.read_csv(path, encoding="utf-8-sig")
    row = best[best["model_name"].eq(model_name)]
    if row.empty:
        return None
    return int(row.iloc[0]["candidate_id"])


def load_tuning_validation(model_name: str) -> pd.DataFrame | None:
    candidate_id = best_candidate_id(model_name)
    if candidate_id is None:
        return None
    frames = []
    for fold in range(5):
        path = ROOT / "results" / "predictions" / "tuning" / f"{model_name}_fold{fold}_candidate{candidate_id}_val_predictions.csv"
        if not path.exists():
            return None
        frame = pd.read_csv(path)
        frame["fold"] = fold
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    out["patient_id"] = out["patient_id"].astype(str)
    out["fold"] = out["fold"].astype(int)
    out["endpoint_window"] = out["endpoint_window"].astype(int)
    return out


def load_test_prediction(model_name: str) -> pd.DataFrame | None:
    path = SOURCE / "predictions" / f"{model_name}_stabilized_predictions.csv"
    if not path.exists():
        return None
    out = pd.read_csv(path)
    out["patient_id"] = out["patient_id"].astype(str)
    out["fold"] = out["fold"].astype(int)
    out["endpoint_window"] = out["endpoint_window"].astype(int)
    return out


def selected_v4b_keys() -> dict[int, str]:
    selected = pd.read_csv(V4B / "tables" / "kg_v4b_horizon_selected_keys.csv")
    sub = selected[selected["policy"].eq("horizon_rho_within_1se")]
    return {int(row["endpoint_window"]): str(row["selection_key"]) for _, row in sub.iterrows()}


def build_validation_base() -> tuple[pd.DataFrame, list[str]]:
    rows = pd.read_csv(V4B / "tables" / "kg_v4b_validation_prediction_rows.csv")
    rows["patient_id"] = rows["patient_id"].astype(str)
    rows["fold"] = rows["fold"].astype(int)
    rows["endpoint_window"] = rows["endpoint_window"].astype(int)
    primary = rows[rows["selection_key"].eq(PRIMARY_KEY)][KEY + ["y_true", "y_pred"]].rename(columns={"y_pred": "pred_kg"})
    keys = selected_v4b_keys()
    long_parts = []
    for window, key in keys.items():
        long_parts.append(rows[rows["endpoint_window"].eq(window) & rows["selection_key"].eq(key)][KEY + ["y_pred"]])
    v4b = pd.concat(long_parts, ignore_index=True).rename(columns={"y_pred": "pred_v4b"})
    rf = load_tuning_validation("random_forest")
    assert rf is not None
    base = primary.merge(v4b, on=KEY).merge(rf[KEY + ["y_pred", "absolute_error"]].rename(columns={"y_pred": "pred_rf", "absolute_error": "rf_abs_error"}), on=KEY)
    readouts = ["pred_kg", "pred_v4b", "pred_rf"]
    for model_name in OPTIONAL_AUX:
        aux = load_tuning_validation(model_name)
        if aux is None:
            continue
        col = f"pred_{model_name}"
        base = base.merge(aux[KEY + ["y_pred"]].rename(columns={"y_pred": col}), on=KEY, how="left")
        if base[col].notna().all():
            readouts.append(col)
        else:
            base = base.drop(columns=[col])
    return base, readouts


def build_test_base(readouts: list[str]) -> pd.DataFrame:
    kg = pd.read_csv(SOURCE / "predictions" / "kg_latentnet_calibrated_predictions.csv")
    v4b = pd.read_csv(V4B / "predictions" / "horizon_rho_within_1se_train_only_predictions.csv")
    rf = load_test_prediction("random_forest")
    assert rf is not None
    for frame in [kg, v4b, rf]:
        frame["patient_id"] = frame["patient_id"].astype(str)
        frame["fold"] = frame["fold"].astype(int)
        frame["endpoint_window"] = frame["endpoint_window"].astype(int)
    base = kg[KEY + ["y_true", "y_pred"]].rename(columns={"y_pred": "pred_kg"})
    base = base.merge(v4b[KEY + ["y_pred"]].rename(columns={"y_pred": "pred_v4b"}), on=KEY)
    base = base.merge(rf[KEY + ["y_pred", "absolute_error"]].rename(columns={"y_pred": "pred_rf", "absolute_error": "rf_abs_error"}), on=KEY)
    for model_name in OPTIONAL_AUX:
        col = f"pred_{model_name}"
        if col not in readouts:
            continue
        aux = load_test_prediction(model_name)
        if aux is None:
            continue
        base = base.merge(aux[KEY + ["y_pred"]].rename(columns={"y_pred": col}), on=KEY, how="left")
    return base


def base_policy_prediction(df: pd.DataFrame, policy: str) -> np.ndarray:
    pred = df["pred_kg"].to_numpy(float).copy()
    if policy == "kg":
        return pred
    if policy == "v4b_all":
        return df["pred_v4b"].to_numpy(float)
    if policy == "hybrid24":
        mask = df["endpoint_window"].eq(24).to_numpy()
    elif policy == "hybrid18_24":
        mask = df["endpoint_window"].isin([18, 24]).to_numpy()
    else:
        raise ValueError(policy)
    pred[mask] = df.loc[mask, "pred_v4b"].to_numpy(float)
    return pred


def design_matrix(df: pd.DataFrame, readouts: list[str]) -> np.ndarray:
    preds = df[readouts].to_numpy(float)
    mean = preds.mean(axis=1, keepdims=True)
    std = preds.std(axis=1, keepdims=True)
    minv = preds.min(axis=1, keepdims=True)
    maxv = preds.max(axis=1, keepdims=True)
    windows = df["endpoint_window"].to_numpy(int)
    one_hot = np.column_stack([(windows == w).astype(float) for w in WINDOWS])
    # Differences to RF and KG often encode where the knowledge readout diverges.
    diffs = []
    for col in readouts:
        diffs.append((df[col].to_numpy(float) - df["pred_rf"].to_numpy(float)).reshape(-1, 1))
        diffs.append((df[col].to_numpy(float) - df["pred_kg"].to_numpy(float)).reshape(-1, 1))
    return np.nan_to_num(np.concatenate([preds, mean, std, minv, maxv, one_hot, *diffs], axis=1), nan=0.0, posinf=0.0, neginf=0.0)


def group_masks(df: pd.DataFrame, scope: str) -> list[tuple[str, np.ndarray, bool]]:
    if scope == "long_only":
        return [("long", df["endpoint_window"].isin([18, 24]).to_numpy(), True)]
    if scope == "horizon_long_only":
        return [("18", df["endpoint_window"].eq(18).to_numpy(), True), ("24", df["endpoint_window"].eq(24).to_numpy(), True)]
    if scope == "horizon":
        return [(str(w), df["endpoint_window"].eq(w).to_numpy(), False) for w in WINDOWS]
    raise ValueError(scope)


def fit_predict_group(
    train: pd.DataFrame,
    target: pd.DataFrame,
    readouts: list[str],
    method: str,
    param: float,
    seed: int,
) -> np.ndarray:
    x_train = design_matrix(train, readouts)
    x_target = design_matrix(target, readouts)
    y = train["y_true"].to_numpy(float)
    pred_matrix_train = train[readouts].to_numpy(float)
    pred_matrix_target = target[readouts].to_numpy(float)
    if method == "best_single":
        mae = np.mean(np.abs(pred_matrix_train - y.reshape(-1, 1)), axis=0)
        return pred_matrix_target[:, int(np.argmin(mae))]
    if method == "ridge":
        model = make_pipeline(StandardScaler(), Ridge(alpha=param))
        model.fit(x_train, y)
        return model.predict(x_target)
    if method == "extra":
        model = ExtraTreesRegressor(n_estimators=160, max_depth=int(param), min_samples_leaf=3, max_features=0.8, random_state=seed, n_jobs=-1)
        model.fit(x_train, y)
        return model.predict(x_target)
    if method == "gbr":
        model = GradientBoostingRegressor(n_estimators=80, learning_rate=0.035, max_depth=int(param), random_state=seed)
        model.fit(x_train, y)
        return model.predict(x_target)
    target_class = np.argmin(np.abs(pred_matrix_train - y.reshape(-1, 1)), axis=1)
    if len(np.unique(target_class)) < 2:
        mae = np.mean(np.abs(pred_matrix_train - y.reshape(-1, 1)), axis=0)
        return pred_matrix_target[:, int(np.argmin(mae))]
    if method == "extra_gate":
        clf = ExtraTreesClassifier(n_estimators=160, max_depth=int(param), min_samples_leaf=3, max_features=0.8, class_weight="balanced", random_state=seed, n_jobs=-1)
    elif method == "gbr_gate":
        clf = GradientBoostingClassifier(n_estimators=80, learning_rate=0.035, max_depth=int(param), random_state=seed)
    else:
        raise ValueError(method)
    clf.fit(x_train, target_class)
    probs_raw = clf.predict_proba(x_target)
    probs = np.zeros((len(target), len(readouts)), dtype=float)
    for idx, cls in enumerate(clf.classes_):
        probs[:, int(cls)] = probs_raw[:, idx]
    row_sum = probs.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    probs /= row_sum
    return np.sum(probs * pred_matrix_target, axis=1)


def apply_candidate(
    train_meta: pd.DataFrame,
    target: pd.DataFrame,
    readouts: list[str],
    base_policy: str,
    method: str,
    scope: str,
    param: float,
    seed: int,
) -> np.ndarray:
    pred = base_policy_prediction(target, base_policy)
    for label, target_mask, keep_short in group_masks(target, scope):
        train_label_mask = dict((name, mask) for name, mask, _ in group_masks(train_meta, scope))[label]
        if train_label_mask.sum() < 8 or target_mask.sum() == 0:
            continue
        train_group = train_meta.loc[train_label_mask].reset_index(drop=True)
        target_group = target.loc[target_mask].reset_index(drop=True)
        pred_group = fit_predict_group(train_group, target_group, readouts, method, param, seed + len(label))
        lo = float(np.quantile(train_group["y_true"], 0.005)) - 0.05
        hi = float(np.quantile(train_group["y_true"], 0.995)) + 0.05
        pred[target_mask] = np.clip(pred_group, lo, hi)
    return pred


def specs() -> list[dict[str, Any]]:
    out = []
    for base_policy in ["kg", "hybrid24", "hybrid18_24", "v4b_all"]:
        out.append({"base_policy": base_policy, "method": "identity", "scope": "none", "param": 0.0})
        for scope in ["long_only", "horizon_long_only", "horizon"]:
            out.append({"base_policy": base_policy, "method": "best_single", "scope": scope, "param": 0.0})
            for alpha in [0.1, 1.0, 10.0, 50.0]:
                out.append({"base_policy": base_policy, "method": "ridge", "scope": scope, "param": alpha})
            for depth in [2.0, 3.0]:
                for method in ["extra", "gbr", "extra_gate", "gbr_gate"]:
                    out.append({"base_policy": base_policy, "method": method, "scope": scope, "param": depth})
    return out


def spec_name(spec: dict[str, Any]) -> str:
    if spec["method"] == "identity":
        return f"{spec['base_policy']}:identity"
    return f"{spec['base_policy']}:{spec['method']}:{spec['scope']}:{spec['param']:g}"


def nested_validation_predictions(val: pd.DataFrame, readouts: list[str], spec: dict[str, Any]) -> pd.DataFrame:
    frames = []
    for fold in sorted(val["fold"].unique()):
        target = val[val["fold"].eq(fold)].reset_index(drop=True)
        train = val[~val["fold"].eq(fold)].reset_index(drop=True)
        if spec["method"] == "identity":
            pred = base_policy_prediction(target, str(spec["base_policy"]))
        else:
            pred = apply_candidate(train, target, readouts, str(spec["base_policy"]), str(spec["method"]), str(spec["scope"]), float(spec["param"]), 20260618 + int(fold) * 97)
        frame = target[KEY + ["y_true", "rf_abs_error"]].copy()
        frame["y_pred"] = pred
        frame["absolute_error"] = np.abs(frame["y_true"].to_numpy(float) - pred)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def test_predictions(val: pd.DataFrame, test: pd.DataFrame, readouts: list[str], spec: dict[str, Any]) -> pd.DataFrame:
    frames = []
    for fold in sorted(test["fold"].unique()):
        target = test[test["fold"].eq(fold)].reset_index(drop=True)
        test_ids = set(target["patient_id"].astype(str))
        train = val[~val["patient_id"].astype(str).isin(test_ids)].reset_index(drop=True)
        if spec["method"] == "identity":
            pred = base_policy_prediction(target, str(spec["base_policy"]))
        else:
            pred = apply_candidate(train, target, readouts, str(spec["base_policy"]), str(spec["method"]), str(spec["scope"]), float(spec["param"]), 20260618 + int(fold) * 97)
        frame = target[KEY + ["y_true", "rf_abs_error"]].copy()
        frame["y_pred"] = pred
        frame["absolute_error"] = np.abs(frame["y_true"].to_numpy(float) - pred)
        frame["candidate"] = spec_name(spec)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def summary_metrics(df: pd.DataFrame, prefix: str = "") -> dict[str, float]:
    err = df["absolute_error"].to_numpy(float)
    rferr = df["rf_abs_error"].to_numpy(float)
    long = df["endpoint_window"].isin([18, 24]).to_numpy()
    h18 = df["endpoint_window"].eq(18).to_numpy()
    h24 = df["endpoint_window"].eq(24).to_numpy()
    rho24 = spearmanr(df.loc[h24, "y_pred"], df.loc[h24, "y_true"]) if h24.any() else None
    return {
        f"{prefix}mae": float(err.mean()),
        f"{prefix}delta_vs_rf": float(rferr.mean() - err.mean()),
        f"{prefix}long_delta_vs_rf": float(rferr[long].mean() - err[long].mean()) if long.any() else math.nan,
        f"{prefix}18_delta_vs_rf": float(rferr[h18].mean() - err[h18].mean()) if h18.any() else math.nan,
        f"{prefix}24_delta_vs_rf": float(rferr[h24].mean() - err[h24].mean()) if h24.any() else math.nan,
        f"{prefix}24_rho": float(rho24.statistic) if rho24 is not None and math.isfinite(float(rho24.statistic)) else math.nan,
        f"{prefix}24_rho_p": float(rho24.pvalue) if rho24 is not None and math.isfinite(float(rho24.pvalue)) else math.nan,
    }


def paired_stats(df: pd.DataFrame, scope: str, mask: pd.Series) -> dict[str, Any]:
    sub = df[mask].copy()
    d = sub["rf_abs_error"].to_numpy(float) - sub["absolute_error"].to_numpy(float)
    low, high = bootstrap_ci(d, 20260618 + len(scope) * 31 + int(sub["endpoint_window"].sum()) if len(sub) else 20260618)
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


def evaluate(preds: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
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
        row["candidate"] = spec_name(spec)
        row.update(spec)
        row["readout_scope"] = row.pop("scope_y", None) if "scope_y" in row else spec["scope"]
        row["scope"] = scope
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
        rows.append(
            {
                "candidate": cand,
                "validation_selected": bool(rank.loc[rank["candidate"].eq(cand), "validation_selected"].any()),
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
    ax.set_ylabel("RF MAE - V7 meta-readout MAE")
    ax.set_title("V7 cross-fold meta-readout: test margins")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_v7_crossfold_meta_margins.png", dpi=180)
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    val, readouts = build_validation_base()
    test = build_test_base(readouts)
    val.to_csv(TABLES / "kg_v7_validation_base_predictions.csv", index=False, encoding="utf-8-sig")
    test.to_csv(TABLES / "kg_v7_test_base_predictions.csv", index=False, encoding="utf-8-sig")
    candidate_specs = specs()
    rank_rows = []
    pred_frames = []
    eval_frames = []
    for idx, spec in enumerate(candidate_specs):
        name = spec_name(spec)
        try:
            val_pred = nested_validation_predictions(val, readouts, spec)
            val_metrics = summary_metrics(val_pred, "nested_val_")
            test_pred = test_predictions(val, test, readouts, spec)
            ev = evaluate(test_pred, spec)
            rank_rows.append({"candidate": name, **spec, **val_metrics, "status": "ok", "error": ""})
            pred_frames.append(test_pred)
            eval_frames.append(ev)
        except Exception as exc:
            rank_rows.append({"candidate": name, **spec, "status": "error", "error": str(exc)[:300]})
        if (idx + 1) % 40 == 0:
            print(f"[v7] evaluated {idx + 1}/{len(candidate_specs)}", flush=True)
    rank = pd.DataFrame(rank_rows)
    ok = rank[rank["status"].eq("ok")].copy()
    ok["nested_validation_objective"] = (
        ok["nested_val_mae"]
        - 0.25 * ok["nested_val_delta_vs_rf"]
        - 0.45 * ok["nested_val_long_delta_vs_rf"].fillna(0)
        - 0.20 * ok["nested_val_24_delta_vs_rf"].fillna(0)
        - 0.02 * ok["nested_val_24_rho"].fillna(0)
    )
    selected = str(ok.sort_values(["nested_validation_objective", "nested_val_mae"]).iloc[0]["candidate"]) if not ok.empty else ""
    rank = rank.merge(ok[["candidate", "nested_validation_objective"]], on="candidate", how="left")
    rank["validation_selected"] = rank["candidate"].eq(selected)
    rank.to_csv(TABLES / "kg_v7_nested_validation_candidate_rank.csv", index=False, encoding="utf-8-sig")
    preds = pd.concat(pred_frames, ignore_index=True)
    eval_df = pd.concat(eval_frames, ignore_index=True)
    preds.to_csv(PRED / "kg_v7_all_candidate_predictions.csv", index=False, encoding="utf-8-sig")
    eval_df.to_csv(TABLES / "kg_v7_test_evaluation_by_scope.csv", index=False, encoding="utf-8-sig")
    conf = conformance(eval_df, rank)
    conf.to_csv(TABLES / "kg_v7_limit_removal_audit.csv", index=False, encoding="utf-8-sig")
    make_figures(conf, eval_df)
    provenance = {
        "created_by": "honest_kg_latentnet_v7_crossfold_meta_readout.py",
        "integrity_note": "V7 uses nested validation predictions to select a meta-readout and trains each test-fold meta model on validation patients excluding current test patient IDs. It is exploratory because previous test audits were already inspected.",
        "readouts": readouts,
        "candidate_count": int(len(candidate_specs)),
        "test_set_used_for_training": False,
        "test_set_used_for_validation_selection": False,
        "validation_selected_candidate": selected,
        "uses_auxiliary_baseline_readouts": True,
    }
    (OUT / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(OUT), "selected": selected, "readouts": readouts, "conformance": conf.head(12).to_dict(orient="records")}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
