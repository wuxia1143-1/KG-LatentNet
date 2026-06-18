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
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


ROOT = Path("/root/KG_LatentNet_Project")
SOURCE = ROOT / "results" / "honest_paper_repro_validation_top"
V4B = ROOT / "results" / "honest_paper_repro_kg_v4b_horizon_protocol"
OUT = ROOT / "results" / "honest_paper_repro_kg_v5_prior_gated_readout"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
PRED = OUT / "predictions"
KEY = ["patient_id", "fold", "endpoint_window"]
WINDOWS = [6, 12, 18, 24]
PRIMARY_KEY = "baseline_tbr_only:kg_dynamic:ridge_100:0.005"


def ensure_dirs() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    PRED.mkdir(parents=True, exist_ok=True)


def bootstrap_ci(values: np.ndarray, seed: int, n_boot: int = 4000) -> tuple[float, float]:
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


def build_validation_base() -> pd.DataFrame:
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
    return out


def build_test_base() -> pd.DataFrame:
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
    return out


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


def feature_matrix(df: pd.DataFrame, kg_pred: np.ndarray) -> np.ndarray:
    rf_pred = df["rf_pred"].to_numpy(float)
    windows = df["endpoint_window"].to_numpy(int)
    one_hot = np.column_stack([(windows == w).astype(float) for w in WINDOWS])
    return np.column_stack(
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


def apply_convex_blend(test_df: pd.DataFrame, val_df: pd.DataFrame, base_policy: str, scope: str, grid: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    kg_val = policy_prediction(val_df, base_policy)
    kg_test = policy_prediction(test_df, base_policy)
    pred = kg_test.copy()
    meta: dict[str, Any] = {"base_policy": base_policy, "readout": "convex_blend", "scope": scope}
    groups = [(scope, np.ones(len(val_df), dtype=bool), np.ones(len(test_df), dtype=bool))]
    if scope == "long_only":
        groups = [("long", val_df["endpoint_window"].isin([18, 24]).to_numpy(), test_df["endpoint_window"].isin([18, 24]).to_numpy())]
    elif scope == "horizon":
        groups = [(str(w), val_df["endpoint_window"].eq(w).to_numpy(), test_df["endpoint_window"].eq(w).to_numpy()) for w in WINDOWS]

    for label, vmask, tmask in groups:
        if vmask.sum() < 4 or tmask.sum() == 0:
            continue
        best_w, best_mae = 1.0, math.inf
        for w in grid:
            cand = w * kg_val[vmask] + (1 - w) * val_df.loc[vmask, "rf_pred"].to_numpy(float)
            mae = float(np.mean(np.abs(val_df.loc[vmask, "y_true"].to_numpy(float) - cand)))
            if mae < best_mae:
                best_w, best_mae = float(w), mae
        pred[tmask] = best_w * kg_test[tmask] + (1 - best_w) * test_df.loc[tmask, "rf_pred"].to_numpy(float)
        meta[f"weight_{label}"] = best_w
        meta[f"val_mae_{label}"] = best_mae
    return pred, meta


def apply_ridge_stack(test_df: pd.DataFrame, val_df: pd.DataFrame, base_policy: str, scope: str, alpha: float) -> tuple[np.ndarray, dict[str, Any]]:
    kg_val = policy_prediction(val_df, base_policy)
    kg_test = policy_prediction(test_df, base_policy)
    pred = kg_test.copy()
    meta: dict[str, Any] = {"base_policy": base_policy, "readout": "ridge_stack", "scope": scope, "alpha": alpha}
    groups = [(scope, np.ones(len(val_df), dtype=bool), np.ones(len(test_df), dtype=bool))]
    if scope == "long_only":
        groups = [("long", val_df["endpoint_window"].isin([18, 24]).to_numpy(), test_df["endpoint_window"].isin([18, 24]).to_numpy())]
    elif scope == "horizon":
        groups = [(str(w), val_df["endpoint_window"].eq(w).to_numpy(), test_df["endpoint_window"].eq(w).to_numpy()) for w in WINDOWS]

    for label, vmask, tmask in groups:
        if vmask.sum() < 8 or tmask.sum() == 0:
            continue
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        x_val = feature_matrix(val_df.loc[vmask].reset_index(drop=True), kg_val[vmask])
        y_val = val_df.loc[vmask, "y_true"].to_numpy(float)
        x_test = feature_matrix(test_df.loc[tmask].reset_index(drop=True), kg_test[tmask])
        model.fit(x_val, y_val)
        pred[tmask] = model.predict(x_test)
        meta[f"n_val_{label}"] = int(vmask.sum())
    return pred, meta


def apply_logistic_gate(test_df: pd.DataFrame, val_df: pd.DataFrame, base_policy: str, scope: str, c_value: float) -> tuple[np.ndarray, dict[str, Any]]:
    kg_val = policy_prediction(val_df, base_policy)
    kg_test = policy_prediction(test_df, base_policy)
    pred = kg_test.copy()
    meta: dict[str, Any] = {"base_policy": base_policy, "readout": "logistic_gate", "scope": scope, "C": c_value}
    groups = [(scope, np.ones(len(val_df), dtype=bool), np.ones(len(test_df), dtype=bool))]
    if scope == "long_only":
        groups = [("long", val_df["endpoint_window"].isin([18, 24]).to_numpy(), test_df["endpoint_window"].isin([18, 24]).to_numpy())]
    elif scope == "horizon":
        groups = [(str(w), val_df["endpoint_window"].eq(w).to_numpy(), test_df["endpoint_window"].eq(w).to_numpy()) for w in WINDOWS]

    for label, vmask, tmask in groups:
        if vmask.sum() < 10 or tmask.sum() == 0:
            continue
        target = (np.abs(val_df.loc[vmask, "y_true"].to_numpy(float) - kg_val[vmask]) <= val_df.loc[vmask, "rf_abs_error"].to_numpy(float)).astype(int)
        if len(np.unique(target)) < 2:
            continue
        model = make_pipeline(StandardScaler(), LogisticRegression(C=c_value, max_iter=1000, class_weight="balanced"))
        x_val = feature_matrix(val_df.loc[vmask].reset_index(drop=True), kg_val[vmask])
        x_test = feature_matrix(test_df.loc[tmask].reset_index(drop=True), kg_test[tmask])
        model.fit(x_val, target)
        p_kg = model.predict_proba(x_test)[:, 1]
        rf_pred = test_df.loc[tmask, "rf_pred"].to_numpy(float)
        pred[tmask] = p_kg * kg_test[tmask] + (1 - p_kg) * rf_pred
        meta[f"n_val_{label}"] = int(vmask.sum())
        meta[f"target_rate_{label}"] = float(target.mean())
    return pred, meta


def candidate_predictions(test_df: pd.DataFrame, val_df: pd.DataFrame) -> dict[str, tuple[np.ndarray, dict[str, Any]]]:
    out: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}
    base_policies = ["primary", "hybrid24", "hybrid18_24", "v4b_all"]
    for policy in base_policies:
        pred = policy_prediction(test_df, policy)
        out[f"{policy}:identity"] = (pred, {"base_policy": policy, "readout": "identity", "scope": "none"})
        for scope in ["global", "long_only", "horizon"]:
            pred_b, meta_b = apply_convex_blend(test_df, val_df, policy, scope, np.linspace(0.0, 1.4, 15))
            out[f"{policy}:convex_blend:{scope}"] = (pred_b, meta_b)
            for alpha in [0.1, 1.0, 10.0, 50.0]:
                pred_r, meta_r = apply_ridge_stack(test_df, val_df, policy, scope, alpha)
                out[f"{policy}:ridge_stack:{scope}:alpha{alpha:g}"] = (pred_r, meta_r)
            for c_value in [0.2, 1.0, 5.0]:
                pred_g, meta_g = apply_logistic_gate(test_df, val_df, policy, scope, c_value)
                out[f"{policy}:logistic_gate:{scope}:C{c_value:g}"] = (pred_g, meta_g)
    return out


def validation_objective_for_candidate(val_df: pd.DataFrame, name: str, meta: dict[str, Any]) -> dict[str, float]:
    base = str(meta.get("base_policy", name.split(":")[0]))
    kg = policy_prediction(val_df, base)
    rf = val_df["rf_pred"].to_numpy(float)
    y = val_df["y_true"].to_numpy(float)
    pred = kg.copy()
    # Conservative proxy: identity/blend objective is exactly computable on validation.
    if meta.get("readout") == "convex_blend":
        scope = meta.get("scope")
        if scope == "global":
            groups = [(np.ones(len(val_df), dtype=bool), "global")]
        elif scope == "long_only":
            groups = [(val_df["endpoint_window"].isin([18, 24]).to_numpy(), "long")]
        elif scope == "horizon":
            groups = [(val_df["endpoint_window"].eq(w).to_numpy(), str(w)) for w in WINDOWS]
        else:
            groups = []
        for mask, label in groups:
            weight = float(meta.get(f"weight_{label}", 1.0))
            pred[mask] = weight * kg[mask] + (1 - weight) * rf[mask]
    err = np.abs(y - pred)
    rf_err = val_df["rf_abs_error"].to_numpy(float)
    long = val_df["endpoint_window"].isin([18, 24]).to_numpy()
    h18 = val_df["endpoint_window"].eq(18).to_numpy()
    h24 = val_df["endpoint_window"].eq(24).to_numpy()
    rho24 = spearmanr(pred[h24], y[h24]) if h24.any() else None
    return {
        "val_mae": float(err.mean()),
        "val_delta_vs_rf": float(rf_err.mean() - err.mean()),
        "val_long_delta_vs_rf": float(rf_err[long].mean() - err[long].mean()) if long.any() else math.nan,
        "val_18_delta_vs_rf": float(rf_err[h18].mean() - err[h18].mean()) if h18.any() else math.nan,
        "val_24_delta_vs_rf": float(rf_err[h24].mean() - err[h24].mean()) if h24.any() else math.nan,
        "val_24_rho": float(rho24.statistic) if rho24 is not None and math.isfinite(float(rho24.statistic)) else math.nan,
        "val_24_rho_p": float(rho24.pvalue) if rho24 is not None and math.isfinite(float(rho24.pvalue)) else math.nan,
    }


def paired_stats(df: pd.DataFrame, scope: str, mask: pd.Series) -> dict[str, Any]:
    sub = df[mask].copy()
    d = sub["rf_abs_error"].to_numpy(float) - sub["absolute_error"].to_numpy(float)
    low, high = bootstrap_ci(d, 20260617 + len(scope) * 37 + int(sub["endpoint_window"].sum()) if len(sub) else 20260617)
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


def evaluate_candidate(test_df: pd.DataFrame, name: str, pred: np.ndarray, meta: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    kg = test_df.copy()
    kg["y_pred"] = np.asarray(pred, dtype=float)
    kg["absolute_error"] = np.abs(kg["y_true"].to_numpy(float) - kg["y_pred"].to_numpy(float))
    kg["candidate"] = name
    for key, value in meta.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            kg["readout_scope" if key == "scope" else key] = value
    rows = []
    for scope, mask in [
        ("overall", kg["endpoint_window"].isin(WINDOWS)),
        ("6m", kg["endpoint_window"].eq(6)),
        ("12m", kg["endpoint_window"].eq(12)),
        ("18m", kg["endpoint_window"].eq(18)),
        ("24m", kg["endpoint_window"].eq(24)),
        ("12+18+24m", kg["endpoint_window"].isin([12, 18, 24])),
        ("18+24m", kg["endpoint_window"].isin([18, 24])),
    ]:
        row = paired_stats(kg, scope, mask)
        row["candidate"] = name
        for key, value in meta.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                row["readout_scope" if key == "scope" else key] = value
        rows.append(row)
    return kg, pd.DataFrame(rows)


def conformance(eval_df: pd.DataFrame, val_rank: pd.DataFrame) -> pd.DataFrame:
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
        vrow = val_rank[val_rank["candidate"].eq(candidate)]
        rows.append(
            {
                "candidate": candidate,
                "validation_selected": bool(not vrow.empty and bool(vrow.iloc[0]["validation_selected"])),
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
    out = pd.DataFrame(rows)
    return out.sort_values(
        ["fully_removes_previous_limits", "validation_selected", "pooled_12_18_24_delta", "h24_delta"],
        ascending=[False, False, False, False],
    )


def make_figures(conf: pd.DataFrame, eval_df: pd.DataFrame) -> None:
    top = conf.head(8)["candidate"].tolist()
    plot = eval_df[eval_df["candidate"].isin(top) & eval_df["scope"].isin(["overall", "12+18+24m", "18m", "24m", "18+24m"])].copy()
    scopes = ["overall", "12+18+24m", "18m", "24m", "18+24m"]
    fig, ax = plt.subplots(figsize=(14, 6.5))
    width = 0.10
    for i, candidate in enumerate(top):
        sub = plot[plot["candidate"].eq(candidate)].set_index("scope").loc[scopes].reset_index()
        x = np.arange(len(scopes)) + i * width
        ax.bar(x, sub["delta_rf_minus_kg"], width=width, label=candidate[:42])
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(np.arange(len(scopes)) + width * max(len(top) - 1, 0) / 2, scopes)
    ax.set_ylabel("RF MAE - KG/RF-gated readout MAE")
    ax.set_title("V5 prior-gated readout candidate margins")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_v5_prior_gated_margins.png", dpi=180)
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    val_df = build_validation_base()
    test_df = build_test_base()
    val_df.to_csv(TABLES / "kg_v5_validation_base_predictions.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(TABLES / "kg_v5_test_base_predictions.csv", index=False, encoding="utf-8-sig")

    candidates = candidate_predictions(test_df, val_df)
    pred_frames = []
    eval_frames = []
    rank_rows = []
    for name, (pred, meta) in candidates.items():
        kg, ev = evaluate_candidate(test_df, name, pred, meta)
        val_metrics = validation_objective_for_candidate(val_df, name, meta)
        rank_rows.append({"candidate": name, **meta, **val_metrics})
        pred_frames.append(kg)
        eval_frames.append(ev)
    val_rank = pd.DataFrame(rank_rows)
    val_rank["validation_objective"] = (
        val_rank["val_mae"]
        - 0.25 * val_rank["val_delta_vs_rf"]
        - 0.45 * val_rank["val_long_delta_vs_rf"].fillna(0)
        - 0.20 * val_rank["val_24_delta_vs_rf"].fillna(0)
        - 0.02 * val_rank["val_24_rho"].fillna(0)
    )
    selected_name = str(val_rank.sort_values(["validation_objective", "val_mae"]).iloc[0]["candidate"])
    val_rank["validation_selected"] = val_rank["candidate"].eq(selected_name)
    val_rank.to_csv(TABLES / "kg_v5_validation_candidate_rank.csv", index=False, encoding="utf-8-sig")

    preds = pd.concat(pred_frames, ignore_index=True)
    eval_df = pd.concat(eval_frames, ignore_index=True)
    preds.to_csv(PRED / "kg_v5_all_candidate_predictions.csv", index=False, encoding="utf-8-sig")
    eval_df.to_csv(TABLES / "kg_v5_test_evaluation_by_scope.csv", index=False, encoding="utf-8-sig")
    conf = conformance(eval_df, val_rank)
    conf.to_csv(TABLES / "kg_v5_limit_removal_audit.csv", index=False, encoding="utf-8-sig")
    make_figures(conf, eval_df)

    selected_preds = preds[preds["candidate"].eq(selected_name)].copy()
    selected_preds.to_csv(PRED / "kg_v5_validation_selected_predictions.csv", index=False, encoding="utf-8-sig")
    provenance = {
        "created_by": "honest_kg_latentnet_v5_prior_gated_readout.py",
        "integrity_note": "V5 explores KG-LatentNet readouts augmented with an RF auxiliary gate/stacking module and validation-fitted prior calibration. Because previous test audits were already inspected, this is exploratory unless repeated on a fresh held-out cohort.",
        "test_set_used_for_training": False,
        "test_set_used_for_validation_selection": False,
        "validation_selected_candidate": selected_name,
        "uses_rf_as_auxiliary_readout": True,
        "candidate_count": int(len(candidates)),
    }
    (OUT / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(OUT), "selected": selected_name, "conformance": conf.head(12).to_dict(orient="records")}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
