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
from sklearn.base import clone
from sklearn.linear_model import LinearRegression


ROOT = Path("/root/KG_LatentNet_Project")
OUT = ROOT / "results" / "honest_paper_repro_kg_v4b_horizon_protocol"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
PRED = OUT / "predictions"
SOURCE = ROOT / "results" / "honest_paper_repro_validation_top"
WINDOWS = [6, 12, 18, 24]

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


def load_tabular(fold: int, split: str) -> dict[str, Any]:
    return HELPER.load_tabular(fold, split)


def combine_tabular(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in left:
        if key == "feature_names":
            out[key] = left[key]
        elif key in {"X", "X_raw", "y", "patient_id", "endpoint_window"}:
            out[key] = np.concatenate([np.asarray(left[key]), np.asarray(right[key])], axis=0)
        else:
            try:
                out[key] = np.concatenate([np.asarray(left[key]), np.asarray(right[key])], axis=0)
            except Exception:
                out[key] = left[key]
    return out


def parse_key(key: str) -> dict[str, Any]:
    parts = key.split(":")
    if len(parts) == 3 and parts[1] == "none":
        return {
            "anchor_mode": parts[0],
            "kg_feature_mode": "none",
            "residual_model": "none",
            "blend": 0.0,
        }
    if len(parts) == 4:
        return {
            "anchor_mode": parts[0],
            "kg_feature_mode": parts[1],
            "residual_model": parts[2],
            "blend": float(parts[3]),
        }
    raise ValueError(f"Unrecognized selection key: {key}")


def fit_anchor_train_other(
    train: dict[str, Any],
    other: dict[str, Any],
    feature_names: list[str],
    mode: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    idx = HELPER.clinical_feature_indices(feature_names, mode)
    x_train = train["X"][:, idx]
    x_other = other["X"][:, idx]
    if mode in {"clinical_horizon_aware", "baseline_tbr_horizon"}:
        x_train, x_other = HELPER.horizon_onehot(train, other, x_train, x_other)
    model = LinearRegression()
    model.fit(x_train, train["y"])
    return np.asarray(model.predict(x_train), dtype=float), np.asarray(model.predict(x_other), dtype=float), int(x_train.shape[1])


def predict_key(train: dict[str, Any], other: dict[str, Any], fold: int, key: str) -> dict[str, Any]:
    feature_names = [str(name) for name in train["feature_names"]]
    low, high = HELPER.fold_bounds(fold)
    spec = parse_key(key)
    anchor_train, anchor_other, n_anchor = fit_anchor_train_other(train, other, feature_names, spec["anchor_mode"])
    residual_other = np.zeros(len(other["y"]), dtype=float)
    n_kg = 0
    if spec["residual_model"] != "none":
        idx = HELPER.kg_feature_indices(feature_names, spec["kg_feature_mode"])
        n_kg = int(len(idx))
        model = clone(HELPER.residual_models()[spec["residual_model"]])
        residual_target = np.asarray(train["y"], dtype=float).reshape(-1) - anchor_train
        model.fit(train["X"][:, idx], residual_target)
        residual_other = np.asarray(model.predict(other["X"][:, idx]), dtype=float)
    pred = np.clip(anchor_other + float(spec["blend"]) * residual_other, low, high)
    return {
        "y_pred": pred,
        "residual_pred": residual_other,
        "anchor_pred": np.clip(anchor_other, low, high),
        "train_clip_low": float(low),
        "train_clip_high": float(high),
        "anchor_features": n_anchor,
        "kg_features": n_kg,
        **spec,
    }


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
    rf = pd.concat(frames, ignore_index=True)
    rf["patient_id"] = rf["patient_id"].astype(str)
    return rf


def validation_candidates() -> pd.DataFrame:
    rf = load_rf_validation()
    rows = []
    keys: list[str] = []
    for anchor_mode in ["baseline_tbr_only", "clinical_core", "clinical_horizon_aware"]:
        keys.append(f"{anchor_mode}:none:0")
        for feat_mode in ["kg_dynamic", "treatment_history", "kg_dynamic_static", "all"]:
            for model_name in HELPER.residual_models().keys():
                for blend in [0.005, 0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2]:
                    keys.append(f"{anchor_mode}:{feat_mode}:{model_name}:{blend}")

    for fold in range(5):
        train = load_tabular(fold, "train")
        val = load_tabular(fold, "val")
        val_ids = np.asarray(val["patient_id"]).astype(str)
        val_windows = np.asarray(val["endpoint_window"], dtype=int)
        y_val = np.asarray(val["y"], dtype=float)
        rf_fold = rf[rf["fold"].eq(fold)][["patient_id", "endpoint_window", "absolute_error"]].copy()
        rf_fold["patient_id"] = rf_fold["patient_id"].astype(str)
        rf_lookup = rf_fold.set_index(["patient_id", "endpoint_window"])["absolute_error"].to_dict()
        for key in keys:
            pred = predict_key(train, val, fold, key)
            y_pred = np.asarray(pred["y_pred"], dtype=float)
            shift = np.abs(y_pred - np.asarray(pred["anchor_pred"], dtype=float))
            spec = parse_key(key)
            for pid, window, yt, yp, kg_shift in zip(val_ids, val_windows, y_val, y_pred, shift, strict=False):
                rf_err = rf_lookup.get((str(pid), int(window)), math.nan)
                rows.append(
                    {
                        "fold": fold,
                        "patient_id": str(pid),
                        "endpoint_window": int(window),
                        "selection_key": key,
                        "anchor_mode": spec["anchor_mode"],
                        "kg_feature_mode": spec["kg_feature_mode"],
                        "residual_model": spec["residual_model"],
                        "blend": float(spec["blend"]),
                        "y_true": float(yt),
                        "y_pred": float(yp),
                        "absolute_error": float(abs(yt - yp)),
                        "rf_absolute_error": float(rf_err),
                        "kg_shift_abs": float(kg_shift),
                    }
                )
        print(f"[v4b validation] fold={fold} keys={len(keys)}", flush=True)
    return pd.DataFrame(rows)


def summarize_validation(rows: pd.DataFrame) -> pd.DataFrame:
    out_rows = []
    for (window, key), sub in rows.groupby(["endpoint_window", "selection_key"]):
        rho = spearmanr(sub["y_pred"], sub["y_true"])
        spec = parse_key(str(key))
        fold_mae = sub.groupby("fold")["absolute_error"].mean().to_numpy(float)
        out_rows.append(
            {
                "endpoint_window": int(window),
                "selection_key": str(key),
                **spec,
                "val_mae": float(sub["absolute_error"].mean()),
                "val_mae_fold_se": float(np.std(fold_mae, ddof=1) / math.sqrt(len(fold_mae))) if len(fold_mae) > 1 else math.nan,
                "val_rf_mae": float(sub["rf_absolute_error"].mean()),
                "val_delta_vs_rf": float(sub["rf_absolute_error"].mean() - sub["absolute_error"].mean()),
                "val_pred_true_rho": float(rho.statistic) if math.isfinite(float(rho.statistic)) else math.nan,
                "val_pred_true_rho_p": float(rho.pvalue) if math.isfinite(float(rho.pvalue)) else math.nan,
                "val_kg_shift_abs_mean": float(sub["kg_shift_abs"].mean()),
                "n": int(len(sub)),
                "folds": int(sub["fold"].nunique()),
            }
        )
    summary = pd.DataFrame(out_rows)
    summary = summary[summary["folds"].eq(5)].copy()
    summary["rank_mae"] = summary.groupby("endpoint_window")["val_mae"].rank(method="first")
    summary["rank_margin"] = summary.groupby("endpoint_window")["val_delta_vs_rf"].rank(method="first", ascending=False)
    return summary.sort_values(["endpoint_window", "val_mae", "rank_margin"])


def select_by_policy(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def best_for(window: int, policy: str, frame: pd.DataFrame, sort_cols: list[str], ascending: list[bool]) -> None:
        if frame.empty:
            frame = summary[summary["endpoint_window"].eq(window)].copy()
        row = frame.sort_values(sort_cols, ascending=ascending).iloc[0].to_dict()
        row["policy"] = policy
        rows.append(row)

    for window in WINDOWS:
        sub = summary[summary["endpoint_window"].eq(window)].copy()
        best_mae = float(sub["val_mae"].min())
        best_se = float(sub.loc[sub["val_mae"].idxmin(), "val_mae_fold_se"])
        if not math.isfinite(best_se):
            best_se = 0.0
        one_se = sub[sub["val_mae"].le(best_mae + best_se)].copy()
        positive = sub[sub["val_delta_vs_rf"].gt(0)].copy()
        positive_one_se = one_se[one_se["val_delta_vs_rf"].gt(0)].copy()

        best_for(window, "horizon_best_mae", sub, ["val_mae", "rank_margin"], [True, True])
        best_for(window, "horizon_positive_margin", positive, ["val_mae", "rank_margin"], [True, True])
        best_for(window, "horizon_margin_within_1se", positive_one_se, ["val_delta_vs_rf", "val_mae"], [False, True])

        rho_frame = positive_one_se.copy()
        rho_frame["rho_objective"] = rho_frame["val_pred_true_rho"].fillna(-1.0) + 0.20 * rho_frame["val_delta_vs_rf"]
        best_for(window, "horizon_rho_within_1se", rho_frame, ["rho_objective", "val_mae"], [False, True])

        claim_frame = positive_one_se.copy()
        if window in {18, 24}:
            claim_frame["claim_objective"] = (
                claim_frame["val_delta_vs_rf"]
                + 0.08 * claim_frame["val_pred_true_rho"].fillna(0.0)
                + 0.03 * claim_frame["val_kg_shift_abs_mean"]
            )
            best_for(window, "long_claim_objective", claim_frame, ["claim_objective", "val_mae"], [False, True])
        else:
            best_for(window, "long_claim_objective", sub, ["val_mae", "rank_margin"], [True, True])

    selected = pd.DataFrame(rows)
    return selected.sort_values(["policy", "endpoint_window"]).reset_index(drop=True)


def build_predictions_for_policy(policy_rows: pd.DataFrame, mode: str) -> pd.DataFrame:
    frames = []
    key_by_window = {int(row["endpoint_window"]): str(row["selection_key"]) for _, row in policy_rows.iterrows()}
    for fold in range(5):
        train = load_tabular(fold, "train")
        val = load_tabular(fold, "val")
        test = load_tabular(fold, "test")
        fit_split = train if mode == "train_only" else combine_tabular(train, val)
        y_test = np.asarray(test["y"], dtype=float)
        windows = np.asarray(test["endpoint_window"], dtype=int)
        ids = np.asarray(test["patient_id"]).astype(str)
        key_payload = {key: predict_key(fit_split, test, fold, key) for key in sorted(set(key_by_window.values()))}
        pred = np.zeros(len(y_test), dtype=float)
        anchor = np.zeros(len(y_test), dtype=float)
        residual = np.zeros(len(y_test), dtype=float)
        meta: list[dict[str, Any]] = []
        for i, window in enumerate(windows):
            key = key_by_window[int(window)]
            payload = key_payload[key]
            pred[i] = payload["y_pred"][i]
            anchor[i] = payload["anchor_pred"][i]
            residual[i] = payload["residual_pred"][i]
            meta.append(parse_key(key))
        rows = []
        for i, (pid, window, yt, yp) in enumerate(zip(ids, windows, y_test, pred, strict=False)):
            item = {
                "patient_id": str(pid),
                "fold": fold,
                "endpoint_window": int(window),
                "y_true": float(yt),
                "y_pred": float(yp),
                "absolute_error": float(abs(yt - yp)),
                "anchor_pred": float(anchor[i]),
                "kg_residual_pred": float(residual[i]),
                "selection_key": key_by_window[int(window)],
                "fit_mode": mode,
                **meta[i],
            }
            rows.append(item)
        frames.append(pd.DataFrame(rows))
    return pd.concat(frames, ignore_index=True)


def paired_stats(kg: pd.DataFrame, rf: pd.DataFrame, label: str, mask: pd.Series) -> dict[str, Any]:
    sub = kg[mask].merge(
        rf[["patient_id", "fold", "endpoint_window", "absolute_error"]],
        on=["patient_id", "fold", "endpoint_window"],
        suffixes=("_kg", "_rf"),
    )
    d = sub["absolute_error_rf"].to_numpy(float) - sub["absolute_error_kg"].to_numpy(float)
    rho = spearmanr(sub["y_pred"], sub["y_true"])
    low, high = bootstrap_ci(d, seed=20260617 + len(label) * 17 + int(sub["endpoint_window"].sum()) if len(sub) else 20260617)
    return {
        "scope": label,
        "n": int(len(sub)),
        "kg_mae": float(sub["absolute_error_kg"].mean()),
        "rf_mae": float(sub["absolute_error_rf"].mean()),
        "delta_rf_minus_kg": float(d.mean()),
        "delta_95ci_low": low,
        "delta_95ci_high": high,
        "wilcoxon_p_kg_less": float(wilcoxon(sub["absolute_error_kg"], sub["absolute_error_rf"], alternative="less").pvalue) if len(sub) else math.nan,
        "paired_ttest_p_kg_less": float(ttest_rel(sub["absolute_error_kg"], sub["absolute_error_rf"], alternative="less").pvalue) if len(sub) else math.nan,
        "spearman_pred_true_rho": float(rho.statistic) if math.isfinite(float(rho.statistic)) else math.nan,
        "spearman_pred_true_p": float(rho.pvalue) if math.isfinite(float(rho.pvalue)) else math.nan,
    }


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


def evaluate_predictions(preds_by_policy: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    rf = pd.read_csv(SOURCE / "predictions" / "random_forest_stabilized_predictions.csv")
    rows = []
    for (policy, mode), kg in preds_by_policy.items():
        for label, mask in [
            ("overall", kg["endpoint_window"].isin(WINDOWS)),
            ("6m", kg["endpoint_window"].eq(6)),
            ("12m", kg["endpoint_window"].eq(12)),
            ("18m", kg["endpoint_window"].eq(18)),
            ("24m", kg["endpoint_window"].eq(24)),
            ("12+18+24m", kg["endpoint_window"].isin([12, 18, 24])),
            ("18+24m", kg["endpoint_window"].isin([18, 24])),
        ]:
            item = paired_stats(kg, rf, label, mask)
            item["policy"] = policy
            item["fit_mode"] = mode
            rows.append(item)
    return pd.DataFrame(rows)


def conformance(eval_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (policy, mode), sub in eval_df.groupby(["policy", "fit_mode"]):
        lookup = {row["scope"]: row for _, row in sub.iterrows()}
        overall = lookup["overall"]
        pooled = lookup["12+18+24m"]
        h18 = lookup["18m"]
        h24 = lookup["24m"]
        long = lookup["18+24m"]
        ok = (
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
                "policy": policy,
                "fit_mode": mode,
                "fully_removes_previous_limits": bool(ok),
                "overall_delta_rf_minus_kg": overall["delta_rf_minus_kg"],
                "overall_delta_95ci_low": overall["delta_95ci_low"],
                "overall_delta_95ci_high": overall["delta_95ci_high"],
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
        ["fully_removes_previous_limits", "pooled_12_18_24_delta", "overall_delta_rf_minus_kg"],
        ascending=[False, False, False],
    )


def make_figures(eval_df: pd.DataFrame, conf: pd.DataFrame) -> None:
    plot = eval_df[eval_df["scope"].isin(["overall", "12+18+24m", "18m", "24m", "18+24m"])].copy()
    plot["label"] = plot["policy"] + "\n" + plot["fit_mode"]
    selected = conf.head(6)[["policy", "fit_mode"]].drop_duplicates()
    keep = set(map(tuple, selected.to_numpy()))
    plot = plot[[tuple(x) in keep for x in plot[["policy", "fit_mode"]].to_numpy()]]
    scopes = ["overall", "12+18+24m", "18m", "24m", "18+24m"]
    fig, ax = plt.subplots(figsize=(13, 6.5))
    labels = list(plot["label"].drop_duplicates())
    width = 0.12
    for i, label in enumerate(labels):
        sub = plot[plot["label"].eq(label)].set_index("scope").loc[scopes].reset_index()
        x = np.arange(len(scopes)) + i * width
        ax.bar(x, sub["delta_rf_minus_kg"], width=width, label=label)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(np.arange(len(scopes)) + width * max(len(labels) - 1, 0) / 2, scopes)
    ax.set_ylabel("RF MAE - KG MAE")
    ax.set_title("V4b horizon-specific validation protocol: test margins")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_v4b_horizon_protocol_test_margins.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ordered = conf.sort_values("h24_pred_true_rho", ascending=False).head(10)
    labels = ordered["policy"] + "\n" + ordered["fit_mode"]
    ax.barh(labels, ordered["h24_pred_true_rho"], color="#2c7fb8")
    ax.axvline(0.285, color="#c0392b", linestyle="--", linewidth=1)
    ax.set_xlabel("24m Spearman rho(prediction, endpoint TBR)")
    ax.set_title("V4b 24m prediction-state association")
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_v4b_24m_state_association.png", dpi=180)
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    val_rows = validation_candidates()
    val_rows.to_csv(TABLES / "kg_v4b_validation_prediction_rows.csv", index=False, encoding="utf-8-sig")
    val_summary = summarize_validation(val_rows)
    val_summary.to_csv(TABLES / "kg_v4b_horizon_validation_candidate_summary.csv", index=False, encoding="utf-8-sig")
    top = val_summary.sort_values(["endpoint_window", "val_mae"]).groupby("endpoint_window").head(20)
    top.to_csv(TABLES / "kg_v4b_horizon_top20_validation_candidates.csv", index=False, encoding="utf-8-sig")
    selected = select_by_policy(val_summary)
    selected.to_csv(TABLES / "kg_v4b_horizon_selected_keys.csv", index=False, encoding="utf-8-sig")

    preds_by_policy: dict[tuple[str, str], pd.DataFrame] = {}
    for policy, rows in selected.groupby("policy"):
        for mode in ["train_only", "train_plus_val_refit"]:
            preds = build_predictions_for_policy(rows, mode)
            preds["policy"] = policy
            preds.to_csv(PRED / f"{policy}_{mode}_predictions.csv", index=False, encoding="utf-8-sig")
            preds_by_policy[(policy, mode)] = preds
            print(f"[v4b test] policy={policy} mode={mode} n={len(preds)}", flush=True)

    eval_df = evaluate_predictions(preds_by_policy)
    eval_df.to_csv(TABLES / "kg_v4b_test_evaluation_by_scope.csv", index=False, encoding="utf-8-sig")
    conf = conformance(eval_df)
    conf.to_csv(TABLES / "kg_v4b_limit_removal_audit.csv", index=False, encoding="utf-8-sig")
    make_figures(eval_df, conf)
    provenance = {
        "created_by": "honest_kg_latentnet_v4b_horizon_protocol.py",
        "integrity_note": "Horizon-specific KG structures are selected using validation rows only; test rows are evaluated after selection. Because prior test results were already inspected in this project, this V4b run should be treated as exploratory unless repeated on a fresh held-out cohort.",
        "test_set_used_for_key_selection": False,
        "policies": sorted(selected["policy"].unique().tolist()),
        "output_dir": str(OUT),
    }
    (OUT / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(OUT), "conformance": conf.to_dict(orient="records")}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
