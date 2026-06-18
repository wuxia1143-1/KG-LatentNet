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


ROOT = Path("/root/KG_LatentNet_Project")
OUT = ROOT / "results" / "honest_paper_repro_nested_ensemble"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
PRED = OUT / "predictions"
SOURCE = ROOT / "results" / "honest_paper_repro_validation_top"
STRUCT_OUT = ROOT / "results" / "honest_paper_repro_structural_revision"

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
STRUCT = load_module("kg_structural_revision", ROOT / "scripts" / "honest_structural_revision_experiments.py")
REFIT = load_module("kg_structural_refit", ROOT / "scripts" / "honest_structural_revision_refit.py")


def ensure_dirs() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    PRED.mkdir(parents=True, exist_ok=True)


def parse_primary_key(key: str) -> tuple[str, str, str, float]:
    parts = key.split(":")
    if len(parts) == 3:
        return parts[0], "none", "none", 0.0
    if len(parts) != 4:
        raise ValueError(key)
    return parts[0], parts[1], parts[2], float(parts[3])


def primary_fold_predictions(fold: int, key: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    anchor_mode, feature_mode, residual_name, blend = parse_primary_key(key)
    train = HELPER.load_tabular(fold, "train")
    val = HELPER.load_tabular(fold, "val")
    test = HELPER.load_tabular(fold, "test")
    names = [str(name) for name in train["feature_names"]]
    low, high = HELPER.fold_bounds(fold)
    anchor = HELPER.fit_anchor(train, val, test, names, anchor_mode)
    if feature_mode == "none":
        val_pred = np.clip(anchor["val"], low, high)
        test_pred = np.clip(anchor["test"], low, high)
    else:
        idx = HELPER.kg_feature_indices(names, feature_mode)
        residual_target = np.asarray(train["y"], dtype=float).reshape(-1) - anchor["train"]
        model = HELPER.residual_models()[residual_name]
        model.fit(train["X"][:, idx], residual_target)
        r_val = np.asarray(model.predict(val["X"][:, idx]), dtype=float)
        r_test = np.asarray(model.predict(test["X"][:, idx]), dtype=float)
        val_pred = np.clip(anchor["val"] + blend * r_val, low, high)
        test_pred = np.clip(anchor["test"] + blend * r_test, low, high)
    return rows_for_split(fold, val, val_pred, key), rows_for_split(fold, test, test_pred, key)


def structural_train_only_predictions(fold: int, selected: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = STRUCT.load_tabular(fold, "train")
    val = STRUCT.load_tabular(fold, "val")
    test = STRUCT.load_tabular(fold, "test")
    names = [str(name) for name in train["feature_names"]]
    y_train = np.asarray(train["y"], dtype=float).reshape(-1)
    y_val = np.asarray(val["y"], dtype=float).reshape(-1)
    w_train = np.asarray(train["endpoint_window"])
    w_val = np.asarray(val["endpoint_window"])
    w_test = np.asarray(test["endpoint_window"])
    low, high = STRUCT.fold_bounds(fold)
    anchor = STRUCT.fit_anchor(train, val, test, names, str(selected["anchor_mode"]))
    idx = STRUCT.kg_feature_indices(names, str(selected["feature_mode"]))
    xtr0 = train["X"][:, idx]
    xva0 = val["X"][:, idx]
    xte0 = test["X"][:, idx]
    xtr, xva = STRUCT.augment_features(xtr0, xva0, w_train, w_val, str(selected["feature_variant"]))
    _, xte = STRUCT.augment_features(xtr0, xte0, w_train, w_test, str(selected["feature_variant"]))
    residual_target = y_train - anchor["train"]
    spec = REFIT.spec_by_name(str(selected["residual_model"]), 20260617 + fold * 100)
    weights = STRUCT.sample_weights(w_train, str(selected["weight_scheme"]))
    r_val, r_test = STRUCT.fit_predict_residual(spec, xtr, residual_target, xva, xte, weights)
    blend = float(selected["blend"]) if pd.notna(selected["blend"]) else 0.0
    base_val = np.clip(anchor["val"] + blend * r_val, low, high)
    base_test = np.clip(anchor["test"] + blend * r_test, low, high)
    cal = STRUCT.fit_calibrator(y_val, base_val, w_val, str(selected["calibration"]))
    val_pred = np.clip(cal(base_val, w_val), low, high)
    test_pred = np.clip(cal(base_test, w_test), low, high)
    return rows_for_split(fold, val, val_pred, str(selected["selection_key"])), rows_for_split(fold, test, test_pred, str(selected["selection_key"]))


def structural_refit_test_predictions(rule: str) -> pd.DataFrame:
    path = STRUCT_OUT / "predictions" / "structural_revision_final_refit_predictions.csv"
    df = pd.read_csv(path)
    return df[df["selection_rule"].eq(rule)][["patient_id", "fold", "endpoint_window", "y_true", "y_pred"]].copy()


def rows_for_split(fold: int, split: dict[str, Any], pred: np.ndarray, key: str) -> pd.DataFrame:
    y = np.asarray(split["y"], dtype=float).reshape(-1)
    return pd.DataFrame(
        {
            "patient_id": [str(pid) for pid in split["patient_id"]],
            "fold": fold,
            "endpoint_window": np.asarray(split["endpoint_window"], dtype=int),
            "y_true": y,
            "y_pred": np.asarray(pred, dtype=float).reshape(-1),
            "component_key": key,
        }
    )


def load_rf_validation_best() -> pd.DataFrame:
    best = pd.read_csv(ROOT / "results" / "tables" / "tuning" / "validation_tuning_best_by_model.csv", encoding="utf-8-sig")
    cid = int(best[best["model_name"].eq("random_forest")].iloc[0]["candidate_id"])
    frames = []
    for fold in range(5):
        path = ROOT / "results" / "predictions" / "tuning" / f"random_forest_fold{fold}_candidate{cid}_val_predictions.csv"
        df = pd.read_csv(path)
        df["fold"] = fold
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def collect_components() -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    selected = pd.read_csv(STRUCT_OUT / "tables" / "structural_revision_expanded_selected_rules.csv")
    primary_keys = {
        "primary_validation_top": "baseline_tbr_only:kg_dynamic:ridge_100:0.005",
        "knowledge_sensitive": "baseline_tbr_only:kg_dynamic:ridge_0.01:0.01",
    }
    val_components: dict[str, list[pd.DataFrame]] = {name: [] for name in primary_keys}
    test_components: dict[str, list[pd.DataFrame]] = {name: [] for name in primary_keys}
    for fold in range(5):
        for name, key in primary_keys.items():
            val, test = primary_fold_predictions(fold, key)
            val_components[name].append(val)
            test_components[name].append(test)

    structural_rules = ["best_18m_validation_margin", "best_overall_validation_mae", "best_long_weighted_validation_mae"]
    for rule in structural_rules:
        row = selected[selected["selection_rule"].eq(rule)].iloc[0]
        val_components[rule] = []
        test_components[f"{rule}_train_only"] = []
        for fold in range(5):
            val, test = structural_train_only_predictions(fold, row)
            val_components[rule].append(val)
            test_components[f"{rule}_train_only"].append(test)
        test_components[f"{rule}_final_refit"] = [structural_refit_test_predictions(rule)]

    return (
        {name: pd.concat(frames, ignore_index=True) for name, frames in val_components.items()},
        {name: pd.concat(frames, ignore_index=True) for name, frames in test_components.items()},
    )


def merge_prediction_components(components: dict[str, pd.DataFrame]) -> pd.DataFrame:
    keys = ["patient_id", "fold", "endpoint_window"]
    merged = None
    for name, df in components.items():
        sub = df[keys + ["y_true", "y_pred"]].rename(columns={"y_pred": name})
        if merged is None:
            merged = sub
        else:
            merged = merged.merge(sub[keys + [name]], on=keys, how="inner")
    assert merged is not None
    return merged


def weighted_prediction(df: pd.DataFrame, names: list[str], weights: np.ndarray) -> np.ndarray:
    mat = df[names].to_numpy(float)
    return mat @ weights


def validation_objective(y: np.ndarray, pred: np.ndarray, windows: np.ndarray, rf_err: np.ndarray) -> dict[str, float]:
    err = np.abs(y - pred)
    out = {
        "val_mae": float(err.mean()),
        "val_delta_vs_rf": float(rf_err.mean() - err.mean()),
        "val_weighted_mae": float(np.average(err, weights=STRUCT.sample_weights(windows, "long4"))),
    }
    for horizon in [18, 24]:
        mask = windows == horizon
        out[f"val_{horizon}_delta_vs_rf"] = float(rf_err[mask].mean() - err[mask].mean())
        rho = spearmanr(pred[mask], y[mask])
        out[f"val_{horizon}_rho"] = float(rho.statistic) if math.isfinite(float(rho.statistic)) else math.nan
        out[f"val_{horizon}_rho_p"] = float(rho.pvalue) if math.isfinite(float(rho.pvalue)) else math.nan
    out["objective"] = (
        out["val_weighted_mae"]
        - 0.25 * out["val_delta_vs_rf"]
        - 0.35 * min(out["val_18_delta_vs_rf"], out["val_24_delta_vs_rf"])
        - 0.02 * max(out["val_24_rho"], 0)
    )
    return out


def grid_weights(n: int, step: float = 0.1) -> np.ndarray:
    values = np.arange(0, 1 + 1e-9, step)
    out = []
    if n == 1:
        return np.ones((1, 1))
    def rec(prefix: list[float], remaining: int, left: float) -> None:
        if remaining == 1:
            out.append(prefix + [round(left, 10)])
            return
        for v in values:
            if v <= left + 1e-9:
                rec(prefix + [float(v)], remaining - 1, round(left - float(v), 10))
    rec([], n, 1.0)
    return np.asarray(out, dtype=float)


def select_ensembles(val_merged: pd.DataFrame, rf_val: pd.DataFrame) -> pd.DataFrame:
    keys = ["patient_id", "fold", "endpoint_window"]
    val = val_merged.merge(rf_val[keys + ["absolute_error"]], on=keys, how="inner")
    y = val["y_true"].to_numpy(float)
    windows = val["endpoint_window"].to_numpy(int)
    rf_err = val["absolute_error"].to_numpy(float)
    component_names = [c for c in val_merged.columns if c not in keys + ["y_true"]]
    candidate_sets = [
        ["primary_validation_top", "knowledge_sensitive"],
        ["primary_validation_top", "best_18m_validation_margin"],
        ["primary_validation_top", "knowledge_sensitive", "best_18m_validation_margin"],
        ["primary_validation_top", "knowledge_sensitive", "best_18m_validation_margin", "best_long_weighted_validation_mae"],
    ]
    rows = []
    for set_id, names in enumerate(candidate_sets):
        names = [name for name in names if name in component_names]
        for weights in grid_weights(len(names), step=0.05):
            pred = weighted_prediction(val, names, weights)
            obj = validation_objective(y, pred, windows, rf_err)
            rows.append(
                {
                    "candidate_set": set_id,
                    "components": "|".join(names),
                    "weights": "|".join(f"{w:.2f}" for w in weights),
                    **obj,
                }
            )
    all_rows = pd.DataFrame(rows)
    all_rows.to_csv(TABLES / "nested_ensemble_validation_weight_grid.csv", index=False, encoding="utf-8-sig")
    selected = []
    for set_id, sub in all_rows.groupby("candidate_set"):
        selected.append(sub.sort_values(["objective", "val_weighted_mae", "val_mae"]).iloc[0])
    sel = pd.DataFrame(selected).sort_values(["objective", "val_weighted_mae"])
    sel.to_csv(TABLES / "nested_ensemble_selected_weights.csv", index=False, encoding="utf-8-sig")
    return sel


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
        "wilcoxon_p_kg_less": float(wilcoxon(sub["absolute_error_kg"], sub["absolute_error_rf"], alternative="less").pvalue),
        "paired_ttest_p_kg_less": float(ttest_rel(sub["absolute_error_kg"], sub["absolute_error_rf"], alternative="less").pvalue),
        "spearman_pred_true_rho": float(rho.statistic) if math.isfinite(float(rho.statistic)) else math.nan,
        "spearman_pred_true_p": float(rho.pvalue) if math.isfinite(float(rho.pvalue)) else math.nan,
    }


def evaluate_selected(test_components: dict[str, pd.DataFrame], selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rf = pd.read_csv(SOURCE / "predictions" / "random_forest_stabilized_predictions.csv")
    eval_rows = []
    pred_rows = []
    for _, row in selected.iterrows():
        names = str(row["components"]).split("|")
        weights = np.asarray([float(x) for x in str(row["weights"]).split("|")], dtype=float)
        for mode in ["train_only", "final_refit"]:
            comps = {}
            for name in names:
                if name in test_components:
                    comps[name] = test_components[name]
                elif f"{name}_{mode}" in test_components:
                    comps[name] = test_components[f"{name}_{mode}"]
                elif f"{name}_train_only" in test_components:
                    comps[name] = test_components[f"{name}_train_only"]
            if set(comps) != set(names):
                continue
            test = merge_prediction_components(comps)
            pred = weighted_prediction(test, names, weights)
            kg = test[["patient_id", "fold", "endpoint_window", "y_true"]].copy()
            kg["y_pred"] = pred
            kg["absolute_error"] = np.abs(kg["y_true"].to_numpy(float) - pred)
            kg["candidate_set"] = int(row["candidate_set"])
            kg["components"] = row["components"]
            kg["weights"] = row["weights"]
            kg["test_mode"] = mode
            pred_rows.append(kg)
            for label, mask in [
                ("overall", kg["endpoint_window"].isin([6, 12, 18, 24])),
                ("18m", kg["endpoint_window"].eq(18)),
                ("24m", kg["endpoint_window"].eq(24)),
                ("18+24m", kg["endpoint_window"].isin([18, 24])),
                ("12+18+24m", kg["endpoint_window"].isin([12, 18, 24])),
            ]:
                item = paired_stats(kg, rf, label, mask)
                item.update({"candidate_set": int(row["candidate_set"]), "components": row["components"], "weights": row["weights"], "test_mode": mode})
                eval_rows.append(item)
    preds = pd.concat(pred_rows, ignore_index=True)
    eval_df = pd.DataFrame(eval_rows)
    preds.to_csv(PRED / "nested_ensemble_selected_test_predictions.csv", index=False, encoding="utf-8-sig")
    eval_df.to_csv(TABLES / "nested_ensemble_test_evaluation_by_scope.csv", index=False, encoding="utf-8-sig")
    return preds, eval_df


def conformance(eval_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (set_id, mode), sub in eval_df.groupby(["candidate_set", "test_mode"]):
        lookup = {row["scope"]: row for _, row in sub.iterrows()}
        overall, pooled, h18, h24, long = lookup["overall"], lookup["12+18+24m"], lookup["18m"], lookup["24m"], lookup["18+24m"]
        full = (
            overall["delta_rf_minus_kg"] > 0
            and pooled["wilcoxon_p_kg_less"] < 0.05
            and h18["delta_rf_minus_kg"] > 0
            and h18["wilcoxon_p_kg_less"] < 0.05
            and h24["delta_rf_minus_kg"] > 0
            and h24["wilcoxon_p_kg_less"] < 0.05
            and h24["spearman_pred_true_p"] < 0.05
        )
        rows.append(
            {
                "candidate_set": set_id,
                "test_mode": mode,
                "fully_removes_previous_limits": bool(full),
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
                "components": sub.iloc[0]["components"],
                "weights": sub.iloc[0]["weights"],
            }
        )
    out = pd.DataFrame(rows).sort_values(["fully_removes_previous_limits", "pooled_12_18_24_delta"], ascending=[False, False])
    out.to_csv(TABLES / "nested_ensemble_limit_removal_audit.csv", index=False, encoding="utf-8-sig")
    return out


def make_figures(eval_df: pd.DataFrame, conf: pd.DataFrame) -> None:
    plot = eval_df[eval_df["scope"].isin(["overall", "18m", "24m", "12+18+24m"])].copy()
    fig, ax = plt.subplots(figsize=(12, 5.5))
    labels = []
    x = np.arange(len(plot["scope"].unique()))
    width = 0.1
    scopes = ["overall", "18m", "24m", "12+18+24m"]
    for i, ((set_id, mode), sub) in enumerate(plot.groupby(["candidate_set", "test_mode"])):
        sub = sub.set_index("scope").loc[scopes].reset_index()
        ax.bar(x + i * width, sub["delta_rf_minus_kg"], width=width, label=f"set{set_id}-{mode}")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x + width * 2.5, scopes)
    ax.set_ylabel("RF MAE - nested KG ensemble MAE")
    ax.set_title("Nested ensemble test margins")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_nested_ensemble_test_margins.png", dpi=180)
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    val_components, test_components = collect_components()
    for name, frame in val_components.items():
        frame.to_csv(PRED / f"val_component_{name}.csv", index=False, encoding="utf-8-sig")
    val_merged = merge_prediction_components(val_components)
    rf_val = load_rf_validation_best()
    selected = select_ensembles(val_merged, rf_val)
    preds, eval_df = evaluate_selected(test_components, selected)
    conf = conformance(eval_df)
    make_figures(eval_df, conf)
    provenance = {
        "created_by": "honest_nested_ensemble_experiments.py",
        "integrity_note": "Ensemble weights are selected using validation predictions only; RF is a comparator, not an ensemble component.",
        "test_set_used_for_selection": False,
        "component_models": list(val_components),
        "outputs": {
            "validation_rows": int(len(val_merged)),
            "selected_weight_sets": int(len(selected)),
            "test_prediction_rows": int(len(preds)),
            "evaluation_rows": int(len(eval_df)),
        },
    }
    (OUT / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(OUT), "conformance": conf.to_dict(orient="records")}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
