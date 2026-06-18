from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scipy.stats import spearmanr, ttest_rel, wilcoxon
from sklearn.linear_model import LinearRegression


ROOT = Path("/root/KG_LatentNet_Project")
OUT = ROOT / "results" / "honest_paper_repro_structural_revision"
TABLES = OUT / "tables"
SOURCE = ROOT / "results" / "honest_paper_repro_validation_top"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


STRUCT = load_module("structural_revision", ROOT / "scripts" / "honest_structural_revision_experiments.py")
HELPER = STRUCT.HELPER


def combine_tabular(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key in left:
        if key == "feature_names":
            out[key] = left[key]
        elif key in {"X", "y", "patient_id", "endpoint_window"}:
            out[key] = np.concatenate([np.asarray(left[key]), np.asarray(right[key])], axis=0)
        else:
            try:
                out[key] = np.concatenate([np.asarray(left[key]), np.asarray(right[key])], axis=0)
            except Exception:
                out[key] = left[key]
    return out


def fit_anchor_train_test(train: dict[str, Any], test: dict[str, Any], feature_names: list[str], mode: str) -> tuple[np.ndarray, np.ndarray]:
    idx = HELPER.clinical_feature_indices(feature_names, mode)
    x_train = train["X"][:, idx]
    x_test = test["X"][:, idx]
    if mode in {"clinical_horizon_aware", "baseline_tbr_horizon"}:
        x_train, x_test = HELPER.horizon_onehot(train, test, x_train, x_test)
    model = LinearRegression()
    model.fit(x_train, train["y"])
    return np.asarray(model.predict(x_train), dtype=float), np.asarray(model.predict(x_test), dtype=float)


def spec_by_name(name: str, seed: int):
    for spec in STRUCT.residual_specs(seed):
        if spec.name == name:
            return spec
    raise KeyError(name)


def refit_rule(row: pd.Series) -> pd.DataFrame:
    frames = []
    for fold in range(5):
        train = STRUCT.load_tabular(fold, "train")
        val = STRUCT.load_tabular(fold, "val")
        test = STRUCT.load_tabular(fold, "test")
        trainval = combine_tabular(train, val)
        feature_names = [str(name) for name in train["feature_names"]]
        y_trainval = np.asarray(trainval["y"], dtype=float).reshape(-1)
        y_test = np.asarray(test["y"], dtype=float).reshape(-1)
        w_trainval = np.asarray(trainval["endpoint_window"])
        w_test = np.asarray(test["endpoint_window"])
        low, high = STRUCT.fold_bounds(fold)

        anchor_trainval, anchor_test = fit_anchor_train_test(trainval, test, feature_names, str(row["anchor_mode"]))
        if str(row["residual_model"]) == "none":
            base_trainval = np.clip(anchor_trainval, low, high)
            base_test = np.clip(anchor_test, low, high)
            residual_test = np.zeros_like(y_test)
        else:
            idx = STRUCT.kg_feature_indices(feature_names, str(row["feature_mode"]))
            xtr0 = trainval["X"][:, idx]
            xte0 = test["X"][:, idx]
            xtr, xte = STRUCT.augment_features(xtr0, xte0, w_trainval, w_test, str(row["feature_variant"]))
            residual_target = y_trainval - anchor_trainval
            spec = spec_by_name(str(row["residual_model"]), 20260617 + fold * 100)
            weights = STRUCT.sample_weights(w_trainval, str(row["weight_scheme"]))
            r_trainval, r_test = STRUCT.fit_predict_residual(spec, xtr, residual_target, xtr, xte, weights)
            blend = float(row["blend"]) if pd.notna(row["blend"]) else 0.0
            base_trainval = np.clip(anchor_trainval + blend * r_trainval, low, high)
            base_test = np.clip(anchor_test + blend * r_test, low, high)
            residual_test = r_test

        cal_mode = str(row["calibration"])
        calibrator = STRUCT.fit_calibrator(y_trainval, base_trainval, w_trainval, cal_mode)
        pred_test = np.clip(calibrator(base_test, w_test), low, high)
        rows = []
        for i, (pid, window, yt, yp) in enumerate(zip(test["patient_id"], w_test, y_test, pred_test, strict=False)):
            rows.append(
                {
                    "patient_id": str(pid),
                    "fold": fold,
                    "endpoint_window": int(window),
                    "y_true": float(yt),
                    "y_pred": float(yp),
                    "absolute_error": float(abs(yt - yp)),
                    "selection_rule": str(row["selection_rule"]),
                    "selection_key": str(row["selection_key"]),
                    "anchor_mode": str(row["anchor_mode"]),
                    "feature_mode": str(row["feature_mode"]),
                    "feature_variant": str(row["feature_variant"]),
                    "residual_model": str(row["residual_model"]),
                    "weight_scheme": str(row["weight_scheme"]),
                    "blend": float(row["blend"]) if pd.notna(row["blend"]) else math.nan,
                    "calibration": cal_mode,
                    "final_refit": "train_plus_val",
                    "kg_residual_pred": float(residual_test[i]),
                    "train_clip_low": float(low),
                    "train_clip_high": float(high),
                }
            )
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


def evaluate(preds: pd.DataFrame) -> pd.DataFrame:
    rf = pd.read_csv(SOURCE / "predictions" / "random_forest_stabilized_predictions.csv")
    rows = []
    for rule, kg in preds.groupby("selection_rule"):
        for label, mask in [
            ("overall", kg["endpoint_window"].isin([6, 12, 18, 24])),
            ("6m", kg["endpoint_window"].eq(6)),
            ("12m", kg["endpoint_window"].eq(12)),
            ("18m", kg["endpoint_window"].eq(18)),
            ("24m", kg["endpoint_window"].eq(24)),
            ("12+18+24m", kg["endpoint_window"].isin([12, 18, 24])),
            ("18+24m", kg["endpoint_window"].isin([18, 24])),
        ]:
            item = paired_stats(kg, rf, label, mask)
            item["selection_rule"] = rule
            rows.append(item)
    return pd.DataFrame(rows)


def conformance(eval_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rule, sub in eval_df.groupby("selection_rule"):
        lookup = {row["scope"]: row for _, row in sub.iterrows()}
        overall, pooled, h18, h24, long = lookup["overall"], lookup["12+18+24m"], lookup["18m"], lookup["24m"], lookup["18+24m"]
        ok = (
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
                "selection_rule": rule,
                "fully_removes_previous_limits_after_refit": bool(ok),
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
    return pd.DataFrame(rows).sort_values(["fully_removes_previous_limits_after_refit", "overall_delta_rf_minus_kg"], ascending=[False, False])


def expanded_selected_rules() -> pd.DataFrame:
    selected = pd.read_csv(TABLES / "structural_revision_selected_rules.csv")
    grouped = pd.read_csv(TABLES / "structural_revision_global_validation_rank.csv")
    rows = [row.to_dict() for _, row in selected.iterrows()]

    def add_rule(rule_name: str, frame: pd.DataFrame, sort_cols: list[str], ascending: list[bool]) -> None:
        if frame.empty:
            frame = grouped.copy()
        row = frame.sort_values(sort_cols, ascending=ascending).iloc[0].to_dict()
        row["selection_rule"] = rule_name
        rows.append(row)

    eligible = grouped[grouped["mean_val_delta_vs_rf"].gt(0)].copy()
    add_rule(
        "best_18m_validation_margin",
        eligible,
        ["mean_val_18_delta_vs_rf", "mean_val_mae"],
        [False, True],
    )
    add_rule(
        "best_24m_validation_margin",
        eligible,
        ["mean_val_24_delta_vs_rf", "mean_val_mae"],
        [False, True],
    )
    both = eligible[(eligible["mean_val_18_delta_vs_rf"].gt(0)) & (eligible["mean_val_24_delta_vs_rf"].gt(0))].copy()
    both["min_18_24_val_delta"] = both[["mean_val_18_delta_vs_rf", "mean_val_24_delta_vs_rf"]].min(axis=1)
    add_rule(
        "max_min_18m_24m_validation_margin",
        both,
        ["min_18_24_val_delta", "mean_val_mae"],
        [False, True],
    )
    out = pd.DataFrame(rows)
    out = out.drop_duplicates(["selection_rule"], keep="last").reset_index(drop=True)
    out.to_csv(TABLES / "structural_revision_expanded_selected_rules.csv", index=False, encoding="utf-8-sig")
    return out


def main() -> None:
    selected = expanded_selected_rules()
    frames = []
    for _, row in selected.iterrows():
        frames.append(refit_rule(row))
    preds = pd.concat(frames, ignore_index=True)
    preds.to_csv(OUT / "predictions" / "structural_revision_final_refit_predictions.csv", index=False, encoding="utf-8-sig")
    eval_df = evaluate(preds)
    eval_df.to_csv(TABLES / "structural_revision_final_refit_test_evaluation_by_scope.csv", index=False, encoding="utf-8-sig")
    conf = conformance(eval_df)
    conf.to_csv(TABLES / "structural_revision_final_refit_limit_removal_audit.csv", index=False, encoding="utf-8-sig")
    provenance = {
        "created_by": "honest_structural_revision_refit.py",
        "selection_source": str(TABLES / "structural_revision_selected_rules.csv"),
        "final_training_protocol": "After validation-only model selection, refit selected structures on train+val for each fold; test remains held out.",
        "test_set_used_for_selection": False,
    }
    (OUT / "final_refit_provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(OUT), "conformance": conf.to_dict(orient="records")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
