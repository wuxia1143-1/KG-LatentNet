from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from scipy.stats import wilcoxon
except Exception:  # pragma: no cover - scipy is available in the project env.
    wilcoxon = None


ROOT = Path("/root/KG_LatentNet_Project")
OUT = ROOT / "results" / "paper_ready_single_model_results"
TABLES = OUT / "tables"
PRED = OUT / "predictions"
PROV = OUT / "provenance"

FOLDS = range(5)
FULL_FEATURE_TOKENS = ["dynamic::", "treatment::", "history::"]


def load_helper():
    path = ROOT / "scripts" / "honest_real_final_outputs_validation_top.py"
    spec = importlib.util.spec_from_file_location("validation_top_helper", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["validation_top_helper"] = module
    spec.loader.exec_module(module)
    return module


HELPER = load_helper()


VARIANTS: list[dict[str, Any]] = [
    {
        "Variant": "Full KG-LatentNet",
        "mode": "residual_fusion",
        "include_tokens": FULL_FEATURE_TOKENS,
        "all_features": False,
        "anchor_mode": "baseline_tbr_only",
        "residual_model": "ridge_100",
        "blend": 0.005,
        "Component setting": "short-term + delayed pathways, KG-guided feature mask, contribution-aware residual fusion",
    },
    {
        "Variant": "w/o short-term pathway",
        "mode": "residual_fusion",
        "include_tokens": ["treatment::", "history::"],
        "all_features": False,
        "anchor_mode": "baseline_tbr_only",
        "residual_model": "ridge_100",
        "blend": 0.005,
        "Component setting": "dynamic short-term marker pathway removed; delayed treatment/history pathway retained",
    },
    {
        "Variant": "w/o delayed pathway",
        "mode": "direct_state",
        "include_tokens": ["dynamic::"],
        "all_features": False,
        "anchor_mode": "none",
        "residual_model": "ridge_100",
        "blend": math.nan,
        "Component setting": "baseline/delayed anchor and treatment/history pathway removed; dynamic short-term pathway retained",
    },
    {
        "Variant": "w/o structured knowledge guidance",
        "mode": "residual_fusion",
        "include_tokens": [],
        "all_features": True,
        "anchor_mode": "baseline_tbr_only",
        "residual_model": "ridge_100",
        "blend": 0.005,
        "Component setting": "KG-guided feature mask removed; residual readout trained on all processed tabular features",
    },
    {
        "Variant": "single-path state update",
        "mode": "direct_state",
        "include_tokens": FULL_FEATURE_TOKENS,
        "all_features": False,
        "anchor_mode": "none",
        "residual_model": "ridge_100",
        "blend": math.nan,
        "Component setting": "single direct latent-state readout without anchor-residual state update",
    },
    {
        "Variant": "w/o contribution-aware fusion",
        "mode": "residual_fusion",
        "include_tokens": FULL_FEATURE_TOKENS,
        "all_features": False,
        "anchor_mode": "baseline_tbr_only",
        "residual_model": "ridge_100",
        "blend": 1.0,
        "Component setting": "contribution-aware blend removed; unweighted residual update used before train-range clipping",
    },
]


def ensure_dirs() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    PRED.mkdir(parents=True, exist_ok=True)
    PROV.mkdir(parents=True, exist_ok=True)


def feature_indices(feature_names: list[str], variant: dict[str, Any]) -> list[int]:
    if variant.get("all_features"):
        return list(range(len(feature_names)))

    tokens = [str(token).lower() for token in variant["include_tokens"]]
    selected = [idx for idx, name in enumerate(feature_names) if any(token in str(name).lower() for token in tokens)]
    if not selected:
        raise RuntimeError(f"No features selected for {variant['Variant']}.")
    return selected


def fit_variant_fold(fold: int, variant: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    train = HELPER.load_tabular(fold, "train")
    val = HELPER.load_tabular(fold, "val")
    test = HELPER.load_tabular(fold, "test")
    feature_names = [str(name) for name in train["feature_names"]]
    idx = feature_indices(feature_names, variant)

    x_train = np.asarray(train["X"], dtype=np.float64)[:, idx]
    x_val = np.asarray(val["X"], dtype=np.float64)[:, idx]
    x_test = np.asarray(test["X"], dtype=np.float64)[:, idx]
    y_train = np.asarray(train["y"], dtype=np.float64).reshape(-1)
    y_val = np.asarray(val["y"], dtype=np.float64).reshape(-1)
    y_test = np.asarray(test["y"], dtype=np.float64).reshape(-1)
    low, high = HELPER.fold_bounds(fold)

    model = HELPER.residual_models()[variant["residual_model"]]
    if variant["mode"] == "direct_state":
        model.fit(x_train, y_train)
        pred_val_raw = np.asarray(model.predict(x_val), dtype=np.float64)
        pred_test_raw = np.asarray(model.predict(x_test), dtype=np.float64)
        anchor_test = np.full_like(y_test, np.nan, dtype=np.float64)
        residual_test = pred_test_raw.copy()
    else:
        anchor = HELPER.fit_anchor(train, val, test, feature_names, variant["anchor_mode"])
        residual_target = y_train - np.asarray(anchor["train"], dtype=np.float64)
        model.fit(x_train, residual_target)
        r_val = np.asarray(model.predict(x_val), dtype=np.float64)
        residual_test = np.asarray(model.predict(x_test), dtype=np.float64)
        blend = float(variant["blend"])
        pred_val_raw = np.asarray(anchor["val"], dtype=np.float64) + blend * r_val
        pred_test_raw = np.asarray(anchor["test"], dtype=np.float64) + blend * residual_test
        anchor_test = np.asarray(anchor["test"], dtype=np.float64)

    pred_val = np.clip(pred_val_raw, low, high)
    pred_test = np.clip(pred_test_raw, low, high)
    val_metric = HELPER.metrics(y_val, pred_val)
    test_metric = HELPER.metrics(y_test, pred_test)

    rows = []
    for i, (pid, window, yt, yp) in enumerate(zip(test["patient_id"], test["endpoint_window"], y_test, pred_test, strict=False)):
        rows.append(
            {
                "Variant": variant["Variant"],
                "patient_id": str(pid),
                "fold": fold,
                "endpoint_window": int(window),
                "y_true": float(yt),
                "y_pred": float(yp),
                "absolute_error": float(abs(yt - yp)),
                "raw_y_pred": float(pred_test_raw[i]),
                "anchor_pred": float(anchor_test[i]) if np.isfinite(anchor_test[i]) else math.nan,
                "kg_component_pred": float(residual_test[i]),
                "train_clip_low": float(low),
                "train_clip_high": float(high),
                "was_clipped": bool(float(pred_test_raw[i]) != float(yp)),
                "mode": variant["mode"],
                "anchor_mode": variant["anchor_mode"],
                "residual_model": variant["residual_model"],
                "blend": variant["blend"],
                "n_features": int(len(idx)),
            }
        )
    fold_info = {
        "Variant": variant["Variant"],
        "fold": fold,
        "val_mae": val_metric["mae"],
        "val_rmse": val_metric["rmse"],
        "val_r2": val_metric["r2"],
        "test_mae": test_metric["mae"],
        "test_rmse": test_metric["rmse"],
        "test_r2": test_metric["r2"],
        "test_n": test_metric["n"],
        "n_features": int(len(idx)),
        "train_clip_low": float(low),
        "train_clip_high": float(high),
    }
    return pd.DataFrame(rows), fold_info


def ci_text(low: Any, high: Any, digits: int = 4) -> str:
    low_s = HELPER.fmt(low, digits)
    high_s = HELPER.fmt(high, digits)
    return f"[{low_s}, {high_s}]" if low_s and high_s else ""


def paired_p_value(full: pd.DataFrame, variant: pd.DataFrame) -> float:
    if wilcoxon is None:
        return math.nan
    keys = ["fold", "patient_id", "endpoint_window"]
    merged = full[keys + ["absolute_error"]].merge(
        variant[keys + ["absolute_error"]],
        on=keys,
        suffixes=("_full", "_variant"),
    )
    if merged.empty:
        return math.nan
    diff = merged["absolute_error_variant"].to_numpy(float) - merged["absolute_error_full"].to_numpy(float)
    if np.allclose(diff, 0):
        return 1.0
    try:
        return float(wilcoxon(diff, alternative="greater", zero_method="wilcox").pvalue)
    except Exception:
        return math.nan


def summarize_variant(
    variant: dict[str, Any],
    pred: pd.DataFrame,
    fold_df: pd.DataFrame,
    full_pred: pd.DataFrame | None,
    full_mae: float | None,
) -> dict[str, Any]:
    y = pred["y_true"].to_numpy(np.float64)
    yp = pred["y_pred"].to_numpy(np.float64)
    m = HELPER.metrics(y, yp)
    ci = HELPER.bootstrap_metric_ci(y, yp, seed=HELPER.stable_seed("component_ablation:" + variant["Variant"]))
    p_value = math.nan if full_pred is None or variant["Variant"] == "Full KG-LatentNet" else paired_p_value(full_pred, pred)
    delta = math.nan if full_mae is None else float(m["mae"] - full_mae)
    rel = math.nan if full_mae in (None, 0) else float(delta / full_mae * 100.0)
    return {
        "Variant": variant["Variant"],
        "Component setting": variant["Component setting"],
        "MAE": m["mae"],
        "MAE_95CI": ci_text(ci["mae_95ci_low"], ci["mae_95ci_high"]),
        "MAE_95CI_low": ci["mae_95ci_low"],
        "MAE_95CI_high": ci["mae_95ci_high"],
        "MAE_delta_vs_full": delta,
        "Relative_MAE_increase_vs_full_pct": rel,
        "RMSE": m["rmse"],
        "RMSE_95CI": ci_text(ci["rmse_95ci_low"], ci["rmse_95ci_high"]),
        "RMSE_95CI_low": ci["rmse_95ci_low"],
        "RMSE_95CI_high": ci["rmse_95ci_high"],
        "R2": m["r2"],
        "R2_95CI": ci_text(ci["r2_95ci_low"], ci["r2_95ci_high"]),
        "R2_95CI_low": ci["r2_95ci_low"],
        "R2_95CI_high": ci["r2_95ci_high"],
        "paired_wilcoxon_p_vs_full": p_value,
        "mean_validation_MAE": float(fold_df.loc[fold_df["Variant"].eq(variant["Variant"]), "val_mae"].mean()),
        "fold_mae_mean": float(fold_df.loc[fold_df["Variant"].eq(variant["Variant"]), "test_mae"].mean()),
        "fold_mae_std": float(fold_df.loc[fold_df["Variant"].eq(variant["Variant"]), "test_mae"].std(ddof=1)),
        "n_features_mean": float(fold_df.loc[fold_df["Variant"].eq(variant["Variant"]), "n_features"].mean()),
        "n": m["n"],
        "n_clipped": int(pred["was_clipped"].sum()),
        "prediction_source": "server rerun: same KG-LatentNet residual-readout code with component-level switches",
    }


def main() -> None:
    ensure_dirs()

    pred_frames = []
    fold_rows = []
    predictions_by_variant: dict[str, pd.DataFrame] = {}
    for variant in VARIANTS:
        variant_frames = []
        for fold in FOLDS:
            fold_pred, fold_info = fit_variant_fold(fold, variant)
            variant_frames.append(fold_pred)
            fold_rows.append(fold_info)
        variant_pred = pd.concat(variant_frames, ignore_index=True)
        predictions_by_variant[variant["Variant"]] = variant_pred
        pred_frames.append(variant_pred)

    all_predictions = pd.concat(pred_frames, ignore_index=True)
    fold_df = pd.DataFrame(fold_rows)
    full_pred = predictions_by_variant["Full KG-LatentNet"]
    full_mae = HELPER.metrics(full_pred["y_true"], full_pred["y_pred"])["mae"]

    summary_rows = [
        summarize_variant(variant, predictions_by_variant[variant["Variant"]], fold_df, full_pred, full_mae)
        for variant in VARIANTS
    ]
    summary = pd.DataFrame(summary_rows)

    order = {variant["Variant"]: idx for idx, variant in enumerate(VARIANTS)}
    summary["Display_order"] = summary["Variant"].map(order)
    summary = summary.sort_values("Display_order").drop(columns=["Display_order"])

    summary.to_csv(TABLES / "Table2_ablation_experiment.csv", index=False, encoding="utf-8-sig")
    fold_df.to_csv(TABLES / "Table2_component_ablation_fold_results.csv", index=False, encoding="utf-8-sig")
    all_predictions.to_csv(PRED / "component_ablation_predictions.csv", index=False, encoding="utf-8-sig")

    provenance = {
        "created_by": "regenerate_component_ablation_table.py",
        "data_sources": [
            "data/processed/tabular/fold_*_tabular_{train,val,test}.pkl",
            "scripts/honest_real_final_outputs_validation_top.py",
        ],
        "full_variant": "baseline_tbr_only anchor + ridge_100 KG residual over dynamic/treatment/history features + blend 0.005",
        "integrity_note": "All rows are recomputed from the same fold splits and KG-LatentNet residual-readout code; no table values are hand edited.",
        "outputs": {
            "summary_table": str(TABLES / "Table2_ablation_experiment.csv"),
            "fold_table": str(TABLES / "Table2_component_ablation_fold_results.csv"),
            "prediction_rows": str(PRED / "component_ablation_predictions.csv"),
        },
    }
    (PROV / "component_ablation_provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(TABLES / "Table2_ablation_experiment.csv"), "rows": int(len(summary))}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
