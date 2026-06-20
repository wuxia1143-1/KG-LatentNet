from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("/root/KG_LatentNet_Project")
OUT = ROOT / "results" / "paper_ready_single_model_results"
TABLES = OUT / "tables"
PREDICTIONS = OUT / "predictions"
FIGURES = OUT / "figures"
PROVENANCE = OUT / "provenance"

STATIC_CLINICAL_TOKENS = [
    "static::年龄",
    "static::性别",
    "static::体重",
    "static::民族",
    "既往疾病史",
    "static::烟龄",
    "static::酒龄",
    "static::BMI",
    "static::血压",
    "肿瘤临床分期",
    "肿瘤TNM分期",
    "肿瘤病理类型",
    "肿瘤病理-分化程度",
    "手术类型",
]

LABORATORY_TOKENS = [
    "CRP",
    "IL-6",
    "NLR",
    "D-二聚体",
    "血小板",
    "中性粒细胞",
    "淋巴细胞",
    "葡萄糖",
    "胆固醇",
    "低密度脂蛋白胆固醇",
    "高密度脂蛋白胆固醇",
    "甘油三酯",
    "BNP",
    "高敏肌钙蛋白T",
    "肾小球滤过率",
    "肌酐",
]

LABORATORY_AGGREGATIONS = [
    "::last",
    "::mean",
    "::max",
    "::slope",
    "::missing_indicator",
]

BASELINE_DEFINITIONS: list[dict[str, Any]] = [
    {
        "model_name": "baseline_tbr_only",
        "Method": "Baseline TBR only",
        "Category": "Clinical RF baseline",
        "feature_mode": "baseline_tbr_only",
        "Input variables": "Baseline TBR only.",
        "Purpose": "Imaging-only clinical baseline evaluating incremental value beyond initial PET/CT-derived vascular status.",
    },
    {
        "model_name": "static_clinical_variables_only",
        "Method": "Static clinical variables only",
        "Category": "Clinical RF baseline",
        "feature_mode": "static_clinical_only",
        "Input variables": "Age, sex, smoking/drinking history, BMI, blood pressure, comorbidity history, cancer stage, TNM, pathology, differentiation, and surgery type when available.",
        "Purpose": "Routine-clinical-information baseline without longitudinal laboratory changes or follow-up treatment trajectories.",
    },
    {
        "model_name": "clinical_treatment_history",
        "Method": "Clinical variables + treatment history",
        "Category": "Clinical RF baseline",
        "feature_mode": "clinical_treatment_history",
        "Input variables": "Static clinical variables plus treatment/history features for CT/RT/IO/TT/OP and cumulative intensity.",
        "Purpose": "Clinical-treatment baseline testing whether treatment process information predicts post-treatment vascular metabolic response.",
    },
    {
        "model_name": "clinical_imaging_baseline",
        "Method": "Baseline TBR + clinical variables + treatment history",
        "Category": "Clinical RF baseline",
        "feature_mode": "clinical_imaging_baseline",
        "Input variables": "Baseline TBR plus static clinical variables and treatment/history features.",
        "Purpose": "Main clinical-imaging baseline approximating information available after baseline PET/CT and during treatment follow-up.",
    },
    {
        "model_name": "strong_clinical_laboratory_baseline",
        "Method": "Baseline TBR + clinical + treatment + latest labs",
        "Category": "Clinical RF baseline",
        "feature_mode": "strong_clinical_laboratory_baseline",
        "Input variables": "Baseline TBR, static clinical variables, treatment/history features, and pre-endpoint laboratory summaries: latest, mean, max, slope, and missing indicators.",
        "Purpose": "Strongest tabular clinical baseline simulating baseline imaging plus recent laboratory and treatment information.",
    },
]


def load_helper():
    path = ROOT / "scripts" / "honest_real_final_outputs_validation_top.py"
    spec = importlib.util.spec_from_file_location("validation_top_helper", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["validation_top_helper"] = module
    spec.loader.exec_module(module)
    return module


HELPER = load_helper()


def ci_text(low: Any, high: Any, digits: int = 4) -> str:
    low_s = HELPER.fmt(low, digits)
    high_s = HELPER.fmt(high, digits)
    return f"[{low_s}, {high_s}]" if low_s and high_s else ""


def contains_any(text: str, tokens: list[str]) -> bool:
    lowered = text.lower()
    return any(token.lower() in lowered for token in tokens)


def selected_feature_names(feature_names: list[str], mode: str) -> tuple[list[int], list[str]]:
    names = [str(name) for name in feature_names]
    lowered = [name.lower() for name in names]

    baseline_idx = [idx for idx, name in enumerate(lowered) if "baseline_tbr_b" in name]
    static_idx = [idx for idx, name in enumerate(names) if contains_any(name, STATIC_CLINICAL_TOKENS)]
    treatment_history_idx = [
        idx for idx, name in enumerate(lowered) if "treatment::" in name or "history::" in name
    ]
    lab_idx = [
        idx
        for idx, name in enumerate(names)
        if name.startswith("dynamic::")
        and contains_any(name, LABORATORY_TOKENS)
        and any(name.endswith(suffix) for suffix in LABORATORY_AGGREGATIONS)
    ]

    if mode == "baseline_tbr_only":
        selected = baseline_idx
    elif mode == "static_clinical_only":
        selected = static_idx
    elif mode == "clinical_treatment_history":
        selected = static_idx + treatment_history_idx
    elif mode == "clinical_imaging_baseline":
        selected = baseline_idx + static_idx + treatment_history_idx
    elif mode == "strong_clinical_laboratory_baseline":
        selected = baseline_idx + static_idx + treatment_history_idx + lab_idx
    else:
        raise ValueError(f"Unsupported feature mode: {mode}")

    selected = sorted(set(selected))
    if not selected:
        raise RuntimeError(f"No features selected for mode={mode}")
    return selected, [names[idx] for idx in selected]


def make_rf_model(seed: int):
    return make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestRegressor(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=4,
            random_state=seed,
            n_jobs=-1,
        ),
    )


def fit_predict_fold(fold: int, definition: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    train = HELPER.load_tabular(fold, "train")
    test = HELPER.load_tabular(fold, "test")
    feature_names = [str(name) for name in train["feature_names"]]
    idx, used_names = selected_feature_names(feature_names, str(definition["feature_mode"]))

    x_train = np.asarray(train["X"], dtype=np.float64)[:, idx]
    y_train = np.asarray(train["y"], dtype=np.float64).reshape(-1)
    x_test = np.asarray(test["X"], dtype=np.float64)[:, idx]
    y_test = np.asarray(test["y"], dtype=np.float64).reshape(-1)

    seed = 20260620 + 1009 * fold + sum(ord(ch) for ch in str(definition["model_name"]))
    model = make_rf_model(seed)
    model.fit(x_train, y_train)
    raw_pred = np.asarray(model.predict(x_test), dtype=np.float64)
    low, high = HELPER.fold_bounds(fold)
    pred = np.clip(raw_pred, low, high)
    metric = HELPER.metrics(y_test, pred)

    rows = []
    for i, (pid, window, yt, yp) in enumerate(zip(test["patient_id"], test["endpoint_window"], y_test, pred, strict=False)):
        rows.append(
            {
                "model_name": definition["model_name"],
                "Method": definition["Method"],
                "fold": fold,
                "patient_id": str(pid),
                "endpoint_window": int(window),
                "y_true": float(yt),
                "y_pred": float(yp),
                "absolute_error": float(abs(yt - yp)),
                "raw_y_pred": float(raw_pred[i]),
                "train_clip_low": float(low),
                "train_clip_high": float(high),
                "was_clipped": bool(float(raw_pred[i]) != float(yp)),
                "feature_mode": definition["feature_mode"],
                "model_type": "RandomForestRegressor",
                "n_features": int(len(idx)),
            }
        )
    fold_row = {
        "model_name": definition["model_name"],
        "Method": definition["Method"],
        "fold": fold,
        "MAE": metric["mae"],
        "RMSE": metric["rmse"],
        "R2": metric["r2"],
        "N": metric["n"],
        "n_features": int(len(idx)),
        "model_type": "RandomForestRegressor",
    }
    return pd.DataFrame(rows), fold_row, used_names


def summarize_predictions(definition: dict[str, Any], pred: pd.DataFrame, fold_df: pd.DataFrame) -> dict[str, Any]:
    y = pred["y_true"].to_numpy(np.float64)
    yp = pred["y_pred"].to_numpy(np.float64)
    m = HELPER.metrics(y, yp)
    ci = HELPER.bootstrap_metric_ci(y, yp, seed=HELPER.stable_seed(str(definition["model_name"])))
    folds = fold_df[fold_df["model_name"].eq(definition["model_name"])]
    return {
        "model_name": definition["model_name"],
        "Method": definition["Method"],
        "Category": definition["Category"],
        "MAE": m["mae"],
        "MAE_95CI": ci_text(ci["mae_95ci_low"], ci["mae_95ci_high"]),
        "MAE_95CI_low": ci["mae_95ci_low"],
        "MAE_95CI_high": ci["mae_95ci_high"],
        "RMSE": m["rmse"],
        "RMSE_95CI": ci_text(ci["rmse_95ci_low"], ci["rmse_95ci_high"]),
        "RMSE_95CI_low": ci["rmse_95ci_low"],
        "RMSE_95CI_high": ci["rmse_95ci_high"],
        "R2": m["r2"],
        "R2_95CI": ci_text(ci["r2_95ci_low"], ci["r2_95ci_high"]),
        "R2_95CI_low": ci["r2_95ci_low"],
        "R2_95CI_high": ci["r2_95ci_high"],
        "N": m["n"],
        "n_features_mean": float(folds["n_features"].mean()) if not folds.empty else math.nan,
        "model_type": "RandomForestRegressor",
        "feature_set": definition["Input variables"],
        "Input variables": definition["Input variables"],
        "Purpose": definition["Purpose"],
    }


def update_comparison_figure(table: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    ordered = table.sort_values("MAE", ascending=True)
    fig_height = max(5.5, 0.38 * len(ordered) + 1.2)
    fig, ax = plt.subplots(figsize=(9.8, fig_height))
    colors = []
    for _, row in ordered.iterrows():
        if row["Category"] == "Proposed":
            colors.append("#c0392b")
        elif "Clinical" in str(row["Category"]):
            colors.append("#7f8c8d")
        else:
            colors.append("#2c7fb8")
    ax.barh(ordered["Method"], ordered["MAE"].astype(float), color=colors, edgecolor="white")
    ax.invert_yaxis()
    ax.set_xlabel("MAE")
    ax.set_title("Five-fold model comparison")
    for y, value in enumerate(ordered["MAE"].astype(float)):
        ax.text(value + 0.004, y, f"{value:.4f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "Figure1_complete_model_comparison.png", dpi=180)
    plt.close(fig)


def load_base_table() -> pd.DataFrame:
    base_path = TABLES / "Table1_complete_model_comparison_with_ci.csv"
    fallback_path = ROOT / "results" / "honest_paper_repro_validation_top" / "tables" / "table1_main_model_comparison_real.csv"
    source_path = base_path if base_path.exists() else fallback_path
    base = pd.read_csv(source_path)
    if "N" not in base.columns and "n" in base.columns:
        base["N"] = base["n"]
    if "n" in base.columns:
        base = base.drop(columns=["n"])
    base.attrs["source_path"] = str(source_path)
    return base


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    PROVENANCE.mkdir(parents=True, exist_ok=True)

    pred_frames = []
    fold_rows = []
    summary_rows = []
    features_by_model: dict[str, list[str]] = {}
    for definition in BASELINE_DEFINITIONS:
        frames = []
        all_names: set[str] = set()
        for fold in range(5):
            pred, fold_row, used_names = fit_predict_fold(fold, definition)
            frames.append(pred)
            fold_rows.append(fold_row)
            all_names.update(used_names)
        all_pred = pd.concat(frames, ignore_index=True)
        pred_frames.append(all_pred)
        fold_df_partial = pd.DataFrame(fold_rows)
        summary_rows.append(summarize_predictions(definition, all_pred, fold_df_partial))
        features_by_model[str(definition["model_name"])] = sorted(all_names)

    new_summary = pd.DataFrame(summary_rows)
    all_predictions = pd.concat(pred_frames, ignore_index=True)
    fold_df = pd.DataFrame(fold_rows)

    base = load_base_table()
    for col in ["n_features_mean", "model_type", "feature_set", "Input variables", "Purpose"]:
        if col not in base.columns:
            base[col] = np.nan if col == "n_features_mean" else ""

    drop_names = set(new_summary["model_name"].astype(str))
    combined = pd.concat([base[~base["model_name"].astype(str).isin(drop_names)], new_summary], ignore_index=True, sort=False)
    combined["MAE"] = combined["MAE"].astype(float)
    combined = combined.sort_values(["MAE", "Method"], ascending=[True, True]).reset_index(drop=True)
    combined["Rank_by_MAE"] = np.arange(1, len(combined) + 1)

    preferred_cols = [
        "model_name",
        "Method",
        "Category",
        "MAE",
        "MAE_95CI",
        "RMSE",
        "RMSE_95CI",
        "R2",
        "R2_95CI",
        "N",
        "Rank_by_MAE",
        "MAE_95CI_low",
        "MAE_95CI_high",
        "RMSE_95CI_low",
        "RMSE_95CI_high",
        "R2_95CI_low",
        "R2_95CI_high",
        "n_features_mean",
        "model_type",
        "feature_set",
        "Input variables",
        "Purpose",
    ]
    extra_cols = [col for col in combined.columns if col not in preferred_cols]
    combined = combined[preferred_cols + extra_cols]

    comparison_path = TABLES / "Table1_complete_model_comparison_with_ci.csv"
    combined.to_csv(comparison_path, index=False, encoding="utf-8-sig")
    new_summary.to_csv(TABLES / "Table1_clinical_rf_baselines.csv", index=False, encoding="utf-8-sig")
    fold_df.to_csv(TABLES / "Table1_clinical_rf_baseline_fold_results.csv", index=False, encoding="utf-8-sig")
    all_predictions.to_csv(PREDICTIONS / "comparison_clinical_rf_baseline_predictions.csv", index=False, encoding="utf-8-sig")
    update_comparison_figure(combined)

    provenance = {
        "created_by": "regenerate_comparison_table_with_clinical_baselines.py",
        "data_sources": ["data/processed/tabular/fold_*_tabular_{train,test}.pkl"],
        "base_comparison_table_source": base.attrs.get("source_path"),
        "integrity_note": "The five added clinical comparison baselines are Random Forest models refit on the original five training folds and evaluated on the corresponding held-out test folds. No table values are hand edited.",
        "static_clinical_tokens": STATIC_CLINICAL_TOKENS,
        "laboratory_tokens": LABORATORY_TOKENS,
        "laboratory_aggregations": LABORATORY_AGGREGATIONS,
        "features_by_model": features_by_model,
        "outputs": {
            "comparison_table": str(comparison_path),
            "added_summary": str(TABLES / "Table1_clinical_rf_baselines.csv"),
            "fold_results": str(TABLES / "Table1_clinical_rf_baseline_fold_results.csv"),
            "predictions": str(PREDICTIONS / "comparison_clinical_rf_baseline_predictions.csv"),
        },
    }
    (PROVENANCE / "comparison_clinical_rf_baselines_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"updated": str(comparison_path), "added_rows": len(new_summary), "total_rows": len(combined)}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
