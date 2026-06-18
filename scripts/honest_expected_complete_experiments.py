from __future__ import annotations

import json
import math
import pickle
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import rankdata, spearmanr, wilcoxon
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path("/root/KG_LatentNet_Project")
SOURCE = ROOT / "results" / "honest_paper_repro_validation_top"
SOURCE_OPT = ROOT / "results" / "honest_paper_repro_validation_top_optimized"
OUT = ROOT / "results" / "honest_paper_repro_expected_complete"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
PRED = SOURCE / "predictions"
LATENT = ROOT / "results" / "latent" / "full_5fold"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DISPLAY = {
    "kg_latentnet_calibrated": "KG-LatentNet",
    "random_forest": "Random Forest (RF)",
    "xgboost": "XGBoost (XGB)",
    "hyperimts": "HyperIMTS",
    "trans": "TRANS",
    "dhgas": "DHGAS",
    "grud": "GRU-D",
    "graphcare": "GraphCare",
    "tgnn4i": "TGNN4I",
    "kedgn": "KEDGN",
}


VARIABLES = {
    "CRP": "CRP",
    "IL-6": "IL-6",
    "NLR": "NLR",
    "D-dimer": "D-二聚体",
    "Platelet": "血小板",
    "Neutrophil": "中性粒细胞",
    "Lymphocyte": "淋巴细胞",
    "LDL-C": "低密度脂蛋白",
    "Triglyceride": "甘油三酯",
    "Cholesterol": "胆固醇",
    "BMI": "BMI",
    "Age": "年龄",
}


def ensure_dirs() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)


def load_prediction(model: str) -> pd.DataFrame:
    if model == "kg_latentnet_calibrated":
        path = PRED / "kg_latentnet_calibrated_predictions.csv"
    else:
        path = PRED / f"{model}_stabilized_predictions.csv"
    df = pd.read_csv(path)
    if "absolute_error" not in df.columns:
        df["absolute_error"] = (df["y_true"] - df["y_pred"]).abs()
    return df


def metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float]:
    y_true = np.asarray(list(y_true), dtype=float)
    y_pred = np.asarray(list(y_pred), dtype=float)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def bootstrap_ci(values: np.ndarray, seed: int, n_boot: int = 2000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(values), len(values))
        stats.append(float(values[idx].mean()))
    return tuple(np.percentile(stats, [2.5, 97.5]).astype(float))


def corr_ci(x: pd.Series, y: pd.Series, seed: int, n_boot: int = 1000) -> dict[str, float]:
    sub = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(sub) < 8:
        return {"n": int(len(sub)), "spearman_r": math.nan, "p_value": math.nan, "ci_low": math.nan, "ci_high": math.nan}
    r, p = spearmanr(sub["x"], sub["y"])
    rng = np.random.default_rng(seed)
    vals = []
    arr = sub.to_numpy(float)
    for _ in range(n_boot):
        idx = rng.integers(0, len(arr), len(arr))
        rr, _ = spearmanr(arr[idx, 0], arr[idx, 1])
        if np.isfinite(rr):
            vals.append(float(rr))
    low, high = np.percentile(vals, [2.5, 97.5]) if vals else (math.nan, math.nan)
    return {"n": int(len(sub)), "spearman_r": float(r), "p_value": float(p), "ci_low": float(low), "ci_high": float(high)}


def rank_z(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    out = pd.Series(np.nan, index=series.index, dtype=float)
    mask = x.notna()
    if mask.sum() < 2:
        return out
    ranks = rankdata(x[mask]) / (mask.sum() + 1.0)
    out.loc[mask] = (ranks - np.nanmean(ranks)) / np.nanstd(ranks)
    return out


def collect_clinical_features() -> pd.DataFrame:
    frames = []
    for fold in range(5):
        with (ROOT / "data" / "processed" / "tabular" / f"fold_{fold}_tabular_test.pkl").open("rb") as handle:
            test = pickle.load(handle)
        names = list(test["feature_names"])
        X = test.get("X_raw", test["X"])
        frame = pd.DataFrame(
            {
                "patient_id": test["patient_id"],
                "fold": fold,
                "endpoint_window": test["endpoint_window"],
            }
        )
        for out_name, needle in VARIABLES.items():
            idx = [i for i, name in enumerate(names) if name.startswith("static::") and needle in name]
            if idx:
                frame[out_name] = X[:, idx[0]]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def collect_raw_latent() -> pd.DataFrame:
    rows = []
    for fold in range(5):
        path = LATENT / f"kg_latentnet_fold{fold}_latent_states.pkl"
        if not path.exists():
            continue
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        rows.extend(payload.get("patient_latent_rows", []))
    if not rows:
        return pd.DataFrame()
    latent = pd.DataFrame(rows)
    keep = [
        "patient_id",
        "fold",
        "endpoint_window",
        "latent_state_score",
        "short_contribution_score",
        "delayed_contribution_score",
    ]
    return latent[[c for c in keep if c in latent.columns]]


def all_model_comparison() -> pd.DataFrame:
    source = pd.read_csv(SOURCE / "tables" / "table1_main_model_comparison_real.csv")
    rows = []
    for model, display in DISPLAY.items():
        row = source[source["model_name"].eq(model)]
        if row.empty:
            continue
        r = row.iloc[0].to_dict()
        rows.append(
            {
                "model_name": model,
                "Method": display,
                "Category": r.get("Category", ""),
                "N": int(r["n"]),
                "MAE": float(r["MAE"]),
                "MAE_95CI": r["MAE_95CI"],
                "RMSE": float(r["RMSE"]),
                "RMSE_95CI": r["RMSE_95CI"],
                "R2": float(r["R2"]),
                "Rank_by_MAE": int(r["Rank_by_MAE"]),
            }
        )
    df = pd.DataFrame(rows).sort_values("Rank_by_MAE")
    df.to_csv(TABLES / "table_all_comparison_models_full.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(12, 7))
    colors = ["#c0392b" if m == "KG-LatentNet" else "#2c7fb8" for m in df["Method"]]
    xerr = np.vstack(
        [
            df["MAE"] - source.set_index("model_name").loc[df["model_name"], "MAE_95CI_low"].to_numpy(float),
            source.set_index("model_name").loc[df["model_name"], "MAE_95CI_high"].to_numpy(float) - df["MAE"],
        ]
    )
    ax.barh(df["Method"], df["MAE"], xerr=xerr, color=colors, edgecolor="white", capsize=4)
    ax.invert_yaxis()
    ax.set_xlabel("MAE")
    ax.set_title("All comparison models")
    for y, v in enumerate(df["MAE"]):
        ax.text(v + 0.004, y, f"{v:.4f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_all_comparison_models_full.png", dpi=180)
    plt.close(fig)
    return df


def small_sample_vs_rf() -> pd.DataFrame:
    kg = load_prediction("kg_latentnet_calibrated")
    rf = load_prediction("random_forest")
    merged = kg[["patient_id", "fold", "endpoint_window", "y_true", "y_pred"]].merge(
        rf[["patient_id", "fold", "endpoint_window", "y_pred"]],
        on=["patient_id", "fold", "endpoint_window"],
        suffixes=("_kg", "_rf"),
    )
    merged["kg_abs_error"] = (merged["y_true"] - merged["y_pred_kg"]).abs()
    merged["rf_abs_error"] = (merged["y_true"] - merged["y_pred_rf"]).abs()
    groups = [
        ("Overall", merged),
        ("6m", merged[merged["endpoint_window"].eq(6)]),
        ("12m", merged[merged["endpoint_window"].eq(12)]),
        ("18m", merged[merged["endpoint_window"].eq(18)]),
        ("24m", merged[merged["endpoint_window"].eq(24)]),
        ("18m+24m", merged[merged["endpoint_window"].isin([18, 24])]),
    ]
    rows = []
    for i, (label, sub) in enumerate(groups):
        diff = sub["rf_abs_error"].to_numpy(float) - sub["kg_abs_error"].to_numpy(float)
        low, high = bootstrap_ci(diff, seed=20260617 + i)
        try:
            p = float(wilcoxon(sub["kg_abs_error"], sub["rf_abs_error"], alternative="less").pvalue)
        except ValueError:
            p = math.nan
        rows.append(
            {
                "group": label,
                "n": int(len(sub)),
                "kg_mae": float(sub["kg_abs_error"].mean()),
                "rf_mae": float(sub["rf_abs_error"].mean()),
                "mae_reduction_vs_rf": float(diff.mean()),
                "mae_reduction_95ci_low": low,
                "mae_reduction_95ci_high": high,
                "wilcoxon_p_kg_less": p,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "table_small_sample_vs_rf.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(df))
    err = np.vstack(
        [
            df["mae_reduction_vs_rf"] - df["mae_reduction_95ci_low"],
            df["mae_reduction_95ci_high"] - df["mae_reduction_vs_rf"],
        ]
    )
    ax.bar(x, df["mae_reduction_vs_rf"], yerr=err, capsize=4, color="#c0392b", edgecolor="white")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x, df["group"])
    ax.set_ylabel("RF MAE - KG MAE")
    ax.set_title("Small-sample comparison against RF")
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_small_sample_vs_rf.png", dpi=180)
    plt.close(fig)
    return df


def knowledge_missingness_robustness() -> pd.DataFrame:
    kg = load_prediction("kg_latentnet_calibrated")
    anchor = kg["y_pred"].to_numpy(float) - kg["blend"].to_numpy(float) * kg["kg_residual_pred"].to_numpy(float)
    residual = kg["kg_residual_pred"].to_numpy(float)
    blend = kg["blend"].to_numpy(float)
    y = kg["y_true"].to_numpy(float)
    full_pred = kg["y_pred"].to_numpy(float)
    configs = [
        ("Full KG-LatentNet", "full", 0.0, 0),
        ("Knowledge-entry masking 30%", "mask", 0.3, 100),
        ("Knowledge-entry masking 60%", "mask", 0.6, 100),
        ("No structured knowledge contribution", "mask", 1.0, 1),
        ("Randomized KG residual entries", "permute_fold", 1.0, 100),
    ]
    rows = []
    for cidx, (setting, mode, rate, repeats) in enumerate(configs):
        preds = []
        if mode == "full":
            preds.append(full_pred)
        elif mode == "mask" and rate >= 1.0:
            preds.append(anchor)
        elif mode == "mask":
            for rep in range(repeats):
                rng = np.random.default_rng(20260617 + cidx * 1000 + rep)
                keep = (rng.random(len(kg)) > rate).astype(float)
                preds.append(anchor + blend * residual * keep)
        elif mode == "permute_fold":
            for rep in range(repeats):
                rng = np.random.default_rng(20260617 + cidx * 1000 + rep)
                shuffled = np.zeros(len(kg), dtype=float)
                for _fold, idx in kg.groupby("fold").groups.items():
                    arr = residual[list(idx)].copy()
                    rng.shuffle(arr)
                    shuffled[list(idx)] = arr
                preds.append(anchor + blend * shuffled)
        maes, rmses, r2s, cons = [], [], [], []
        for pred in preds:
            m = metrics(y, pred)
            maes.append(m["MAE"])
            rmses.append(m["RMSE"])
            r2s.append(m["R2"])
            cons.append(float(spearmanr(full_pred, pred).statistic))
        low, high = bootstrap_ci(np.array(maes), seed=20260617 + cidx, n_boot=1000) if len(maes) > 1 else (maes[0], maes[0])
        rows.append(
            {
                "analysis_type": "Structured knowledge missingness",
                "setting": setting,
                "repeats": int(len(preds)),
                "MAE": float(np.mean(maes)),
                "MAE_repeat_95CI_low": float(low),
                "MAE_repeat_95CI_high": float(high),
                "RMSE": float(np.mean(rmses)),
                "R2": float(np.mean(r2s)),
                "delta_MAE_vs_full": float(np.mean(maes) - metrics(y, full_pred)["MAE"]),
                "representation_consistency": float(np.mean(cons)),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "table_knowledge_missingness_robustness.csv", index=False, encoding="utf-8-sig")

    fig, ax1 = plt.subplots(figsize=(11, 5.5))
    ax1.bar(df["setting"], df["MAE"], color=["#c0392b"] + ["#2c7fb8"] * (len(df) - 1), edgecolor="white")
    ax1.set_ylabel("MAE")
    ax1.set_title("Structured knowledge missingness robustness")
    ax1.tick_params(axis="x", rotation=18)
    for i, v in enumerate(df["MAE"]):
        ax1.text(i, v + 0.0002, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_knowledge_missingness_robustness.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(df["setting"], df["delta_MAE_vs_full"], color=["#c0392b"] + ["#2c7fb8"] * (len(df) - 1), edgecolor="white")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("Delta MAE vs Full KG-LatentNet")
    ax.set_title("Structured knowledge missingness: zoomed degradation")
    ax.tick_params(axis="x", rotation=18)
    for i, v in enumerate(df["delta_MAE_vs_full"]):
        ax.text(i, v + 0.00001, f"{v:.6f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_knowledge_missingness_delta_mae.png", dpi=180)
    plt.close(fig)
    return df


def build_state_dataset() -> pd.DataFrame:
    base = pd.read_csv(SOURCE_OPT / "tables" / "optimized_state_score_dataset.csv")
    clinical = collect_clinical_features()
    latent = collect_raw_latent()
    df = base.merge(clinical, on=["patient_id", "fold", "endpoint_window"], how="left")
    if not latent.empty:
        df = df.merge(latent, on=["patient_id", "fold", "endpoint_window"], how="left")
    for col in ["endpoint_tbr_y", "baseline_tbr_b", "delta_tbr", *VARIABLES.keys()]:
        if col in df.columns:
            df[f"rz_{col}"] = rank_z(df[col])
    feature_cols = [
        "endpoint_state_score",
        "progression_state_score",
        "baseline_tbr_b",
        "latent_state_score",
        "short_contribution_score",
        "delayed_contribution_score",
        *VARIABLES.keys(),
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]
    targets = {
        "burden_inflammatory_state_score": ["rz_endpoint_tbr_y", "rz_CRP", "rz_IL-6", "rz_NLR"],
        "progression_inflammatory_state_score": ["rz_delta_tbr", "rz_CRP", "rz_IL-6", "rz_NLR"],
    }
    for score_name, target_cols in targets.items():
        usable_targets = [c for c in target_cols if c in df.columns]
        target = df[usable_targets].mean(axis=1, skipna=True)
        pred = np.full(len(df), np.nan)
        for fold in sorted(df["fold"].unique()):
            train_mask = df["fold"].ne(fold) & target.notna()
            test_mask = df["fold"].eq(fold)
            model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))
            model.fit(df.loc[train_mask, feature_cols], target.loc[train_mask])
            pred[test_mask] = model.predict(df.loc[test_mask, feature_cols])
        df[score_name] = pred
    df.to_csv(TABLES / "table_state_score_dataset_with_aligned_readouts.csv", index=False, encoding="utf-8-sig")
    return df


def state_associations(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    indicators = {
        "Endpoint TBR": "endpoint_tbr_y",
        "Baseline TBR": "baseline_tbr_b",
        "Delta TBR": "delta_tbr",
    }
    indicators.update({label: label for label in VARIABLES})
    scores = [
        "endpoint_state_score",
        "progression_state_score",
        "burden_inflammatory_state_score",
        "progression_inflammatory_state_score",
    ]
    rows, stage_rows = [], []
    for sidx, score in enumerate(scores):
        if score not in df.columns:
            continue
        for iidx, (label, col) in enumerate(indicators.items()):
            if col not in df.columns:
                continue
            row = corr_ci(df[score], df[col], seed=20260617 + sidx * 100 + iidx)
            rows.append({"state_score": score, "indicator": label, "variable": col, **row})
            for window, sub in df.groupby("endpoint_window"):
                st = corr_ci(sub[score], sub[col], seed=20260617 + int(window) + sidx * 100 + iidx, n_boot=400)
                stage_rows.append({"endpoint_window": int(window), "state_score": score, "indicator": label, "variable": col, **st})
    assoc = pd.DataFrame(rows)
    stage = pd.DataFrame(stage_rows)
    assoc.to_csv(TABLES / "table_state_score_clinical_associations_aligned.csv", index=False, encoding="utf-8-sig")
    stage.to_csv(TABLES / "table_variable_state_stage_level_associations.csv", index=False, encoding="utf-8-sig")

    plot_assoc = assoc[assoc["state_score"].isin(["burden_inflammatory_state_score", "progression_inflammatory_state_score"])]
    plot_assoc = plot_assoc[plot_assoc["indicator"].isin(["Endpoint TBR", "Delta TBR", "CRP", "IL-6", "NLR"])]
    labels = plot_assoc["state_score"].str.replace("_", " ") + " vs " + plot_assoc["indicator"]
    colors = ["#c0392b" if "burden" in s else "#31a354" for s in plot_assoc["state_score"]]
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(labels, plot_assoc["spearman_r"], color=colors)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Spearman r")
    ax.set_title("Aligned state-readout clinical associations")
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_state_score_clinical_associations_aligned.png", dpi=180)
    plt.close(fig)
    return assoc, stage


def variable_state_figures(stage: pd.DataFrame) -> None:
    selected = [
        "Endpoint TBR",
        "Baseline TBR",
        "Delta TBR",
        "CRP",
        "IL-6",
        "NLR",
        "D-dimer",
        "Platelet",
        "Neutrophil",
        "Lymphocyte",
        "LDL-C",
        "Triglyceride",
        "Cholesterol",
        "BMI",
        "Age",
    ]
    for score, filename, title in [
        ("burden_inflammatory_state_score", "figure_variable_state_stage_heatmap_burden.png", "Stage-level variable-state relations: burden-inflammatory state"),
        ("progression_inflammatory_state_score", "figure_variable_state_stage_heatmap_progression.png", "Stage-level variable-state relations: progression-inflammatory state"),
    ]:
        sub = stage[stage["state_score"].eq(score) & stage["indicator"].isin(selected)]
        pivot = sub.pivot_table(index="indicator", columns="endpoint_window", values="spearman_r")
        pivot = pivot.reindex([v for v in selected if v in pivot.index])
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(pivot.to_numpy(float), aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_yticks(range(len(pivot.index)), pivot.index)
        ax.set_xticks(range(len(pivot.columns)), [f"{int(c)}m" for c in pivot.columns])
        ax.set_title(title)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Spearman r")
        fig.tight_layout()
        fig.savefig(FIGURES / filename, dpi=180)
        plt.close(fig)


def main() -> None:
    ensure_dirs()
    comp = all_model_comparison()
    small = small_sample_vs_rf()
    robust = knowledge_missingness_robustness()
    state_df = build_state_dataset()
    assoc, stage = state_associations(state_df)
    variable_state_figures(stage)
    provenance = {
        "created_by": "honest_expected_complete_experiments.py",
        "integrity_note": "All reported values are recomputed from existing real out-of-fold predictions, processed fold data, and train-fold fitted post-hoc readouts. No table cells are hand-edited.",
        "main_source": str(SOURCE),
        "optimized_source": str(SOURCE_OPT),
        "small_sample": "Fixed comparator is Random Forest (RF), as requested.",
        "knowledge_missingness": "Counterfactual masking/randomization of the validation-selected KG residual contribution. This is not a full retraining no-knowledge ablation.",
        "aligned_state_readouts": "Post-hoc clinical alignment readouts trained on non-held-out folds. They are reported separately from the raw latent_state_score and model prediction scores.",
        "limitation": "Endpoint burden and longitudinal progression remain partially opposing axes in this dataset because baseline TBR is strongly associated with endpoint TBR and inversely associated with delta TBR.",
        "outputs": {
            "comparison_rows": int(len(comp)),
            "small_rows": int(len(small)),
            "knowledge_rows": int(len(robust)),
            "association_rows": int(len(assoc)),
            "stage_rows": int(len(stage)),
        },
    }
    (OUT / "expected_complete_provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(OUT), **provenance["outputs"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
