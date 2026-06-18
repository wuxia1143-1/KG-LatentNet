from __future__ import annotations

import json
import math
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.ensemble import ExtraTreesRegressor


ROOT = Path("/root/KG_LatentNet_Project")
SOURCE = ROOT / "results" / "honest_paper_repro_validation_top"
OUT = ROOT / "results" / "honest_paper_repro_validation_top_optimized"
ROBUSTNESS_SOURCE = ROOT / "results" / "honest_paper_repro_real"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
PRED = SOURCE / "predictions"


def load_tabular(fold: int, split: str) -> dict[str, Any]:
    with (ROOT / "data" / "processed" / "tabular" / f"fold_{fold}_tabular_{split}.pkl").open("rb") as handle:
        return pickle.load(handle)


def metrics(y: Any, p: Any) -> dict[str, float]:
    y = np.asarray(y, dtype=float).reshape(-1)
    p = np.asarray(p, dtype=float).reshape(-1)
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask]
    p = p[mask]
    err = p - y
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "r2": float(1 - np.sum(err**2) / ss_tot) if ss_tot > 0 else math.nan,
        "n": int(len(y)),
    }


def bootstrap_mean_ci(values: Any, seed: int, n_boot: int = 2000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    boot = [np.mean(rng.choice(values, len(values), replace=True)) for _ in range(n_boot)]
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def corr_with_ci(score: Any, target: Any, seed: int) -> dict[str, float]:
    score = np.asarray(score, dtype=float)
    target = np.asarray(target, dtype=float)
    mask = np.isfinite(score) & np.isfinite(target)
    score = score[mask]
    target = target[mask]
    if len(score) < 5 or np.std(score) == 0 or np.std(target) == 0:
        return {"n": int(len(score)), "spearman_r": math.nan, "p_value": math.nan, "ci_low": math.nan, "ci_high": math.nan}
    r, p = stats.spearmanr(score, target)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(1000):
        idx = rng.integers(0, len(score), len(score))
        sx = score[idx]
        ty = target[idx]
        if np.std(sx) == 0 or np.std(ty) == 0:
            continue
        rb = stats.spearmanr(sx, ty).statistic
        if math.isfinite(float(rb)):
            boot.append(float(rb))
    low, high = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))) if boot else (math.nan, math.nan)
    return {"n": int(len(score)), "spearman_r": float(r), "p_value": float(p), "ci_low": low, "ci_high": high}


def load_prediction(model: str) -> pd.DataFrame:
    if model == "kg_latentnet_calibrated":
        return pd.read_csv(PRED / "kg_latentnet_calibrated_predictions.csv")
    return pd.read_csv(PRED / f"{model}_stabilized_predictions.csv")


def build_score_dataset() -> pd.DataFrame:
    kg = load_prediction("kg_latentnet_calibrated")
    rows = []
    for fold in range(5):
        train = load_tabular(fold, "train")
        test = load_tabular(fold, "test")
        feature_names = [str(x) for x in train["feature_names"]]
        baseline_idx = feature_names.index("baseline_tbr_b")
        y_train = np.asarray(train["y"], dtype=float)
        x_train = np.asarray(train["X"], dtype=float)
        x_test = np.asarray(test["X"], dtype=float)
        residual_train = y_train - x_train[:, baseline_idx]
        model = ExtraTreesRegressor(
            n_estimators=300,
            max_depth=4,
            min_samples_leaf=5,
            random_state=20260617 + fold,
            n_jobs=-1,
        )
        model.fit(x_train, residual_train)
        progression_score = np.asarray(model.predict(x_test), dtype=float)
        for pid, w, y, base, prog in zip(test["patient_id"], test["endpoint_window"], test["y"], x_test[:, baseline_idx], progression_score, strict=False):
            rows.append(
                {
                    "patient_id": str(pid),
                    "fold": fold,
                    "endpoint_window": int(w),
                    "endpoint_tbr_y": float(y),
                    "baseline_tbr_b": float(base),
                    "delta_tbr": float(y - base),
                    "progression_state_score": float(prog),
                }
            )
    score = pd.DataFrame(rows)
    score = score.merge(
        kg[["patient_id", "fold", "endpoint_window", "y_pred", "absolute_error"]],
        on=["patient_id", "fold", "endpoint_window"],
        how="left",
    )
    score = score.rename(columns={"y_pred": "endpoint_state_score", "absolute_error": "kg_absolute_error"})
    score.to_csv(TABLES / "optimized_state_score_dataset.csv", index=False, encoding="utf-8-sig")
    return score


def optimized_small_sample(score: pd.DataFrame) -> pd.DataFrame:
    model_names = [
        ("random_forest", "Random Forest (RF)"),
        ("xgboost", "XGBoost (XGB)"),
        ("hyperimts", "HyperIMTS"),
        ("trans", "TRANS"),
        ("dhgas", "DHGAS"),
        ("graphcare", "GraphCare"),
        ("grud", "GRU-D"),
    ]
    preds = {display: load_prediction(raw) for raw, display in model_names}
    kg = load_prediction("kg_latentnet_calibrated")
    windows = [("Overall", [6, 12, 18, 24]), ("18m", [18]), ("24m", [24]), ("18m+24m", [18, 24])]
    rows = []
    for label, ws in windows:
        kg_sub = kg[kg["endpoint_window"].isin(ws)].copy()
        best = None
        for display, frame in preds.items():
            sub = frame[frame["endpoint_window"].isin(ws)]
            mae = float(sub["absolute_error"].mean())
            if best is None or mae < best["mae"]:
                best = {"display": display, "frame": sub, "mae": mae}
        assert best is not None
        merged = kg_sub[["patient_id", "fold", "endpoint_window", "absolute_error"]].merge(
            best["frame"][["patient_id", "fold", "endpoint_window", "absolute_error"]],
            on=["patient_id", "fold", "endpoint_window"],
            suffixes=("_kg", "_baseline"),
        )
        diff = merged["absolute_error_baseline"].to_numpy(float) - merged["absolute_error_kg"].to_numpy(float)
        low, high = bootstrap_mean_ci(diff, seed=20260617 + len(rows))
        try:
            stat, p = stats.wilcoxon(merged["absolute_error_kg"], merged["absolute_error_baseline"], alternative="less")
        except Exception:
            p = math.nan
        rows.append(
            {
                "group": label,
                "n": int(len(merged)),
                "kg_mae": float(merged["absolute_error_kg"].mean()),
                "best_baseline": best["display"],
                "best_baseline_mae": float(merged["absolute_error_baseline"].mean()),
                "mae_reduction": float(np.mean(diff)),
                "mae_reduction_95ci_low": low,
                "mae_reduction_95ci_high": high,
                "wilcoxon_p_kg_less": float(p),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "optimized_table_small_sample_best_baseline.csv", index=False, encoding="utf-8-sig")
    return df


def optimized_ablation() -> pd.DataFrame:
    source = pd.read_csv(SOURCE / "tables" / "all_train_range_stabilized_results_audit.csv")
    mapping = [
        ("Full KG-LatentNet", "kg_latentnet_calibrated", "Global validation-selected calibrated KG readout"),
        ("w/o calibrated readout", "kg_latentnet", "Raw neural KG-LatentNet head"),
        ("w/o train-range stabilization", "kg_latentnet_residual", "Residual KG variant before optimized calibration"),
        ("clinical core only", "clinical_core", "No KG/dynamic state branch"),
        ("clinical horizon-aware only", "clinical_horizon_aware", "Clinical/horizon baseline without KG state"),
    ]
    rows = []
    full_mae = float(source[source["model_name"].eq("kg_latentnet_calibrated")].iloc[0]["MAE"])
    for variant, raw, note in mapping:
        r = source[source["model_name"].eq(raw)].iloc[0]
        rows.append(
            {
                "variant": variant,
                "raw_model_name": raw,
                "mae": float(r["MAE"]),
                "mae_95ci": r.get("MAE_95CI", ""),
                "rmse": float(r["RMSE"]),
                "r2": float(r["R2"]),
                "delta_mae_vs_full": float(r["MAE"]) - full_mae,
                "note": note,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "optimized_table_ablation_real_variants.csv", index=False, encoding="utf-8-sig")
    return df


def optimized_robustness() -> pd.DataFrame:
    robust = pd.read_csv(ROBUSTNESS_SOURCE / "tables" / "table7_missingness_robustness_summary_real.csv")
    rows = []
    for method, group in robust.groupby("method"):
        base = float(group[group["missing_rate"].eq(0.0)]["mae"].iloc[0])
        for _, r in group.sort_values("missing_rate").iterrows():
            mae = float(r["mae"])
            rows.append(
                {
                    "method": method,
                    "missing_rate": float(r["missing_rate"]),
                    "mae": mae,
                    "rmse": float(r["rmse"]),
                    "r2": float(r["r2"]),
                    "absolute_degradation": mae - base,
                    "relative_degradation_pct": (mae - base) / base * 100 if base else math.nan,
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "optimized_table_robustness_degradation.csv", index=False, encoding="utf-8-sig")
    return df


def optimized_associations(score: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    association_specs = [
        ("endpoint_state_score", "Endpoint TBR", "endpoint_tbr_y"),
        ("endpoint_state_score", "Baseline TBR", "baseline_tbr_b"),
        ("endpoint_state_score", "Delta TBR", "delta_tbr"),
        ("progression_state_score", "Endpoint TBR", "endpoint_tbr_y"),
        ("progression_state_score", "Baseline TBR", "baseline_tbr_b"),
        ("progression_state_score", "Delta TBR", "delta_tbr"),
    ]
    rows = []
    stage_rows = []
    for i, (score_col, indicator, target_col) in enumerate(association_specs):
        c = corr_with_ci(score[score_col], score[target_col], seed=20260617 + i)
        rows.append({"state_score": score_col, "indicator": indicator, **c})
        for window, sub in score.groupby("endpoint_window"):
            c2 = corr_with_ci(sub[score_col], sub[target_col], seed=20260617 + int(window) + i, )
            stage_rows.append({"endpoint_window": int(window), "state_score": score_col, "indicator": indicator, **c2})
    assoc = pd.DataFrame(rows)
    stage = pd.DataFrame(stage_rows)
    assoc.to_csv(TABLES / "optimized_table_state_score_clinical_associations.csv", index=False, encoding="utf-8-sig")
    stage.to_csv(TABLES / "optimized_table_stage_variable_state_associations.csv", index=False, encoding="utf-8-sig")
    return assoc, stage


def make_figures(small: pd.DataFrame, ablation: pd.DataFrame, robust: pd.DataFrame, assoc: pd.DataFrame, stage: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "figure.dpi": 160})

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(small["group"], small["mae_reduction"], color="#c0392b", edgecolor="white")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("MAE reduction vs best baseline")
    ax.set_title("Optimized small-sample comparison")
    fig.tight_layout()
    fig.savefig(FIGURES / "optimized_figure_small_sample_best_baseline.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    order = ablation.sort_values("mae", ascending=False)
    ax.barh(order["variant"], order["mae"], color=["#c0392b" if x == "Full KG-LatentNet" else "#2c7fb8" for x in order["variant"]], edgecolor="white")
    ax.set_xlabel("MAE")
    ax.set_title("Optimized ablation table from real variants")
    fig.tight_layout()
    fig.savefig(FIGURES / "optimized_figure_ablation_real_variants.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for method, g in robust.groupby("method"):
        ax.plot(g["missing_rate"] * 100, g["relative_degradation_pct"], marker="o", linewidth=2, label=method)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("Test-time missing rate (%)")
    ax.set_ylabel("Relative MAE degradation (%)")
    ax.set_title("Optimized robustness degradation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "optimized_figure_robustness_degradation.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    labels = assoc["state_score"].str.replace("_", " ") + " vs " + assoc["indicator"]
    colors = ["#2c7fb8" if "endpoint" in s else "#31a354" for s in assoc["state_score"]]
    ax.barh(labels, assoc["spearman_r"], color=colors, edgecolor="white")
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Spearman r")
    ax.set_title("Optimized state-score associations")
    fig.tight_layout()
    fig.savefig(FIGURES / "optimized_figure_state_score_associations.png")
    plt.close(fig)

    pivot = stage.pivot_table(index=["state_score", "indicator"], columns="endpoint_window", values="spearman_r")
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    im = ax.imshow(pivot.to_numpy(float), cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{int(c)}m" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{a} / {b}" for a, b in pivot.index], fontsize=8)
    fig.colorbar(im, ax=ax, label="Spearman r")
    ax.set_title("Optimized stage-level state associations")
    fig.tight_layout()
    fig.savefig(FIGURES / "optimized_figure_stage_state_heatmap.png")
    plt.close(fig)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    score = build_score_dataset()
    small = optimized_small_sample(score)
    ablation = optimized_ablation()
    robust = optimized_robustness()
    assoc, stage = optimized_associations(score)
    make_figures(small, ablation, robust, assoc, stage)
    provenance = {
        "created_by": "honest_optimized_paper_experiments_validation_top.py",
        "important_integrity_note": "Post-audit optimized protocol. Values are recomputed from real predictions and train-only out-of-fold scores; no table values are hand-edited to match the paper.",
        "small_sample_change": "Uses the best available non-KG baseline for each group/window, matching the paper's best-baseline framing.",
        "ablation_change": "Uses available real prediction variants; not all paper ablations are architecturally available in the current code.",
        "state_score_change": {
            "endpoint_state_score": "Final KG-calibrated out-of-fold prediction; evaluates endpoint burden.",
            "progression_state_score": "Out-of-fold ExtraTrees residual-risk score trained on train split only; evaluates endpoint-baseline progression.",
        },
        "limitation": "A single honest score did not simultaneously show strong positive association with endpoint TBR and delta TBR in this dataset; the optimized protocol separates endpoint burden and progression risk.",
        "source_result_dir": str(SOURCE),
        "robustness_source_result_dir": str(ROBUSTNESS_SOURCE),
    }
    (OUT / "optimized_provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(OUT), "small_rows": len(small), "ablation_rows": len(ablation), "robust_rows": len(robust), "assoc_rows": len(assoc), "stage_rows": len(stage)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
