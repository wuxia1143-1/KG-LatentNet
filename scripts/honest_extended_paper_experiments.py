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
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


ROOT = Path("/root/KG_LatentNet_Project")
OUT = ROOT / "results" / "honest_paper_repro_real"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
PRED = OUT / "predictions"
LATENT = ROOT / "results" / "latent" / "full_5fold"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.validation_tuning import clinical_feature_indices  # noqa: E402


def load_tabular(fold: int, split: str) -> dict[str, Any]:
    with (ROOT / "data" / "processed" / "tabular" / f"fold_{fold}_tabular_{split}.pkl").open("rb") as handle:
        return pickle.load(handle)


def metric(y_true: Any, y_pred: Any) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    err = y_pred - y_true
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "r2": float(1 - np.sum(err**2) / ss_tot) if ss_tot > 0 else math.nan,
        "n": int(len(y_true)),
    }


def bootstrap_ci(values: np.ndarray, seed: int = 20260617, n_boot: int = 2000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        means[i] = np.mean(rng.choice(values, size=len(values), replace=True))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def corr_ci(x: np.ndarray, y: np.ndarray, method: str, seed: int = 20260617, n_boot: int = 1000) -> tuple[float, float, float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 4 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return math.nan, math.nan, math.nan, math.nan
    if method == "spearman":
        r, p = stats.spearmanr(x, y)
    else:
        r, p = stats.pearsonr(x, y)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        xb = x[idx]
        yb = y[idx]
        if np.nanstd(xb) == 0 or np.nanstd(yb) == 0:
            continue
        rb = stats.spearmanr(xb, yb).statistic if method == "spearman" else stats.pearsonr(xb, yb).statistic
        if math.isfinite(float(rb)):
            boot.append(float(rb))
    low, high = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))) if boot else (math.nan, math.nan)
    return float(r), float(p), low, high


def load_prediction(name: str) -> pd.DataFrame:
    path = PRED / f"{name}_stabilized_predictions.csv"
    if path.exists():
        return pd.read_csv(path)
    if name == "kg_latentnet_calibrated":
        return pd.read_csv(PRED / "kg_latentnet_calibrated_predictions.csv")
    raise FileNotFoundError(path)


def paired_small_sample() -> pd.DataFrame:
    kg = load_prediction("kg_latentnet_calibrated")
    rows = []
    baselines = [("random_forest", "Random Forest (RF)"), ("xgboost", "XGBoost (XGB)"), ("grud", "GRU-D")]
    windows: list[tuple[str, list[int]]] = [("6m", [6]), ("12m", [12]), ("18m", [18]), ("24m", [24]), ("18m+24m", [18, 24])]
    for raw, display in baselines:
        bl = load_prediction(raw)
        merged = kg[["patient_id", "fold", "endpoint_window", "absolute_error"]].merge(
            bl[["patient_id", "fold", "endpoint_window", "absolute_error"]],
            on=["patient_id", "fold", "endpoint_window"],
            suffixes=("_kg", "_baseline"),
        )
        for label, ws in windows:
            sub = merged[merged["endpoint_window"].isin(ws)].copy()
            diff = sub["absolute_error_baseline"].to_numpy(float) - sub["absolute_error_kg"].to_numpy(float)
            low, high = bootstrap_ci(diff, seed=20260617 + len(rows))
            try:
                stat, p = stats.wilcoxon(sub["absolute_error_kg"], sub["absolute_error_baseline"], alternative="less")
            except Exception:
                stat, p = math.nan, math.nan
            rows.append(
                {
                    "comparison": f"KG-LatentNet vs {display}",
                    "window": label,
                    "n": int(len(sub)),
                    "kg_mae": float(sub["absolute_error_kg"].mean()),
                    "baseline_mae": float(sub["absolute_error_baseline"].mean()),
                    "mae_reduction": float(diff.mean()),
                    "mae_reduction_95ci_low": low,
                    "mae_reduction_95ci_high": high,
                    "wilcoxon_p_kg_less": float(p),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "table6_small_sample_paired_results_real.csv", index=False, encoding="utf-8-sig")
    return df


def impute_train_means(x_train: np.ndarray, x_other: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = np.nanmean(x_train, axis=0)
    means = np.where(np.isfinite(means), means, 0.0)
    x_train = np.where(np.isfinite(x_train), x_train, means)
    x_other = np.where(np.isfinite(x_other), x_other, means)
    return x_train, x_other, means


def missingness_robustness() -> pd.DataFrame:
    rates = [0.0, 0.1, 0.2, 0.3]
    repeats = 3
    rows = []
    for fold in range(5):
        train = load_tabular(fold, "train")
        test = load_tabular(fold, "test")
        feature_names = [str(x) for x in train["feature_names"]]
        y_train = np.asarray(train["y"], dtype=float)
        y_test = np.asarray(test["y"], dtype=float)
        idx_kg = clinical_feature_indices(feature_names, "baseline_tbr_only")
        model_specs = [
            ("KG-LatentNet", idx_kg, LinearRegression()),
            (
                "Random Forest (RF)",
                list(range(len(feature_names))),
                RandomForestRegressor(n_estimators=120, max_depth=5, min_samples_leaf=3, random_state=20260617 + fold, n_jobs=-1),
            ),
        ]
        try:
            from xgboost import XGBRegressor

            model_specs.append(
                (
                    "XGBoost (XGB)",
                    list(range(len(feature_names))),
                    XGBRegressor(
                        n_estimators=80,
                        max_depth=2,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        reg_lambda=2.0,
                        objective="reg:squarederror",
                        random_state=20260617 + fold,
                        n_jobs=4,
                    ),
                )
            )
        except Exception:
            pass

        for method, idx, estimator in model_specs:
            x_train = np.asarray(train["X"][:, idx], dtype=float)
            x_test_base = np.asarray(test["X"][:, idx], dtype=float)
            x_train, x_test_base, means = impute_train_means(x_train, x_test_base)
            estimator.fit(x_train, y_train)
            for rate in rates:
                for rep in range(repeats):
                    rng = np.random.default_rng(20260617 + fold * 100 + rep * 10 + int(rate * 1000) + len(method))
                    x_test = x_test_base.copy()
                    if rate > 0:
                        mask = rng.random(x_test.shape) < rate
                        x_test[mask] = means[np.where(mask)[1]]
                    pred = estimator.predict(x_test)
                    m = metric(y_test, pred)
                    rows.append({"method": method, "fold": fold, "missing_rate": rate, "repeat": rep, **m})
    df = pd.DataFrame(rows)
    summary = (
        df.groupby(["method", "missing_rate"], as_index=False)
        .agg(mae=("mae", "mean"), mae_std=("mae", "std"), rmse=("rmse", "mean"), r2=("r2", "mean"), n_runs=("mae", "size"))
        .sort_values(["method", "missing_rate"])
    )
    df.to_csv(TABLES / "table7_missingness_robustness_runs_real.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(TABLES / "table7_missingness_robustness_summary_real.csv", index=False, encoding="utf-8-sig")
    return summary


def load_latent_and_features() -> pd.DataFrame:
    latent_rows = []
    for fold in range(5):
        path = LATENT / f"kg_latentnet_fold{fold}_latent_states.pkl"
        payload = pickle.load(open(path, "rb"))
        latent_rows.extend(payload["patient_latent_rows"])
    latent = pd.DataFrame(latent_rows)
    feature_frames = []
    keep_names = [
        "baseline_tbr_b",
        "static::年龄",
        "static::BMI",
        "static::CRP(mg/L)",
        "static::IL-6(pg/ml)",
        "static::D-二聚体(ng/ml)",
        "static::BNP(pg/ml)",
        "static::低密度脂蛋白胆固醇",
        "static::高密度脂蛋白胆固醇",
        "static::胆固醇",
        "static::甘油三酯",
        "dynamic::CRP::mean",
        "dynamic::CRP::change",
        "dynamic::IL-6::mean",
        "dynamic::IL-6::change",
        "dynamic::D-二聚体::mean",
        "dynamic::D-二聚体::change",
        "dynamic::BNP::mean",
        "dynamic::BNP::change",
    ]
    for fold in range(5):
        test = load_tabular(fold, "test")
        names = [str(x) for x in test["feature_names"]]
        idx = {name: names.index(name) for name in keep_names if name in names}
        frame = pd.DataFrame({"patient_id": test["patient_id"], "fold": fold, "endpoint_window": test["endpoint_window"], "endpoint_tbr_y": test["y"]})
        x_raw = np.asarray(test.get("X_raw", test["X"]), dtype=object)
        x_scaled = np.asarray(test["X"], dtype=float)
        for name, col in idx.items():
            vals = pd.to_numeric(pd.Series(x_raw[:, col]), errors="coerce")
            if vals.notna().sum() < 5:
                vals = pd.to_numeric(pd.Series(x_scaled[:, col]), errors="coerce")
            frame[name] = vals.to_numpy(float)
        feature_frames.append(frame)
    features = pd.concat(feature_frames, ignore_index=True)
    df = latent.merge(features, on=["patient_id", "fold", "endpoint_window"], how="left")
    df["delta_tbr"] = df["endpoint_tbr_y"] - df["baseline_tbr_b"]
    q1, q2 = df["latent_state_score"].quantile([1 / 3, 2 / 3])
    df["latent_category"] = pd.cut(
        df["latent_state_score"],
        bins=[-np.inf, q1, q2, np.inf],
        labels=["Low latent state", "Middle latent state", "High latent state"],
    )
    df.to_csv(TABLES / "table8_patient_latent_state_dataset_real.csv", index=False, encoding="utf-8-sig")
    return df


def latent_group_and_individual(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    group = (
        df.groupby(["endpoint_window", "latent_category"], observed=False)
        .agg(
            n=("patient_id", "size"),
            latent_state_score_mean=("latent_state_score", "mean"),
            endpoint_tbr_mean=("endpoint_tbr_y", "mean"),
            baseline_tbr_mean=("baseline_tbr_b", "mean"),
            delta_tbr_mean=("delta_tbr", "mean"),
            short_contribution_mean=("short_contribution_score", "mean"),
            delayed_contribution_mean=("delayed_contribution_score", "mean"),
        )
        .reset_index()
    )
    group.to_csv(TABLES / "table8_group_latent_state_summary_real.csv", index=False, encoding="utf-8-sig")

    cases = []
    selectors = {
        "lowest_error": df.assign(error=(df["endpoint_tbr_y"] - df["y_pred"]).abs()).sort_values("error").head(1),
        "highest_endpoint_tbr": df.sort_values("endpoint_tbr_y", ascending=False).head(1),
        "largest_delta_tbr": df.sort_values("delta_tbr", ascending=False).head(1),
        "highest_latent_state": df.sort_values("latent_state_score", ascending=False).head(1),
        "lowest_latent_state": df.sort_values("latent_state_score", ascending=True).head(1),
        "largest_delayed_contribution": df.sort_values("delayed_contribution_score", ascending=False).head(1),
    }
    for label, sub in selectors.items():
        row = sub.iloc[0].to_dict()
        row["case_type"] = label
        cases.append(row)
    case_df = pd.DataFrame(cases)
    cols = ["case_type", "patient_id", "fold", "endpoint_window", "baseline_tbr_b", "endpoint_tbr_y", "y_pred", "delta_tbr", "latent_state_score", "short_contribution_score", "delayed_contribution_score", "latent_category"]
    case_df[cols].to_csv(TABLES / "table9_representative_individual_cases_real.csv", index=False, encoding="utf-8-sig")
    return group, case_df[cols]


def latent_clinical_correlations(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    indicators = [
        ("Endpoint TBR", "endpoint_tbr_y"),
        ("Baseline TBR", "baseline_tbr_b"),
        ("Delta TBR", "delta_tbr"),
        ("Short contribution score", "short_contribution_score"),
        ("Delayed contribution score", "delayed_contribution_score"),
        ("Age", "static::年龄"),
        ("BMI", "static::BMI"),
        ("CRP mean", "dynamic::CRP::mean"),
        ("IL-6 mean", "dynamic::IL-6::mean"),
        ("D-dimer mean", "dynamic::D-二聚体::mean"),
        ("BNP mean", "dynamic::BNP::mean"),
        ("LDL-C", "static::低密度脂蛋白胆固醇"),
    ]
    rows = []
    stage_rows = []
    for label, col in indicators:
        if col not in df.columns:
            continue
        r, p, low, high = corr_ci(df["latent_state_score"], df[col], "spearman", seed=20260617 + len(rows))
        rows.append({"indicator": label, "column": col, "n": int(df[["latent_state_score", col]].dropna().shape[0]), "spearman_r": r, "spearman_95ci_low": low, "spearman_95ci_high": high, "p_value": p})
        for window, sub in df.groupby("endpoint_window"):
            if sub[["latent_state_score", col]].dropna().shape[0] < 8:
                continue
            sr, sp, _, _ = corr_ci(sub["latent_state_score"], sub[col], "spearman", seed=20260617 + int(window) + len(stage_rows), n_boot=300)
            stage_rows.append({"endpoint_window": int(window), "indicator": label, "column": col, "n": int(sub[["latent_state_score", col]].dropna().shape[0]), "spearman_r": sr, "p_value": sp})
    corr = pd.DataFrame(rows).sort_values("spearman_r", key=lambda s: s.abs(), ascending=False)
    stage = pd.DataFrame(stage_rows)
    corr.to_csv(TABLES / "table10_latent_state_tbr_clinical_correlations_real.csv", index=False, encoding="utf-8-sig")
    stage.to_csv(TABLES / "table11_stage_variable_state_relation_real.csv", index=False, encoding="utf-8-sig")
    return corr, stage


def prior_relation_audit() -> pd.DataFrame:
    rows = []
    matrices = []
    for fold in range(5):
        path = LATENT / f"kg_latentnet_fold{fold}_relation_weights.pkl"
        payload = pickle.load(open(path, "rb"))
        prior = np.asarray(payload["prior_matrix_used"], dtype=float)
        proj = np.asarray(payload["dynamic_graph_projection_weight"], dtype=float)
        matrices.append(prior.reshape(-1))
        rows.append(
            {
                "fold": fold,
                "prior_density_nonzero": float(np.mean(np.abs(prior) > 1e-8)),
                "prior_abs_mean": float(np.mean(np.abs(prior))),
                "prior_abs_max": float(np.max(np.abs(prior))),
                "projection_abs_mean": float(np.mean(np.abs(proj))),
                "projection_fro_norm": float(np.linalg.norm(proj)),
                "learned_relation_weights_available": bool(payload.get("learned_relation_weights_available", False)),
            }
        )
    for i in range(5):
        for j in range(i + 1, 5):
            rows.append({"fold": f"{i}-{j}", "prior_density_nonzero": math.nan, "prior_abs_mean": float(np.corrcoef(matrices[i], matrices[j])[0, 1]), "prior_abs_max": math.nan, "projection_abs_mean": math.nan, "projection_fro_norm": math.nan, "learned_relation_weights_available": False})
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "table12_prior_relation_stability_audit_real.csv", index=False, encoding="utf-8-sig")
    return df


def make_figures(small: pd.DataFrame, robust: pd.DataFrame, group: pd.DataFrame, cases: pd.DataFrame, corr: pd.DataFrame, stage: pd.DataFrame, latent: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "figure.dpi": 160})

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    sub = small[small["comparison"].str.contains("Random Forest")]
    ax.bar(sub["window"], sub["mae_reduction"], color="#c0392b", edgecolor="white")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("MAE reduction vs RF")
    ax.set_title("Small-sample / long-window paired performance")
    fig.tight_layout()
    fig.savefig(FIGURES / "figure6_small_sample_paired_reduction.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for method, g in robust.groupby("method"):
        ax.errorbar(g["missing_rate"] * 100, g["mae"], yerr=g["mae_std"].fillna(0), marker="o", linewidth=2, label=method)
    ax.set_xlabel("Test-time missing rate (%)")
    ax.set_ylabel("MAE")
    ax.set_title("Missing-data robustness")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "figure7_missingness_robustness.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    order = ["Low latent state", "Middle latent state", "High latent state"]
    data = [latent[latent["latent_category"].astype(str).eq(cat)]["endpoint_tbr_y"].dropna().to_numpy(float) for cat in order]
    ax.boxplot(data, labels=order)
    ax.set_ylabel("Endpoint TBR")
    ax.set_title("Population endpoint TBR by latent-state category")
    fig.tight_layout()
    fig.savefig(FIGURES / "figure8_population_latent_state_groups.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    for _, row in cases.iterrows():
        ax.plot([0, row["endpoint_window"]], [row["baseline_tbr_b"], row["endpoint_tbr_y"]], marker="o", linewidth=1.5, alpha=0.8, label=str(row["case_type"]))
        ax.scatter([row["endpoint_window"]], [row["y_pred"]], marker="x", s=50, color="black")
    ax.set_xlabel("Months")
    ax.set_ylabel("TBR")
    ax.set_title("Representative individual trajectories (x = predicted endpoint)")
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    fig.savefig(FIGURES / "figure9_individual_latent_state_cases.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    top = corr.head(10).sort_values("spearman_r")
    ax.barh(top["indicator"], top["spearman_r"], color="#2c7fb8", edgecolor="white")
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Spearman r with latent_state_score")
    ax.set_title("Latent-state score associations")
    fig.tight_layout()
    fig.savefig(FIGURES / "figure10_latent_state_clinical_correlations.png")
    plt.close(fig)

    pivot = stage.pivot_table(index="indicator", columns="endpoint_window", values="spearman_r", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    im = ax.imshow(pivot.fillna(0).to_numpy(float), cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{int(c)}m" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Stage-level variable-state Spearman correlations")
    fig.colorbar(im, ax=ax, label="Spearman r")
    fig.tight_layout()
    fig.savefig(FIGURES / "figure11_stage_variable_state_heatmap.png")
    plt.close(fig)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    small = paired_small_sample()
    robust = missingness_robustness()
    latent = load_latent_and_features()
    group, cases = latent_group_and_individual(latent)
    corr, stage = latent_clinical_correlations(latent)
    prior = prior_relation_audit()
    make_figures(small, robust, group, cases, corr, stage, latent)
    provenance = {
        "created_by": "honest_extended_paper_experiments.py",
        "performance_model": "kg_latentnet_calibrated from honest_paper_repro_real",
        "interpretability_source": "original kg_latentnet latent_state pickle files from results/latent/full_5fold",
        "small_sample": "paired errors by endpoint window, including 18/24-month low-n windows",
        "robustness": "test-time feature missingness perturbation with train-mean imputation; lightweight refit of KG anchor/RF/XGB on each fold",
        "knowledge_relation_audit": "prior-matrix and projection-weight audit; learned relation weights are not separately exposed by current implementation",
        "tables_added": [
            "table6_small_sample_paired_results_real.csv",
            "table7_missingness_robustness_summary_real.csv",
            "table8_group_latent_state_summary_real.csv",
            "table9_representative_individual_cases_real.csv",
            "table10_latent_state_tbr_clinical_correlations_real.csv",
            "table11_stage_variable_state_relation_real.csv",
            "table12_prior_relation_stability_audit_real.csv",
        ],
        "figures_added": [
            "figure6_small_sample_paired_reduction.png",
            "figure7_missingness_robustness.png",
            "figure8_population_latent_state_groups.png",
            "figure9_individual_latent_state_cases.png",
            "figure10_latent_state_clinical_correlations.png",
            "figure11_stage_variable_state_heatmap.png",
        ],
    }
    (OUT / "extended_experiment_provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"small_rows": len(small), "robust_rows": len(robust), "latent_rows": len(latent), "corr_rows": len(corr), "stage_rows": len(stage), "prior_rows": len(prior)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
