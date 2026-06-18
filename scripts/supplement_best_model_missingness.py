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

from sklearn.base import clone


ROOT = Path("/root/KG_LatentNet_Project")
OUT = ROOT / "results" / "paper_ready_single_model_results"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
PRED = OUT / "predictions"

SELECTED_KEY = "baseline_tbr_only:kg_dynamic:ridge_100:0.005"
RATES = [0.0, 0.1, 0.2, 0.3, 0.4]
REPEATS = 100


def load_helper():
    path = ROOT / "scripts" / "honest_real_final_outputs_validation_top.py"
    spec = importlib.util.spec_from_file_location("validation_top_helper", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["validation_top_helper"] = module
    spec.loader.exec_module(module)
    return module


HELPER = load_helper()


def ensure_dirs() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    PRED.mkdir(parents=True, exist_ok=True)


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


def metric(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    err = y_pred - y_true
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "R2": float(1.0 - np.sum(err**2) / ss_tot) if ss_tot > 0 else math.nan,
    }


def bootstrap_ci(values: np.ndarray, seed: int, n_boot: int = 2000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, len(values), len(values))
        stats[i] = float(values[idx].mean())
    low, high = np.percentile(stats, [2.5, 97.5])
    return float(low), float(high)


def perturb_test(test: dict[str, Any], train_means: np.ndarray, rate: float, seed: int) -> dict[str, Any]:
    out = dict(test)
    x = np.asarray(test["X"], dtype=float).copy()
    if rate > 0:
        rng = np.random.default_rng(seed)
        mask = rng.random(x.shape) < rate
        x[mask] = train_means[np.where(mask)[1]]
    out["X"] = x
    return out


def predict_selected(train: dict[str, Any], test: dict[str, Any], fold: int) -> np.ndarray:
    spec = parse_key(SELECTED_KEY)
    feature_names = [str(name) for name in train["feature_names"]]
    low, high = HELPER.fold_bounds(fold)
    anchor = HELPER.fit_anchor(train, test, test, feature_names, spec["anchor_mode"])
    anchor_train = np.asarray(anchor["train"], dtype=float)
    anchor_test = np.asarray(anchor["test"], dtype=float)
    residual_test = np.zeros(len(test["y"]), dtype=float)
    if spec["residual_model"] != "none":
        idx = HELPER.kg_feature_indices(feature_names, spec["kg_feature_mode"])
        model = clone(HELPER.residual_models()[spec["residual_model"]])
        residual_target = np.asarray(train["y"], dtype=float).reshape(-1) - anchor_train
        model.fit(np.asarray(train["X"])[:, idx], residual_target)
        residual_test = np.asarray(model.predict(np.asarray(test["X"])[:, idx]), dtype=float)
    pred = np.clip(anchor_test + float(spec["blend"]) * residual_test, low, high)
    return np.asarray(pred, dtype=float)


def run_missingness() -> tuple[pd.DataFrame, pd.DataFrame]:
    run_rows = []
    pred_rows = []
    for rate in RATES:
        repeats = 1 if rate == 0 else REPEATS
        for rep in range(repeats):
            frames = []
            for fold in range(5):
                train = HELPER.load_tabular(fold, "train")
                test = HELPER.load_tabular(fold, "test")
                train_means = np.nanmean(np.asarray(train["X"], dtype=float), axis=0)
                train_means = np.where(np.isfinite(train_means), train_means, 0.0)
                test_p = perturb_test(test, train_means, rate, 20260618 + fold * 100000 + rep * 101 + int(rate * 1000))
                pred = predict_selected(train, test_p, fold)
                frame = pd.DataFrame(
                    {
                        "patient_id": np.asarray(test["patient_id"]).astype(str),
                        "fold": fold,
                        "endpoint_window": np.asarray(test["endpoint_window"], dtype=int),
                        "missing_rate": rate,
                        "repeat": rep,
                        "y_true": np.asarray(test["y"], dtype=float),
                        "y_pred": pred,
                    }
                )
                frame["absolute_error"] = np.abs(frame["y_true"].to_numpy(float) - frame["y_pred"].to_numpy(float))
                frames.append(frame)
            all_pred = pd.concat(frames, ignore_index=True)
            m = metric(all_pred["y_true"].to_numpy(float), all_pred["y_pred"].to_numpy(float))
            run_rows.append({"missing_rate": rate, "repeat": rep, **m, "n": int(len(all_pred))})
            pred_rows.append(all_pred)
    runs = pd.DataFrame(run_rows)
    preds = pd.concat(pred_rows, ignore_index=True)
    return runs, preds


def summarize_runs(runs: pd.DataFrame) -> pd.DataFrame:
    full_mae = float(runs[runs["missing_rate"].eq(0.0)]["MAE"].iloc[0])
    rows = []
    for rate, sub in runs.groupby("missing_rate"):
        mae = sub["MAE"].to_numpy(float)
        rmse = sub["RMSE"].to_numpy(float)
        r2 = sub["R2"].to_numpy(float)
        mae_low, mae_high = bootstrap_ci(mae, 20260618 + int(float(rate) * 1000))
        rmse_low, rmse_high = bootstrap_ci(rmse, 20260619 + int(float(rate) * 1000))
        rows.append(
            {
                "analysis_type": "Missing-data variation",
                "model": "KG-LatentNet",
                "selection_key": SELECTED_KEY,
                "missing_rate": float(rate),
                "repeats": int(len(sub)),
                "n_per_repeat": int(sub["n"].iloc[0]),
                "MAE": float(np.mean(mae)),
                "MAE_repeat_95CI_low": mae_low,
                "MAE_repeat_95CI_high": mae_high,
                "RMSE": float(np.mean(rmse)),
                "RMSE_repeat_95CI_low": rmse_low,
                "RMSE_repeat_95CI_high": rmse_high,
                "R2": float(np.mean(r2)),
                "delta_MAE_vs_full": float(np.mean(mae) - full_mae),
            }
        )
    return pd.DataFrame(rows)


def merge_robustness(missing_summary: pd.DataFrame) -> pd.DataFrame:
    knowledge_path = ROOT / "results" / "honest_paper_repro_expected_complete" / "tables" / "table_knowledge_missingness_robustness.csv"
    knowledge = pd.read_csv(knowledge_path)
    k_rows = []
    for _, row in knowledge.iterrows():
        k_rows.append(
            {
                "analysis_type": "Structured knowledge perturbation",
                "condition": row["setting"],
                "model": "KG-LatentNet",
                "selection_key": SELECTED_KEY,
                "repeats": int(row["repeats"]),
                "MAE": float(row["MAE"]),
                "MAE_95CI_low": float(row["MAE_repeat_95CI_low"]),
                "MAE_95CI_high": float(row["MAE_repeat_95CI_high"]),
                "RMSE": float(row["RMSE"]),
                "R2": float(row["R2"]),
                "delta_MAE_vs_full": float(row["delta_MAE_vs_full"]),
                "representation_consistency": float(row["representation_consistency"]),
                "note": "Counterfactual masking/randomization of validation-selected KG residual contribution; no hand-edited metrics.",
            }
        )
    m_rows = []
    for _, row in missing_summary.iterrows():
        m_rows.append(
            {
                "analysis_type": "Missing-data variation",
                "condition": f"Feature missing rate {float(row['missing_rate']):.0%}",
                "model": "KG-LatentNet",
                "selection_key": SELECTED_KEY,
                "repeats": int(row["repeats"]),
                "MAE": float(row["MAE"]),
                "MAE_95CI_low": float(row["MAE_repeat_95CI_low"]),
                "MAE_95CI_high": float(row["MAE_repeat_95CI_high"]),
                "RMSE": float(row["RMSE"]),
                "R2": float(row["R2"]),
                "delta_MAE_vs_full": float(row["delta_MAE_vs_full"]),
                "representation_consistency": math.nan,
                "note": "Test-time feature masking with train-fold mean imputation using the validation-selected KG-LatentNet readout.",
            }
        )
    return pd.DataFrame(k_rows + m_rows)


def plot_missingness(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(summary["missing_rate"] * 100, summary["MAE"], marker="o", linewidth=2.2, color="#c0392b")
    ax.fill_between(
        summary["missing_rate"] * 100,
        summary["MAE_repeat_95CI_low"],
        summary["MAE_repeat_95CI_high"],
        color="#c0392b",
        alpha=0.16,
    )
    ax.set_xlabel("Feature missing rate (%)")
    ax.set_ylabel("MAE")
    ax.set_title("KG-LatentNet missing-data robustness")
    fig.tight_layout()
    fig.savefig(FIGURES / "Figure8_missing_data_robustness_best_model.png", dpi=180)
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    runs, preds = run_missingness()
    summary = summarize_runs(runs)
    combined = merge_robustness(summary)
    runs.to_csv(TABLES / "best_model_missing_data_robustness_runs.csv", index=False, encoding="utf-8-sig")
    preds.to_csv(PRED / "best_model_missing_data_robustness_predictions.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(TABLES / "best_model_missing_data_robustness_summary.csv", index=False, encoding="utf-8-sig")
    combined.to_csv(TABLES / "Table8_structured_knowledge_and_missing_data_robustness.csv", index=False, encoding="utf-8-sig")
    plot_missingness(summary)
    provenance = {
        "created_by": "supplement_best_model_missingness.py",
        "selected_model": "kg_latentnet_calibrated / KG-LatentNet",
        "selection_key": SELECTED_KEY,
        "test_set_used_for_selection": False,
        "missingness_experiment": "Fixed validation-selected KG-LatentNet readout; test-time feature masking with train-fold mean imputation.",
        "knowledge_robustness_source": "honest_paper_repro_expected_complete/table_knowledge_missingness_robustness.csv",
    }
    (OUT / "provenance" / "best_model_missingness_provenance.json").parent.mkdir(parents=True, exist_ok=True)
    (OUT / "provenance" / "best_model_missingness_provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary_rows": len(summary), "combined_rows": len(combined), "out": str(OUT)}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
