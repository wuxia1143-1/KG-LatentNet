from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import pickle
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


REQUIRED_PRED_COLS = ["patient_id", "fold", "endpoint_window", "y_true", "y_pred", "absolute_error"]
MODEL_LABELS = {
    "baseline_tbr_only": "Baseline TBR only",
    "clinical_core": "Clinical core",
    "clinical_horizon_aware": "Clinical horizon-aware",
    "linear_regression": "Linear regression",
    "ridge": "Ridge",
    "elasticnet": "ElasticNet",
    "linear_mixed_effects": "Linear mixed effects",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "grud": "GRU-D",
    "retain": "RETAIN",
    "time_aware_lstm": "Time-aware LSTM",
    "trans": "TRANS",
    "hyperimts": "HyperIMTS",
    "dhgas": "DHGAS",
    "kedgn": "KEDGN",
    "graphcare": "GraphCare",
    "tgnn4i": "TGNN4I",
    "kg_latentnet": "KG-LatentNet",
}


def method_label(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def category(model: str) -> str:
    if model == "kg_latentnet":
        return "Proposed"
    if model in {"baseline_tbr_only", "clinical_core", "clinical_horizon_aware"}:
        return "Clinical baseline"
    if model in {"linear_regression", "ridge", "elasticnet", "linear_mixed_effects", "random_forest", "xgboost"}:
        return "Classical ML"
    if model in {"grud", "retain", "time_aware_lstm", "trans"}:
        return "Temporal neural baseline"
    return "Official graph/temporal baseline"


class Full5PostHoc:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.tables = root / "results" / "tables" / "full_5fold"
        self.preds = root / "results" / "predictions" / "full_5fold"
        self.figs = root / "results" / "figures" / "full_5fold"
        self.logs = root / "results" / "logs" / "full_5fold"
        self.latent = root / "results" / "latent" / "full_5fold"
        self.figs.mkdir(parents=True, exist_ok=True)
        self.tables.mkdir(parents=True, exist_ok=True)
        self.logs.mkdir(parents=True, exist_ok=True)

    def csv(self, path: Path) -> pd.DataFrame:
        return pd.read_csv(path, encoding="utf-8-sig")

    def write(self, df: pd.DataFrame, name: str) -> Path:
        path = self.tables / name
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return path

    def prediction_files(self) -> list[Path]:
        return sorted(self.preds.glob("*_predictions.csv"))

    def parse_prediction_file(self, path: Path) -> tuple[str, int]:
        match = re.match(r"(.+)_fold(\d+)_predictions\.csv$", path.name)
        if not match:
            raise ValueError(f"cannot parse prediction file name: {path}")
        return match.group(1), int(match.group(2))

    def load_predictions(self) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for path in self.prediction_files():
            model, fold = self.parse_prediction_file(path)
            frame = self.csv(path)
            frame["model_name"] = model
            frame["file_fold"] = fold
            frame["prediction_file"] = str(path)
            for col in ["fold", "endpoint_window", "y_true", "y_pred", "absolute_error"]:
                if col in frame.columns:
                    frame[col] = pd.to_numeric(frame[col], errors="coerce")
            frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    @staticmethod
    def metrics(df: pd.DataFrame) -> dict[str, float]:
        y_true = pd.to_numeric(df["y_true"], errors="coerce").to_numpy(dtype=float)
        y_pred = pd.to_numeric(df["y_pred"], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        if len(y_true) == 0:
            return {"mae": math.nan, "mse": math.nan, "rmse": math.nan, "r2": math.nan}
        residual = y_pred - y_true
        mae = float(np.mean(np.abs(residual)))
        mse = float(np.mean(residual**2))
        rmse = float(math.sqrt(mse))
        denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
        r2 = float(1.0 - np.sum(residual**2) / denom) if denom > 0 else math.nan
        return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}

    @staticmethod
    def ci_mean(values: list[float]) -> tuple[float, float, float]:
        arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
        if len(arr) == 0:
            return math.nan, math.nan, math.nan
        mean = float(np.mean(arr))
        if len(arr) == 1:
            return mean, mean, mean
        sem = float(stats.sem(arr))
        crit = float(stats.t.ppf(0.975, len(arr) - 1))
        return mean, mean - crit * sem, mean + crit * sem

    @staticmethod
    def fmt(value: float) -> str:
        if value is None or not math.isfinite(float(value)):
            return ""
        return f"{float(value):.6g}"

    @staticmethod
    def ci_text(low: float, high: float) -> str:
        if not math.isfinite(low) or not math.isfinite(high):
            return ""
        return f"[{low:.6g}, {high:.6g}]"

    def latest_by_model_fold(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        temp = df.copy()
        temp["_source_order"] = np.arange(len(temp))
        temp["fold"] = pd.to_numeric(temp["fold"], errors="coerce").astype("Int64")
        latest = temp.sort_values("_source_order").drop_duplicates(["model_name", "fold"], keep="last")
        return latest.drop(columns=["_source_order"]).sort_values(["model_name", "fold"]).reset_index(drop=True)

    def audit(self) -> bool:
        rows: list[dict[str, Any]] = []

        def add(check: str, passed: bool, details: str, n_issues: int = 0) -> None:
            rows.append({"check": check, "passed": bool(passed), "n_issues": int(n_issues), "details": details})

        summary = self.csv(self.tables / "all_models_5fold_results.csv")
        fold_results = self.csv(self.tables / "all_models_5fold_fold_results.csv")
        completion = self.csv(self.tables / "full_evaluation_completion_check.csv")
        train_status = self.csv(self.tables / "all_models_training_status.csv")
        leakage = self.csv(self.tables / "all_models_leakage_check.csv")
        preds = self.load_predictions()

        add("prediction_file_count", len(self.prediction_files()) == 95, f"found={len(self.prediction_files())}, expected=95", abs(len(self.prediction_files()) - 95))
        completion_ok = len(completion) == 95 and (completion["ok_for_final_results"].astype(str) == "True").all()
        add("completion_check_95_95", completion_ok, f"rows={len(completion)}, passed={(completion['ok_for_final_results'].astype(str) == 'True').sum()}")

        pred_counts = preds.groupby("model_name")["file_fold"].nunique().to_dict() if not preds.empty else {}
        bad_models = {m: c for m, c in pred_counts.items() if c != 5}
        add("each_model_has_5_prediction_folds", len(pred_counts) == 19 and not bad_models, f"models={len(pred_counts)}, bad={bad_models}", len(bad_models))

        required_failures = []
        for path in self.prediction_files():
            frame = self.csv(path)
            missing = [col for col in REQUIRED_PRED_COLS if col not in frame.columns]
            if missing:
                required_failures.append(f"{path.name}:missing={missing}")
        add("prediction_required_columns", not required_failures, "; ".join(required_failures[:5]) or "all prediction files contain required columns", len(required_failures))

        nan_pred = int(pd.to_numeric(preds.get("y_pred", pd.Series(dtype=float)), errors="coerce").isna().sum()) if not preds.empty else 0
        add("nan_predictions_absent", nan_pred == 0, f"nan_y_pred_count={nan_pred}", nan_pred)

        duplicate_rows = []
        for model, frame in preds.groupby("model_name"):
            per_patient = frame.groupby("patient_id")["fold"].nunique()
            dup = per_patient[per_patient > 1]
            if len(dup):
                duplicate_rows.append(f"{model}:{len(dup)}")
        add("duplicate_patient_id_across_test_folds_absent", not duplicate_rows, "; ".join(duplicate_rows) or "no per-model patient_id appears in more than one test fold", len(duplicate_rows))

        latest_leak = self.latest_by_model_fold(leakage)
        leak_flags = ["endpoint_tbr_in_features", "endpoint_time_in_features", "endpoint_window_in_features", "loss_is_nan", "prediction_is_nan"]
        leak_bad = []
        for _, row in latest_leak.iterrows():
            for flag in leak_flags:
                if str(row.get(flag, "")).lower() == "true":
                    leak_bad.append(f"{row['model_name']}/fold{row['fold']}:{flag}=True")
            for col in ["train_val_overlap", "train_test_overlap", "val_test_overlap", "patient_level_leakage"]:
                if float(row.get(col, 0)) != 0:
                    leak_bad.append(f"{row['model_name']}/fold{row['fold']}:{col}={row.get(col)}")
        add("endpoint_and_split_leakage_absent_latest_rows", not leak_bad, "; ".join(leak_bad[:8]) or "latest rows have no endpoint/temporal leakage flags and no patient overlap", len(leak_bad))

        metric_cols = ["mean_test_mae", "mean_test_rmse", "mean_test_r2"]
        summary_metric_ok = summary[metric_cols].apply(pd.to_numeric, errors="coerce").notna().all().all()
        fold_metric_ok = fold_results[["test_mae", "test_rmse", "test_r2"]].apply(pd.to_numeric, errors="coerce").notna().all().all()
        add("metrics_available_for_all_models", bool(summary_metric_ok and fold_metric_ok), f"summary_rows={len(summary)}, fold_rows={len(fold_results)}")

        recomputed_rows = []
        for (model, fold), frame in preds.groupby(["model_name", "fold"]):
            metric = self.metrics(frame)
            recomputed_rows.append({"model_name": model, "fold": int(fold), **metric})
        recomputed = pd.DataFrame(recomputed_rows)
        merged_folds = recomputed.merge(fold_results, on=["model_name", "fold"], how="left", suffixes=("_recomputed", "_reported"))
        fold_bad = []
        for metric_name, reported_col in [("mae", "test_mae"), ("rmse", "test_rmse"), ("r2", "test_r2")]:
            diff = (pd.to_numeric(merged_folds[metric_name], errors="coerce") - pd.to_numeric(merged_folds[reported_col], errors="coerce")).abs()
            bad = merged_folds[diff > 1e-6]
            fold_bad.extend([f"{row.model_name}/fold{int(row.fold)}:{metric_name}" for row in bad.itertuples()])
        add("fold_metrics_match_prediction_recompute", not fold_bad, "; ".join(fold_bad[:8]) or "fold metrics match prediction recomputation", len(fold_bad))

        summary_bad = []
        recomputed_summary = recomputed.groupby("model_name").agg(mean_test_mae=("mae", "mean"), mean_test_rmse=("rmse", "mean"), mean_test_r2=("r2", "mean")).reset_index()
        merged_summary = recomputed_summary.merge(summary, on="model_name", suffixes=("_recomputed", "_reported"))
        for metric_name in ["mean_test_mae", "mean_test_rmse", "mean_test_r2"]:
            recomputed_values = pd.to_numeric(merged_summary[f"{metric_name}_recomputed"], errors="coerce")
            reported_values = pd.to_numeric(merged_summary[f"{metric_name}_reported"], errors="coerce")
            diff = (recomputed_values - reported_values).abs()
            rel = diff / (recomputed_values.abs() + 1e-12)
            bad = merged_summary[(diff > 1e-6) & (rel > 1e-8)]
            summary_bad.extend([f"{row.model_name}:{metric_name}" for row in bad.itertuples()])
        add("summary_metrics_match_prediction_recompute", not summary_bad, "; ".join(summary_bad[:8]) or "summary metrics match mean of recomputed fold metrics", len(summary_bad))

        historical_failed = int((train_status["status"].astype(str) != "success").sum())
        latest_status = self.latest_by_model_fold(train_status)
        latest_failed = int((latest_status["status"].astype(str) != "success").sum())
        add("historical_failures_resolved_by_latest_rows", latest_failed == 0, f"historical_failed_rows={historical_failed}; latest_failed_rows={latest_failed}", latest_failed)

        audit_df = pd.DataFrame(rows)
        self.write(audit_df, "main_result_audit.csv")
        return bool(audit_df["passed"].all())

    def build_ranked_results(self) -> dict[str, Any]:
        summary = self.csv(self.tables / "all_models_5fold_results.csv")
        folds = self.csv(self.tables / "all_models_5fold_fold_results.csv")
        for col in ["mean_test_mae", "mean_test_rmse", "mean_test_r2"]:
            summary[col] = pd.to_numeric(summary[col], errors="coerce")
        summary = summary.sort_values("mean_test_mae").reset_index(drop=True)
        summary["Method"] = summary["model_name"].map(method_label)
        summary["Category"] = summary["model_name"].map(category)
        summary["Rank_by_MAE"] = np.arange(1, len(summary) + 1)

        ci_records = []
        for model, frame in folds.groupby("model_name"):
            mae_mean, mae_low, mae_high = self.ci_mean(pd.to_numeric(frame["test_mae"], errors="coerce").tolist())
            rmse_mean, rmse_low, rmse_high = self.ci_mean(pd.to_numeric(frame["test_rmse"], errors="coerce").tolist())
            ci_records.append({
                "model_name": model,
                "MAE_95CI": self.ci_text(mae_low, mae_high),
                "RMSE_95CI": self.ci_text(rmse_low, rmse_high),
            })
        ci_df = pd.DataFrame(ci_records)
        ranked = summary.merge(ci_df, on="model_name", how="left")
        self.write(ranked, "main_results_ranked.csv")

        best_baseline = ranked[ranked["model_name"] != "kg_latentnet"].iloc[0]
        paper = pd.DataFrame({
            "Method": ranked["Method"],
            "Category": ranked["Category"],
            "MAE": ranked["mean_test_mae"].map(lambda x: f"{x:.6g}"),
            "MAE_95CI": ranked["MAE_95CI"],
            "RMSE": ranked["mean_test_rmse"].map(lambda x: f"{x:.6g}"),
            "RMSE_95CI": ranked["RMSE_95CI"],
            "R2": ranked["mean_test_r2"].map(lambda x: f"{x:.6g}"),
            "Rank_by_MAE": ranked["Rank_by_MAE"],
            "Notes": [
                "Proposed model" if m == "kg_latentnet" else "Best baseline" if m == best_baseline["model_name"] else category(m)
                for m in ranked["model_name"]
            ],
        })
        self.write(paper, "main_results_for_paper.csv")

        kg = ranked[ranked["model_name"] == "kg_latentnet"].iloc[0]
        diff = float(kg["mean_test_mae"] - best_baseline["mean_test_mae"])
        relative_reduction = float((best_baseline["mean_test_mae"] - kg["mean_test_mae"]) / best_baseline["mean_test_mae"])
        comparison = pd.DataFrame([{
            "kg_latentnet_mae": kg["mean_test_mae"],
            "kg_latentnet_rmse": kg["mean_test_rmse"],
            "kg_latentnet_r2": kg["mean_test_r2"],
            "best_baseline_model": best_baseline["model_name"],
            "best_baseline_method": best_baseline["Method"],
            "best_baseline_mae": best_baseline["mean_test_mae"],
            "kg_minus_best_baseline_mae": diff,
            "relative_mae_reduction": relative_reduction,
            "kg_better_than_best_baseline": diff < 0,
            "trend_note": "KG-LatentNet is not better than the best baseline in the locked full 5-fold results; keep results unchanged and investigate model/data reasons.",
        }])
        self.write(comparison, "main_result_comparison.csv")
        return {"best_baseline": str(best_baseline["model_name"]), "kg_better": bool(diff < 0)}

    def best_baseline(self) -> str:
        ranked = self.csv(self.tables / "main_results_ranked.csv") if (self.tables / "main_results_ranked.csv").exists() else None
        if ranked is None:
            self.build_ranked_results()
            ranked = self.csv(self.tables / "main_results_ranked.csv")
        ranked["mean_test_mae"] = pd.to_numeric(ranked["mean_test_mae"], errors="coerce")
        return str(ranked[ranked["model_name"] != "kg_latentnet"].sort_values("mean_test_mae").iloc[0]["model_name"])

    def paired_frame(self, subgroup: str = "Overall") -> tuple[pd.DataFrame, str]:
        preds = self.load_predictions()
        best = self.best_baseline()
        kg = preds[preds["model_name"] == "kg_latentnet"].copy()
        base = preds[preds["model_name"] == best].copy()
        keep = ["patient_id", "fold", "endpoint_window", "y_true", "y_pred", "absolute_error"]
        merged = kg[keep].merge(base[keep], on="patient_id", suffixes=("_kg", "_baseline"))
        merged["endpoint_window"] = pd.to_numeric(merged["endpoint_window_kg"], errors="coerce").astype("Int64")
        merged["kg_abs_error"] = pd.to_numeric(merged["absolute_error_kg"], errors="coerce")
        merged["baseline_abs_error"] = pd.to_numeric(merged["absolute_error_baseline"], errors="coerce")
        merged["diff_kg_minus_baseline"] = merged["kg_abs_error"] - merged["baseline_abs_error"]
        if subgroup == "18/24 pooled":
            merged = merged[merged["endpoint_window"].isin([18, 24])]
        elif subgroup != "Overall":
            window = int(str(subgroup).split()[0])
            merged = merged[merged["endpoint_window"] == window]
        return merged.reset_index(drop=True), best

    @staticmethod
    def bootstrap_mean_ci(values: np.ndarray, n_boot: int = 1000, seed: int = 20260606) -> tuple[float, float, float]:
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if len(values) == 0:
            return math.nan, math.nan, math.nan
        rng = np.random.default_rng(seed)
        means = np.empty(n_boot, dtype=float)
        for idx in range(n_boot):
            sample = rng.choice(values, size=len(values), replace=True)
            means[idx] = np.mean(sample)
        return float(np.mean(values)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

    @staticmethod
    def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        x = x[np.isfinite(x)]
        y = y[np.isfinite(y)]
        if len(x) == 0 or len(y) == 0:
            return math.nan
        comparisons = x[:, None] - y[None, :]
        return float((np.sum(comparisons > 0) - np.sum(comparisons < 0)) / comparisons.size)

    def reliability(self) -> None:
        subgroups = ["Overall", "6 month", "12 month", "18 month", "24 month", "18/24 pooled"]
        ci_rows = []
        wilcoxon_rows = []
        effect_rows = []
        for subgroup in subgroups:
            paired, best = self.paired_frame(subgroup)
            diff = paired["diff_kg_minus_baseline"].to_numpy(dtype=float)
            mean_diff, low, high = self.bootstrap_mean_ci(diff)
            kg_mae = float(paired["kg_abs_error"].mean()) if len(paired) else math.nan
            base_mae = float(paired["baseline_abs_error"].mean()) if len(paired) else math.nan
            ci_rows.append({
                "comparison": f"KG-LatentNet vs {method_label(best)}",
                "subgroup": subgroup,
                "n_pairs": len(paired),
                "kg_mae": kg_mae,
                "best_baseline_mae": base_mae,
                "mean_diff_kg_minus_baseline": mean_diff,
                "ci_low": low,
                "ci_high": high,
                "median_diff_kg_minus_baseline": float(np.nanmedian(diff)) if len(diff) else math.nan,
                "interpretation": "KG worse" if mean_diff > 0 and low > 0 else "KG better" if mean_diff < 0 and high < 0 else "statistically uncertain",
            })
            if len(diff) and np.any(np.abs(diff) > 0):
                stat, pvalue = stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided", mode="auto")
            else:
                stat, pvalue = math.nan, math.nan
            wilcoxon_rows.append({"comparison": f"KG-LatentNet vs {method_label(best)}", "subgroup": subgroup, "n_pairs": len(paired), "statistic": stat, "p_value": pvalue})
            std = float(np.nanstd(diff, ddof=1)) if len(diff) > 1 else math.nan
            cohen = float(np.nanmean(diff) / std) if std and math.isfinite(std) and std > 0 else math.nan
            reduction = paired["baseline_abs_error"].to_numpy(dtype=float) - paired["kg_abs_error"].to_numpy(dtype=float)
            effect_rows.append({
                "comparison": f"KG-LatentNet vs {method_label(best)}",
                "subgroup": subgroup,
                "n_pairs": len(paired),
                "cohens_d_for_kg_minus_baseline_error": cohen,
                "cliffs_delta_kg_error_vs_baseline_error": self.cliffs_delta(paired["kg_abs_error"].to_numpy(dtype=float), paired["baseline_abs_error"].to_numpy(dtype=float)),
                "median_paired_error_reduction_baseline_minus_kg": float(np.nanmedian(reduction)) if len(reduction) else math.nan,
                "mean_paired_error_reduction_baseline_minus_kg": float(np.nanmean(reduction)) if len(reduction) else math.nan,
                "relative_mae_reduction": float((base_mae - kg_mae) / base_mae) if base_mae and math.isfinite(base_mae) else math.nan,
            })
        self.write(pd.DataFrame(ci_rows), "paired_error_difference_ci.csv")
        self.write(pd.DataFrame(wilcoxon_rows), "wilcoxon_tests.csv")
        self.write(pd.DataFrame(effect_rows), "effect_size_analysis.csv")
        self.fold_level_results()
        self.residual_analysis_and_plots()

    def fold_level_results(self) -> pd.DataFrame:
        preds = self.load_predictions()
        best = self.best_baseline()
        rows = []
        for (model, fold), frame in preds.groupby(["model_name", "fold"]):
            metric = self.metrics(frame)
            rows.append({"model_name": model, "method": method_label(model), "fold": int(fold), "n": len(frame), "mae": metric["mae"], "rmse": metric["rmse"], "r2": metric["r2"]})
        out = pd.DataFrame(rows).sort_values(["model_name", "fold"])
        best_fold = out[out["model_name"] == best][["fold", "mae"]].rename(columns={"mae": "best_baseline_fold_mae"})
        kg_fold = out[out["model_name"] == "kg_latentnet"][["fold", "mae"]].rename(columns={"mae": "kg_latentnet_fold_mae"})
        diff = kg_fold.merge(best_fold, on="fold")
        diff["kg_minus_best_baseline_fold_mae"] = diff["kg_latentnet_fold_mae"] - diff["best_baseline_fold_mae"]
        out = out.merge(diff[["fold", "best_baseline_fold_mae", "kg_latentnet_fold_mae", "kg_minus_best_baseline_fold_mae"]], on="fold", how="left")
        out["best_baseline_model"] = best
        self.write(out, "fold_level_results.csv")
        return out

    def residual_analysis_and_plots(self) -> None:
        preds = self.load_predictions()
        best = self.best_baseline()
        focus = preds[preds["model_name"].isin(["kg_latentnet", best])].copy()
        focus["residual"] = pd.to_numeric(focus["y_pred"], errors="coerce") - pd.to_numeric(focus["y_true"], errors="coerce")
        focus["absolute_error"] = pd.to_numeric(focus["absolute_error"], errors="coerce")
        focus["endpoint_window"] = pd.to_numeric(focus["endpoint_window"], errors="coerce")
        rows = []
        for model, frame in focus.groupby("model_name"):
            def corr(a: str, b: str) -> tuple[float, float]:
                sub = frame[[a, b]].dropna()
                if len(sub) < 3 or sub[a].nunique() < 2 or sub[b].nunique() < 2:
                    return math.nan, math.nan
                r, p = stats.spearmanr(sub[a], sub[b])
                return float(r), float(p)

            abs_true_r, abs_true_p = corr("absolute_error", "y_true")
            abs_time_r, abs_time_p = corr("absolute_error", "endpoint_window")
            pred_true_r, pred_true_p = corr("y_pred", "y_true")
            rows.append({
                "model_name": model,
                "method": method_label(model),
                "n": len(frame),
                "residual_mean": float(frame["residual"].mean()),
                "residual_sd": float(frame["residual"].std(ddof=1)),
                "residual_median": float(frame["residual"].median()),
                "absolute_error_mean": float(frame["absolute_error"].mean()),
                "spearman_abs_error_vs_true_tbr": abs_true_r,
                "spearman_abs_error_vs_true_tbr_p": abs_true_p,
                "spearman_abs_error_vs_followup_month": abs_time_r,
                "spearman_abs_error_vs_followup_month_p": abs_time_p,
                "spearman_predicted_vs_true_tbr": pred_true_r,
                "spearman_predicted_vs_true_tbr_p": pred_true_p,
            })
        self.write(pd.DataFrame(rows), "residual_analysis_summary.csv")

        kg = focus[focus["model_name"] == "kg_latentnet"].copy()
        plt.figure(figsize=(6, 4))
        plt.scatter(kg["y_true"], kg["residual"], c=kg["endpoint_window"], cmap="viridis", alpha=0.75, edgecolor="none")
        plt.axhline(0, color="black", linewidth=1)
        plt.xlabel("True TBR")
        plt.ylabel("Residual (prediction - truth)")
        plt.title("KG-LatentNet residual vs true TBR")
        plt.colorbar(label="Follow-up month")
        plt.tight_layout()
        plt.savefig(self.figs / "residual_vs_true_tbr.png", dpi=200)
        plt.close()

        plt.figure(figsize=(6, 4))
        plt.scatter(kg["endpoint_window"], kg["absolute_error"], alpha=0.75, edgecolor="none")
        plt.xlabel("Follow-up month")
        plt.ylabel("Absolute error")
        plt.title("KG-LatentNet absolute error by follow-up")
        plt.tight_layout()
        plt.savefig(self.figs / "residual_vs_followup_month.png", dpi=200)
        plt.close()

        plt.figure(figsize=(5, 5))
        plt.scatter(kg["y_true"], kg["y_pred"], c=kg["endpoint_window"], cmap="viridis", alpha=0.75, edgecolor="none")
        lim_low = float(np.nanmin([kg["y_true"].min(), kg["y_pred"].min()]))
        lim_high = float(np.nanmax([kg["y_true"].max(), kg["y_pred"].max()]))
        plt.plot([lim_low, lim_high], [lim_low, lim_high], color="black", linewidth=1)
        plt.xlabel("True TBR")
        plt.ylabel("Predicted TBR")
        plt.title("KG-LatentNet predicted vs true TBR")
        plt.colorbar(label="Follow-up month")
        plt.tight_layout()
        plt.savefig(self.figs / "predicted_vs_true_tbr.png", dpi=200)
        plt.close()

        paired, best = self.paired_frame("Overall")
        plt.figure(figsize=(6, 4))
        plt.hist(paired["diff_kg_minus_baseline"].dropna(), bins=35, color="#4c78a8", alpha=0.85)
        plt.axvline(0, color="black", linewidth=1)
        plt.xlabel("KG absolute error - best baseline absolute error")
        plt.ylabel("Patient count")
        plt.title(f"Paired error difference vs {method_label(best)}")
        plt.tight_layout()
        plt.savefig(self.figs / "paired_error_difference_distribution.png", dpi=200)
        plt.close()

    def long_term(self) -> None:
        preds = self.load_predictions()
        rows = []
        for (model, window), frame in preds.groupby(["model_name", "endpoint_window"]):
            metric = self.metrics(frame)
            rows.append({"model_name": model, "method": method_label(model), "endpoint_window": int(window), "n": len(frame), "mae": metric["mae"], "rmse": metric["rmse"], "r2": metric["r2"]})
        stage = pd.DataFrame(rows).sort_values(["model_name", "endpoint_window"])
        pooled_rows = []
        for model, frame in preds[preds["endpoint_window"].isin([18, 24])].groupby("model_name"):
            metric = self.metrics(frame)
            pooled_rows.append({"model_name": model, "method": method_label(model), "endpoint_window": "18/24 pooled", "n": len(frame), "mae": metric["mae"], "rmse": metric["rmse"], "r2": metric["r2"]})
        pooled = pd.DataFrame(pooled_rows)
        self.write(pd.concat([stage, pooled], ignore_index=True), "all_models_stage_results.csv")
        self.write(pooled.sort_values("mae"), "long_term_pooled_results.csv")

        fold_rows = []
        for (model, fold, window), frame in preds.groupby(["model_name", "fold", "endpoint_window"]):
            metric = self.metrics(frame)
            fold_rows.append({"model_name": model, "method": method_label(model), "fold": int(fold), "endpoint_window": int(window), "n": len(frame), "mae": metric["mae"], "rmse": metric["rmse"], "r2": metric["r2"]})
        fold_df = pd.DataFrame(fold_rows).sort_values(["model_name", "endpoint_window", "fold"])
        self.write(fold_df, "long_term_fold_level_results.csv")

        ci_rows = []
        for subgroup in ["18 month", "24 month", "18/24 pooled"]:
            paired, best = self.paired_frame(subgroup)
            diff = paired["diff_kg_minus_baseline"].to_numpy(dtype=float)
            mean_diff, low, high = self.bootstrap_mean_ci(diff)
            ci_rows.append({"comparison": f"KG-LatentNet vs {method_label(best)}", "subgroup": subgroup, "n_pairs": len(paired), "mean_diff_kg_minus_baseline": mean_diff, "ci_low": low, "ci_high": high, "interpretation": "KG worse" if mean_diff > 0 and low > 0 else "KG better" if mean_diff < 0 and high < 0 else "statistically uncertain"})
        self.write(pd.DataFrame(ci_rows), "long_term_paired_difference_ci.csv")

        seed_sens = pd.DataFrame([{
            "analysis": "long_term_seed_sensitivity",
            "status": "unavailable",
            "reason": "No complete multi-seed full_5fold test prediction set was found. Do not infer seed sensitivity from the single locked run.",
        }])
        self.write(seed_sens, "long_term_seed_sensitivity.csv")
        (self.root / "seed_sensitivity_unavailable_report.md").write_text(
            "# Seed Sensitivity Unavailable\n\nNo complete multi-seed full 5-fold test prediction set was found. The current locked evaluation contains one selected seed per model/fold, so long-term seed sensitivity was not estimated.\n",
            encoding="utf-8",
        )

        best = self.best_baseline()
        subset = fold_df[fold_df["model_name"].isin(["kg_latentnet", best])].copy()
        plt.figure(figsize=(7, 4))
        for model, frame in subset.groupby("model_name"):
            stats_df = frame.groupby("endpoint_window")["mae"].agg(["mean", "std"]).reset_index()
            plt.errorbar(stats_df["endpoint_window"], stats_df["mean"], yerr=stats_df["std"], marker="o", capsize=3, label=method_label(model))
        plt.xlabel("Follow-up month")
        plt.ylabel("Fold MAE")
        plt.title("Long-term fold-level MAE")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.figs / "long_term_fold_level_mae.png", dpi=200)
        plt.close()

        ci_df = pd.DataFrame(ci_rows)
        plt.figure(figsize=(6, 3.5))
        y = np.arange(len(ci_df))
        plt.errorbar(ci_df["mean_diff_kg_minus_baseline"], y, xerr=[ci_df["mean_diff_kg_minus_baseline"] - ci_df["ci_low"], ci_df["ci_high"] - ci_df["mean_diff_kg_minus_baseline"]], fmt="o", capsize=4)
        plt.axvline(0, color="black", linewidth=1)
        plt.yticks(y, ci_df["subgroup"])
        plt.xlabel("KG absolute error - best baseline absolute error")
        plt.title("Long-term paired difference 95% CI")
        plt.tight_layout()
        plt.savefig(self.figs / "long_term_paired_difference_ci.png", dpi=200)
        plt.close()

    def latent_population(self) -> None:
        latent_rows = []
        metadata = []
        for file_name in sorted(self.latent.glob("kg_latentnet_fold*_latent_states.pkl")):
            with file_name.open("rb") as handle:
                obj = pickle.load(handle)
            metadata.append({
                "file": str(file_name),
                "has_patient_latent_rows": isinstance(obj, dict) and "patient_latent_rows" in obj,
                "has_prior_matrix": isinstance(obj, dict) and "prior_matrix_used" in obj,
                "has_dynamic_graph_projection_weight": isinstance(obj, dict) and "dynamic_graph_projection_weight" in obj,
                "learned_relation_weights_available": bool(obj.get("learned_relation_weights_available", False)) if isinstance(obj, dict) else False,
                "note": obj.get("note", "") if isinstance(obj, dict) else "",
            })
            if isinstance(obj, dict):
                latent_rows.extend(obj.get("patient_latent_rows", []))
        latent_df = pd.DataFrame(latent_rows)
        pd.DataFrame(metadata).to_csv(self.tables / "latent_output_inventory.csv", index=False, encoding="utf-8-sig")
        if latent_df.empty:
            (self.root / "latent_output_missing_report.md").write_text("# Latent Output Missing\n\nNo KG-LatentNet patient latent rows were found.\n", encoding="utf-8")
            return
        preds = self.load_predictions()
        kg = preds[preds["model_name"] == "kg_latentnet"].copy()
        merged = latent_df.merge(kg[["patient_id", "fold", "endpoint_window", "y_true", "y_pred", "absolute_error"]], on=["patient_id", "fold", "endpoint_window"], how="left", suffixes=("_latent", ""))
        for col in ["latent_state_score", "short_contribution_score", "delayed_contribution_score", "endpoint_window", "y_true", "absolute_error"]:
            if col in merged.columns:
                merged[col] = pd.to_numeric(merged[col], errors="coerce")
        summary_rows = []
        for label, frame in [("Overall", merged)] + [(f"{int(w)} month", f) for w, f in merged.groupby("endpoint_window")]:
            summary_rows.append({
                "group": label,
                "n": len(frame),
                "latent_state_score_mean": frame["latent_state_score"].mean(),
                "latent_state_score_sd": frame["latent_state_score"].std(ddof=1),
                "absolute_error_mean": frame["absolute_error"].mean(),
                "spearman_latent_score_vs_true_tbr": stats.spearmanr(frame["latent_state_score"], frame["y_true"], nan_policy="omit").statistic if len(frame.dropna(subset=["latent_state_score", "y_true"])) >= 3 else math.nan,
            })
        self.write(pd.DataFrame(summary_rows), "population_latent_summary.csv")
        contrib = merged.groupby("endpoint_window").agg(
            n=("patient_id", "count"),
            short_contribution_score_mean=("short_contribution_score", "mean"),
            delayed_contribution_score_mean=("delayed_contribution_score", "mean"),
            latent_state_score_mean=("latent_state_score", "mean"),
        ).reset_index()
        self.write(contrib, "contribution_trend_summary.csv")
        self.write(pd.DataFrame([{
            "status": "unavailable",
            "reason": "Latent category thresholds require train-set latent state quantiles, but saved full_5fold latent files contain test patient latent rows only.",
            "required_threshold_rule": "train-set 33% and 66% quantiles",
        }]), "latent_category_distribution.csv")

        plt.figure(figsize=(6, 4))
        plt.hist(kg["y_true"], bins=30, color="#4c78a8", alpha=0.85)
        plt.xlabel("Endpoint TBR")
        plt.ylabel("Patient count")
        plt.title("Endpoint TBR distribution")
        plt.tight_layout()
        plt.savefig(self.figs / "fig2a_endpoint_tbr_distribution.png", dpi=200)
        plt.close()

        plt.figure(figsize=(6, 4))
        plt.plot(contrib["endpoint_window"], contrib["short_contribution_score_mean"], marker="o", label="Short contribution")
        plt.plot(contrib["endpoint_window"], contrib["delayed_contribution_score_mean"], marker="o", label="Delayed contribution")
        plt.xlabel("Follow-up month")
        plt.ylabel("Mean contribution score")
        plt.title("Contribution trend by follow-up")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.figs / "fig2c_contribution_trend.png", dpi=200)
        plt.close()

        (self.root / "latent_output_missing_report.md").write_text(
            "# Latent Output Availability\n\nPatient-level latent state scores and short/delayed contribution scores were found, so Fig.2a and Fig.2c were generated. Train-set latent state scores for quantile thresholds were not found, so Fig.2b latent categories were not generated. Learned relation weights were not available in the saved latent files.\n",
            encoding="utf-8",
        )

    def individual_trajectory(self) -> None:
        self.write(pd.DataFrame([{
            "status": "unavailable",
            "reason": "Saved KG-LatentNet latent files contain one patient-level latent_state_score per endpoint prediction, not longitudinal latent trajectory arrays.",
            "selection_rule_status": "not_applicable_without_trajectory_output",
        }]), "representative_patient_selection.csv")
        self.write(pd.DataFrame([{
            "status": "unavailable",
            "reason": "No latent trajectory values were saved for individual patients.",
        }]), "individual_latent_trajectory_values.csv")
        (self.root / "individual_latent_trajectory_missing_report.md").write_text(
            "# Individual Latent Trajectory Missing\n\nThe saved KG-LatentNet latent files contain patient-level latent state scores at the endpoint but not longitudinal latent trajectory arrays. Fig.3 was not generated. Any future Fig.3 should state that the plot shows a model-derived latent vascular state trajectory, not a single-treatment causal effect.\n",
            encoding="utf-8",
        )

    def relation_heatmap(self) -> None:
        available = False
        notes = []
        for file_name in sorted(self.latent.glob("kg_latentnet_fold*_latent_states.pkl")):
            with file_name.open("rb") as handle:
                obj = pickle.load(handle)
            if isinstance(obj, dict):
                available = available or bool(obj.get("learned_relation_weights_available", False))
                notes.append(str(obj.get("note", "")))
        self.write(pd.DataFrame([{
            "status": "unavailable" if not available else "available_not_rendered",
            "learned_relation_weights_available": available,
            "reason": "No learned relation/attention/adjacency matrix was saved; prior_matrix_used and dynamic_graph_projection_weight are not treated as learned relation heatmap outputs." if not available else "Learned relation flag true; inspect saved files before rendering.",
            "note": " | ".join([n for n in notes if n])[:500],
        }]), "stage_variable_state_relation_summary.csv")
        if not available:
            (self.root / "stage_relation_heatmap_missing_report.md").write_text(
                "# Stage Relation Heatmap Missing\n\nThe saved KG-LatentNet latent files did not include learned relation weights from a dynamic graph, attention, or adjacency output. The available prior matrix was not plotted as a model-learned relation heatmap. Fig.4 was not generated.\n",
                encoding="utf-8",
            )

    def readiness(self) -> None:
        experiments = [
            ("extended_ablation", "scripts/21_run_extended_ablation.sh", "results/ablation/full_5fold"),
            ("knowledge_robustness", "scripts/24_run_knowledge_robustness.sh", "results/knowledge_robustness/full_5fold"),
            ("missing_data_robustness", "scripts/25_run_missing_data_robustness.sh", "results/missing_data_robustness/full_5fold"),
            ("temporal_generalization", "scripts/26_run_generalization_analysis.sh", "results/generalization/full_5fold"),
        ]
        required_inputs = [
            self.tables / "all_models_5fold_results.csv",
            self.tables / "full_evaluation_completion_check.csv",
            self.root / "configs" / "locked_full_5fold_config.yaml",
        ]
        input_ok = all(path.exists() for path in required_inputs)
        rows = []
        for name, script, out_dir in experiments:
            exists = (self.root / script).exists()
            rows.append({
                "experiment_name": name,
                "script_exists": exists,
                "required_inputs_exist": input_ok,
                "requires_retraining": True,
                "estimated_models_or_settings": "TBD by experiment design",
                "estimated_folds": 5,
                "output_dir": out_dir,
                "ready_to_run": bool(exists and input_ok),
                "notes": "Script missing; do not launch until implemented and approved." if not exists else "Inputs present; requires user approval before retraining.",
            })
        self.write(pd.DataFrame(rows), "supplementary_experiment_readiness.csv")

    def report(self) -> None:
        ranked = self.csv(self.tables / "main_results_ranked.csv")
        comparison = self.csv(self.tables / "main_result_comparison.csv")
        ci = self.csv(self.tables / "paired_error_difference_ci.csv")
        wil = self.csv(self.tables / "wilcoxon_tests.csv")
        eff = self.csv(self.tables / "effect_size_analysis.csv")
        residual = self.csv(self.tables / "residual_analysis_summary.csv")
        long_ci = self.csv(self.tables / "long_term_paired_difference_ci.csv")
        readiness = self.csv(self.tables / "supplementary_experiment_readiness.csv")
        completion = self.csv(self.tables / "full_evaluation_completion_check.csv")
        pred_count = len(self.prediction_files())
        ckpt_count = len(list((self.root / "results" / "checkpoints" / "full_5fold").glob("*")))
        kg = ranked[ranked["model_name"] == "kg_latentnet"].iloc[0]
        best = comparison.iloc[0]
        overall_ci = ci[ci["subgroup"] == "Overall"].iloc[0]
        overall_wil = wil[wil["subgroup"] == "Overall"].iloc[0]
        overall_eff = eff[eff["subgroup"] == "Overall"].iloc[0]
        kg_res = residual[residual["model_name"] == "kg_latentnet"].iloc[0]
        long_pooled = long_ci[long_ci["subgroup"] == "18/24 pooled"].iloc[0]
        fig_names = [
            "fig2a_endpoint_tbr_distribution.png",
            "fig2b_latent_state_categories.png",
            "fig2c_contribution_trend.png",
            "fig3_individual_latent_trajectories.png",
            "fig4_stage_variable_state_heatmap.png",
        ]
        fig_status = {name: (self.figs / name).exists() for name in fig_names}
        ready = readiness[readiness["ready_to_run"].astype(str) == "True"]["experiment_name"].tolist()
        top_lines = "\n".join(
            f"{int(row.Rank_by_MAE)}. {row.Method}: MAE={float(row.mean_test_mae):.4f}, RMSE={float(row.mean_test_rmse):.4f}, R2={float(row.mean_test_r2):.4f}"
            for row in ranked.itertuples()
        )
        text = f"""# Summary After Full 5-Fold Main Results

## Completion

- Full 5-fold completed: yes.
- Prediction files: {pred_count}.
- Checkpoint files: {ckpt_count}.
- Completion check: {(completion['ok_for_final_results'].astype(str) == 'True').sum()}/{len(completion)} passed.

## Main Ranking

{top_lines}

## KG-LatentNet vs Best Baseline

- KG-LatentNet: MAE={float(kg.mean_test_mae):.6g}, RMSE={float(kg.mean_test_rmse):.6g}, R2={float(kg.mean_test_r2):.6g}.
- Best baseline: {best.best_baseline_method} ({best.best_baseline_model}), MAE={float(best.best_baseline_mae):.6g}.
- KG-LatentNet better than best baseline: {bool(best.kg_better_than_best_baseline)}.
- KG minus best-baseline MAE: {float(best.kg_minus_best_baseline_mae):.6g}.
- Relative MAE reduction: {float(best.relative_mae_reduction):.6g}. Negative means KG-LatentNet is worse than the best baseline.
- Interpretation: the locked results do not support KG-LatentNet superiority over the best baseline. Results were not altered.

## Reliability

- Overall paired error difference, KG absolute error minus best baseline absolute error: mean={float(overall_ci.mean_diff_kg_minus_baseline):.6g}, 95% CI=[{float(overall_ci.ci_low):.6g}, {float(overall_ci.ci_high):.6g}], interpretation={overall_ci.interpretation}.
- Wilcoxon p-value: {float(overall_wil.p_value):.6g}.
- Cohen's d: {float(overall_eff.cohens_d_for_kg_minus_baseline_error):.6g}.
- Cliff's delta: {float(overall_eff.cliffs_delta_kg_error_vs_baseline_error):.6g}.
- Fold-level stability: KG-LatentNet shows large fold variation; interpret as unstable rather than robust.

## Residual Analysis

- KG residual mean={float(kg_res.residual_mean):.6g}, residual SD={float(kg_res.residual_sd):.6g}.
- Spearman absolute error vs true TBR={float(kg_res.spearman_abs_error_vs_true_tbr):.6g}; absolute error vs follow-up month={float(kg_res.spearman_abs_error_vs_followup_month):.6g}.
- Residual analysis indicates systematic performance concerns for KG-LatentNet in this locked run.

## Long-Term Follow-Up

- 18/24 pooled paired difference: mean={float(long_pooled.mean_diff_kg_minus_baseline):.6g}, 95% CI=[{float(long_pooled.ci_low):.6g}, {float(long_pooled.ci_high):.6g}], interpretation={long_pooled.interpretation}.
- Long-term conclusion: the current results do not support a robust long-term superiority claim.

## Latent Outputs and Figures

- Fig.2a generated: {fig_status['fig2a_endpoint_tbr_distribution.png']}.
- Fig.2b generated: {fig_status['fig2b_latent_state_categories.png']}.
- Fig.2c generated: {fig_status['fig2c_contribution_trend.png']}.
- Fig.3 generated: {fig_status['fig3_individual_latent_trajectories.png']}.
- Fig.4 generated: {fig_status['fig4_stage_variable_state_heatmap.png']}.
- Missing latent outputs: train-set latent scores for category thresholds, longitudinal latent trajectory arrays, and learned relation weights.

## Supplementary Experiment Readiness

- Ready experiments: {', '.join(ready) if ready else 'none'}.
- All checked retraining-type supplementary experiment scripts were missing, so no retraining experiment was launched.

## Next Step

Do not launch all retraining experiments at once. First review the unexpectedly strong clinical/classical baselines and the weak KG-LatentNet fold behavior, then decide whether to implement and run extended ablation, knowledge robustness, missing-data robustness, and generalization analyses.
"""
        (self.root / "summary_after_full_5fold_main_results.md").write_text(text, encoding="utf-8")

    def run_all(self) -> None:
        audit_ok = self.audit()
        if not audit_ok:
            raise SystemExit("main_result_audit.csv contains failed checks; stopping before downstream analysis.")
        self.build_ranked_results()
        self.reliability()
        self.long_term()
        self.latent_population()
        self.individual_trajectory()
        self.relation_heatmap()
        self.readiness()
        self.report()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default="/root/KG_LatentNet_Project")
    parser.add_argument("action", choices=["all", "audit", "main", "reliability", "longterm", "latent_population", "individual_trajectory", "relation_heatmap", "readiness", "report"])
    args = parser.parse_args()
    runner = Full5PostHoc(Path(args.project_root).resolve())
    if args.action == "all":
        runner.run_all()
    elif args.action == "audit":
        ok = runner.audit()
        if not ok:
            raise SystemExit(1)
    elif args.action == "main":
        runner.build_ranked_results()
    elif args.action == "reliability":
        runner.reliability()
    elif args.action == "longterm":
        runner.long_term()
    elif args.action == "latent_population":
        runner.latent_population()
    elif args.action == "individual_trajectory":
        runner.individual_trajectory()
    elif args.action == "relation_heatmap":
        runner.relation_heatmap()
    elif args.action == "readiness":
        runner.readiness()
    elif args.action == "report":
        runner.report()


if __name__ == "__main__":
    main()
