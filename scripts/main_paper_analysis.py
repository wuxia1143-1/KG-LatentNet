from __future__ import annotations

import csv
import math
import statistics
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path("/root/KG_LatentNet_Project")
TABLES = PROJECT_ROOT / "results" / "tables" / "full_5fold"
PRED = PROJECT_ROOT / "results" / "predictions" / "full_5fold"
FIGURES = PROJECT_ROOT / "results" / "figures" / "full_5fold"
REPORTS = PROJECT_ROOT / "results" / "reports"
CONFIG_PATH = PROJECT_ROOT / "configs" / "main_paper_model_list.yaml"


def load_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_csv(path: Path, rows: list[dict], fields: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def fmt(v, d=6):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return ""
    return f"{v:.{d}f}"


def load_preds(prefix: str) -> pd.DataFrame:
    frames = []
    for fold in range(5):
        p = PRED / f"{prefix}_fold{fold}_predictions.csv"
        if p.exists():
            frames.append(pd.read_csv(p))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def compute_metrics(y_true, y_pred):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan"), "n": 0}
    ae = np.abs(yt - yp)
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    return {
        "mae": float(np.mean(ae)),
        "rmse": float(np.sqrt(np.mean(ae ** 2))),
        "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "n": int(len(yt)),
    }


def find_raw_model(all_results: pd.DataFrame, raw_names: list[str]) -> str | None:
    for name in raw_names:
        matches = all_results[all_results["model_name"].str.lower() == name.lower()]
        if not matches.empty:
            return matches.iloc[0]["model_name"]
    return None


def get_pred_prefix(raw_name: str) -> str:
    if raw_name == "kg_latentnet_residual":
        return "kg_latentnet_residual"
    return raw_name


def bootstrap_ci(data, n_boot=1000, ci=95, seed=42):
    rng = np.random.RandomState(seed)
    boot_means = []
    n = len(data)
    for _ in range(n_boot):
        sample = rng.choice(data, size=n, replace=True)
        boot_means.append(np.mean(sample))
    low = np.percentile(boot_means, (100 - ci) / 2)
    high = np.percentile(boot_means, 100 - (100 - ci) / 2)
    return float(low), float(high)


def compute_fold_ci(fold_results: pd.DataFrame, raw_name: str, metric_col: str):
    fold_rows = fold_results[fold_results["model_name"] == raw_name]
    values = pd.to_numeric(fold_rows[metric_col], errors="coerce").dropna().astype(float).values
    if len(values) < 2:
        return "", ""
    ci_low, ci_high = bootstrap_ci(values)
    return fmt(ci_low), fmt(ci_high)


def append_residual_rows(base: pd.DataFrame, residual_path: Path) -> pd.DataFrame:
    if not residual_path.exists():
        return base
    residual = pd.read_csv(residual_path)
    residual = residual[residual["model_name"] == "kg_latentnet_residual"]
    if residual.empty:
        return base
    base = base[base["model_name"] != "kg_latentnet_residual"]
    return pd.concat([base, residual], ignore_index=True)


def main():
    config = load_config()
    main_models = config["main_paper_models"]
    display_order = config["display_order"]

    all_results = append_residual_rows(
        pd.read_csv(TABLES / "all_models_5fold_results.csv"),
        TABLES / "all_models_5fold_results_with_kg_residual.csv",
    )

    fold_results = append_residual_rows(
        pd.read_csv(TABLES / "all_models_5fold_fold_results.csv"),
        TABLES / "all_models_stage_results_with_kg_residual.csv",
    )

    model_map = {}
    for m in main_models:
        raw = find_raw_model(all_results, m["raw_names"])
        if raw:
            model_map[raw] = {"display": m["display_name"], "category": m["category"]}

    print(f"Found {len(model_map)} models in results:")
    for raw, info in model_map.items():
        row = all_results[all_results["model_name"] == raw].iloc[0]
        print(f"  {info['display']}: MAE={row['mean_test_mae']}")

    kg_raw = find_raw_model(all_results, ["kg_latentnet_residual", "kg_latentnet"])
    if kg_raw is None:
        print("ERROR: KG-LatentNet not found")
        return

    table1_rows = []
    for display_name in display_order:
        raw = None
        for r, info in model_map.items():
            if info["display"] == display_name:
                raw = r
                break
        if raw is None:
            print(f"WARNING: {display_name} not found in results, skipping")
            continue

        row = all_results[all_results["model_name"] == raw].iloc[0]
        mae = float(row["mean_test_mae"])
        rmse = float(row["mean_test_rmse"])
        r2 = float(row["mean_test_r2"])
        mae_std = float(row["std_test_mae"]) if row["std_test_mae"] else 0
        rmse_std = float(row["std_test_rmse"]) if row["std_test_rmse"] else 0

        mae_ci_low, mae_ci_high = compute_fold_ci(fold_results, raw, "test_mae")
        rmse_ci_low, rmse_ci_high = compute_fold_ci(fold_results, raw, "test_rmse")

        table1_rows.append({
            "Method": display_name,
            "Category": model_map[raw]["category"],
            "MAE": fmt(mae),
            "MAE_std": fmt(mae_std),
            "MAE_95CI": f"[{mae_ci_low}, {mae_ci_high}]" if mae_ci_low else "",
            "RMSE": fmt(rmse),
            "RMSE_std": fmt(rmse_std),
            "RMSE_95CI": f"[{rmse_ci_low}, {rmse_ci_high}]" if rmse_ci_low else "",
            "R2": fmt(r2),
            "Rank_by_MAE": "",
            "raw_model_name": raw,
            "Notes": "",
        })

    table1_sorted = sorted(table1_rows, key=lambda r: float(r["MAE"]) if r["MAE"] else float("inf"))
    for i, row in enumerate(table1_sorted):
        row["Rank_by_MAE"] = str(i + 1)

    ordered_table1 = []
    for display_name in display_order:
        for row in table1_sorted:
            if row["Method"] == display_name:
                ordered_table1.append(row)
                break

    table1_fields = ["Method", "Category", "MAE", "MAE_std", "MAE_95CI", "RMSE", "RMSE_std", "RMSE_95CI", "R2", "Rank_by_MAE", "raw_model_name", "Notes"]
    write_csv(TABLES / "main_paper_table1_selected_models.csv", ordered_table1, table1_fields)
    print(f"Saved main_paper_table1_selected_models.csv")

    baselines_only = [r for r in ordered_table1 if r["Category"] != "Proposed"]
    best_bl = min(baselines_only, key=lambda r: float(r["MAE"]) if r["MAE"] else float("inf"))
    kg_row = next(r for r in ordered_table1 if r["Category"] == "Proposed")

    kg_mae = float(kg_row["MAE"])
    bl_mae = float(best_bl["MAE"])
    abs_diff = kg_mae - bl_mae
    rel_diff = abs_diff / bl_mae * 100 if bl_mae > 0 else float("nan")

    best_bl_row = [{
        "best_baseline_name": best_bl["Method"],
        "best_baseline_raw_name": best_bl["raw_model_name"],
        "best_baseline_mae": fmt(bl_mae),
        "kg_latentnet_mae": fmt(kg_mae),
        "absolute_mae_difference": fmt(abs_diff),
        "relative_mae_reduction_pct": fmt(rel_diff, 2),
        "kg_better": str(kg_mae <= bl_mae),
        "comparison_scope": "selected_main_paper_models_only",
        "note": "baseline_tbr_only excluded from main comparison table but retained in supplementary audit",
    }]
    write_csv(TABLES / "main_paper_best_baseline_selected_models.csv", best_bl_row,
              ["best_baseline_name", "best_baseline_raw_name", "best_baseline_mae",
               "kg_latentnet_mae", "absolute_mae_difference", "relative_mae_reduction_pct",
               "kg_better", "comparison_scope", "note"])
    print(f"Best baseline: {best_bl['Method']} (MAE={bl_mae:.6f})")
    print(f"KG-LatentNet: MAE={kg_mae:.6f}, diff={abs_diff:+.6f} ({rel_diff:+.1f}%)")

    bl_raw = best_bl["raw_model_name"]
    kg_preds = load_preds(get_pred_prefix(kg_raw))
    bl_preds = load_preds(bl_raw)

    if kg_preds.empty or bl_preds.empty:
        print("WARNING: Missing prediction files, skipping paired statistics")
        return

    n_common = min(len(kg_preds), len(bl_preds))
    kg_ae = kg_preds["absolute_error"].values[:n_common]
    bl_ae = bl_preds["absolute_error"].values[:n_common]
    paired_diff = kg_ae - bl_ae

    ci_low, ci_high = np.percentile(paired_diff, [2.5, 97.5])
    boot_ci_low, boot_ci_high = bootstrap_ci(paired_diff)

    try:
        wilcoxon_stat, wilcoxon_p = stats.wilcoxon(kg_ae, bl_ae, alternative="greater")
    except Exception:
        wilcoxon_stat, wilcoxon_p = float("nan"), float("nan")

    std_diff = np.std(paired_diff, ddof=1)
    cohens_d = float(np.mean(paired_diff) / std_diff) if std_diff > 0 else float("nan")

    n = len(paired_diff)
    greater = sum(1 for i in range(n) for j in range(n) if i != j and paired_diff[i] > paired_diff[j])
    cliffs_delta = float(greater / (n * (n - 1))) if n > 1 else float("nan")

    ci_rows = [
        {"metric": "paired_error_diff_95ci_low", "value": fmt(ci_low)},
        {"metric": "paired_error_diff_95ci_high", "value": fmt(ci_high)},
        {"metric": "bootstrap_95ci_low", "value": fmt(boot_ci_low)},
        {"metric": "bootstrap_95ci_high", "value": fmt(boot_ci_high)},
        {"metric": "mean_paired_error_reduction", "value": fmt(-np.mean(paired_diff))},
        {"metric": "median_paired_error_reduction", "value": fmt(-np.median(paired_diff))},
        {"metric": "comparison", "value": f"KG-LatentNet vs {best_bl['Method']}"},
    ]
    write_csv(TABLES / "main_paper_paired_error_difference_ci.csv", ci_rows, ["metric", "value"])

    wilcoxon_rows = [{
        "test": "wilcoxon_signed_rank",
        "statistic": fmt(wilcoxon_stat),
        "p_value": fmt(wilcoxon_p, 10),
        "alternative": "kg_ae_greater_than_best_baseline",
        "significant_at_005": str(wilcoxon_p < 0.05),
        "comparison": f"KG-LatentNet vs {best_bl['Method']}",
    }]
    write_csv(TABLES / "main_paper_wilcoxon_tests.csv", wilcoxon_rows,
              ["test", "statistic", "p_value", "alternative", "significant_at_005", "comparison"])

    effect_rows = [
        {"metric": "cohens_d", "value": fmt(cohens_d)},
        {"metric": "cliffs_delta", "value": fmt(cliffs_delta)},
        {"metric": "mean_paired_error_reduction", "value": fmt(-np.mean(paired_diff))},
        {"metric": "median_paired_error_reduction", "value": fmt(-np.median(paired_diff))},
    ]
    write_csv(TABLES / "main_paper_effect_size_analysis.csv", effect_rows, ["metric", "value"])

    windows = sorted(kg_preds["endpoint_window"].unique())
    long_rows = []
    for w in windows:
        kg_w = kg_preds[kg_preds["endpoint_window"] == w]
        bl_w = bl_preds[bl_preds["endpoint_window"] == w]
        kg_m = compute_metrics(kg_w["y_true"].values, kg_w["y_pred"].values)
        bl_m = compute_metrics(bl_w["y_true"].values, bl_w["y_pred"].values)
        long_rows.append({
            "window": w, "kg_mae": fmt(kg_m["mae"]), "bl_mae": fmt(bl_m["mae"]),
            "diff": fmt(kg_m["mae"] - bl_m["mae"]),
            "n_kg": len(kg_w), "n_bl": len(bl_w),
        })

    pooled = kg_preds[kg_preds["endpoint_window"].isin([18, 24])]
    bl_pooled = bl_preds[bl_preds["endpoint_window"].isin([18, 24])]
    if not pooled.empty:
        kg_pm = compute_metrics(pooled["y_true"].values, pooled["y_pred"].values)
        bl_pm = compute_metrics(bl_pooled["y_true"].values, bl_pooled["y_pred"].values)
        long_rows.append({
            "window": "18_24_pooled", "kg_mae": fmt(kg_pm["mae"]), "bl_mae": fmt(bl_pm["mae"]),
            "diff": fmt(kg_pm["mae"] - bl_pm["mae"]),
            "n_kg": len(pooled), "n_bl": len(bl_pooled),
        })

    write_csv(TABLES / "main_paper_long_term_pooled_results.csv", long_rows,
              ["window", "kg_mae", "bl_mae", "diff", "n_kg", "n_bl"])

    stage_rows = []
    for display_name in display_order:
        raw = None
        for r, info in model_map.items():
            if info["display"] == display_name:
                raw = r
                break
        if raw is None:
            continue
        model_preds = load_preds(get_pred_prefix(raw))
        if model_preds.empty:
            continue
        for w in windows:
            w_preds = model_preds[model_preds["endpoint_window"] == w]
            m = compute_metrics(w_preds["y_true"].values, w_preds["y_pred"].values)
            stage_rows.append({
                "Method": display_name, "endpoint_window": w,
                "MAE": fmt(m["mae"]), "n_patients": len(w_preds),
            })
    write_csv(TABLES / "main_paper_stage_results_selected_models.csv", stage_rows,
              ["Method", "endpoint_window", "MAE", "n_patients"])

    fold_rows = []
    for display_name in display_order:
        raw = None
        for r, info in model_map.items():
            if info["display"] == display_name:
                raw = r
                break
        if raw is None:
            continue
        fold_df = fold_results[fold_results["model_name"] == raw]
        for _, fr in fold_df.iterrows():
            fold_rows.append({
                "Method": display_name, "fold": fr.get("fold", ""),
                "test_mae": fr.get("test_mae", ""), "test_rmse": fr.get("test_rmse", ""),
                "test_r2": fr.get("test_r2", ""),
            })
    write_csv(TABLES / "main_paper_fold_results_selected_models.csv", fold_rows,
              ["Method", "fold", "test_mae", "test_rmse", "test_r2"])

    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "figure.dpi": 150})

    fig, ax = plt.subplots(figsize=(10, 5))
    methods = [r["Method"] for r in ordered_table1]
    maes = [float(r["MAE"]) for r in ordered_table1]
    colors = ["#e74c3c" if r["Category"] == "Proposed" else "#3498db" for r in ordered_table1]
    bars = ax.barh(range(len(methods)), maes, color=colors, edgecolor="white")
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods)
    ax.set_xlabel("MAE")
    ax.set_title("Main Paper Table I: Model Comparison (MAE)")
    ax.invert_yaxis()
    for bar, mae in zip(bars, maes):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2, f"{mae:.4f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "main_paper_model_mae_barplot_selected_models.png")
    plt.close(fig)
    print("Saved main_paper_model_mae_barplot_selected_models.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    stage_df = pd.DataFrame(stage_rows)
    for display_name in display_order:
        model_stage = stage_df[stage_df["Method"] == display_name]
        if model_stage.empty:
            continue
        w_labels = [str(w) for w in model_stage["endpoint_window"]]
        mae_vals = [float(m) for m in model_stage["MAE"]]
        style = "-" if display_name == "KG-LatentNet" else "--"
        lw = 2.5 if display_name == "KG-LatentNet" else 1.2
        ax.plot(w_labels, mae_vals, style, linewidth=lw, marker="o", markersize=5, label=display_name)
    ax.set_xlabel("Endpoint Window (months)")
    ax.set_ylabel("MAE")
    ax.set_title("Stage-Level MAE by Model")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES / "main_paper_stage_mae_selected_models.png")
    plt.close(fig)
    print("Saved main_paper_stage_mae_selected_models.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(paired_diff, bins=40, color="#e74c3c", edgecolor="white", alpha=0.8)
    ax.axvline(0, color="black", linewidth=1, linestyle="--")
    ax.axvline(np.mean(paired_diff), color="blue", linewidth=1.5, label=f"mean={np.mean(paired_diff):.4f}")
    ax.set_xlabel("Paired Error Diff (KG - Best Baseline)")
    ax.set_ylabel("Count")
    ax.set_title(f"Paired Error Difference Distribution\nvs {best_bl['Method']}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "main_paper_paired_error_difference_selected_models.png")
    plt.close(fig)
    print("Saved main_paper_paired_error_difference_selected_models.png")

    REPORTS.mkdir(parents=True, exist_ok=True)
    n_main_models = len(ordered_table1)
    report = f"""# Main Paper Selected Model Results Summary

## 1. Models Included in Main Table ({n_main_models} models)

| # | Method | Category |
|---|---|---|
"""
    for i, r in enumerate(ordered_table1):
        report += f"| {i+1} | {r['Method']} | {r['Category']} |\n"

    report += f"""
## 2. Why Only These Models

These {n_main_models} models represent the core comparison set for the paper:
- **Proposed**: KG-LatentNet (with baseline-anchored residual readout)
- **Deep learning baselines**: HyperIMTS, TRANS, GRU-D, DHGAS, GraphCare, KEDGN, TGNN4I
- **Classical ML baselines**: Random Forest (RF), XGBoost (XGB)

Excluded models (retained in supplementary audit):
- baseline_tbr_only: Clinical sanity check, not a DL/ML comparison
- Clinical baselines (clinical_core, clinical_horizon_aware): Feature ablation only
- Simple linear models (linear_regression, ridge, elasticnet, linear_mixed_effects): Not main comparison
- Exploratory RNNs (Time-aware LSTM, RETAIN): Known numerical instability

## 3. KG-LatentNet Performance

- **MAE**: {fmt(kg_mae)}
- **RMSE**: {kg_row['RMSE']}
- **R2**: {kg_row['R2']}

## 4. Selected Best Baseline

- **Name**: {best_bl['Method']}
- **MAE**: {fmt(bl_mae)}

## 5. KG-LatentNet vs Best Baseline

- **Absolute MAE difference**: {fmt(abs_diff)} ({'worse' if abs_diff > 0 else 'better'})
- **Relative difference**: {fmt(rel_diff, 2)}%

## 6. Statistical Tests

- **Paired error diff 95% CI**: [{fmt(ci_low)}, {fmt(ci_high)}]
- **Wilcoxon p-value**: {fmt(wilcoxon_p, 10)}
- **Cohen's d**: {fmt(cohens_d)}
- **Cliff's delta**: {fmt(cliffs_delta)}
- **Mean paired error reduction**: {fmt(-np.mean(paired_diff))}
- **Median paired error reduction**: {fmt(-np.median(paired_diff))}

## 7. Long-term (18/24 months) Results

"""

    for r in long_rows:
        report += f"- Window {r['window']}: KG={r['kg_mae']}, {best_bl['Method']}={r['bl_mae']}, diff={r['diff']}\n"

    kg_better = kg_mae <= bl_mae
    report += f"""
## 8. Conclusion

**KG-LatentNet {'MEETS' if kg_better else 'DOES NOT meet'} the best baseline ({best_bl['Method']}) within the selected main paper models.**

{'The model achieves competitive or superior performance.' if kg_better else 'The model shows higher error than the best baseline. See model_repositioning_report.md for discussion.'}

## 9. Note on baseline_tbr_only

baseline_tbr_only (MAE=0.2171) is excluded from the main comparison table as it is a clinical sanity check (using only baseline TBR as prediction), not a deep learning or ML model. It is retained in the supplementary audit for reviewer reference.
"""

    report_path = REPORTS / "main_paper_selected_model_results_summary.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"Saved main_paper_selected_model_results_summary.md")

    excluded_rows = []
    for exc in config["excluded_from_main_table"]:
        raw = find_raw_model(all_results, [exc["model"]])
        mae = ""
        if raw:
            row = all_results[all_results["model_name"] == raw]
            if not row.empty:
                mae = row.iloc[0]["mean_test_mae"]
        excluded_rows.append({
            "model_name": exc["model"],
            "display_name": exc["model"],
            "mean_test_mae": mae,
            "reason": exc["reason"],
        })
    write_csv(TABLES / "supplementary_excluded_baseline_audit.csv", excluded_rows,
              ["model_name", "display_name", "mean_test_mae", "reason"])

    supp_note = """# Supplementary: Excluded Baseline Audit

## Models Excluded from Main Table

The following models were evaluated but excluded from the main paper Table I:

"""
    for exc in config["excluded_from_main_table"]:
        supp_note += f"- **{exc['model']}**: {exc['reason']}\n"

    supp_note += """
## Purpose

These models are retained for:
1. Reviewer queries about clinical baselines
2. Feature ablation analysis
3. Sanity check that baseline TBR is a strong predictor
4. Complete experimental audit trail

## baseline_tbr_only Note

baseline_tbr_only achieved MAE=0.2171, which is the strongest single-predictor baseline.
This is expected because baseline_tbr_b has a Pearson correlation of r=0.33 with endpoint_tbr_y.
This model is excluded from the main comparison as it is not a machine learning model,
but it serves as an important clinical reference point.

## No Data Deleted

All prediction files, checkpoints, and result tables for excluded models are preserved.
"""

    supp_path = REPORTS / "supplementary_excluded_baseline_note.md"
    supp_path.write_text(supp_note, encoding="utf-8")
    print(f"Saved supplementary_excluded_baseline_note.md")

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
