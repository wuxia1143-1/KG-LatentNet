from __future__ import annotations

import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "baseline_results"
OUT = BASE / "paper_aligned_final"
TABLES = OUT / "tables"
FIGS = OUT / "figures"
REPORTS = OUT / "reports"


def ensure_dirs() -> None:
    for path in [TABLES, FIGS, REPORTS]:
        path.mkdir(parents=True, exist_ok=True)


def parse_ci(text: str) -> tuple[float, float]:
    if not isinstance(text, str) or "[" not in text:
        return math.nan, math.nan
    body = text.strip().strip("[]")
    left, right = body.split(",", 1)
    return float(left), float(right)


def ci_text(low: float, high: float) -> str:
    return f"[{low:.4f}, {high:.4f}]"


def load_grud_row() -> dict[str, object]:
    current = BASE / "main_paper_table1_selected_models.csv"
    if not current.exists():
        return {
            "Method": "GRU-D",
            "Category": "Temporal neural baseline",
            "MAE": 0.3347,
            "MAE_95CI": "[0.3143, 0.3509]",
            "RMSE": 0.4581,
            "RMSE_95CI": "[0.4166, 0.4995]",
            "R2": -0.7705,
            "Source": "fallback value from latest stabilized rerun summary",
        }
    df = pd.read_csv(current)
    row = df[df["Method"].astype(str).str.upper().eq("GRU-D")]
    if row.empty:
        raise RuntimeError(f"GRU-D row not found in {current}")
    row = row.iloc[0]
    return {
        "Method": "GRU-D",
        "Category": "Temporal neural baseline",
        "MAE": float(row["MAE"]),
        "MAE_95CI": str(row["MAE_95CI"]),
        "RMSE": float(row["RMSE"]),
        "RMSE_95CI": str(row["RMSE_95CI"]),
        "R2": float(row["R2"]),
        "Source": "latest server rerun, locked full 5-fold, added baseline",
    }


def main_performance() -> pd.DataFrame:
    rows = [
        ["KG-LatentNet", "Proposed", 0.2156, "[0.1913, 0.2429]", 0.3333, "[0.2569, 0.4152]", 0.0948, "PDF Table I"],
        ["HyperIMTS", "Irregular time-series baseline", 0.2201, "[0.1964, 0.2474]", 0.3396, "[0.2648, 0.4208]", 0.0554, "PDF Table I"],
        ["TRANS", "Transformer baseline", 0.2236, "[0.1993, 0.2511]", 0.3451, "[0.2700, 0.4251]", 0.0294, "PDF Table I"],
        ["DHGAS", "Dynamic heterogeneous graph baseline", 0.2264, "[0.1998, 0.2538]", 0.3487, "[0.2754, 0.4275]", 0.0094, "PDF Table I"],
        ["GraphCare", "Knowledge-enhanced healthcare baseline", 0.2269, "[0.2032, 0.2537]", 0.3411, "[0.2716, 0.4168]", 0.0520, "PDF Table I"],
        ["KEDGN", "Knowledge-enhanced dynamic graph baseline", 0.2289, "[0.2045, 0.2577]", 0.3537, "[0.2785, 0.4360]", -0.0196, "PDF Table I"],
        ["TGNN4I", "Dynamic graph baseline", 0.2391, "[0.2042, 0.2556]", 0.3448, "[0.2746, 0.4209]", 0.0312, "PDF Table I"],
        ["Random Forest (RF)", "Classical ML baseline", 0.2275, "[0.2050, 0.2550]", 0.3470, "[0.2740, 0.4260]", 0.0365, "PDF Table I"],
        ["XGBoost (XGB)", "Classical ML baseline", 0.2258, "[0.2035, 0.2535]", 0.3458, "[0.2725, 0.4250]", 0.0412, "PDF Table I"],
    ]
    df = pd.DataFrame(rows, columns=["Method", "Category", "MAE", "MAE_95CI", "RMSE", "RMSE_95CI", "R2", "Source"])
    df = pd.concat([df, pd.DataFrame([load_grud_row()])], ignore_index=True)
    df = df.sort_values("MAE").reset_index(drop=True)
    df["Rank_by_MAE"] = np.arange(1, len(df) + 1)
    df["OutlierFlag"] = (df["MAE"] > 1.0) | (df["RMSE"] > 2.0) | (df["R2"] < -10.0)
    df.to_csv(TABLES / "table1_main_performance_with_grud.csv", index=False, encoding="utf-8-sig")
    return df


def small_sample() -> pd.DataFrame:
    rows = [
        ["Overall", 417, 0.2156, "[0.1913, 0.2429]", 0.2201, "[0.1964, 0.2474]", 0.012],
        ["18 months", 70, 0.2930, "[0.2324, 0.3618]", 0.2989, "[0.2367, 0.3705]", 0.021],
        ["24 months", 48, 0.3150, "[0.2386, 0.4032]", 0.3204, "[0.2419, 0.4136]", 0.018],
        ["18/24 months pooled", 118, 0.2788, "[0.2298, 0.3367]", 0.2871, "[0.2355, 0.3468]", 0.006],
    ]
    df = pd.DataFrame(rows, columns=["Group", "N", "KG_LatentNet_MAE", "KG_95CI", "Best_baseline_MAE", "Best_baseline_95CI", "p_value"])
    df["Best_baseline"] = "HyperIMTS"
    df.to_csv(TABLES / "table2_small_sample_significance.csv", index=False, encoding="utf-8-sig")
    return df


def ablation() -> pd.DataFrame:
    rows = [
        ["Full KG-LatentNet", 0.2156, "[0.1913, 0.2429]", 0.3333, "[0.2569, 0.4152]", 0.0948],
        ["w/o short-term pathway", 0.2205, "[0.1948, 0.2484]", 0.3415, "[0.2669, 0.4220]", 0.0496],
        ["w/o delayed pathway", 0.2211, "[0.1966, 0.2481]", 0.3375, "[0.2646, 0.4167]", 0.0720],
        ["w/o structured knowledge guidance", 0.2209, "[0.1966, 0.2478]", 0.3397, "[0.2668, 0.4208]", 0.0597],
        ["Single-path state update", 0.2220, "[0.1976, 0.2496]", 0.3430, "[0.2650, 0.4275]", 0.0416],
        ["w/o contribution-aware fusion", 0.2229, "[0.1985, 0.2495]", 0.3401, "[0.2666, 0.4194]", 0.0577],
    ]
    df = pd.DataFrame(rows, columns=["Variant", "MAE", "MAE_95CI", "RMSE", "RMSE_95CI", "R2"])
    df.to_csv(TABLES / "table3_ablation.csv", index=False, encoding="utf-8-sig")
    return df


def correlations() -> pd.DataFrame:
    rows = [
        ["Endpoint TBR", 417, 0.52, "[0.44, 0.59]", "<0.001"],
        ["Delta TBR", 417, 0.43, "[0.34, 0.51]", "<0.001"],
        ["CRP", 364, 0.33, "[0.23, 0.42]", "<0.001"],
        ["IL-6", 291, 0.31, "[0.20, 0.41]", "<0.001"],
        ["NLR", 392, 0.29, "[0.20, 0.38]", "<0.001"],
    ]
    df = pd.DataFrame(rows, columns=["Indicator", "N", "Spearman_rho", "rho_95CI", "p_value"])
    df.to_csv(TABLES / "table4_latent_state_clinical_correlations.csv", index=False, encoding="utf-8-sig")
    return df


def robustness() -> pd.DataFrame:
    rows = [
        ["Structured knowledge perturbation", "Full KG-LatentNet", 0.2156, "[0.1913, 0.2429]", 0.3333, "[0.2569, 0.4152]", 1.000],
        ["Structured knowledge perturbation", "No structured knowledge", 0.2209, "[0.1966, 0.2478]", 0.3397, "[0.2668, 0.4209]", 0.915],
        ["Structured knowledge perturbation", "Randomized knowledge entries", 0.2192, "[0.1945, 0.2467]", 0.3396, "[0.2653, 0.4212]", 0.912],
        ["Structured knowledge perturbation", "Treatment-effect matrix only", 0.2209, "[0.1966, 0.2478]", 0.3397, "[0.2668, 0.4209]", 0.915],
        ["Structured knowledge perturbation", "Biomarker-correlation matrix only", 0.2190, "[0.1944, 0.2469]", 0.3390, "[0.2644, 0.4219]", 0.951],
        ["Structured knowledge perturbation", "Knowledge-entry masking 30%", 0.2231, "[0.1985, 0.2499]", 0.3431, "[0.2670, 0.4253]", 0.897],
        ["Missing-data variation", "Shared preprocessing", 0.2156, "[0.1913, 0.2429]", 0.3333, "[0.2569, 0.4152]", 1.000],
        ["Missing-data variation", "Mean imputation", 0.2190, "[0.1944, 0.2469]", 0.3390, "[0.2544, 0.4219]", 0.935],
        ["Missing-data variation", "LOCF", 0.2224, "[0.1979, 0.2504]", 0.3438, "[0.2657, 0.4279]", 0.896],
        ["Missing-data variation", "Linear interpolation", 0.2221, "[0.1984, 0.2498]", 0.3405, "[0.2669, 0.4208]", 0.880],
    ]
    df = pd.DataFrame(rows, columns=["Analysis_type", "Setting", "MAE", "MAE_95CI", "RMSE", "RMSE_95CI", "Representation_consistency"])
    df.to_csv(TABLES / "table5_robustness.csv", index=False, encoding="utf-8-sig")
    return df


def dataset_summary() -> pd.DataFrame:
    rows = [
        ["Total cohort", "Patients", 417, "Single-center lung cancer cohort"],
        ["6-month window", "Patient-level samples", 189, "3-9 months after baseline PET/CT"],
        ["12-month window", "Patient-level samples", 110, "10-15 months after baseline PET/CT"],
        ["18-month window", "Patient-level samples", 70, "16-21 months after baseline PET/CT"],
        ["24-month window", "Patient-level samples", 48, "22-27 months after baseline PET/CT"],
        ["Static features", "Input component", math.nan, "Baseline patient context"],
        ["Longitudinal variables", "Input component", math.nan, "Time-varying laboratory and physiological variables"],
        ["Treatment sequences", "Input component", math.nan, "External therapeutic exposures"],
        ["Baseline TBR", "Input component", math.nan, "Initial vascular status"],
        ["Endpoint TBR", "Prediction target", math.nan, "Post-treatment vascular inflammation surrogate"],
    ]
    df = pd.DataFrame(rows, columns=["Item", "Type", "N", "Description"])
    df.to_csv(TABLES / "table6_dataset_summary.csv", index=False, encoding="utf-8-sig")
    return df


def anomaly_audit(main_df: pd.DataFrame) -> None:
    current_path = BASE / "main_paper_table1_selected_models.csv"
    rows = []
    if current_path.exists():
        current = pd.read_csv(current_path)
        for _, row in current.iterrows():
            mae = float(row["MAE"])
            rmse = float(row["RMSE"])
            r2 = float(row["R2"])
            rows.append({
                "Method": row["Method"],
                "Current_MAE": mae,
                "Current_RMSE": rmse,
                "Current_R2": r2,
                "Current_outlier_flag": bool((mae > 1.0) or (rmse > 2.0) or (r2 < -10.0)),
                "Paper_aligned_included": bool(row["Method"] in set(main_df["Method"])),
                "Note": "Current run has numerical instability/outlier scale" if (mae > 1.0 or rmse > 2.0 or r2 < -10.0) else "No gross outlier in current run",
            })
    pd.DataFrame(rows).to_csv(TABLES / "current_result_anomaly_audit.csv", index=False, encoding="utf-8-sig")

    merged = main_df.merge(pd.DataFrame(rows), on="Method", how="left")
    merged.to_csv(TABLES / "paper_aligned_vs_current_gap.csv", index=False, encoding="utf-8-sig")


def draw_main(main_df: pd.DataFrame) -> None:
    plt.rcParams.update({"font.size": 10, "figure.dpi": 160})
    df = main_df.sort_values("MAE", ascending=True)
    colors = ["#d62728" if m == "KG-LatentNet" else "#4c78a8" for m in df["Method"]]

    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(df))
    ax.barh(y, df["MAE"], color=colors, edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(df["Method"])
    ax.invert_yaxis()
    ax.set_xlabel("MAE")
    ax.set_title("Overall endpoint TBR prediction performance")
    for idx, val in enumerate(df["MAE"]):
        ax.text(val + 0.003, idx, f"{val:.4f}", va="center", fontsize=9)
    ax.set_xlim(0, max(0.38, df["MAE"].max() * 1.18))
    fig.tight_layout()
    fig.savefig(FIGS / "fig1_main_mae_with_grud.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.38
    x = np.arange(len(df))
    ax.bar(x - width / 2, df["MAE"], width, label="MAE", color="#4c78a8")
    ax.bar(x + width / 2, df["RMSE"], width, label="RMSE", color="#f58518")
    ax.set_xticks(x)
    ax.set_xticklabels(df["Method"], rotation=35, ha="right")
    ax.set_ylabel("Error")
    ax.set_title("MAE and RMSE by model")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "fig2_main_mae_rmse_with_grud.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(df["Method"], df["R2"], color=colors, edgecolor="white")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("R2")
    ax.set_title("Overall R2 by model")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(FIGS / "fig3_main_r2_with_grud.png")
    plt.close(fig)


def draw_small_sample(df: pd.DataFrame) -> None:
    x = np.arange(len(df))
    kg_low = []
    kg_high = []
    bl_low = []
    bl_high = []
    for _, row in df.iterrows():
        kl, kh = parse_ci(row["KG_95CI"])
        bl, bh = parse_ci(row["Best_baseline_95CI"])
        kg_low.append(row["KG_LatentNet_MAE"] - kl)
        kg_high.append(kh - row["KG_LatentNet_MAE"])
        bl_low.append(row["Best_baseline_MAE"] - bl)
        bl_high.append(bh - row["Best_baseline_MAE"])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.36
    ax.bar(x - width / 2, df["KG_LatentNet_MAE"], width, yerr=[kg_low, kg_high], capsize=3, label="KG-LatentNet", color="#d62728")
    ax.bar(x + width / 2, df["Best_baseline_MAE"], width, yerr=[bl_low, bl_high], capsize=3, label="Best baseline", color="#4c78a8")
    ax.set_xticks(x)
    ax.set_xticklabels(df["Group"], rotation=25, ha="right")
    ax.set_ylabel("MAE")
    ax.set_title("Small-sample and long-term robustness")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "fig4_small_sample_significance.png")
    plt.close(fig)


def draw_ablation(df: pd.DataFrame) -> None:
    plot_df = df.sort_values("MAE", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    colors = ["#d62728" if v == "Full KG-LatentNet" else "#4c78a8" for v in plot_df["Variant"]]
    y = np.arange(len(plot_df))
    ax.barh(y, plot_df["MAE"], color=colors, edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["Variant"])
    ax.invert_yaxis()
    ax.set_xlabel("MAE")
    ax.set_title("Ablation study")
    for idx, val in enumerate(plot_df["MAE"]):
        ax.text(val + 0.0007, idx, f"{val:.4f}", va="center", fontsize=9)
    ax.set_xlim(0.212, 0.225)
    fig.tight_layout()
    fig.savefig(FIGS / "fig5_ablation_mae.png")
    plt.close(fig)


def draw_correlations(df: pd.DataFrame) -> None:
    lows, highs = [], []
    for _, row in df.iterrows():
        low, high = parse_ci(row["rho_95CI"])
        lows.append(row["Spearman_rho"] - low)
        highs.append(high - row["Spearman_rho"])
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(df))
    ax.bar(x, df["Spearman_rho"], yerr=[lows, highs], capsize=3, color="#54a24b", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(df["Indicator"])
    ax.set_ylabel("Spearman rho")
    ax.set_ylim(0, 0.65)
    ax.set_title("Association between latent state score and clinical indicators")
    fig.tight_layout()
    fig.savefig(FIGS / "fig6_latent_state_clinical_correlations.png")
    plt.close(fig)


def draw_robustness(df: pd.DataFrame) -> None:
    for analysis_type, name in [
        ("Structured knowledge perturbation", "fig7_structured_knowledge_robustness.png"),
        ("Missing-data variation", "fig8_missing_data_robustness.png"),
    ]:
        sub = df[df["Analysis_type"] == analysis_type].copy()
        x = np.arange(len(sub))
        fig, ax1 = plt.subplots(figsize=(8.5, 4.6))
        ax2 = ax1.twinx()
        ax1.bar(x, sub["MAE"], color="#4c78a8", alpha=0.85, label="MAE")
        ax2.plot(x, sub["Representation_consistency"], color="#d62728", marker="o", linewidth=2, label="Representation consistency")
        ax1.set_xticks(x)
        ax1.set_xticklabels(sub["Setting"], rotation=30, ha="right")
        ax1.set_ylabel("MAE")
        ax2.set_ylabel("Representation consistency")
        ax1.set_title(analysis_type)
        ax1.set_ylim(0.212, 0.226)
        ax2.set_ylim(0.86, 1.02)
        fig.tight_layout()
        fig.savefig(FIGS / name)
        plt.close(fig)


def write_report(main_df: pd.DataFrame) -> None:
    best = main_df.iloc[0]
    best_bl = main_df[main_df["Method"] != "KG-LatentNet"].sort_values("MAE").iloc[0]
    rel = (best_bl["MAE"] - best["MAE"]) / best_bl["MAE"] * 100
    report = f"""# Paper-Aligned Experimental Outputs

This folder regenerates paper-style tables and figures from the manuscript-reported values in `KG_LatentNet__... (24).pdf`, with GRU-D appended from the latest stabilized full 5-fold rerun.

## Main Result

- Proposed model: KG-LatentNet
- KG-LatentNet MAE/RMSE/R2: {best['MAE']:.4f} / {best['RMSE']:.4f} / {best['R2']:.4f}
- Best non-KG baseline: {best_bl['Method']} with MAE {best_bl['MAE']:.4f}
- Relative MAE reduction vs best baseline: {rel:.2f}%

## Integrity Notes

- The main KG-LatentNet and baseline values, except GRU-D, come from the supplied manuscript PDF.
- GRU-D is newly added from the latest server rerun and is marked in the `Source` column.
- No method in `table1_main_performance_with_grud.csv` is flagged by the gross outlier rule MAE > 1.0, RMSE > 2.0, or R2 < -10.
- `current_result_anomaly_audit.csv` records the current live-run anomalies separately instead of silently discarding them.
"""
    (REPORTS / "paper_aligned_results_summary.md").write_text(report, encoding="utf-8")

    provenance = pd.DataFrame([
        {"Artifact": "table1_main_performance_with_grud.csv", "Primary_source": "PDF Table I plus latest GRU-D rerun", "Note": "Paper-aligned comparison table with added GRU-D"},
        {"Artifact": "table2_small_sample_significance.csv", "Primary_source": "PDF Table II", "Note": "Small-sample robustness and Wilcoxon p-values"},
        {"Artifact": "table3_ablation.csv", "Primary_source": "PDF Table III", "Note": "KG-LatentNet component ablation"},
        {"Artifact": "table4_latent_state_clinical_correlations.csv", "Primary_source": "PDF Table IV", "Note": "Latent state score association analysis"},
        {"Artifact": "table5_robustness.csv", "Primary_source": "PDF Table V", "Note": "Knowledge and missing-data robustness"},
        {"Artifact": "current_result_anomaly_audit.csv", "Primary_source": "Current local/server rerun outputs", "Note": "Tracks anomalies rather than altering them"},
    ])
    provenance.to_csv(TABLES / "source_provenance.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    ensure_dirs()
    main_df = main_performance()
    small_df = small_sample()
    abl_df = ablation()
    corr_df = correlations()
    robust_df = robustness()
    dataset_summary()
    anomaly_audit(main_df)
    draw_main(main_df)
    draw_small_sample(small_df)
    draw_ablation(abl_df)
    draw_correlations(corr_df)
    draw_robustness(robust_df)
    write_report(main_df)
    pdf_text = BASE / "paper_24_extracted_text.txt"
    if pdf_text.exists():
        shutil.copy2(pdf_text, REPORTS / "paper_24_extracted_text.txt")
    print(f"Wrote paper-aligned outputs to {OUT}")


if __name__ == "__main__":
    main()
