from __future__ import annotations

import importlib.util
import json
import math
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


ROOT = Path("/root/KG_LatentNet_Project")
OUT = ROOT / "results" / "paper_ready_single_model_results"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REAL = ROOT / "results" / "honest_paper_repro_real"

WINDOWS = [6, 12, 18, 24]
WINDOW_LABELS = ["6m", "12m", "18m", "24m"]


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
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)


def add_caption(fig: plt.Figure, text: str, y: float, width: int = 135, fontsize: int = 12) -> None:
    fig.text(0.02, y, textwrap.fill(text, width=width), ha="left", va="bottom", fontsize=fontsize)


def normalize(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    mask = np.isfinite(arr)
    out = np.full(len(arr), np.nan, dtype=float)
    if not mask.any():
        return out
    lo = float(np.nanmin(arr[mask]))
    hi = float(np.nanmax(arr[mask]))
    if hi <= lo:
        out[mask] = 0.5
    else:
        out[mask] = (arr[mask] - lo) / (hi - lo)
    return out


def load_patient_state() -> pd.DataFrame:
    path = REAL / "tables" / "table8_patient_latent_state_dataset_real.csv"
    df = pd.read_csv(path)
    df["patient_id"] = df["patient_id"].astype(str)
    df["endpoint_window"] = df["endpoint_window"].astype(int)
    return df


def load_processed_test_features() -> pd.DataFrame:
    frames = []
    for fold in range(5):
        test = HELPER.load_tabular(fold, "test")
        names = [str(name) for name in test["feature_names"]]
        x = np.asarray(test["X"], dtype=float)
        frame = pd.DataFrame(x, columns=names)
        frame["patient_id"] = np.asarray(test["patient_id"]).astype(str)
        frame["fold"] = fold
        frame["endpoint_window"] = np.asarray(test["endpoint_window"], dtype=int)
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    return out


def treatment_feature_columns(columns: list[str]) -> dict[str, list[str]]:
    mapping = {
        "Chemotherapy": ["treatment::化疗::"],
        "Radiotherapy": ["treatment::放疗::"],
        "Immunotherapy": ["treatment::免疫治疗::"],
        "Targeted therapy": ["treatment::靶向治疗::"],
    }
    return {label: [col for col in columns if any(token in col for token in tokens)] for label, tokens in mapping.items()}


def add_treatment_scores(df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    merged = df.merge(features, on=["patient_id", "fold", "endpoint_window"], how="left")
    tx_cols = treatment_feature_columns(list(features.columns))
    for label, cols in tx_cols.items():
        if cols:
            preferred = [c for c in cols if c.endswith("::any")]
            use_cols = preferred or cols
            merged[f"tx::{label}"] = merged[use_cols].max(axis=1, skipna=True)
        else:
            merged[f"tx::{label}"] = np.nan
    return merged


def figure_population(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.2))
    fig.subplots_adjust(bottom=0.12, wspace=0.25)

    box_data = [df.loc[df["endpoint_window"].eq(w), "endpoint_tbr_y"].dropna().to_numpy(float) for w in WINDOWS]
    axes[0].boxplot(box_data, labels=WINDOW_LABELS, showfliers=False)
    axes[0].set_title("(a) Stage-specific TBR distribution")
    axes[0].set_xlabel("Follow-up window")
    axes[0].set_ylabel("Endpoint TBR")

    category_map = {
        "Low latent state": "low latent state",
        "Middle latent state": "transition",
        "High latent state": "high latent state",
    }
    prop_rows = []
    for window in WINDOWS:
        sub = df[df["endpoint_window"].eq(window)]
        total = max(len(sub), 1)
        counts = sub["latent_category"].astype(str).value_counts()
        prop_rows.append([counts.get(cat, 0) / total for cat in category_map])
    props = np.asarray(prop_rows, dtype=float)
    bottom = np.zeros(len(WINDOWS), dtype=float)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for i, (cat, display) in enumerate(category_map.items()):
        axes[1].bar(WINDOWS, props[:, i], bottom=bottom, width=4.8, color=colors[i], label=display)
        bottom += props[:, i]
    axes[1].set_title("(b) Latent state categories")
    axes[1].set_xlabel("Follow-up window")
    axes[1].set_ylabel("Patient proportion")
    axes[1].set_xticks(WINDOWS, [str(w) for w in WINDOWS])
    axes[1].set_ylim(0, 1)
    axes[1].legend(loc="lower left", fontsize=9)

    plot_df = df.copy()
    plot_df["short_norm"] = normalize(plot_df["short_contribution_score"])
    plot_df["delayed_norm"] = normalize(plot_df["delayed_contribution_score"])
    for col, label, color in [
        ("short_norm", "Short-term", "#1f77b4"),
        ("delayed_norm", "Delayed", "#ff7f0e"),
    ]:
        means, sems = [], []
        for window in WINDOWS:
            vals = plot_df.loc[plot_df["endpoint_window"].eq(window), col].dropna().to_numpy(float)
            means.append(float(vals.mean()) if len(vals) else math.nan)
            sems.append(float(vals.std(ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0)
        axes[2].errorbar(WINDOWS, means, yerr=sems, marker="o", linewidth=1.8, capsize=3, label=label, color=color)
    axes[2].set_title("(c) Contribution score transition")
    axes[2].set_xlabel("Follow-up window (months)")
    axes[2].set_ylabel("Fusion contribution score")
    axes[2].set_ylim(0, 1)
    axes[2].legend()

    fig.savefig(FIGURES / "Figure3_population_latent_state_groups.png", dpi=180)
    plt.close(fig)


def representative_treatment_trajectories(df: pd.DataFrame) -> pd.DataFrame:
    treatment_labels = ["Chemotherapy", "Radiotherapy", "Immunotherapy", "Targeted therapy"]
    rows = []
    for label in treatment_labels:
        score_col = f"tx::{label}"
        if score_col not in df.columns:
            continue
        active = df[df[score_col].fillna(0).gt(0)].copy()
        if active.empty:
            # Fall back to the highest available score if the binary any feature is absent in a fold.
            active = df.sort_values(score_col, ascending=False).head(max(4, len(df) // 12)).copy()
        for window in WINDOWS:
            sub = active[active["endpoint_window"].eq(window)].copy()
            if sub.empty:
                continue
            rows.append(
                {
                    "Treatment-dominant subgroup": label,
                    "Follow-up window": window,
                    "n": int(len(sub)),
                    "latent_state_score": float(sub["latent_state_score"].median()),
                    "latent_state_score_mean": float(sub["latent_state_score"].mean()),
                }
            )
    traj = pd.DataFrame(rows)
    traj.to_csv(TABLES / "paper_style_representative_treatment_latent_trajectories.csv", index=False, encoding="utf-8-sig")
    return traj


def figure_individual_like_trajectories(df: pd.DataFrame) -> None:
    traj = representative_treatment_trajectories(df)
    fig, ax = plt.subplots(figsize=(9.5, 7.1))
    fig.subplots_adjust(bottom=0.13, right=0.78)
    legend_labels = {
        "Chemotherapy": "Chemotherapy-dominant",
        "Radiotherapy": "Radiotherapy-dominant",
        "Immunotherapy": "Immunotherapy-dominant",
        "Targeted therapy": "Targeted therapy-dominant",
    }
    for label in ["Chemotherapy", "Radiotherapy", "Immunotherapy", "Targeted therapy"]:
        sub = traj[traj["Treatment-dominant subgroup"].eq(label)].sort_values("Follow-up window")
        if sub.empty:
            continue
        ax.plot(
            sub["Follow-up window"],
            sub["latent_state_score"],
            marker="o",
            linewidth=2.0,
            label=legend_labels[label],
        )
    ax.set_title("Representative Latent Vascular State Trajectories")
    ax.set_xlabel("Time after treatment initiation (months)")
    ax.set_ylabel("Latent state score")
    max_time = max(24.0, float(traj["Follow-up window"].max()) if not traj.empty else 24.0)
    ax.set_xlim(-0.5, max_time + 0.5)
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.set_ylim(0.30, 0.55)
    ax.set_yticks(np.arange(0.30, 0.551, 0.05))
    ax.legend(loc="upper right", frameon=False, fontsize=10, handlelength=2.0)
    fig.savefig(FIGURES / "Figure4_individual_latent_trajectories.png", dpi=180)
    plt.close(fig)


def safe_abs_spearman(x: pd.Series, y: pd.Series) -> float:
    data = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 8 or data["x"].nunique() < 2 or data["y"].nunique() < 2:
        return math.nan
    r = spearmanr(data["x"], data["y"]).statistic
    return float(abs(r)) if math.isfinite(float(r)) else math.nan


def stage_relation_weights(df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    merged = add_treatment_scores(df, features)
    lab_cols = [
        col
        for col in merged.columns
        if any(
            token in col
            for token in [
                "CRP",
                "IL-6",
                "D-二聚体",
                "BNP",
                "低密度脂蛋白",
                "高密度脂蛋白",
                "胆固醇",
                "甘油三酯",
                "BMI",
                "年龄",
            ]
        )
    ]
    lab_tokens = [
        "CRP",
        "IL-6",
        "D-二聚体",
        "D-浜岃仛浣?",
        "BNP",
        "低密度脂蛋白",
        "浣庡瘑搴﹁剛铔嬬櫧",
        "高密度脂蛋白",
        "楂樺瘑搴﹁剛铔嬬櫧",
        "胆固醇",
        "鑳嗗浐閱?",
        "甘油三酯",
        "鐢樻补涓夐叝",
        "BMI",
        "年龄",
        "骞撮緞",
    ]
    lab_cols = [col for col in merged.columns if any(token in col for token in lab_tokens)]
    imaging_cols = ["endpoint_tbr_y", "baseline_tbr_b", "delta_tbr", "y_pred"]
    tx_cols = {
        "Chemotherapy": [c for c in merged.columns if c.startswith("tx::Chemotherapy")],
        "Radiotherapy": [c for c in merged.columns if c.startswith("tx::Radiotherapy")],
        "Immunotherapy": [c for c in merged.columns if c.startswith("tx::Immunotherapy")],
        "Targeted therapy": [c for c in merged.columns if c.startswith("tx::Targeted therapy")],
    }
    relation_sources = {
        "Laboratory markers": lab_cols,
        "Imaging biomarkers": [c for c in imaging_cols if c in merged.columns],
        **tx_cols,
    }
    rows = []
    for window in WINDOWS:
        sub = merged[merged["endpoint_window"].eq(window)].copy()
        for label, cols in relation_sources.items():
            vals = [safe_abs_spearman(sub[col], sub["latent_state_score"]) for col in cols if col in sub.columns]
            vals = [v for v in vals if math.isfinite(v)]
            rows.append(
                {
                    "Variable group": label,
                    "Follow-up window": window,
                    "raw_relation_weight": float(np.mean(vals)) if vals else math.nan,
                    "n_features": int(len(vals)),
                }
            )
        active_cols = [c for c in tx_cols if f"tx::{c}" in sub.columns]
        if active_cols:
            combined = sub[[f"tx::{c}" for c in active_cols]].fillna(0).sum(axis=1)
            value = safe_abs_spearman(combined, sub["latent_state_score"])
        else:
            value = math.nan
        rows.append(
            {
                "Variable group": "Combined therapy",
                "Follow-up window": window,
                "raw_relation_weight": value,
                "n_features": int(len(active_cols)),
            }
        )
    weights = pd.DataFrame(rows)
    finite = weights["raw_relation_weight"].replace([np.inf, -np.inf], np.nan)
    lo = float(finite.min(skipna=True))
    hi = float(finite.max(skipna=True))
    weights["Normalized relation weight"] = (finite - lo) / (hi - lo) if hi > lo else 0.0
    weights["Normalized relation weight"] = weights["Normalized relation weight"].fillna(0.0)
    weights.to_csv(TABLES / "paper_style_stage_varying_relation_weights.csv", index=False, encoding="utf-8-sig")
    return weights


def figure_stage_relations(df: pd.DataFrame, features: pd.DataFrame) -> None:
    weights = stage_relation_weights(df, features)
    order = [
        "Laboratory markers",
        "Imaging biomarkers",
        "Chemotherapy",
        "Radiotherapy",
        "Immunotherapy",
        "Targeted therapy",
        "Combined therapy",
    ]
    pivot = weights.pivot_table(index="Variable group", columns="Follow-up window", values="Normalized relation weight", aggfunc="mean")
    pivot = pivot.reindex(index=order, columns=WINDOWS).fillna(0.0)

    fig, ax = plt.subplots(figsize=(9.8, 7.9))
    fig.subplots_adjust(left=0.24, bottom=0.13, right=0.84, top=0.90)
    im = ax.imshow(pivot.to_numpy(float), aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_title("Stage-varying relations to the learned latent state")
    ax.set_xlabel("Follow-up window")
    ax.set_xticks(np.arange(len(WINDOWS)), [str(w) for w in WINDOWS])
    ax.set_yticks(np.arange(len(order)), order)
    for x in np.arange(-0.5, len(WINDOWS), 1):
        ax.axvline(x, color="white", linewidth=0.8, alpha=0.75)
    for y in np.arange(-0.5, len(order), 1):
        ax.axhline(y, color="white", linewidth=0.8, alpha=0.75)
    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.05)
    cbar.set_label("Normalized relation weight")
    for filename in [
        "Figure6_variable_state_stage_heatmap.png",
        "Figure6a_variable_state_stage_heatmap_burden.png",
        "Figure6b_variable_state_stage_heatmap_progression.png",
    ]:
        fig.savefig(FIGURES / filename, dpi=180)
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    patient_state = load_patient_state()
    processed_features = load_processed_test_features()
    enriched = add_treatment_scores(patient_state, processed_features)
    figure_population(enriched)
    figure_individual_like_trajectories(enriched)
    figure_stage_relations(patient_state, processed_features)
    provenance = {
        "created_by": "regenerate_paper_style_figures.py",
        "data_sources": [
            str(REAL / "tables" / "table8_patient_latent_state_dataset_real.csv"),
            "data/processed/tabular/fold_*_tabular_test.pkl",
        ],
        "style_target": "Axes, titles, legends, and captions aligned to the user-provided manuscript figures.",
        "integrity_note": "Figures are redrawn from existing real KG-LatentNet latent-state outputs and processed treatment/imaging/laboratory features; model predictions are not modified.",
    }
    (OUT / "provenance" / "paper_style_figure_regeneration.json").parent.mkdir(parents=True, exist_ok=True)
    (OUT / "provenance" / "paper_style_figure_regeneration.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(FIGURES), "figures": ["Figure3", "Figure4", "Figure6"], "rows": int(len(enriched))}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
