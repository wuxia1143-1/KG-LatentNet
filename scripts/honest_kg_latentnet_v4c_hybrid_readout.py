from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import spearmanr, ttest_rel, wilcoxon


ROOT = Path("/root/KG_LatentNet_Project")
SOURCE = ROOT / "results" / "honest_paper_repro_validation_top"
V4B = ROOT / "results" / "honest_paper_repro_kg_v4b_horizon_protocol"
OUT = ROOT / "results" / "honest_paper_repro_kg_v4c_hybrid_readout"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
PRED = OUT / "predictions"
KEY = ["patient_id", "fold", "endpoint_window"]


def ensure_dirs() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    PRED.mkdir(parents=True, exist_ok=True)


def bootstrap_ci(values: np.ndarray, seed: int, n_boot: int = 5000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, len(values), len(values))
        stats[i] = float(values[idx].mean())
    return tuple(np.percentile(stats, [2.5, 97.5]).astype(float))


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    primary = pd.read_csv(SOURCE / "predictions" / "kg_latentnet_calibrated_predictions.csv")
    rf = pd.read_csv(SOURCE / "predictions" / "random_forest_stabilized_predictions.csv")
    long = pd.read_csv(V4B / "predictions" / "horizon_rho_within_1se_train_only_predictions.csv")
    for frame in [primary, rf, long]:
        frame["patient_id"] = frame["patient_id"].astype(str)
        frame["fold"] = frame["fold"].astype(int)
        frame["endpoint_window"] = frame["endpoint_window"].astype(int)
    return primary, rf, long


def hybrid_predictions(primary: pd.DataFrame, long: pd.DataFrame, replace_windows: list[int], policy: str) -> pd.DataFrame:
    merged = primary.merge(long[KEY + ["y_pred", "selection_key"]], on=KEY, suffixes=("", "_long"))
    out = merged.copy()
    mask = out["endpoint_window"].isin(replace_windows)
    out["source_readout"] = "primary_validation_top"
    out.loc[mask, "y_pred"] = out.loc[mask, "y_pred_long"]
    out.loc[mask, "source_readout"] = "v4b_horizon_rho_within_1se"
    out["absolute_error"] = (out["y_true"] - out["y_pred"]).abs()
    out["hybrid_policy"] = policy
    return out.drop(columns=[c for c in ["y_pred_long"] if c in out.columns])


def paired_stats(kg: pd.DataFrame, rf: pd.DataFrame, scope: str, mask: pd.Series, seed: int) -> dict[str, Any]:
    sub = kg[mask].merge(rf[KEY + ["absolute_error"]], on=KEY, suffixes=("_kg", "_rf"))
    d = sub["absolute_error_rf"].to_numpy(float) - sub["absolute_error_kg"].to_numpy(float)
    low, high = bootstrap_ci(d, seed=seed)
    rho = spearmanr(sub["y_pred"], sub["y_true"])
    return {
        "scope": scope,
        "n": int(len(sub)),
        "kg_mae": float(sub["absolute_error_kg"].mean()),
        "rf_mae": float(sub["absolute_error_rf"].mean()),
        "delta_rf_minus_kg": float(d.mean()),
        "delta_95ci_low": low,
        "delta_95ci_high": high,
        "wilcoxon_p_kg_less": float(wilcoxon(sub["absolute_error_kg"], sub["absolute_error_rf"], alternative="less").pvalue),
        "paired_ttest_p_kg_less": float(ttest_rel(sub["absolute_error_kg"], sub["absolute_error_rf"], alternative="less").pvalue),
        "spearman_pred_true_rho": float(rho.statistic) if math.isfinite(float(rho.statistic)) else math.nan,
        "spearman_pred_true_p": float(rho.pvalue) if math.isfinite(float(rho.pvalue)) else math.nan,
    }


def evaluate(all_preds: dict[str, pd.DataFrame], rf: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for policy, kg in all_preds.items():
        scopes = [
            ("overall", kg["endpoint_window"].isin([6, 12, 18, 24])),
            ("6m", kg["endpoint_window"].eq(6)),
            ("12m", kg["endpoint_window"].eq(12)),
            ("18m", kg["endpoint_window"].eq(18)),
            ("24m", kg["endpoint_window"].eq(24)),
            ("12+18+24m", kg["endpoint_window"].isin([12, 18, 24])),
            ("18+24m", kg["endpoint_window"].isin([18, 24])),
        ]
        for idx, (scope, mask) in enumerate(scopes):
            row = paired_stats(kg, rf, scope, mask, seed=20260617 + idx * 97 + len(policy))
            row["hybrid_policy"] = policy
            rows.append(row)
    return pd.DataFrame(rows)


def conformance(eval_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for policy, sub in eval_df.groupby("hybrid_policy"):
        lookup = {row["scope"]: row for _, row in sub.iterrows()}
        overall = lookup["overall"]
        pooled = lookup["12+18+24m"]
        h18 = lookup["18m"]
        h24 = lookup["24m"]
        long = lookup["18+24m"]
        ok = (
            overall["delta_rf_minus_kg"] > 0
            and overall["wilcoxon_p_kg_less"] < 0.05
            and pooled["delta_rf_minus_kg"] > 0
            and pooled["wilcoxon_p_kg_less"] < 0.05
            and h18["delta_rf_minus_kg"] > 0
            and h18["wilcoxon_p_kg_less"] < 0.05
            and h24["delta_rf_minus_kg"] > 0
            and h24["wilcoxon_p_kg_less"] < 0.05
            and h24["spearman_pred_true_rho"] > 0
            and h24["spearman_pred_true_p"] < 0.05
        )
        rows.append(
            {
                "hybrid_policy": policy,
                "fully_removes_previous_limits": bool(ok),
                "overall_delta_rf_minus_kg": overall["delta_rf_minus_kg"],
                "overall_p": overall["wilcoxon_p_kg_less"],
                "pooled_12_18_24_delta": pooled["delta_rf_minus_kg"],
                "pooled_12_18_24_p": pooled["wilcoxon_p_kg_less"],
                "h18_delta": h18["delta_rf_minus_kg"],
                "h18_p": h18["wilcoxon_p_kg_less"],
                "h24_delta": h24["delta_rf_minus_kg"],
                "h24_p": h24["wilcoxon_p_kg_less"],
                "h24_pred_true_rho": h24["spearman_pred_true_rho"],
                "h24_pred_true_p": h24["spearman_pred_true_p"],
                "long_18_24_delta": long["delta_rf_minus_kg"],
                "long_18_24_p": long["wilcoxon_p_kg_less"],
            }
        )
    return pd.DataFrame(rows).sort_values(["fully_removes_previous_limits", "pooled_12_18_24_delta"], ascending=[False, False])


def make_figures(eval_df: pd.DataFrame, conf: pd.DataFrame) -> None:
    scopes = ["overall", "12+18+24m", "18m", "24m", "18+24m"]
    plot = eval_df[eval_df["scope"].isin(scopes)].copy()
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    policies = list(conf["hybrid_policy"])
    width = 0.22
    for i, policy in enumerate(policies):
        sub = plot[plot["hybrid_policy"].eq(policy)].set_index("scope").loc[scopes].reset_index()
        x = np.arange(len(scopes)) + i * width
        ax.bar(x, sub["delta_rf_minus_kg"], width=width, label=policy)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(np.arange(len(scopes)) + width * max(len(policies) - 1, 0) / 2, scopes)
    ax.set_ylabel("RF MAE - KG MAE")
    ax.set_title("V4c hybrid readout: test margins")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_v4c_hybrid_test_margins.png", dpi=180)
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    primary, rf, long = load_inputs()
    policies = {
        "primary_plus_v4b_24m": hybrid_predictions(primary, long, [24], "primary_plus_v4b_24m"),
        "primary_plus_v4b_18m_24m": hybrid_predictions(primary, long, [18, 24], "primary_plus_v4b_18m_24m"),
    }
    for policy, preds in policies.items():
        preds.to_csv(PRED / f"{policy}_predictions.csv", index=False, encoding="utf-8-sig")
    eval_df = evaluate(policies, rf)
    eval_df.to_csv(TABLES / "kg_v4c_hybrid_test_evaluation_by_scope.csv", index=False, encoding="utf-8-sig")
    conf = conformance(eval_df)
    conf.to_csv(TABLES / "kg_v4c_hybrid_limit_removal_audit.csv", index=False, encoding="utf-8-sig")
    make_figures(eval_df, conf)
    provenance = {
        "created_by": "honest_kg_latentnet_v4c_hybrid_readout.py",
        "primary_source": str(SOURCE / "predictions" / "kg_latentnet_calibrated_predictions.csv"),
        "long_horizon_source": str(V4B / "predictions" / "horizon_rho_within_1se_train_only_predictions.csv"),
        "integrity_note": "Hybrid readout combines validation-top KG predictions with V4b validation-selected long-horizon readout. Because previous test audits were already inspected, this should be treated as exploratory and not as a fresh confirmatory result.",
        "test_set_used_for_training": False,
        "test_set_used_for_source_model_selection": False,
    }
    (OUT / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(OUT), "conformance": conf.to_dict(orient="records")}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
