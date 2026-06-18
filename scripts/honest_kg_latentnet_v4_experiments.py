from __future__ import annotations

import importlib.util
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from scipy.stats import spearmanr, ttest_rel, wilcoxon
from torch import nn


ROOT = Path("/root/KG_LatentNet_Project")
OUT = ROOT / "results" / "honest_paper_repro_kg_v4"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
PRED = OUT / "predictions"
SOURCE = ROOT / "results" / "honest_paper_repro_validation_top"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


HELPER = load_module("kg_validation_top", ROOT / "scripts" / "honest_real_final_outputs_validation_top.py")


WINDOWS = [6, 12, 18, 24]
WINDOW_TO_INDEX = {w: i for i, w in enumerate(WINDOWS)}


def ensure_dirs() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    PRED.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_tabular(fold: int, split: str) -> dict[str, Any]:
    return HELPER.load_tabular(fold, split)


def combine_tabular(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in left:
        if key == "feature_names":
            out[key] = left[key]
        elif key in {"X", "X_raw", "y", "patient_id", "endpoint_window"}:
            out[key] = np.concatenate([np.asarray(left[key]), np.asarray(right[key])], axis=0)
        else:
            try:
                out[key] = np.concatenate([np.asarray(left[key]), np.asarray(right[key])], axis=0)
            except Exception:
                out[key] = left[key]
    return out


def fit_anchor_train_other(train: dict[str, Any], other: dict[str, Any], feature_names: list[str], mode: str) -> tuple[np.ndarray, np.ndarray]:
    idx = HELPER.clinical_feature_indices(feature_names, mode)
    x_train = train["X"][:, idx]
    x_other = other["X"][:, idx]
    if mode in {"clinical_horizon_aware", "baseline_tbr_horizon"}:
        x_train, x_other = HELPER.horizon_onehot(train, other, x_train, x_other)
    from sklearn.linear_model import LinearRegression

    model = LinearRegression()
    model.fit(x_train, train["y"])
    return np.asarray(model.predict(x_train), dtype=float), np.asarray(model.predict(x_other), dtype=float)


def feature_indices(names: list[str], mode: str) -> list[int]:
    if mode in {"kg_dynamic", "treatment_history", "kg_dynamic_static", "all"}:
        return HELPER.kg_feature_indices(names, mode)
    raise KeyError(mode)


def sample_weights(windows: np.ndarray, scheme: str) -> np.ndarray:
    w = np.ones(len(windows), dtype=np.float32)
    if scheme == "balanced":
        return w
    if scheme == "long":
        w[np.asarray(windows) == 18] = 1.8
        w[np.asarray(windows) == 24] = 2.8
        return w
    if scheme == "very_long":
        w[np.asarray(windows) == 6] = 0.8
        w[np.asarray(windows) == 12] = 1.0
        w[np.asarray(windows) == 18] = 2.5
        w[np.asarray(windows) == 24] = 4.0
        return w
    raise KeyError(scheme)


def scale_train_other(x_train: np.ndarray, x_other: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(x_train, axis=0)
    std = np.nanstd(x_train, axis=0)
    std[std < 1e-6] = 1.0
    x_train_s = np.nan_to_num((x_train - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)
    x_other_s = np.nan_to_num((x_other - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)
    return x_train_s.astype(np.float32), x_other_s.astype(np.float32), mean, std


class V4Net(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float, residual_scale: float) -> None:
        super().__init__()
        self.residual_scale = residual_scale
        self.trunk = nn.Sequential(
            nn.Linear(input_dim + 4, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self.shared_head = nn.Linear(hidden_dim, 1)
        self.horizon_heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in WINDOWS])
        self.mix_gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2))
        self.state_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, window_idx: torch.Tensor, anchor: torch.Tensor) -> dict[str, torch.Tensor]:
        one_hot = torch.nn.functional.one_hot(window_idx, num_classes=4).float()
        z = self.trunk(torch.cat([x, one_hot], dim=-1))
        shared = self.shared_head(z).squeeze(-1)
        horizon_values = torch.stack([head(z).squeeze(-1) for head in self.horizon_heads], dim=1)
        horizon = horizon_values.gather(1, window_idx.view(-1, 1)).squeeze(1)
        gate = torch.softmax(self.mix_gate(z), dim=-1)
        residual = self.residual_scale * (gate[:, 0] * shared + gate[:, 1] * horizon)
        y_pred = anchor + residual
        state = torch.sigmoid(self.state_head(z)).squeeze(-1)
        return {
            "y_pred": y_pred,
            "delta_pred": residual,
            "state_score": state,
            "gate_shared": gate[:, 0],
            "gate_horizon": gate[:, 1],
        }


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    anchor_mode: str
    feature_mode: str
    hidden_dim: int
    dropout: float
    lr: float
    weight_decay: float
    long_weight: str
    lambda_delta: float
    lambda_state: float
    residual_scale: float
    max_epochs: int = 220
    patience: int = 30


CANDIDATES = [
    Candidate("v4_a_baseline_all_long", "baseline_tbr_only", "all", 32, 0.10, 1e-3, 1e-4, "long", 0.5, 0.15, 0.35),
    Candidate("v4_b_baseline_all_verylong", "baseline_tbr_only", "all", 32, 0.10, 8e-4, 1e-4, "very_long", 0.7, 0.20, 0.45),
    Candidate("v4_c_baseline_kg_long", "baseline_tbr_only", "kg_dynamic_static", 32, 0.10, 1e-3, 1e-4, "long", 0.5, 0.15, 0.35),
    Candidate("v4_d_core_all_long", "clinical_core", "all", 32, 0.10, 1e-3, 1e-4, "long", 0.5, 0.15, 0.35),
    Candidate("v4_e_core_all_verylong", "clinical_core", "all", 32, 0.10, 8e-4, 1e-4, "very_long", 0.7, 0.20, 0.45),
    Candidate("v4_f_core_kg_long", "clinical_core", "kg_dynamic_static", 32, 0.10, 1e-3, 1e-4, "long", 0.5, 0.15, 0.35),
    Candidate("v4_g_horizon_all_long", "clinical_horizon_aware", "all", 32, 0.10, 8e-4, 1e-4, "long", 0.5, 0.15, 0.35),
    Candidate("v4_h_small_regularized", "baseline_tbr_only", "kg_dynamic_static", 16, 0.20, 1e-3, 5e-4, "balanced", 0.3, 0.10, 0.25),
    Candidate("v4_i_core_small_regularized", "clinical_core", "kg_dynamic_static", 16, 0.20, 1e-3, 5e-4, "balanced", 0.3, 0.10, 0.25),
    Candidate("v4_j_baseline_all_lowdrop", "baseline_tbr_only", "all", 48, 0.05, 6e-4, 1e-4, "long", 0.6, 0.15, 0.30),
]


def to_tensor(values: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32, device=device)


def window_indices(windows: np.ndarray) -> np.ndarray:
    return np.asarray([WINDOW_TO_INDEX[int(w)] for w in windows], dtype=np.int64)


def compute_loss(
    outputs: dict[str, torch.Tensor],
    y: torch.Tensor,
    anchor: torch.Tensor,
    weights: torch.Tensor,
    y_min: float,
    y_max: float,
    cand: Candidate,
) -> torch.Tensor:
    err = torch.nn.functional.smooth_l1_loss(outputs["y_pred"], y, reduction="none", beta=0.05)
    endpoint = torch.mean(err * weights)
    delta_true = y - anchor
    delta_loss = torch.mean(torch.abs(outputs["delta_pred"] - delta_true) * weights)
    state_target = torch.clamp((y - y_min) / max(y_max - y_min, 1e-6), 0.0, 1.0)
    state_loss = torch.mean((outputs["state_score"] - state_target) ** 2 * weights)
    below = torch.relu(y_min - outputs["y_pred"])
    above = torch.relu(outputs["y_pred"] - y_max)
    range_loss = torch.mean(below + above)
    return endpoint + cand.lambda_delta * delta_loss + cand.lambda_state * state_loss + 0.02 * range_loss


def prepare_fold_arrays(fold: int, cand: Candidate, train_split: dict[str, Any], other_split: dict[str, Any]) -> dict[str, Any]:
    names = [str(name) for name in train_split["feature_names"]]
    idx = feature_indices(names, cand.feature_mode)
    anchor_train, anchor_other = fit_anchor_train_other(train_split, other_split, names, cand.anchor_mode)
    x_train, x_other, _, _ = scale_train_other(train_split["X"][:, idx], other_split["X"][:, idx])
    return {
        "x_train": x_train,
        "x_other": x_other,
        "y_train": np.asarray(train_split["y"], dtype=np.float32),
        "y_other": np.asarray(other_split["y"], dtype=np.float32),
        "anchor_train": anchor_train.astype(np.float32),
        "anchor_other": anchor_other.astype(np.float32),
        "widx_train": window_indices(train_split["endpoint_window"]),
        "widx_other": window_indices(other_split["endpoint_window"]),
        "weights_train": sample_weights(np.asarray(train_split["endpoint_window"]), cand.long_weight).astype(np.float32),
        "other_split": other_split,
        "input_dim": int(x_train.shape[1]),
        "y_min": float(np.nanmin(train_split["y"])),
        "y_max": float(np.nanmax(train_split["y"])),
    }


def train_model(
    fold: int,
    cand: Candidate,
    train_split: dict[str, Any],
    val_split: dict[str, Any] | None,
    epochs_override: int | None = None,
) -> tuple[V4Net, int, float, dict[str, Any]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(20260617 + fold * 101 + sum(ord(c) for c in cand.candidate_id))
    if val_split is None:
        val_split = train_split
    arrays = prepare_fold_arrays(fold, cand, train_split, val_split)
    model = V4Net(arrays["input_dim"], cand.hidden_dim, cand.dropout, cand.residual_scale).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cand.lr, weight_decay=cand.weight_decay)
    x = to_tensor(arrays["x_train"], device)
    y = to_tensor(arrays["y_train"], device)
    anchor = to_tensor(arrays["anchor_train"], device)
    weights = to_tensor(arrays["weights_train"], device)
    widx = torch.tensor(arrays["widx_train"], dtype=torch.long, device=device)

    x_val = to_tensor(arrays["x_other"], device)
    y_val = to_tensor(arrays["y_other"], device)
    anchor_val = to_tensor(arrays["anchor_other"], device)
    widx_val = torch.tensor(arrays["widx_other"], dtype=torch.long, device=device)

    best_state = None
    best_mae = float("inf")
    best_epoch = 0
    no_improve = 0
    max_epochs = int(epochs_override or cand.max_epochs)
    for epoch in range(1, max_epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        out = model(x, widx, anchor)
        loss = compute_loss(out, y, anchor, weights, arrays["y_min"], arrays["y_max"], cand)
        if torch.isnan(loss) or torch.isinf(loss):
            break
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        if val_split is not train_split:
            model.eval()
            with torch.no_grad():
                pred = model(x_val, widx_val, anchor_val)["y_pred"].detach().cpu().numpy()
            pred = np.clip(pred, arrays["y_min"], arrays["y_max"])
            mae = float(np.mean(np.abs(arrays["y_other"] - pred)))
            if mae + 1e-6 < best_mae:
                best_mae = mae
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= cand.patience:
                break
        else:
            best_epoch = epoch

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch, best_mae, arrays


def predict_model(fold: int, cand: Candidate, model: V4Net, train_split: dict[str, Any], split: dict[str, Any]) -> pd.DataFrame:
    device = next(model.parameters()).device
    arrays = prepare_fold_arrays(fold, cand, train_split, split)
    model.eval()
    with torch.no_grad():
        pred = model(
            to_tensor(arrays["x_other"], device),
            torch.tensor(arrays["widx_other"], dtype=torch.long, device=device),
            to_tensor(arrays["anchor_other"], device),
        )["y_pred"].detach().cpu().numpy()
    low, high = HELPER.fold_bounds(fold)
    pred = np.clip(pred, low, high)
    y = arrays["y_other"]
    return pd.DataFrame(
        {
            "patient_id": [str(pid) for pid in split["patient_id"]],
            "fold": fold,
            "endpoint_window": np.asarray(split["endpoint_window"], dtype=int),
            "y_true": y,
            "y_pred": pred,
            "absolute_error": np.abs(y - pred),
            "candidate_id": cand.candidate_id,
            "anchor_mode": cand.anchor_mode,
            "feature_mode": cand.feature_mode,
        }
    )


def load_rf_validation_best() -> pd.DataFrame:
    best = pd.read_csv(ROOT / "results" / "tables" / "tuning" / "validation_tuning_best_by_model.csv", encoding="utf-8-sig")
    cid = int(best[best["model_name"].eq("random_forest")].iloc[0]["candidate_id"])
    frames = []
    for fold in range(5):
        path = ROOT / "results" / "predictions" / "tuning" / f"random_forest_fold{fold}_candidate{cid}_val_predictions.csv"
        df = pd.read_csv(path)
        df["fold"] = fold
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def candidate_validation_summary(preds: pd.DataFrame, rf_val: pd.DataFrame) -> dict[str, float]:
    merged = preds.merge(
        rf_val[["patient_id", "fold", "endpoint_window", "absolute_error"]].rename(columns={"absolute_error": "rf_abs_error"}),
        on=["patient_id", "fold", "endpoint_window"],
        how="inner",
    )
    err = merged["absolute_error"].to_numpy(float)
    rferr = merged["rf_abs_error"].to_numpy(float)
    windows = merged["endpoint_window"].to_numpy(int)
    out = {
        "mean_val_mae": float(err.mean()),
        "mean_val_delta_vs_rf": float(rferr.mean() - err.mean()),
        "mean_val_weighted_mae": float(np.average(err, weights=sample_weights(windows, "very_long"))),
    }
    for horizon in [18, 24]:
        mask = windows == horizon
        rho = spearmanr(merged.loc[mask, "y_pred"], merged.loc[mask, "y_true"])
        out[f"mean_val_{horizon}_delta_vs_rf"] = float(rferr[mask].mean() - err[mask].mean())
        out[f"mean_val_{horizon}_rho"] = float(rho.statistic) if math.isfinite(float(rho.statistic)) else math.nan
        out[f"mean_val_{horizon}_rho_p"] = float(rho.pvalue) if math.isfinite(float(rho.pvalue)) else math.nan
    out["objective"] = (
        out["mean_val_weighted_mae"]
        - 0.25 * out["mean_val_delta_vs_rf"]
        - 0.35 * min(out["mean_val_18_delta_vs_rf"], out["mean_val_24_delta_vs_rf"])
        - 0.02 * max(out["mean_val_24_rho"], 0)
    )
    return out


def run_validation_search() -> tuple[pd.DataFrame, dict[str, dict[int, int]]]:
    rf_val = load_rf_validation_best()
    rows = []
    best_epochs: dict[str, dict[int, int]] = {}
    for cand in CANDIDATES:
        fold_preds = []
        best_epochs[cand.candidate_id] = {}
        for fold in range(5):
            train = load_tabular(fold, "train")
            val = load_tabular(fold, "val")
            model, best_epoch, best_mae, _ = train_model(fold, cand, train, val)
            pred = predict_model(fold, cand, model, train, val)
            fold_preds.append(pred)
            best_epochs[cand.candidate_id][fold] = int(max(best_epoch, 20))
            print(f"[v4 search] {cand.candidate_id} fold={fold} val_mae={pred['absolute_error'].mean():.4f} best_epoch={best_epoch}", flush=True)
        preds = pd.concat(fold_preds, ignore_index=True)
        preds.to_csv(PRED / f"v4_validation_predictions_{cand.candidate_id}.csv", index=False, encoding="utf-8-sig")
        summary = candidate_validation_summary(preds, rf_val)
        rows.append({**cand.__dict__, **summary})
    search = pd.DataFrame(rows).sort_values(["objective", "mean_val_weighted_mae", "mean_val_mae"]).reset_index(drop=True)
    search.to_csv(TABLES / "kg_v4_validation_candidate_summary.csv", index=False, encoding="utf-8-sig")
    search.iloc[[0]].to_csv(TABLES / "kg_v4_selected_candidate.csv", index=False, encoding="utf-8-sig")
    return search, best_epochs


def test_predictions_for_selected(search: pd.DataFrame, best_epochs: dict[str, dict[int, int]]) -> pd.DataFrame:
    selected_id = str(search.iloc[0]["candidate_id"])
    cand = next(c for c in CANDIDATES if c.candidate_id == selected_id)
    frames = []
    frames_refit = []
    for fold in range(5):
        train = load_tabular(fold, "train")
        val = load_tabular(fold, "val")
        test = load_tabular(fold, "test")
        model, _, _, _ = train_model(fold, cand, train, val)
        pred = predict_model(fold, cand, model, train, test)
        pred["test_mode"] = "train_with_val_early_stop"
        frames.append(pred)

        trainval = combine_tabular(train, val)
        epochs = int(best_epochs[selected_id].get(fold, 80))
        refit_model, _, _, _ = train_model(fold, cand, trainval, None, epochs_override=epochs)
        refit_pred = predict_model(fold, cand, refit_model, trainval, test)
        refit_pred["test_mode"] = "train_plus_val_refit"
        frames_refit.append(refit_pred)
    out = pd.concat(frames + frames_refit, ignore_index=True)
    out.to_csv(PRED / "kg_v4_selected_test_predictions.csv", index=False, encoding="utf-8-sig")
    return out


def paired_stats(kg: pd.DataFrame, rf: pd.DataFrame, label: str, mask: pd.Series) -> dict[str, Any]:
    sub = kg[mask].merge(
        rf[["patient_id", "fold", "endpoint_window", "absolute_error"]].rename(columns={"absolute_error": "rf_absolute_error"}),
        on=["patient_id", "fold", "endpoint_window"],
        how="inner",
    )
    diff = sub["rf_absolute_error"].to_numpy(float) - sub["absolute_error"].to_numpy(float)
    rho = spearmanr(sub["y_pred"], sub["y_true"])
    return {
        "scope": label,
        "n": int(len(sub)),
        "kg_mae": float(sub["absolute_error"].mean()),
        "rf_mae": float(sub["rf_absolute_error"].mean()),
        "delta_rf_minus_kg": float(diff.mean()),
        "wilcoxon_p_kg_less": float(wilcoxon(sub["absolute_error"], sub["rf_absolute_error"], alternative="less").pvalue),
        "paired_ttest_p_kg_less": float(ttest_rel(sub["absolute_error"], sub["rf_absolute_error"], alternative="less").pvalue),
        "spearman_pred_true_rho": float(rho.statistic) if math.isfinite(float(rho.statistic)) else math.nan,
        "spearman_pred_true_p": float(rho.pvalue) if math.isfinite(float(rho.pvalue)) else math.nan,
    }


def evaluate_test(preds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rf = pd.read_csv(SOURCE / "predictions" / "random_forest_stabilized_predictions.csv")
    rows = []
    for mode, kg in preds.groupby("test_mode"):
        for label, mask in [
            ("overall", kg["endpoint_window"].isin(WINDOWS)),
            ("18m", kg["endpoint_window"].eq(18)),
            ("24m", kg["endpoint_window"].eq(24)),
            ("18+24m", kg["endpoint_window"].isin([18, 24])),
            ("12+18+24m", kg["endpoint_window"].isin([12, 18, 24])),
        ]:
            item = paired_stats(kg, rf, label, mask)
            item["test_mode"] = mode
            rows.append(item)
    eval_df = pd.DataFrame(rows)
    eval_df.to_csv(TABLES / "kg_v4_test_evaluation_by_scope.csv", index=False, encoding="utf-8-sig")
    conf_rows = []
    for mode, sub in eval_df.groupby("test_mode"):
        lookup = {row["scope"]: row for _, row in sub.iterrows()}
        overall, pooled, h18, h24, long = lookup["overall"], lookup["12+18+24m"], lookup["18m"], lookup["24m"], lookup["18+24m"]
        full = (
            overall["delta_rf_minus_kg"] > 0
            and pooled["wilcoxon_p_kg_less"] < 0.05
            and h18["delta_rf_minus_kg"] > 0
            and h18["wilcoxon_p_kg_less"] < 0.05
            and h24["delta_rf_minus_kg"] > 0
            and h24["wilcoxon_p_kg_less"] < 0.05
            and h24["spearman_pred_true_p"] < 0.05
        )
        conf_rows.append(
            {
                "test_mode": mode,
                "fully_removes_previous_limits": bool(full),
                "overall_delta_rf_minus_kg": overall["delta_rf_minus_kg"],
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
    conf = pd.DataFrame(conf_rows).sort_values(["fully_removes_previous_limits", "pooled_12_18_24_delta"], ascending=[False, False])
    conf.to_csv(TABLES / "kg_v4_limit_removal_audit.csv", index=False, encoding="utf-8-sig")
    return eval_df, conf


def make_figures(eval_df: pd.DataFrame) -> None:
    scopes = ["overall", "18m", "24m", "12+18+24m", "18+24m"]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    width = 0.25
    x = np.arange(len(scopes))
    for i, (mode, sub) in enumerate(eval_df.groupby("test_mode")):
        sub = sub.set_index("scope").loc[scopes].reset_index()
        ax.bar(x + i * width, sub["delta_rf_minus_kg"], width=width, label=mode)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x + width / 2, scopes)
    ax.set_ylabel("RF MAE - KG-LatentNet V4 MAE")
    ax.set_title("KG-LatentNet V4 held-out test margins")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_kg_v4_test_margins.png", dpi=180)
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    search, best_epochs = run_validation_search()
    preds = test_predictions_for_selected(search, best_epochs)
    eval_df, conf = evaluate_test(preds)
    make_figures(eval_df)
    provenance = {
        "created_by": "honest_kg_latentnet_v4_experiments.py",
        "integrity_note": "V4 architecture and candidate grid are fixed before held-out test evaluation; selection uses validation predictions only.",
        "test_set_used_for_selection": False,
        "architecture": "anchor residual neural model with shared trunk, horizon-specific heads, gate mixing, long-horizon loss weights, and endpoint-state auxiliary loss.",
        "candidate_count": len(CANDIDATES),
        "selected_candidate": str(search.iloc[0]["candidate_id"]),
        "outputs": {
            "validation_candidates": int(len(search)),
            "test_prediction_rows": int(len(preds)),
            "test_eval_rows": int(len(eval_df)),
        },
    }
    (OUT / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(OUT), "selected_candidate": provenance["selected_candidate"], "conformance": conf.to_dict(orient="records")}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
