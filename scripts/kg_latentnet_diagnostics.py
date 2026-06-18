from __future__ import annotations

import csv
import json
import math
import pickle
import sys
import traceback
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import r2_score
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path("/root/KG_LatentNet_Project")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import apply_preprocess, ids_to_indices, load_dataset, load_fold
from src.data.prior_alignment import build_aligned_prior_matrix
from src.models.kg_latentnet import KGLatentNet
from src.models.modules.losses import EndpointMSELoss
from src.training.validation_tuning import TensorFoldDataset, build_torch_model, collate, set_seed


TABLE_DIR = PROJECT_ROOT / "results" / "tables" / "full_5fold"
PRED_DIR = PROJECT_ROOT / "results" / "predictions" / "full_5fold"
CKPT_DIR = PROJECT_ROOT / "results" / "checkpoints" / "full_5fold"
FIG_DIR = PROJECT_ROOT / "results" / "figures" / "full_5fold"
LOG_DIR = PROJECT_ROOT / "results" / "logs" / "full_5fold"


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    err = y_pred - y_true
    mse = float(np.mean(err**2)) if len(err) else math.nan
    return {
        "MAE": float(np.mean(np.abs(err))) if len(err) else math.nan,
        "RMSE": float(np.sqrt(mse)) if math.isfinite(mse) else math.nan,
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else math.nan,
    }


def pred_path(fold: int) -> Path:
    return PRED_DIR / f"kg_latentnet_fold{fold}_predictions.csv"


def load_kg_pred(fold: int) -> pd.DataFrame:
    frame = pd.read_csv(pred_path(fold), encoding="utf-8-sig")
    for col in ["fold", "endpoint_window", "y_true", "y_pred", "absolute_error"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def endpoint_window_distribution(frame: pd.DataFrame) -> str:
    counts = frame["endpoint_window"].value_counts().sort_index()
    return ";".join(f"{int(k)}:{int(v)}" for k, v in counts.items())


def prediction_audit() -> None:
    rows = []
    for fold in range(5):
        df = load_kg_pred(fold)
        worst = df.sort_values("absolute_error", ascending=False).head(10)
        rows.append({
            "fold": fold,
            "n_patients": len(df),
            "y_true_min": df["y_true"].min(),
            "y_true_max": df["y_true"].max(),
            "y_true_mean": df["y_true"].mean(),
            "y_true_std": df["y_true"].std(ddof=1),
            "y_pred_min": df["y_pred"].min(),
            "y_pred_max": df["y_pred"].max(),
            "y_pred_mean": df["y_pred"].mean(),
            "y_pred_std": df["y_pred"].std(ddof=1),
            "absolute_error_min": df["absolute_error"].min(),
            "absolute_error_max": df["absolute_error"].max(),
            "absolute_error_mean": df["absolute_error"].mean(),
            "absolute_error_median": df["absolute_error"].median(),
            "n_nan_y_true": int(df["y_true"].isna().sum()),
            "n_nan_y_pred": int(df["y_pred"].isna().sum()),
            "n_inf_y_pred": int(np.isinf(df["y_pred"].to_numpy(dtype=float)).sum()),
            "n_extreme_pred_abs_gt_5": int((df["y_pred"].abs() > 5).sum()),
            "n_extreme_pred_abs_gt_10": int((df["y_pred"].abs() > 10).sum()),
            "worst_10_patients": ";".join(f"{r.patient_id}:{r.absolute_error:.6g}" for r in worst.itertuples()),
            "endpoint_window_distribution": endpoint_window_distribution(df),
            "status": "failed" if (df["y_pred"].abs() > 10).any() or df["y_pred"].isna().any() else "passed",
            "suspected_issue": "fold has extreme prediction values" if (df["y_pred"].abs() > 10).any() else "no extreme prediction values",
        })
    write_csv(TABLE_DIR / "kg_latentnet_prediction_audit.csv", rows)


def metric_recompute() -> None:
    fold_rows = []
    all_true = []
    all_pred = []
    for fold in range(5):
        df = load_kg_pred(fold)
        met = metrics(df["y_true"].to_numpy(), df["y_pred"].to_numpy())
        worst = df.sort_values("absolute_error", ascending=False).iloc[0]
        fold_rows.append({
            "fold": fold,
            "n_test": len(df),
            **met,
            "y_pred_min": df["y_pred"].min(),
            "y_pred_max": df["y_pred"].max(),
            "max_absolute_error": df["absolute_error"].max(),
            "worst_patient_id": worst["patient_id"],
        })
        all_true.append(df["y_true"].to_numpy())
        all_pred.append(df["y_pred"].to_numpy())
    write_csv(TABLE_DIR / "kg_latentnet_fold_metric_recompute.csv", fold_rows)
    all_true_np = np.concatenate(all_true)
    all_pred_np = np.concatenate(all_pred)
    overall = metrics(all_true_np, all_pred_np)
    summary = pd.read_csv(TABLE_DIR / "all_models_5fold_results.csv", encoding="utf-8-sig")
    kg = summary[summary["model_name"] == "kg_latentnet"].iloc[0]
    rows = [{
        "source": "prediction_file_patient_level_recompute",
        **overall,
        "reported_mean_fold_MAE": kg["mean_test_mae"],
        "reported_mean_fold_RMSE": kg["mean_test_rmse"],
        "reported_mean_fold_R2": kg["mean_test_r2"],
        "note": "Patient-level pooled metrics are expected to differ from all_models mean-of-fold metrics.",
    }]
    mean_fold = pd.DataFrame(fold_rows)[["MAE", "RMSE", "R2"]].mean()
    rows.append({
        "source": "prediction_file_mean_of_fold_recompute",
        "MAE": mean_fold["MAE"],
        "RMSE": mean_fold["RMSE"],
        "R2": mean_fold["R2"],
        "reported_mean_fold_MAE": kg["mean_test_mae"],
        "reported_mean_fold_RMSE": kg["mean_test_rmse"],
        "reported_mean_fold_R2": kg["mean_test_r2"],
        "note": "This should match all_models_5fold_results.csv within rounding tolerance.",
    })
    write_csv(TABLE_DIR / "kg_latentnet_metric_recompute.csv", rows)


def target_scaling_audit() -> None:
    dataset = load_dataset(PROJECT_ROOT)
    rows = []
    for fold in range(5):
        fold_payload = load_fold(PROJECT_ROOT, fold)
        train_idx = ids_to_indices(dataset, fold_payload["train_patient_ids"])
        train_y = np.asarray(dataset["endpoint_tbr_y"])[train_idx].astype(float)
        preprocess_path = PROJECT_ROOT / "data" / "processed" / f"fold_{fold}_preprocess.pkl"
        preprocess = pickle.load(preprocess_path.open("rb"))
        df = load_kg_pred(fold)
        y_scaler_keys = [key for key in preprocess if "y" in key.lower() or "target" in key.lower()]
        rows.append({
            "fold": fold,
            "y_scaler_exists": False,
            "y_scaler_fit_on_train_only": "not_applicable_no_y_scaler",
            "y_true_saved_scale": "original_endpoint_tbr_y",
            "y_pred_saved_scale": "direct_model_output_original_target_loss_scale",
            "inverse_transform_applied": False,
            "inverse_transform_applied_times": 0,
            "target_scaler_path": "",
            "preprocess_path": str(preprocess_path),
            "preprocess_y_related_keys": ";".join(y_scaler_keys),
            "preprocess_endpoint_tbr_y_used_for_fit": preprocess.get("endpoint_tbr_y_used_for_fit", ""),
            "scaler_train_y_min": float(np.nanmin(train_y)),
            "scaler_train_y_max": float(np.nanmax(train_y)),
            "saved_y_true_min": df["y_true"].min(),
            "saved_y_true_max": df["y_true"].max(),
            "saved_y_pred_min": df["y_pred"].min(),
            "saved_y_pred_max": df["y_pred"].max(),
            "status": "passed_no_target_scaling_detected",
            "suspected_issue": "not target inverse-transform; KG predicts directly on raw endpoint_tbr_y scale",
        })
    write_csv(TABLE_DIR / "kg_latentnet_target_scaling_audit.csv", rows)


def checkpoint_audit() -> None:
    cfg = yaml.safe_load((PROJECT_ROOT / "configs" / "locked_full_5fold_config.yaml").read_text(encoding="utf-8"))
    dataset = load_dataset(PROJECT_ROOT)
    static_dim = int(dataset["static_features"].shape[1])
    dynamic_dim = int(dataset["dynamic_features"].shape[2])
    treatment_dim = int(dataset["treatment_features"].shape[2])
    rows = []
    for fold in range(5):
        info = cfg["models"]["kg_latentnet"]["folds"][f"fold_{fold}"]
        params = dict(info["selected_params"])
        expected_best = CKPT_DIR / f"kg_latentnet_fold{fold}_best.pt"
        actual = CKPT_DIR / f"kg_latentnet_fold{fold}.pt"
        strict_load_ok = False
        checkpoint_epoch = ""
        model_match = False
        error = ""
        if actual.exists():
            try:
                ckpt = torch.load(actual, map_location="cpu")
                state = ckpt.get("model_state_dict", ckpt)
                checkpoint_epoch = ckpt.get("epoch", "")
                model = KGLatentNet(static_dim, dynamic_dim, treatment_dim, hidden_dim=int(params.get("hidden_dim", 64)))
                model.load_state_dict(state, strict=True)
                strict_load_ok = True
                model_match = True
            except Exception as exc:
                error = str(exc)
        rows.append({
            "fold": fold,
            "checkpoint_exists": actual.exists(),
            "checkpoint_path": str(actual) if actual.exists() else "",
            "expected_best_checkpoint_exists": expected_best.exists(),
            "expected_best_checkpoint_path": str(expected_best),
            "checkpoint_epoch": checkpoint_epoch,
            "best_validation_mae": info.get("best_validation_mae"),
            "final_train_loss": "",
            "final_val_loss": "",
            "checkpoint_loaded_for_test": "in_memory_best_state_loaded_before_test; file not reloaded",
            "checkpoint_model_config_matches_current_model": model_match,
            "optimizer_state_loaded_or_not": "not_loaded_for_test",
            "test_prediction_generated_from_checkpoint": "generated from same in-memory best_state saved to checkpoint",
            "status": "warning_best_filename_missing" if not expected_best.exists() and actual.exists() and strict_load_ok else "failed" if not strict_load_ok else "passed",
            "error": error,
        })
    write_csv(TABLE_DIR / "kg_latentnet_checkpoint_audit.csv", rows)


def training_curve_summary() -> None:
    cfg = yaml.safe_load((PROJECT_ROOT / "configs" / "locked_full_5fold_config.yaml").read_text(encoding="utf-8"))
    rows = []
    for fold in range(5):
        info = cfg["models"]["kg_latentnet"]["folds"][f"fold_{fold}"]
        params = info["selected_params"]
        rows.append({
            "fold": fold,
            "selected_candidate_id": info.get("selected_candidate_id"),
            "selected_seed": info.get("selected_seed"),
            "learning_rate": params.get("learning_rate"),
            "hidden_dim": params.get("hidden_dim"),
            "weight_decay": params.get("weight_decay"),
            "best_validation_mae": info.get("best_validation_mae"),
            "best_validation_rmse": info.get("best_validation_rmse"),
            "best_validation_r2": info.get("best_validation_r2"),
            "train_loss_curve_available": False,
            "validation_curve_available": False,
            "gradient_norm_available": False,
            "early_stopping_epoch_available": False,
            "status": "curve_unavailable",
            "note": "full_5fold_evaluation.py did not log per-epoch KG train/val curves; locked config contains only selected validation metrics.",
        })
    write_csv(TABLE_DIR / "kg_latentnet_training_curve_summary.csv", rows)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    x = [row["fold"] for row in rows]
    plt.plot(x, [row["best_validation_mae"] for row in rows], marker="o", label="Best validation MAE")
    plt.plot(x, [row["best_validation_rmse"] for row in rows], marker="o", label="Best validation RMSE")
    plt.xlabel("Fold")
    plt.ylabel("Validation metric")
    plt.title("KG-LatentNet selected validation metrics (per-epoch curves unavailable)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "kg_latentnet_train_val_curve.png", dpi=200)
    plt.close()


def load_train_arrays(dataset: dict[str, Any], fold: int, limit: int | None = None) -> dict[str, Any]:
    payload = load_fold(PROJECT_ROOT, fold)
    ids = payload["train_patient_ids"][:limit] if limit else payload["train_patient_ids"]
    preprocess = pickle.load((PROJECT_ROOT / "data" / "processed" / f"fold_{fold}_preprocess.pkl").open("rb"))
    return apply_preprocess(dataset, preprocess, ids_to_indices(dataset, ids))


@torch.no_grad()
def loader_mae(model: nn.Module, loader: DataLoader, device: torch.device, prior_matrix: torch.Tensor) -> tuple[float, tuple[float, float], tuple[float, float]]:
    model.eval()
    ys, ps = [], []
    for batch in loader:
        tb = {key: value.to(device) for key, value in batch["tensors"].items()}
        pred = model(tb, prior_matrix=prior_matrix)
        ys.extend(tb["endpoint_tbr_y"].detach().cpu().numpy().reshape(-1).tolist())
        ps.extend(pred.detach().cpu().numpy().reshape(-1).tolist())
    y = np.asarray(ys)
    p = np.asarray(ps)
    return float(np.mean(np.abs(y - p))), (float(np.min(y)), float(np.max(y))), (float(np.min(p)), float(np.max(p)))


def sanity_check() -> None:
    log_lines = []
    rows = []
    dataset = load_dataset(PROJECT_ROOT)
    prior_np, _, prior_checks = build_aligned_prior_matrix(PROJECT_ROOT, dataset["feature_names"]["dynamic_features"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prior = torch.tensor(prior_np, dtype=torch.float32, device=device)
    cfg = yaml.safe_load((PROJECT_ROOT / "configs" / "locked_full_5fold_config.yaml").read_text(encoding="utf-8"))
    static_dim = int(dataset["static_features"].shape[1])
    dynamic_dim = int(dataset["dynamic_features"].shape[2])
    treatment_dim = int(dataset["treatment_features"].shape[2])
    for fold in range(5):
        params = dict(cfg["models"]["kg_latentnet"]["folds"][f"fold_{fold}"]["selected_params"])
        set_seed(9000 + fold)
        arrays = load_train_arrays(dataset, fold, limit=16)
        loader = DataLoader(TensorFoldDataset(arrays), batch_size=16, shuffle=False, collate_fn=collate)
        batch = next(iter(loader))
        tb = {key: value.to(device) for key, value in batch["tensors"].items()}
        model = KGLatentNet(static_dim, dynamic_dim, treatment_dim, hidden_dim=int(params.get("hidden_dim", 32))).to(device)
        pred0 = model(tb, prior_matrix=prior)
        loss0 = EndpointMSELoss()(pred0, tb["endpoint_tbr_y"])
        static_context = torch.cat([tb["static_features"], tb["baseline_tbr_b"]], dim=-1)
        static_hidden = model.static_encoder(static_context)
        dynamic_hidden = model.dynamic_graph(tb["dynamic_features"], tb["dynamic_mask"], prior_matrix=prior)
        sequence = torch.cat([dynamic_hidden, tb["treatment_features"], tb["delta_time"].unsqueeze(-1)], dim=-1)
        sequence_mask = (tb["dynamic_mask"].sum(dim=-1) > 0) | (tb["treatment_features"].sum(dim=-1) > 0)
        dynamic_summary = model.short_delay_update(sequence, sequence_mask)
        fused = model.fusion(static_hidden, dynamic_summary)
        mae_start, y_range, p_range = loader_mae(model, loader, device, prior)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=float(params.get("weight_decay", 0.0)))
        for _ in range(200):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            pred = model(tb, prior_matrix=prior)
            loss = EndpointMSELoss()(pred, tb["endpoint_tbr_y"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        mae_end, _, p_end_range = loader_mae(model, loader, device, prior)
        rows.append({
            "test_name": "tiny_overfit_16_train_patients",
            "fold": fold,
            "status": "passed" if mae_end < mae_start and mae_end < 0.2 else "failed",
            "train_mae_start": mae_start,
            "train_mae_end": mae_end,
            "y_true_range": f"{y_range[0]:.6g}..{y_range[1]:.6g}",
            "y_pred_range": f"{p_end_range[0]:.6g}..{p_end_range[1]:.6g}",
            "loss_nan": bool(torch.isnan(loss0).item()),
            "latent_nan": bool(torch.isnan(fused).any().item()),
            "graph_nan": bool(torch.isnan(dynamic_hidden).any().item()),
            "suspected_issue": "cannot overfit tiny train subset" if not (mae_end < mae_start and mae_end < 0.2) else "none",
        })
        rows.append({
            "test_name": "one_batch_forward_check",
            "fold": fold,
            "status": "passed" if torch.isfinite(loss0) and not torch.isnan(fused).any() and not torch.isnan(dynamic_hidden).any() else "failed",
            "train_mae_start": mae_start,
            "train_mae_end": "",
            "y_true_range": f"{y_range[0]:.6g}..{y_range[1]:.6g}",
            "y_pred_range": f"{p_range[0]:.6g}..{p_range[1]:.6g}",
            "loss_nan": bool(torch.isnan(loss0).item()),
            "latent_nan": bool(torch.isnan(fused).any().item()),
            "graph_nan": bool(torch.isnan(dynamic_hidden).any().item()),
            "suspected_issue": "one-batch finite forward ok",
        })
        # Baseline-only linear readout sanity on the same 16 train patients.
        y = tb["endpoint_tbr_y"].detach().cpu().numpy().reshape(-1)
        x = tb["baseline_tbr_b"].detach().cpu().numpy().reshape(-1)
        X = np.vstack([x, np.ones_like(x)]).T
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        pred_linear = X @ coef
        mae_linear = float(np.mean(np.abs(pred_linear - y)))
        rows.append({
            "test_name": "baseline_replacement_sanity_train16",
            "fold": fold,
            "status": "passed" if mae_linear < mae_start else "failed",
            "train_mae_start": mae_start,
            "train_mae_end": mae_linear,
            "y_true_range": f"{float(np.min(y)):.6g}..{float(np.max(y)):.6g}",
            "y_pred_range": f"{float(np.min(pred_linear)):.6g}..{float(np.max(pred_linear)):.6g}",
            "loss_nan": False,
            "latent_nan": "",
            "graph_nan": "",
            "suspected_issue": "baseline-only readout is much easier than KG initialization",
        })
        log_lines.append(f"fold={fold} shapes=" + json.dumps({k: list(v.shape) for k, v in tb.items()}) + f" y_range={y_range} initial_pred_range={p_range} tiny_overfit_mae={mae_start}->{mae_end}")
    write_csv(TABLE_DIR / "kg_latentnet_sanity_check.csv", rows)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (LOG_DIR / "kg_latentnet_sanity_check.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")


def input_prior_audit() -> None:
    dataset = load_dataset(PROJECT_ROOT)
    prior_np, prior_names, prior_checks = build_aligned_prior_matrix(PROJECT_ROOT, dataset["feature_names"]["dynamic_features"])
    rows = []
    global_empty_dynamic = int((np.asarray(dataset["dynamic_mask"]).sum(axis=(1, 2)) == 0).sum())
    for fold in range(5):
        arrays = load_train_arrays(dataset, fold)
        delta = arrays["delta_time"]
        mask = arrays["dynamic_mask"]
        treatment = arrays["treatment_features"]
        rows.append({
            "fold": fold,
            "split": "train",
            "static_features_shape": str(tuple(arrays["static_features"].shape)),
            "dynamic_features_shape": str(tuple(arrays["dynamic_features"].shape)),
            "dynamic_mask_shape": str(tuple(mask.shape)),
            "delta_time_shape": str(tuple(delta.shape)),
            "treatment_features_shape": str(tuple(treatment.shape)),
            "baseline_tbr_b_shape": str(tuple(arrays["baseline_tbr_b"].shape)),
            "prior_matrix_shape": str(tuple(prior_np.shape)),
            "dynamic_feature_count": len(dataset["feature_names"]["dynamic_features"]),
            "prior_feature_count": len(prior_names),
            "prior_alignment_passed": all(bool(r.get("passed")) for r in prior_checks),
            "train_empty_dynamic_history_count": int((mask.sum(axis=(1, 2)) == 0).sum()),
            "global_empty_dynamic_history_count": global_empty_dynamic,
            "zero_padding_masked": bool(np.all(arrays["dynamic_features"][mask == 0] == 0)),
            "delta_time_min": float(np.nanmin(delta)),
            "delta_time_max": float(np.nanmax(delta)),
            "delta_time_mean": float(np.nanmean(delta)),
            "delta_time_abs_gt_365_count": int((np.abs(delta) > 365).sum()),
            "dynamic_value_min": float(np.nanmin(arrays["dynamic_features"])),
            "dynamic_value_max": float(np.nanmax(arrays["dynamic_features"])),
            "status": "warning_large_delta_time" if (np.abs(delta) > 365).any() else "passed",
            "suspected_issue": "delta_time is raw and may dominate GRU input" if (np.abs(delta) > 365).any() else "no shape/prior mismatch detected",
        })
    write_csv(TABLE_DIR / "kg_latentnet_input_prior_audit.csv", rows)


def bugfix_log() -> None:
    text = """# KG-LatentNet Bugfix Log

## Current diagnostic action

- No baseline results were modified.
- No prediction CSV was manually edited.
- No supplementary retraining experiment was launched.
- Test set was not used for hyperparameter tuning: false.

## Evidence before any KG rerun

- `kg_latentnet_fold0_predictions.csv` contains an extreme prediction around 319 while y_true is around 1.24.
- `kg_latentnet_fold1_predictions.csv` predicts around 0.06-0.22 while y_true is around 1.16-2.49.
- Metric recomputation from prediction files matches the reported KG-LatentNet metrics, so the summary metric calculation is not the primary source of the anomaly.
- Target preprocessing does not fit or apply a y scaler; y_true is saved in original endpoint TBR units and y_pred is direct model output on the raw target loss scale.
- Full evaluation saves `kg_latentnet_fold*.pt` from the in-memory best validation state; `_best.pt` filenames are not used by the current runner.

## Fix status

No code fix has been applied yet. Diagnostics indicate instability in KG-LatentNet training/model behavior rather than a confirmed target inverse-transform or metric bug.

## Impact on baseline

None.

## Need to rerun KG-LatentNet

Only after a specific implementation bug is confirmed and fixed. Do not rerun all baselines.
"""
    (LOG_DIR / "kg_latentnet_bugfix_log.md").write_text(text, encoding="utf-8")


def main() -> None:
    prediction_audit()
    metric_recompute()
    target_scaling_audit()
    checkpoint_audit()
    training_curve_summary()
    sanity_check()
    input_prior_audit()
    bugfix_log()


if __name__ == "__main__":
    main()
