from __future__ import annotations

import argparse
import csv
import json
import logging
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import apply_preprocess, ids_to_indices, load_dataset, load_fold
from src.data.prior_alignment import build_aligned_prior_matrix
from src.models.kg_latentnet_residual import KGLatentNetResidual
from src.training.train_kg_latentnet_residual import (
    ResidualTensorDataset,
    build_prior_matrix,
    collate,
    compute_metrics,
    evaluate_model,
    setup_logger,
    write_csv,
)


def load_locked_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_and_eval_fold(
    project_root: Path,
    fold: int,
    fold_config: dict[str, Any],
    device: torch.device,
    logger: logging.Logger,
) -> dict[str, Any]:
    seed = int(fold_config.get("seed", 20260606))
    random.seed(seed + fold)
    np.random.seed(seed + fold)
    torch.manual_seed(seed + fold)

    target_mode = str(fold_config["target_mode"])
    loss_type = str(fold_config["loss"])
    hidden_dim = int(fold_config["hidden_dim"])
    latent_dim = int(fold_config["latent_dim"])
    dropout = float(fold_config["dropout"])
    lr = float(fold_config["learning_rate"])
    wd = float(fold_config["weight_decay"])
    lambda_graph_prior = float(fold_config.get("lambda_graph_prior", 0.0))
    lambda_smooth = float(fold_config.get("lambda_smooth", 0.0))
    lambda_disentangle = float(fold_config.get("lambda_disentangle", 0.0))
    lambda_residual = float(fold_config.get("lambda_residual", 0.0))
    lambda_anchor = float(fold_config.get("lambda_anchor", 0.0))
    gradient_clip = float(fold_config.get("gradient_clip", 1.0))
    patience = int(fold_config.get("early_stopping_patience", 50))
    batch_size = int(fold_config.get("batch_size", 16))
    max_epochs = 200

    logger.info("=== Fold %d Test Evaluation ===", fold)
    logger.info("target_mode=%s loss=%s hidden_dim=%d latent_dim=%d", target_mode, loss_type, hidden_dim, latent_dim)
    logger.info("lr=%.6f wd=%.6f dropout=%.2f gradient_clip=%.1f", lr, wd, dropout, gradient_clip)
    logger.info("lambda_residual=%.4f lambda_anchor=%.4f lambda_graph_prior=%.4f", lambda_residual, lambda_anchor, lambda_graph_prior)
    logger.info("lambda_smooth=%.4f lambda_disentangle=%.4f", lambda_smooth, lambda_disentangle)

    dataset = load_dataset(project_root)
    fold_payload = load_fold(project_root, fold)
    with (project_root / "data" / "processed" / f"fold_{fold}_preprocess.pkl").open("rb") as f:
        preprocess = pickle.load(f)

    train_indices = ids_to_indices(dataset, fold_payload["train_patient_ids"])
    val_indices = ids_to_indices(dataset, fold_payload["val_patient_ids"])
    test_indices = ids_to_indices(dataset, fold_payload["test_patient_ids"])

    train_arrays = apply_preprocess(dataset, preprocess, train_indices)
    val_arrays = apply_preprocess(dataset, preprocess, val_indices)
    test_arrays = apply_preprocess(dataset, preprocess, test_indices)

    logger.info("train=%d val=%d test=%d", len(train_arrays["patient_id"]), len(val_arrays["patient_id"]), len(test_arrays["patient_id"]))

    y_train = train_arrays["endpoint_tbr_y"]
    y_range = (float(np.nanmin(y_train)), float(np.nanmax(y_train)))
    logger.info("train_y_range=[%.4f, %.4f]", y_range[0], y_range[1])

    prior_matrix = build_prior_matrix(project_root, dataset["feature_names"]["dynamic_features"]).to(device)

    static_dim = dataset["static_features"].shape[1]
    dynamic_dim = dataset["dynamic_features"].shape[2]
    treatment_dim = dataset["treatment_features"].shape[2]

    model = KGLatentNetResidual(
        static_dim=static_dim,
        dynamic_dim=dynamic_dim,
        treatment_dim=treatment_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        dropout=dropout,
        target_mode=target_mode,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("model_params=%d", n_params)

    if loss_type == "Huber":
        criterion = nn.HuberLoss(delta=1.0)
    else:
        criterion = nn.L1Loss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    train_loader = DataLoader(ResidualTensorDataset(train_arrays), batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(ResidualTensorDataset(val_arrays), batch_size=64, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(ResidualTensorDataset(test_arrays), batch_size=64, shuffle=False, collate_fn=collate)

    best_val_mae = float("inf")
    best_epoch = 0
    best_state = None
    no_improve = 0

    t_start = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            tensor_batch = {k: v.to(device) for k, v in batch["tensors"].items()}
            outputs = model(tensor_batch, prior_matrix=prior_matrix)
            loss_dict = KGLatentNetResidual.compute_loss(
                outputs, tensor_batch, criterion, target_mode,
                lambda_residual=lambda_residual,
                lambda_anchor=lambda_anchor,
                lambda_graph_prior=lambda_graph_prior,
                lambda_smooth=lambda_smooth,
                lambda_disentangle=lambda_disentangle,
                y_range=y_range,
                prior_matrix=prior_matrix,
            )
            loss = loss_dict["total"]
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if gradient_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            train_losses.append(loss_dict["main"].item())

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")

        _, val_preds, _ = evaluate_model(
            model, val_loader, criterion, device, prior_matrix,
            target_mode, lambda_residual, lambda_anchor,
            lambda_graph_prior, lambda_smooth, lambda_disentangle, fold,
        )
        val_y_true = np.array([r["y_true"] for r in val_preds])
        val_y_pred = np.array([r["y_pred"] for r in val_preds])
        val_metrics = compute_metrics(val_y_true, val_y_pred)

        scheduler.step(val_metrics["mae"])

        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 20 == 0 or epoch == 1:
            logger.info("epoch=%d train_loss=%.6f val_mae=%.6f val_rmse=%.6f val_r2=%.6f",
                        epoch, train_loss, val_metrics["mae"], val_metrics["rmse"], val_metrics["r2"])

        if no_improve >= patience:
            logger.info("Early stopping at epoch %d", epoch)
            break

    elapsed = time.time() - t_start
    logger.info("Training done: best_epoch=%d best_val_mae=%.6f elapsed=%.1fs", best_epoch, best_val_mae, elapsed)

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)

    ckpt_path = project_root / "results" / "checkpoints" / "full_5fold" / f"kg_latentnet_residual_fold{fold}_best.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": best_state,
        "fold": fold,
        "best_epoch": best_epoch,
        "best_val_mae": best_val_mae,
        "config": fold_config,
    }, ckpt_path)
    logger.info("Saved checkpoint: %s", ckpt_path)

    _, test_preds, test_latent = evaluate_model(
        model, test_loader, criterion, device, prior_matrix,
        target_mode, lambda_residual, lambda_anchor,
        lambda_graph_prior, lambda_smooth, lambda_disentangle, fold,
    )

    test_y_true = np.array([r["y_true"] for r in test_preds])
    test_y_pred = np.array([r["y_pred"] for r in test_preds])
    test_metrics = compute_metrics(test_y_true, test_y_pred)
    logger.info("TEST: MAE=%.6f RMSE=%.6f R2=%.6f", test_metrics["mae"], test_metrics["rmse"], test_metrics["r2"])

    pred_fields = ["patient_id", "fold", "endpoint_window", "baseline_tbr_b", "y_true", "delta_true", "delta_pred", "y_pred", "absolute_error"]
    pred_path = project_root / "results" / "predictions" / "full_5fold" / f"kg_latentnet_residual_fold{fold}_predictions.csv"
    write_csv(pred_path, test_preds, pred_fields)
    logger.info("Saved predictions: %s", pred_path)

    latent_path = project_root / "results" / "latent" / "full_5fold" / f"kg_latentnet_residual_fold{fold}_latent_states.pkl"
    latent_path.parent.mkdir(parents=True, exist_ok=True)
    with latent_path.open("wb") as f:
        pickle.dump(test_latent["latent_states"], f)
    logger.info("Saved latent states: %s", latent_path)

    contrib_path = project_root / "results" / "latent" / "full_5fold" / f"kg_latentnet_residual_fold{fold}_contributions.pkl"
    with contrib_path.open("wb") as f:
        pickle.dump(test_latent["contributions"], f)
    logger.info("Saved contributions: %s", contrib_path)

    return {
        "fold": fold,
        "target_mode": target_mode,
        "loss": loss_type,
        "hidden_dim": hidden_dim,
        "latent_dim": latent_dim,
        "learning_rate": lr,
        "weight_decay": wd,
        "dropout": dropout,
        "best_epoch": best_epoch,
        "best_val_mae": best_val_mae,
        "test_mae": test_metrics["mae"],
        "test_rmse": test_metrics["rmse"],
        "test_r2": test_metrics["r2"],
        "runtime_sec": elapsed,
        "status": "success",
        "prediction_path": str(pred_path),
        "checkpoint_path": str(ckpt_path),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="KG-LatentNet-Residual locked config test evaluation")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--config", default="configs/locked_kg_residual_config.yaml")
    parser.add_argument("--folds", type=str, default="0,1,2,3,4")
    args = parser.parse_args(argv)

    project_root = Path(args.project_root).resolve()
    config_path = project_root / args.config
    folds = [int(f) for f in args.folds.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger = setup_logger(project_root, 0, "kg_latentnet_residual_test_eval")
    logger.info("device=%s config=%s folds=%s", device, config_path, folds)

    locked_config = load_locked_config(config_path)
    logger.info("stage=%s test_set_used_for_selection=%s", locked_config["stage"], locked_config["test_set_used_for_selection"])

    fold_configs = locked_config["folds"]
    status_fields = [
        "fold", "target_mode", "loss", "hidden_dim", "latent_dim",
        "learning_rate", "weight_decay", "dropout", "best_epoch",
        "best_val_mae", "test_mae", "test_rmse", "test_r2",
        "runtime_sec", "status", "prediction_path", "checkpoint_path",
    ]
    all_results = []

    for fold in folds:
        fold_key = str(fold)
        if fold_key not in fold_configs:
            logger.error("Fold %d not found in locked config", fold)
            continue

        fold_config = fold_configs[fold_key]
        logger.info("Starting fold %d...", fold)

        try:
            result = train_and_eval_fold(project_root, fold, fold_config, device, logger)
            all_results.append(result)
        except Exception as e:
            logger.error("Fold %d failed: %s", fold, e, exc_info=True)
            all_results.append({
                "fold": fold, "target_mode": "", "loss": "", "hidden_dim": 0,
                "latent_dim": 0, "learning_rate": 0, "weight_decay": 0, "dropout": 0,
                "best_epoch": 0, "best_val_mae": float("nan"), "test_mae": float("nan"),
                "test_rmse": float("nan"), "test_r2": float("nan"), "runtime_sec": 0,
                "status": "failed", "prediction_path": "", "checkpoint_path": "",
            })

    status_path = project_root / "results" / "tables" / "full_5fold" / "kg_latentnet_residual_fold_level_results.csv"
    write_csv(status_path, all_results, status_fields)
    logger.info("Saved fold-level results: %s", status_path)

    success_results = [r for r in all_results if r["status"] == "success"]
    if success_results:
        maes = [r["test_mae"] for r in success_results]
        rmses = [r["test_rmse"] for r in success_results]
        r2s = [r["test_r2"] for r in success_results]
        import statistics
        summary = {
            "model_name": "kg_latentnet_residual",
            "n_folds": len(all_results),
            "n_success": len(success_results),
            "mean_test_mae": statistics.mean(maes),
            "std_test_mae": statistics.stdev(maes) if len(maes) > 1 else 0,
            "mean_test_rmse": statistics.mean(rmses),
            "std_test_rmse": statistics.stdev(rmses) if len(rmses) > 1 else 0,
            "mean_test_r2": statistics.mean(r2s),
            "std_test_r2": statistics.stdev(r2s) if len(r2s) > 1 else 0,
        }
        logger.info("=== SUMMARY ===")
        logger.info("MAE: %.6f +/- %.6f", summary["mean_test_mae"], summary["std_test_mae"])
        logger.info("RMSE: %.6f +/- %.6f", summary["mean_test_rmse"], summary["std_test_rmse"])
        logger.info("R2: %.6f +/- %.6f", summary["mean_test_r2"], summary["std_test_r2"])

    report_path = project_root / "results" / "tables" / "full_5fold" / "learned_relation_weights_unavailable_report.csv"
    write_csv(report_path, [{"learned_relation_weights_available": "false"}], ["learned_relation_weights_available"])
    logger.info("Saved relation weights report: %s", report_path)

    logger.info("=== TEST EVALUATION COMPLETE ===")


if __name__ == "__main__":
    main()
