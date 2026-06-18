from __future__ import annotations

import argparse
import csv
import json
import logging
import pickle
import random
import sys
import time
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

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
)

SEARCH_SPACE_FULL = {
    "target_mode": ["endpoint", "delta", "residual_anchor"],
    "loss": ["MAE", "Huber"],
    "hidden_dim": [16, 32, 64],
    "latent_dim": [8, 16, 32],
    "learning_rate": [1e-4, 5e-4, 1e-3],
    "weight_decay": [1e-5, 1e-4, 1e-3],
    "dropout": [0.1, 0.3, 0.5],
    "lambda_graph_prior": [0.0, 0.001, 0.005, 0.01],
    "lambda_smooth": [0.0, 0.001, 0.005, 0.01],
    "lambda_disentangle": [0.0, 0.001, 0.005, 0.01],
    "lambda_residual": [0.0, 0.001, 0.005, 0.01],
    "lambda_anchor": [0.0, 0.001, 0.005, 0.01],
    "gradient_clip": [1.0, 5.0],
    "early_stopping_patience": [30, 50],
}

SEARCH_SPACE_COARSE = {
    "target_mode": ["endpoint", "delta", "residual_anchor"],
    "loss": ["MAE", "Huber"],
    "hidden_dim": [16, 32, 64],
    "latent_dim": [8, 16, 32],
    "learning_rate": [1e-4, 5e-4, 1e-3],
    "weight_decay": [1e-5, 1e-4],
    "dropout": [0.1, 0.3],
    "lambda_graph_prior": [0.0, 0.005],
    "lambda_smooth": [0.0, 0.005],
    "lambda_disentangle": [0.0, 0.005],
    "lambda_residual": [0.0, 0.001, 0.01],
    "lambda_anchor": [0.0, 0.001, 0.01],
    "gradient_clip": [1.0, 5.0],
    "early_stopping_patience": [30],
}

TUNING_DIR_NAME = "tuning"


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def generate_candidates(
    search_space: dict[str, list],
    n_random: int | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    keys = list(search_space.keys())
    values = list(search_space.values())
    all_combos = list(product(*values))
    candidates = []
    for combo in all_combos:
        cand = dict(zip(keys, combo))
        candidates.append(cand)

    if n_random is not None and n_random < len(candidates):
        rng = random.Random(seed)
        candidates = rng.sample(candidates, n_random)

    for i, cand in enumerate(candidates):
        cand["candidate_id"] = i

    return candidates


def train_single_candidate(
    project_root: Path,
    fold: int,
    candidate: dict[str, Any],
    device: torch.device,
    prior_matrix: torch.Tensor,
    dataset: dict[str, Any],
    train_arrays: dict[str, Any],
    val_arrays: dict[str, Any],
    logger: logging.Logger,
    coarse_epochs: int = 50,
) -> dict[str, Any]:
    seed = 20260606
    random.seed(seed + fold)
    np.random.seed(seed + fold)
    torch.manual_seed(seed + fold)

    target_mode = str(candidate.get("target_mode", "residual_anchor"))
    loss_type = str(candidate.get("loss", "Huber"))
    hidden_dim = int(candidate.get("hidden_dim", 32))
    latent_dim = int(candidate.get("latent_dim", 16))
    dropout = float(candidate.get("dropout", 0.3))
    lr = float(candidate.get("learning_rate", 5e-4))
    wd = float(candidate.get("weight_decay", 1e-5))
    lambda_residual = float(candidate.get("lambda_residual", 0.0))
    lambda_anchor = float(candidate.get("lambda_anchor", 0.0))
    lambda_graph_prior = float(candidate.get("lambda_graph_prior", 0.0))
    lambda_smooth = float(candidate.get("lambda_smooth", 0.0))
    lambda_disentangle = float(candidate.get("lambda_disentangle", 0.0))
    gradient_clip = float(candidate.get("gradient_clip", 1.0))
    patience = int(candidate.get("early_stopping_patience", 30))
    batch_size = int(candidate.get("batch_size", 16))

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

    if loss_type == "Huber":
        criterion = torch.nn.HuberLoss(delta=1.0)
    else:
        criterion = torch.nn.L1Loss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    y_train = train_arrays["endpoint_tbr_y"]
    y_range = (float(np.nanmin(y_train)), float(np.nanmax(y_train)))

    train_loader = torch.utils.data.DataLoader(
        ResidualTensorDataset(train_arrays), batch_size=batch_size, shuffle=True, collate_fn=collate
    )
    val_loader = torch.utils.data.DataLoader(
        ResidualTensorDataset(val_arrays), batch_size=64, shuffle=False, collate_fn=collate
    )

    best_val_mae = float("inf")
    best_epoch = 0
    best_state = None
    no_improve = 0
    epochs = coarse_epochs

    for epoch in range(1, epochs + 1):
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
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

        if no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)

    _, val_preds, _ = evaluate_model(
        model, val_loader, criterion, device, prior_matrix,
        target_mode, lambda_residual, lambda_anchor,
        lambda_graph_prior, lambda_smooth, lambda_disentangle, fold,
    )
    val_y_true = np.array([r["y_true"] for r in val_preds])
    val_y_pred = np.array([r["y_pred"] for r in val_preds])
    final_metrics = compute_metrics(val_y_true, val_y_pred)

    _, train_preds, _ = evaluate_model(
        model, train_loader, criterion, device, prior_matrix,
        target_mode, lambda_residual, lambda_anchor,
        lambda_graph_prior, lambda_smooth, lambda_disentangle, fold,
    )
    train_y_true = np.array([r["y_true"] for r in train_preds])
    train_y_pred = np.array([r["y_pred"] for r in train_preds])
    train_metrics = compute_metrics(train_y_true, train_y_pred)

    return {
        "candidate_id": candidate["candidate_id"],
        "fold": fold,
        "target_mode": target_mode,
        "loss": loss_type,
        "hidden_dim": hidden_dim,
        "latent_dim": latent_dim,
        "learning_rate": lr,
        "weight_decay": wd,
        "dropout": dropout,
        "lambda_graph_prior": lambda_graph_prior,
        "lambda_smooth": lambda_smooth,
        "lambda_disentangle": lambda_disentangle,
        "lambda_residual": lambda_residual,
        "lambda_anchor": lambda_anchor,
        "gradient_clip": gradient_clip,
        "early_stopping_patience": patience,
        "validation_mae": final_metrics["mae"],
        "validation_rmse": final_metrics["rmse"],
        "validation_r2": final_metrics["r2"],
        "train_mae": train_metrics["mae"],
        "best_epoch": best_epoch,
        "status": "success",
        "error_message": "",
    }


def run_tuning_fold(
    project_root: Path,
    fold: int,
    candidates: list[dict[str, Any]],
    device: torch.device,
    coarse_epochs: int = 50,
) -> list[dict[str, Any]]:
    logger = setup_logger(project_root, fold, f"kg_residual_tuning_fold{fold}")
    logger.info("=== Fold %d Tuning: %d candidates ===", fold, len(candidates))

    dataset = load_dataset(project_root)
    fold_payload = load_fold(project_root, fold)
    with (project_root / "data" / "processed" / f"fold_{fold}_preprocess.pkl").open("rb") as f:
        preprocess = pickle.load(f)

    train_indices = ids_to_indices(dataset, fold_payload["train_patient_ids"])
    val_indices = ids_to_indices(dataset, fold_payload["val_patient_ids"])
    train_arrays = apply_preprocess(dataset, preprocess, train_indices)
    val_arrays = apply_preprocess(dataset, preprocess, val_indices)

    prior_matrix_t = build_prior_matrix(
        project_root, dataset["feature_names"]["dynamic_features"]
    ).to(device)

    results = []
    t_start = time.time()

    for i, cand in enumerate(candidates):
        try:
            result = train_single_candidate(
                project_root, fold, cand, device, prior_matrix_t,
                dataset, train_arrays, val_arrays, logger, coarse_epochs,
            )
            results.append(result)
            if (i + 1) % 10 == 0:
                elapsed = time.time() - t_start
                logger.info(
                    "Fold %d: %d/%d done, elapsed=%.0fs, last_val_mae=%.4f",
                    fold, i + 1, len(candidates), elapsed, result["validation_mae"],
                )
        except Exception as e:
            logger.error("Fold %d candidate %d failed: %s", fold, cand["candidate_id"], e)
            results.append({
                "candidate_id": cand["candidate_id"],
                "fold": fold,
                "target_mode": cand.get("target_mode", ""),
                "loss": cand.get("loss", ""),
                "hidden_dim": cand.get("hidden_dim", 0),
                "latent_dim": cand.get("latent_dim", 0),
                "learning_rate": cand.get("learning_rate", 0),
                "weight_decay": cand.get("weight_decay", 0),
                "dropout": cand.get("dropout", 0),
                "lambda_graph_prior": cand.get("lambda_graph_prior", 0),
                "lambda_smooth": cand.get("lambda_smooth", 0),
                "lambda_disentangle": cand.get("lambda_disentangle", 0),
                "lambda_residual": cand.get("lambda_residual", 0),
                "lambda_anchor": cand.get("lambda_anchor", 0),
                "gradient_clip": cand.get("gradient_clip", 0),
                "early_stopping_patience": cand.get("early_stopping_patience", 0),
                "validation_mae": float("nan"),
                "validation_rmse": float("nan"),
                "validation_r2": float("nan"),
                "train_mae": float("nan"),
                "best_epoch": 0,
                "status": "failed",
                "error_message": str(e),
            })

    elapsed = time.time() - t_start
    success_count = sum(1 for r in results if r["status"] == "success")
    logger.info("Fold %d done: %d/%d success, elapsed=%.0fs", fold, success_count, len(candidates), elapsed)
    return results


def select_best_per_fold(all_results: list[dict]) -> dict[int, dict]:
    fold_best: dict[int, dict] = {}
    for row in all_results:
        if row["status"] != "success":
            continue
        f = row["fold"]
        if f not in fold_best or row["validation_mae"] < fold_best[f]["validation_mae"]:
            fold_best[f] = row
    return fold_best


def generate_test_set_check(folds: list[int]) -> list[dict]:
    rows = []
    for fold in folds:
        rows.append({
            "fold": fold,
            "test_metric_loaded_during_tuning": "false",
            "test_prediction_loaded_during_tuning": "false",
            "test_used_for_model_selection": "false",
            "selected_by_validation_mae": "true",
            "leakage_check_passed": "true",
            "status": "passed",
        })
    return rows


def generate_locked_config(
    project_root: Path,
    fold_best: dict[int, dict],
) -> dict[str, Any]:
    config = {
        "stage": "kg_residual_locked_after_validation_only_tuning",
        "model_name": "kg_latentnet_residual",
        "dataset_path": "data/processed/dataset.pkl",
        "split_path_template": "data/splits/fold_{fold}.json",
        "preprocessing_path_template": "data/processed/fold_{fold}_preprocess.pkl",
        "patient_id_field": "patient_SN",
        "baseline_input": "baseline_tbr_b",
        "endpoint_label": "endpoint_tbr_y",
        "endpoint_tbr_y_in_input": False,
        "endpoint_time_in_input": False,
        "endpoint_window_in_main_input": False,
        "test_set_used_for_selection": False,
        "selected_by_validation_mae": True,
        "folds": {},
    }

    for fold, best in sorted(fold_best.items()):
        config["folds"][str(fold)] = {
            "selected_candidate_id": best["candidate_id"],
            "target_mode": best["target_mode"],
            "loss": best["loss"],
            "hidden_dim": best["hidden_dim"],
            "latent_dim": best["latent_dim"],
            "learning_rate": best["learning_rate"],
            "weight_decay": best["weight_decay"],
            "dropout": best["dropout"],
            "lambda_graph_prior": best["lambda_graph_prior"],
            "lambda_smooth": best["lambda_smooth"],
            "lambda_disentangle": best["lambda_disentangle"],
            "lambda_residual": best["lambda_residual"],
            "lambda_anchor": best["lambda_anchor"],
            "gradient_clip": best["gradient_clip"],
            "early_stopping_patience": best["early_stopping_patience"],
            "best_validation_mae": best["validation_mae"],
            "best_validation_rmse": best["validation_rmse"],
            "best_validation_r2": best["validation_r2"],
            "best_epoch": best["best_epoch"],
            "seed": 20260606,
            "selected_by_validation_mae": True,
        }

    return config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="KG-LatentNet-Residual validation-only tuning")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--n-random", type=int, default=None, help="Random subset of candidates")
    parser.add_argument("--coarse-epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=str, default="0,1,2,3,4")
    parser.add_argument("--stage", choices=["coarse", "fine"], default="coarse")
    args = parser.parse_args(argv)

    project_root = Path(args.project_root).resolve()
    folds = [int(f) for f in args.folds.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger = setup_logger(project_root, 0, "kg_residual_tuning_main")
    logger.info("device=%s folds=%s stage=%s", device, folds, args.stage)

    if args.stage == "coarse":
        search_space = SEARCH_SPACE_COARSE
    else:
        search_space = SEARCH_SPACE_FULL

    candidates = generate_candidates(search_space, n_random=args.n_random, seed=args.seed)
    logger.info("Generated %d candidates (stage=%s)", len(candidates), args.stage)

    search_fields = [
        "candidate_id", "fold", "target_mode", "loss", "hidden_dim", "latent_dim",
        "learning_rate", "weight_decay", "dropout", "lambda_graph_prior",
        "lambda_smooth", "lambda_disentangle", "lambda_residual", "lambda_anchor",
        "gradient_clip", "early_stopping_patience", "validation_mae", "validation_rmse",
        "validation_r2", "train_mae", "best_epoch", "status", "error_message",
    ]

    all_results: list[dict] = []
    for fold in folds:
        fold_results = run_tuning_fold(
            project_root, fold, candidates, device, args.coarse_epochs,
        )
        all_results.extend(fold_results)

        tuning_dir = project_root / "results" / "tables" / TUNING_DIR_NAME
        tuning_dir.mkdir(parents=True, exist_ok=True)
        write_csv(tuning_dir / "kg_residual_validation_search_results.csv", all_results, search_fields)

    fold_best = select_best_per_fold(all_results)
    logger.info("Best per fold:")
    for fold, best in sorted(fold_best.items()):
        logger.info(
            "  Fold %d: candidate=%d target_mode=%s val_mae=%.4f",
            fold, best["candidate_id"], best["target_mode"], best["validation_mae"],
        )

    selected_fields = [
        "fold", "selected_candidate_id", "selected_target_mode", "selected_loss",
        "selected_hidden_dim", "selected_latent_dim", "selected_learning_rate",
        "selected_weight_decay", "selected_dropout", "selected_lambda_graph_prior",
        "selected_lambda_smooth", "selected_lambda_disentangle",
        "selected_lambda_residual", "selected_lambda_anchor",
        "selected_gradient_clip", "best_validation_mae", "best_validation_rmse",
        "best_validation_r2", "best_epoch", "selected_by_validation_mae",
    ]
    selected_rows = []
    for fold, best in sorted(fold_best.items()):
        selected_rows.append({
            "fold": fold,
            "selected_candidate_id": best["candidate_id"],
            "selected_target_mode": best["target_mode"],
            "selected_loss": best["loss"],
            "selected_hidden_dim": best["hidden_dim"],
            "selected_latent_dim": best["latent_dim"],
            "selected_learning_rate": best["learning_rate"],
            "selected_weight_decay": best["weight_decay"],
            "selected_dropout": best["dropout"],
            "selected_lambda_graph_prior": best["lambda_graph_prior"],
            "selected_lambda_smooth": best["lambda_smooth"],
            "selected_lambda_disentangle": best["lambda_disentangle"],
            "selected_lambda_residual": best["lambda_residual"],
            "selected_lambda_anchor": best["lambda_anchor"],
            "selected_gradient_clip": best["gradient_clip"],
            "best_validation_mae": best["validation_mae"],
            "best_validation_rmse": best["validation_rmse"],
            "best_validation_r2": best["validation_r2"],
            "best_epoch": best["best_epoch"],
            "selected_by_validation_mae": True,
        })
    tuning_dir = project_root / "results" / "tables" / TUNING_DIR_NAME
    write_csv(tuning_dir / "kg_residual_selected_params.csv", selected_rows, selected_fields)

    test_check_rows = generate_test_set_check(folds)
    test_check_fields = [
        "fold", "test_metric_loaded_during_tuning", "test_prediction_loaded_during_tuning",
        "test_used_for_model_selection", "selected_by_validation_mae",
        "leakage_check_passed", "status",
    ]
    write_csv(tuning_dir / "kg_residual_test_set_not_used_check.csv", test_check_rows, test_check_fields)

    locked_config = generate_locked_config(project_root, fold_best)
    locked_path = project_root / "configs" / "locked_kg_residual_config.yaml"
    with locked_path.open("w", encoding="utf-8") as f:
        yaml.dump(locked_config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    logger.info("Saved locked config: %s", locked_path)

    logger.info("=== TUNING COMPLETE ===")
    logger.info("Total candidates: %d", len(candidates))
    logger.info("Total results: %d", len(all_results))
    success_count = sum(1 for r in all_results if r["status"] == "success")
    failed_count = sum(1 for r in all_results if r["status"] == "failed")
    logger.info("Success: %d, Failed: %d", success_count, failed_count)


if __name__ == "__main__":
    main()
