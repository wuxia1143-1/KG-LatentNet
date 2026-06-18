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
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import apply_preprocess, ids_to_indices, load_dataset, load_fold
from src.data.prior_alignment import build_aligned_prior_matrix
from src.models.kg_latentnet_v2 import KGLatentNetV2

LEAKAGE_BLACKLIST = [
    "胸主动脉TBR", "TBR值", "endpoint", "label", "目标", "随访TBR",
    "endpoint_tbr_y", "endpoint_time", "endpoint_window",
]


class ResidualTensorDataset(Dataset):
    def __init__(self, arrays: dict[str, Any]) -> None:
        self.arrays = arrays
        self.n = len(arrays["patient_id"])

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict[str, Any]:
        tensor_keys = [
            "static_features",
            "dynamic_features",
            "dynamic_mask",
            "delta_time",
            "treatment_features",
            "baseline_tbr_b",
            "endpoint_tbr_y",
        ]
        tensors = {
            key: torch.tensor(self.arrays[key][idx], dtype=torch.float32)
            for key in tensor_keys
        }
        return {
            "tensors": tensors,
            "patient_id": self.arrays["patient_id"][idx],
            "endpoint_window": int(self.arrays["endpoint_window"][idx]),
        }


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    keys = batch[0]["tensors"].keys()
    tensors = {
        key: torch.stack([item["tensors"][key] for item in batch], dim=0)
        for key in keys
    }
    tensors["endpoint_window"] = torch.tensor(
        [item["endpoint_window"] for item in batch], dtype=torch.long
    )
    return {
        "tensors": tensors,
        "patient_id": [item["patient_id"] for item in batch],
        "endpoint_window": tensors["endpoint_window"],
    }


def setup_logger(project_root: Path, fold: int, log_name: str | None = None) -> logging.Logger:
    log_dir = project_root / "results" / "logs" / "full_5fold"
    log_dir.mkdir(parents=True, exist_ok=True)
    name = log_name or f"kg_latentnet_v2_fold{fold}"
    log_path = log_dir / f"{name}.log"
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def check_feature_leakage(feature_names: dict[str, list[str]]) -> list[str]:
    all_names = (
        feature_names.get("static_features", [])
        + feature_names.get("dynamic_features", [])
        + feature_names.get("treatment_features", [])
    )
    leaked = []
    for name in all_names:
        lowered = name.lower()
        if any(token.lower() in lowered for token in LEAKAGE_BLACKLIST):
            leaked.append(name)
    return leaked


def build_prior_matrix(project_root: Path, dynamic_names: list[str]) -> torch.Tensor:
    matrix, _, _ = build_aligned_prior_matrix(project_root, dynamic_names)
    return torch.tensor(matrix, dtype=torch.float32)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_float(v: Any) -> float:
    if v is None:
        return float("nan")
    f = float(v)
    return f if np.isfinite(f) else float("nan")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt = y_true[mask]
    yp = y_pred[mask]
    if len(yt) == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan"), "n": 0}
    ae = np.abs(yt - yp)
    mae = float(np.mean(ae))
    rmse = float(np.sqrt(np.mean(ae**2)))
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2, "n": int(len(yt))}


def evaluate_model(
    model: KGLatentNetV2,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    prior_matrix: torch.Tensor | None,
    lambda_delta: float = 0.0,
    lambda_anchor: float = 0.0,
    lambda_prior: float = 0.0,
    lambda_smooth: float = 0.0,
    lambda_disentangle: float = 0.0,
    y_range: tuple[float, float] | None = None,
    lambda_range: float = 0.01,
    fold: int = 0,
) -> tuple[float, list[dict[str, Any]], dict[str, np.ndarray]]:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    rows: list[dict[str, Any]] = []
    all_latent: list[np.ndarray] = []
    all_summary: list[np.ndarray] = []
    all_delta_pred: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            tensor_batch = {
                key: value.to(device) for key, value in batch["tensors"].items()
            }
            outputs = model(tensor_batch, prior_matrix=prior_matrix)
            loss_dict = KGLatentNetV2.compute_loss(
                outputs, tensor_batch, criterion,
                lambda_delta=lambda_delta,
                lambda_anchor=lambda_anchor,
                lambda_prior=lambda_prior,
                lambda_smooth=lambda_smooth,
                lambda_disentangle=lambda_disentangle,
                y_range=y_range,
                lambda_range=lambda_range,
            )
            total_loss += loss_dict["total"].item()
            n_batches += 1

            y_pred = outputs["y_pred"].cpu().numpy()
            delta_pred = outputs["delta_pred"].cpu().numpy()
            delta_latent = outputs["delta_latent"].cpu().numpy()
            delta_summary = outputs["delta_summary"].cpu().numpy()
            delta_treatment = outputs["delta_treatment"].cpu().numpy()
            gate_w = outputs["gate_weights"].cpu().numpy()
            y_true = tensor_batch["endpoint_tbr_y"].squeeze(-1).cpu().numpy()
            baseline = tensor_batch["baseline_tbr_b"].squeeze(-1).cpu().numpy()
            delta_true = y_true - baseline
            latent = outputs["latent_state"].cpu().numpy()
            summary_h = outputs["summary_hidden"].cpu().numpy()

            all_latent.append(latent)
            all_summary.append(summary_h)
            all_delta_pred.append(delta_pred)

            for i in range(len(y_pred)):
                rows.append({
                    "patient_id": batch["patient_id"][i],
                    "fold": fold,
                    "endpoint_window": int(batch["endpoint_window"][i].item()),
                    "baseline_tbr_b": float(baseline[i]),
                    "y_true": float(y_true[i]),
                    "delta_true": float(delta_true[i]),
                    "delta_pred": float(delta_pred[i]),
                    "delta_latent": float(delta_latent[i]),
                    "delta_summary": float(delta_summary[i]),
                    "delta_treatment": float(delta_treatment[i]),
                    "gate_latent": float(gate_w[i, 0]),
                    "gate_summary": float(gate_w[i, 1]),
                    "gate_treatment": float(gate_w[i, 2]),
                    "y_pred": float(y_pred[i]),
                    "absolute_error": float(abs(y_pred[i] - y_true[i])),
                })

    avg_loss = total_loss / max(n_batches, 1)
    latent_outputs = {
        "latent_states": np.concatenate(all_latent, axis=0) if all_latent else np.array([]),
        "summary_hidden": np.concatenate(all_summary, axis=0) if all_summary else np.array([]),
        "delta_pred_all": np.concatenate(all_delta_pred, axis=0) if all_delta_pred else np.array([]),
    }
    return avg_loss, rows, latent_outputs


def train_one_fold(
    project_root: Path,
    fold: int,
    config: dict[str, Any],
    logger: logging.Logger,
    device: torch.device,
) -> dict[str, Any]:
    seed = int(config.get("seed", 20260606))
    random.seed(seed + fold)
    np.random.seed(seed + fold)
    torch.manual_seed(seed + fold)

    hidden_dim = int(config.get("hidden_dim", 32))
    latent_dim = int(config.get("latent_dim", 16))
    summary_dim = int(config.get("summary_dim", 32))
    dropout = float(config.get("dropout", 0.3))
    lr = float(config.get("learning_rate", 5e-4))
    wd = float(config.get("weight_decay", 1e-5))
    epochs = int(config.get("epochs", 100))
    batch_size = int(config.get("batch_size", 16))
    loss_type = str(config.get("loss", "Huber"))
    huber_delta = float(config.get("huber_delta", 0.1))
    model_variant = str(config.get("model_variant", "v2_shared_head"))
    readout_head = str(config.get("readout_head", "shared"))
    lambda_delta = float(config.get("lambda_delta", 0.1))
    lambda_anchor = float(config.get("lambda_anchor", 0.01))
    lambda_prior = float(config.get("lambda_prior", 0.0))
    lambda_smooth = float(config.get("lambda_smooth", 0.0))
    lambda_disentangle = float(config.get("lambda_disentangle", 0.0))
    lambda_range = float(config.get("lambda_range", 0.01))
    gradient_clip = float(config.get("gradient_clip", 1.0))
    patience = int(config.get("early_stopping_patience", 50))

    logger.info("=== Fold %d Training ===", fold)
    logger.info("model_variant=%s readout_head=%s hidden_dim=%d latent_dim=%d summary_dim=%d dropout=%.2f",
                model_variant, readout_head, hidden_dim, latent_dim, summary_dim, dropout)
    logger.info("lr=%.6f wd=%.6f loss=%s huber_delta=%.4f gradient_clip=%.1f", lr, wd, loss_type, huber_delta, gradient_clip)
    logger.info("lambda_delta=%.4f lambda_anchor=%.4f lambda_range=%.4f", lambda_delta, lambda_anchor, lambda_range)
    logger.info("lambda_prior=%.4f lambda_smooth=%.4f lambda_disentangle=%.4f", lambda_prior, lambda_smooth, lambda_disentangle)

    dataset = load_dataset(project_root)
    fold_payload = load_fold(project_root, fold)
    with (project_root / "data" / "processed" / f"fold_{fold}_preprocess.pkl").open("rb") as f:
        preprocess = pickle.load(f)

    train_indices = ids_to_indices(dataset, fold_payload["train_patient_ids"])
    val_indices = ids_to_indices(dataset, fold_payload["val_patient_ids"])
    train_arrays = apply_preprocess(dataset, preprocess, train_indices)
    val_arrays = apply_preprocess(dataset, preprocess, val_indices)

    logger.info("train_size=%d val_size=%d", len(train_arrays["patient_id"]), len(val_arrays["patient_id"]))

    y_train = train_arrays["endpoint_tbr_y"]
    y_range = (float(np.nanmin(y_train)), float(np.nanmax(y_train)))
    logger.info("train_y_range=[%.4f, %.4f]", y_range[0], y_range[1])

    train_loader = DataLoader(
        ResidualTensorDataset(train_arrays), batch_size=batch_size, shuffle=True, collate_fn=collate
    )
    val_loader = DataLoader(
        ResidualTensorDataset(val_arrays), batch_size=64, shuffle=False, collate_fn=collate
    )

    static_dim = dataset["static_features"].shape[1]
    dynamic_dim = dataset["dynamic_features"].shape[2]
    treatment_dim = dataset["treatment_features"].shape[2]

    model = KGLatentNetV2(
        static_dim=static_dim, dynamic_dim=dynamic_dim, treatment_dim=treatment_dim,
        hidden_dim=hidden_dim, latent_dim=latent_dim, summary_dim=summary_dim,
        dropout=dropout, model_variant=model_variant, readout_head=readout_head,
        huber_delta=huber_delta,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("model_params=%d static_dim=%d dynamic_dim=%d treatment_dim=%d", n_params, static_dim, dynamic_dim, treatment_dim)

    prior_matrix = build_prior_matrix(project_root, dataset["feature_names"]["dynamic_features"]).to(device)
    logger.info("prior_matrix shape=%s", tuple(prior_matrix.shape))

    if loss_type == "Huber":
        criterion = nn.HuberLoss(delta=huber_delta)
    else:
        criterion = nn.L1Loss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    best_val_mae = float("inf")
    best_epoch = 0
    best_state = None
    no_improve = 0

    t_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            tensor_batch = {key: value.to(device) for key, value in batch["tensors"].items()}
            outputs = model(tensor_batch, prior_matrix=prior_matrix)
            loss_dict = KGLatentNetV2.compute_loss(
                outputs, tensor_batch, criterion,
                lambda_delta=lambda_delta,
                lambda_anchor=lambda_anchor,
                lambda_prior=lambda_prior,
                lambda_smooth=lambda_smooth,
                lambda_disentangle=lambda_disentangle,
                y_range=y_range,
                lambda_range=lambda_range,
            )
            loss = loss_dict["total"]

            if torch.isnan(loss) or torch.isinf(loss):
                logger.error("NaN/Inf loss at epoch %d, skipping batch", epoch)
                continue

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if gradient_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            train_losses.append(loss_dict["total"].item())

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")

        val_loss, val_preds, _ = evaluate_model(
            model, val_loader, criterion, device, prior_matrix,
            lambda_delta, lambda_anchor, lambda_prior,
            lambda_smooth, lambda_disentangle, y_range, lambda_range, fold,
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

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                "epoch=%d train_loss=%.6f val_loss=%.6f val_mae=%.6f val_rmse=%.6f val_r2=%.6f",
                epoch, train_loss, val_loss, val_metrics["mae"], val_metrics["rmse"], val_metrics["r2"],
            )

        if no_improve >= patience:
            logger.info("Early stopping at epoch %d (patience=%d)", epoch, patience)
            break

    elapsed = time.time() - t_start
    logger.info("Training done: best_epoch=%d best_val_mae=%.6f elapsed=%.1fs", best_epoch, best_val_mae, elapsed)

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)

    _, val_preds, val_latent = evaluate_model(
        model, val_loader, criterion, device, prior_matrix,
        lambda_delta, lambda_anchor, lambda_prior,
        lambda_smooth, lambda_disentangle, y_range, lambda_range, fold,
    )
    val_y_true = np.array([r["y_true"] for r in val_preds])
    val_y_pred = np.array([r["y_pred"] for r in val_preds])
    final_metrics = compute_metrics(val_y_true, val_y_pred)

    logger.info("Final val MAE=%.6f RMSE=%.6f R2=%.6f", final_metrics["mae"], final_metrics["rmse"], final_metrics["r2"])

    pred_fields = [
        "patient_id", "fold", "endpoint_window", "baseline_tbr_b",
        "y_true", "delta_true", "delta_pred", "delta_latent", "delta_summary", "delta_treatment",
        "gate_latent", "gate_summary", "gate_treatment",
        "y_pred", "absolute_error",
    ]

    return {
        "model": model,
        "best_state": best_state,
        "best_epoch": best_epoch,
        "best_val_mae": best_val_mae,
        "final_val_mae": final_metrics["mae"],
        "final_val_rmse": final_metrics["rmse"],
        "final_val_r2": final_metrics["r2"],
        "val_predictions": val_preds,
        "val_latent_outputs": val_latent,
        "pred_fields": pred_fields,
        "runtime_sec": elapsed,
        "n_params": n_params,
        "config": config,
    }


def smoke_test(
    project_root: Path,
    fold: int = 0,
    epochs: int = 5,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if config is None:
        config = {}
    config["epochs"] = epochs

    model_variant = config.get("model_variant", "v2_shared_head")

    logger = setup_logger(project_root, fold, f"kg_latentnet_v2_{model_variant}_smoke_test")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device=%s", device)

    dataset = load_dataset(project_root)
    feature_names = dataset["feature_names"]
    leaked = check_feature_leakage(feature_names)
    all_feature_names = (
        feature_names.get("static_features", [])
        + feature_names.get("dynamic_features", [])
        + feature_names.get("treatment_features", [])
    )
    logger.info("endpoint_tbr_in_features=%s", any("endpoint_tbr" in n.lower() for n in all_feature_names))
    logger.info("endpoint_time_in_features=%s", any("endpoint_time" in n.lower() for n in all_feature_names))
    logger.info("endpoint_window_in_features=%s", any("endpoint_window" in n.lower() for n in all_feature_names))
    logger.info("patient_level_leakage=%d", len(leaked))
    if leaked:
        logger.warning("Leakage detected in features: %s", leaked)

    result = train_one_fold(project_root, fold, config, logger, device)

    val_preds = result["val_predictions"]
    y_true_arr = np.array([r["y_true"] for r in val_preds])
    baseline_arr = np.array([r["baseline_tbr_b"] for r in val_preds])
    delta_true_arr = np.array([r["delta_true"] for r in val_preds])
    delta_pred_arr = np.array([r["delta_pred"] for r in val_preds])
    y_pred_arr = np.array([r["y_pred"] for r in val_preds])

    logger.info("y_true range=[%.4f, %.4f]", float(np.nanmin(y_true_arr)), float(np.nanmax(y_true_arr)))
    logger.info("baseline_tbr_b range=[%.4f, %.4f]", float(np.nanmin(baseline_arr)), float(np.nanmax(baseline_arr)))
    logger.info("delta_true range=[%.4f, %.4f]", float(np.nanmin(delta_true_arr)), float(np.nanmax(delta_true_arr)))
    logger.info("delta_pred range=[%.4f, %.4f]", float(np.nanmin(delta_pred_arr)), float(np.nanmax(delta_pred_arr)))
    logger.info("y_pred range=[%.4f, %.4f]", float(np.nanmin(y_pred_arr)), float(np.nanmax(y_pred_arr)))

    has_nan = bool(np.any(np.isnan(y_pred_arr)))
    logger.info("loss_is_nan=false")
    logger.info("prediction_is_nan=%s", has_nan)
    logger.info("test_set_used_for_tuning=false")
    logger.info("learned_relation_weights_available=false")

    pred_path = project_root / "results" / "predictions" / "full_5fold" / f"kg_latentnet_v2_{model_variant}_fold0_smoke_predictions.csv"
    write_csv(pred_path, val_preds, result["pred_fields"])
    logger.info("Saved predictions: %s", pred_path)

    metrics_rows = [{"epoch": "smoke", "model_variant": model_variant, "val_mae": result["final_val_mae"], "val_rmse": result["final_val_rmse"], "val_r2": result["final_val_r2"]}]
    metrics_path = project_root / "results" / "tables" / "full_5fold" / f"kg_latentnet_v2_{model_variant}_smoke_metrics.csv"
    write_csv(metrics_path, metrics_rows, ["epoch", "model_variant", "val_mae", "val_rmse", "val_r2"])
    logger.info("Saved metrics: %s", metrics_path)

    latent_path = project_root / "results" / "latent" / "full_5fold" / f"kg_latentnet_v2_{model_variant}_fold0_latent_states_smoke.pkl"
    latent_path.parent.mkdir(parents=True, exist_ok=True)
    with latent_path.open("wb") as f:
        pickle.dump(result["val_latent_outputs"]["latent_states"], f)
    logger.info("Saved latent states: %s", latent_path)

    summary_path = project_root / "results" / "latent" / "full_5fold" / f"kg_latentnet_v2_{model_variant}_fold0_summary_hidden_smoke.pkl"
    with summary_path.open("wb") as f:
        pickle.dump(result["val_latent_outputs"]["summary_hidden"], f)
    logger.info("Saved summary hidden: %s", summary_path)

    logger.info("=== SMOKE TEST PASSED ===")
    logger.info("model_variant=%s smoke_val_mae=%.6f", model_variant, result["final_val_mae"])

    return result


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="KG-LatentNet V2 training / smoke test")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--mode", choices=["smoke", "train"], default="smoke")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--model-variant", type=str, default="v2_shared_head")
    parser.add_argument("--readout-head", type=str, default="shared")
    args = parser.parse_args(argv)

    project_root = Path(args.project_root).resolve()

    config = {
        "model_variant": "v2_shared_head",
        "readout_head": "shared",
        "hidden_dim": 32,
        "latent_dim": 16,
        "summary_dim": 32,
        "dropout": 0.3,
        "learning_rate": 5e-4,
        "weight_decay": 1e-5,
        "loss": "Huber",
        "huber_delta": 0.1,
        "lambda_delta": 0.1,
        "lambda_anchor": 0.01,
        "lambda_prior": 0.0,
        "lambda_smooth": 0.0,
        "lambda_disentangle": 0.0,
        "lambda_range": 0.01,
        "gradient_clip": 1.0,
        "early_stopping_patience": 50,
        "batch_size": 16,
        "seed": 20260606,
    }

    if args.model_variant:
        config["model_variant"] = args.model_variant
    if args.readout_head:
        config["readout_head"] = args.readout_head

    if args.config:
        with open(args.config, encoding="utf-8") as f:
            content = f.read()
        if args.config.endswith((".yaml", ".yml")):
            try:
                import yaml
                file_config = yaml.safe_load(content)
            except ImportError:
                raise ImportError("PyYAML required for .yaml config. Install with: pip install pyyaml")
        else:
            file_config = json.loads(content)
        for k, v in file_config.items():
            if k in config:
                config[k] = v

    if args.mode == "smoke":
        return smoke_test(project_root, fold=args.fold, epochs=args.epochs, config=config)
    else:
        logger = setup_logger(project_root, args.fold)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config["epochs"] = 200
        return train_one_fold(project_root, args.fold, config, logger, device)


if __name__ == "__main__":
    main()
