from __future__ import annotations

import argparse
import csv
import json
import logging
import pickle
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import apply_preprocess, ids_to_indices, load_dataset, load_fold  # noqa: E402
from src.data.prior_alignment import build_aligned_prior_matrix  # noqa: E402
from src.models.kg_latentnet import KGLatentNet  # noqa: E402
from src.models.modules.losses import EndpointMSELoss  # noqa: E402
from src.training.evaluate import evaluate_model  # noqa: E402


LEAKAGE_BLACKLIST = ["胸主动脉TBR", "TBR值", "endpoint", "label", "目标", "随访TBR"]


class KGTensorDataset(Dataset):
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
        tensors = {key: torch.tensor(self.arrays[key][idx], dtype=torch.float32) for key in tensor_keys}
        return {
            "tensors": tensors,
            "patient_id": self.arrays["patient_id"][idx],
            "endpoint_window": int(self.arrays["endpoint_window"][idx]),
        }


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    keys = batch[0]["tensors"].keys()
    tensors = {key: torch.stack([item["tensors"][key] for item in batch], dim=0) for key in keys}
    return {
        "tensors": tensors,
        "patient_id": [item["patient_id"] for item in batch],
        "endpoint_window": torch.tensor([item["endpoint_window"] for item in batch], dtype=torch.long),
    }


def setup_logger(project_root: Path) -> logging.Logger:
    log_dir = project_root / "results" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "kg_latentnet_smoke_test.log"
    logger = logging.getLogger("kg_latentnet_smoke")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def check_feature_leakage(feature_names: dict[str, list[str]]) -> bool:
    names = feature_names.get("static_features", []) + feature_names.get("dynamic_features", []) + feature_names.get("treatment_features", [])
    for name in names:
        lowered = name.lower()
        if any(token.lower() in lowered for token in LEAKAGE_BLACKLIST):
            return True
    return False


def build_prior_matrix(project_root: Path, dynamic_names: list[str]) -> torch.Tensor:
    matrix, _, _ = build_aligned_prior_matrix(project_root, dynamic_names)
    return torch.tensor(matrix, dtype=torch.float32)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_smoke_test(project_root: Path, fold: int = 0, epochs: int = 5, seed: int = 20260605) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    logger = setup_logger(project_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Starting KG-LatentNet smoke test fold=%s epochs=%s device=%s", fold, epochs, device)

    dataset = load_dataset(project_root)
    fold_payload = load_fold(project_root, fold)
    with (project_root / "data" / "processed" / f"fold_{fold}_preprocess.pkl").open("rb") as handle:
        preprocess = pickle.load(handle)

    train_indices = ids_to_indices(dataset, fold_payload["train_patient_ids"])
    val_indices = ids_to_indices(dataset, fold_payload["val_patient_ids"])
    train_arrays = apply_preprocess(dataset, preprocess, train_indices)
    val_arrays = apply_preprocess(dataset, preprocess, val_indices)

    leakage_detected = check_feature_leakage(dataset["feature_names"])
    logger.info("endpoint_tbr_leakage_detected=%s", leakage_detected)
    if leakage_detected:
        raise RuntimeError("Endpoint TBR leakage detected in feature names.")

    train_loader = DataLoader(KGTensorDataset(train_arrays), batch_size=32, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(KGTensorDataset(val_arrays), batch_size=64, shuffle=False, collate_fn=collate)
    first_batch = next(iter(train_loader))
    for key, value in first_batch["tensors"].items():
        logger.info("%s shape=%s", key, tuple(value.shape))
    logger.info("endpoint_window shape=%s", tuple(first_batch["endpoint_window"].shape))

    static_dim = dataset["static_features"].shape[1]
    dynamic_dim = dataset["dynamic_features"].shape[2]
    treatment_dim = dataset["treatment_features"].shape[2]
    model = KGLatentNet(static_dim=static_dim, dynamic_dim=dynamic_dim, treatment_dim=treatment_dim, hidden_dim=64).to(device)
    prior_matrix = build_prior_matrix(project_root, dataset["feature_names"]["dynamic_features"]).to(device)
    logger.info("prior_matrix shape=%s", tuple(prior_matrix.shape))
    criterion = EndpointMSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    metrics_rows: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            tensor_batch = {key: value.to(device) for key, value in batch["tensors"].items()}
            y = tensor_batch["endpoint_tbr_y"]
            optimizer.zero_grad(set_to_none=True)
            pred = model(tensor_batch, prior_matrix=prior_matrix)
            loss = criterion(pred, y)
            if torch.isnan(loss):
                raise RuntimeError("NaN loss detected during smoke test.")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        train_loss = float(np.mean(losses))
        val_loss, val_predictions = evaluate_model(model, val_loader, criterion, device, prior_matrix=prior_matrix)
        logger.info("epoch=%d train_loss=%.6f val_loss=%.6f", epoch, train_loss, val_loss)
        metrics_rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

    prediction_nan = any(not np.isfinite(row["y_pred"]) for row in val_predictions)
    logger.info("prediction_file_columns=patient_id,endpoint_window,y_true,y_pred,absolute_error")
    logger.info("nan_prediction_detected=%s", prediction_nan)
    metrics_path = project_root / "results" / "tables" / "kg_latentnet_smoke_test_metrics.csv"
    preds_path = project_root / "results" / "predictions" / "kg_latentnet_fold0_smoke_predictions.csv"
    ckpt_path = project_root / "results" / "checkpoints" / "kg_latentnet_fold0_smoke.pt"
    write_csv(metrics_path, metrics_rows, ["epoch", "train_loss", "val_loss"])
    write_csv(preds_path, val_predictions, ["patient_id", "endpoint_window", "y_true", "y_pred", "absolute_error"])
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "fold": fold, "epochs": epochs}, ckpt_path)
    logger.info("saved_metrics=%s", metrics_path)
    logger.info("saved_predictions=%s", preds_path)
    logger.info("saved_checkpoint=%s", ckpt_path)
    return {"nan_prediction": prediction_nan, "epochs": epochs, "val_rows": len(val_predictions)}


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Run KG-LatentNet smoke test.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args(argv)
    return run_smoke_test(Path(args.project_root).resolve(), fold=args.fold, epochs=args.epochs)


if __name__ == "__main__":
    main()
