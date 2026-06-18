from __future__ import annotations

import argparse
import csv
import logging
import math
import pickle
import random
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import apply_preprocess, ids_to_indices, load_dataset, load_fold  # noqa: E402
from src.data.prior_alignment import build_aligned_prior_matrix  # noqa: E402
from src.models.baselines import BASELINE_MODEL_REGISTRY  # noqa: E402
from src.models.modules.losses import EndpointMSELoss  # noqa: E402
from src.training.evaluate import evaluate_model  # noqa: E402
from src.training.metrics import regression_metrics  # noqa: E402


DEFAULT_BASELINE_ORDER = ["hyperimts", "trans", "tgnn4i", "dhgas", "kedgn", "graphcare"]


class BaselineTensorDataset(Dataset):
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
            "patient_id": str(self.arrays["patient_id"][idx]),
            "endpoint_window": int(self.arrays["endpoint_window"][idx]),
            "endpoint_time": str(self.arrays["endpoint_time"][idx]),
        }


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    keys = batch[0]["tensors"].keys()
    tensors = {key: torch.stack([item["tensors"][key] for item in batch], dim=0) for key in keys}
    return {
        "tensors": tensors,
        "patient_id": [item["patient_id"] for item in batch],
        "endpoint_window": torch.tensor([item["endpoint_window"] for item in batch], dtype=torch.long),
        "endpoint_time": [item["endpoint_time"] for item in batch],
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def setup_baseline_logger(project_root: Path, baseline_name: str) -> logging.Logger:
    log_dir = project_root / "results" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"baseline_smoke.{baseline_name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_dir / f"{baseline_name}_smoke_test.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def load_baseline_config(project_root: Path) -> dict[str, Any]:
    config_path = project_root / "configs" / "baselines.yaml"
    if not config_path.exists():
        return {}
    return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}


def check_feature_leakage(dataset: dict[str, Any]) -> bool:
    blacklist = [str(token).lower() for token in dataset.get("leakage_blacklist", [])]
    feature_names = dataset["feature_names"]
    names = (
        feature_names.get("static_features", [])
        + feature_names.get("dynamic_features", [])
        + feature_names.get("treatment_features", [])
        + feature_names.get("baseline_tbr_b", [])
    )
    for name in names:
        lowered = str(name).lower()
        if any(token and token in lowered for token in blacklist):
            return True
    return False


def split_leakage_counts(fold_payload: dict[str, Any]) -> dict[str, int]:
    train = set(fold_payload["train_patient_ids"])
    val = set(fold_payload["val_patient_ids"])
    test = set(fold_payload["test_patient_ids"])
    return {
        "train_val_overlap": len(train & val),
        "train_test_overlap": len(train & test),
        "val_test_overlap": len(val & test),
    }


def tensor_shape_summary(batch: dict[str, Any]) -> dict[str, str]:
    return {key: str(tuple(value.shape)) for key, value in batch["tensors"].items()}


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def run_one_baseline(
    project_root: Path,
    baseline_name: str,
    baseline_cfg: dict[str, Any],
    dataset: dict[str, Any],
    fold_payload: dict[str, Any],
    train_arrays: dict[str, Any],
    val_arrays: dict[str, Any],
    prior_matrix: torch.Tensor,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    logger = setup_baseline_logger(project_root, baseline_name)
    implementation = baseline_cfg.get("implementation", {})
    official_code_used = bool(implementation.get("official_code_used", False))
    public_implementation_used = bool(implementation.get("public_implementation_used", False))
    faithful_reimplementation = bool(implementation.get("faithful_reimplementation", True))
    official_code_found = bool(implementation.get("official_code_found", False))
    public_implementation_found = bool(implementation.get("public_implementation_found", False))
    adapter_completed = baseline_name in BASELINE_MODEL_REGISTRY
    used_inputs = baseline_cfg.get("used_inputs", [])
    usage_note = str(baseline_cfg.get("usage_note", ""))

    logger.info("baseline_name=%s", baseline_name)
    logger.info("official_code_used=%s", bool_text(official_code_used))
    logger.info("public_implementation_used=%s", bool_text(public_implementation_used))
    logger.info("faithful_reimplementation=%s", bool_text(faithful_reimplementation))
    logger.info("official_code_found=%s", bool_text(official_code_found))
    logger.info("public_implementation_found=%s", bool_text(public_implementation_found))
    logger.info("adapter_completed=%s", bool_text(adapter_completed))
    logger.info("used_inputs=%s", ",".join(used_inputs))
    logger.info("usage_note=%s", usage_note)

    train_loader = DataLoader(
        BaselineTensorDataset(train_arrays),
        batch_size=int(baseline_cfg.get("batch_size", 32)),
        shuffle=True,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        BaselineTensorDataset(val_arrays),
        batch_size=int(baseline_cfg.get("eval_batch_size", 64)),
        shuffle=False,
        collate_fn=collate,
    )
    first_batch = next(iter(train_loader))
    for key, shape in tensor_shape_summary(first_batch).items():
        logger.info("%s shape=%s", key, shape)
    logger.info("endpoint_window shape=%s", tuple(first_batch["endpoint_window"].shape))
    logger.info("endpoint_time count_in_batch=%d", len(first_batch["endpoint_time"]))
    logger.info(
        "train_patient_count=%d val_patient_count=%d test_patient_count=%d",
        len(fold_payload["train_patient_ids"]),
        len(fold_payload["val_patient_ids"]),
        len(fold_payload["test_patient_ids"]),
    )

    feature_leakage = check_feature_leakage(dataset)
    overlap = split_leakage_counts(fold_payload)
    patient_leakage = any(value > 0 for value in overlap.values())
    logger.info("endpoint_tbr_in_features=%s", bool_text(feature_leakage))
    logger.info("patient_level_leakage=%s", bool_text(patient_leakage))
    if feature_leakage:
        raise RuntimeError("Endpoint TBR leakage detected in baseline input feature names.")
    if patient_leakage:
        raise RuntimeError(f"Patient-level split leakage detected: {overlap}")

    static_dim = int(dataset["static_features"].shape[1])
    dynamic_dim = int(dataset["dynamic_features"].shape[2])
    treatment_dim = int(dataset["treatment_features"].shape[2])
    model_cls = BASELINE_MODEL_REGISTRY[baseline_name]
    model = model_cls(
        static_dim=static_dim,
        dynamic_dim=dynamic_dim,
        treatment_dim=treatment_dim,
        hidden_dim=int(baseline_cfg.get("hidden_dim", 64)),
    ).to(device)
    criterion = EndpointMSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(baseline_cfg.get("learning_rate", 1e-3)))
    epochs = int(baseline_cfg.get("epochs", 3))
    logger.info("fold=0 epochs=%d device=%s prior_matrix_shape=%s", epochs, device, tuple(prior_matrix.shape))

    metrics_rows: list[dict[str, Any]] = []
    loss_nan_detected = False
    val_predictions: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            tensor_batch = {key: value.to(device) for key, value in batch["tensors"].items()}
            y = tensor_batch["endpoint_tbr_y"]
            optimizer.zero_grad(set_to_none=True)
            pred = model(tensor_batch, prior_matrix=prior_matrix)
            loss = criterion(pred, y)
            if torch.isnan(loss) or torch.isinf(loss):
                loss_nan_detected = True
                raise RuntimeError(f"NaN/Inf loss detected for {baseline_name} at epoch {epoch}.")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        train_loss = float(np.mean(losses)) if losses else math.nan
        val_loss, val_predictions = evaluate_model(model, val_loader, criterion, device, prior_matrix=prior_matrix)
        if not math.isfinite(val_loss):
            loss_nan_detected = True
            raise RuntimeError(f"NaN/Inf validation loss detected for {baseline_name} at epoch {epoch}.")
        val_metric = regression_metrics(val_predictions)
        logger.info(
            "epoch=%d train_loss=%.6f val_loss=%.6f val_mae=%.6f val_rmse=%.6f",
            epoch,
            train_loss,
            val_loss,
            val_metric["mae"],
            val_metric["rmse"],
        )
        metrics_rows.append(
            {
                "baseline_name": baseline_name,
                "fold": 0,
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mae": val_metric["mae"],
                "val_rmse": val_metric["rmse"],
                "n_val": int(val_metric["n"]),
            }
        )

    prediction_nan = any((not math.isfinite(float(row["y_pred"]))) for row in val_predictions)
    expected_prediction_columns = {"patient_id", "endpoint_window", "y_true", "y_pred", "absolute_error"}
    prediction_columns_ok = bool(val_predictions) and expected_prediction_columns.issubset(val_predictions[0].keys())
    preds_path = project_root / "results" / "predictions" / f"{baseline_name}_fold0_smoke_predictions.csv"
    write_csv(preds_path, val_predictions, ["patient_id", "endpoint_window", "y_true", "y_pred", "absolute_error"])
    logger.info("loss_nan_detected=%s", bool_text(loss_nan_detected))
    logger.info("prediction_nan_detected=%s", bool_text(prediction_nan))
    logger.info("prediction_file_saved=%s", bool_text(preds_path.exists()))
    logger.info("prediction_path=%s", preds_path)
    logger.info("prediction_columns=patient_id,endpoint_window,y_true,y_pred,absolute_error")
    logger.info("prediction_columns_ok=%s", bool_text(prediction_columns_ok))
    if prediction_nan:
        raise RuntimeError(f"NaN/Inf prediction detected for {baseline_name}.")
    if not prediction_columns_ok:
        raise RuntimeError(f"Prediction columns missing for {baseline_name}.")

    status_row = {
        "baseline_name": baseline_name,
        "status": "success",
        "reason": "fold0_smoke_test_completed",
        "failed_reason": "",
        "error_message": "",
        "official_code_used": official_code_used,
        "public_implementation_used": public_implementation_used,
        "faithful_reimplementation": faithful_reimplementation,
        "whether_official_code_found": official_code_found,
        "whether_public_implementation_found": public_implementation_found,
        "whether_adapter_completed": adapter_completed,
        "next_fix_plan": "",
    }
    usage_row = {
        "baseline_name": baseline_name,
        "patient_id": True,
        "static_features": "static_features" in used_inputs,
        "dynamic_features": "dynamic_features" in used_inputs,
        "dynamic_mask": "dynamic_mask" in used_inputs,
        "delta_time": "delta_time" in used_inputs,
        "treatment_features": "treatment_features" in used_inputs,
        "baseline_tbr_b": "baseline_tbr_b" in used_inputs,
        "endpoint_tbr_y": "loss_only",
        "endpoint_window": "metadata_only",
        "endpoint_time": "metadata_only",
        "prior_matrix": "prior_matrix" in used_inputs,
        "usage_note": usage_note,
    }
    leakage_row = {
        "baseline_name": baseline_name,
        "endpoint_tbr_in_features": feature_leakage,
        "patient_level_leakage": patient_leakage,
        "train_val_overlap": overlap["train_val_overlap"],
        "train_test_overlap": overlap["train_test_overlap"],
        "val_test_overlap": overlap["val_test_overlap"],
        "leakage_blacklist_passed": not feature_leakage,
        "passed": (not feature_leakage) and (not patient_leakage),
    }
    return metrics_rows, status_row, usage_row, leakage_row


def failed_rows(
    baseline_name: str,
    baseline_cfg: dict[str, Any],
    exc: BaseException,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    implementation = baseline_cfg.get("implementation", {})
    error_message = traceback.format_exc()
    status_row = {
        "baseline_name": baseline_name,
        "status": "failed",
        "reason": "smoke_test_failed",
        "failed_reason": str(exc),
        "error_message": error_message,
        "official_code_used": bool(implementation.get("official_code_used", False)),
        "public_implementation_used": bool(implementation.get("public_implementation_used", False)),
        "faithful_reimplementation": bool(implementation.get("faithful_reimplementation", True)),
        "whether_official_code_found": bool(implementation.get("official_code_found", False)),
        "whether_public_implementation_found": bool(implementation.get("public_implementation_found", False)),
        "whether_adapter_completed": baseline_name in BASELINE_MODEL_REGISTRY,
        "next_fix_plan": baseline_cfg.get("next_fix_plan", "Inspect traceback and adapt model/data interface without substituting a placeholder."),
    }
    usage_row = {
        "baseline_name": baseline_name,
        "patient_id": True,
        "static_features": "static_features" in baseline_cfg.get("used_inputs", []),
        "dynamic_features": "dynamic_features" in baseline_cfg.get("used_inputs", []),
        "dynamic_mask": "dynamic_mask" in baseline_cfg.get("used_inputs", []),
        "delta_time": "delta_time" in baseline_cfg.get("used_inputs", []),
        "treatment_features": "treatment_features" in baseline_cfg.get("used_inputs", []),
        "baseline_tbr_b": "baseline_tbr_b" in baseline_cfg.get("used_inputs", []),
        "endpoint_tbr_y": "loss_only",
        "endpoint_window": "metadata_only",
        "endpoint_time": "metadata_only",
        "prior_matrix": "prior_matrix" in baseline_cfg.get("used_inputs", []),
        "usage_note": baseline_cfg.get("usage_note", ""),
    }
    leakage_row = {
        "baseline_name": baseline_name,
        "endpoint_tbr_in_features": "",
        "patient_level_leakage": "",
        "train_val_overlap": "",
        "train_test_overlap": "",
        "val_test_overlap": "",
        "leakage_blacklist_passed": "",
        "passed": False,
    }
    return status_row, usage_row, leakage_row


def run_baseline_smoke_tests(project_root: Path, fold: int = 0, seed: int = 20260605) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    project_root = project_root.resolve()
    config = load_baseline_config(project_root)
    baseline_configs = config.get("baselines", {})
    baseline_names = [name for name in config.get("order", DEFAULT_BASELINE_ORDER) if name in DEFAULT_BASELINE_ORDER]
    if not baseline_names:
        baseline_names = DEFAULT_BASELINE_ORDER

    dataset = load_dataset(project_root)
    fold_payload = load_fold(project_root, fold)
    with (project_root / "data" / "processed" / f"fold_{fold}_preprocess.pkl").open("rb") as handle:
        preprocess = pickle.load(handle)
    train_indices = ids_to_indices(dataset, fold_payload["train_patient_ids"])
    val_indices = ids_to_indices(dataset, fold_payload["val_patient_ids"])
    train_arrays = apply_preprocess(dataset, preprocess, train_indices)
    val_arrays = apply_preprocess(dataset, preprocess, val_indices)

    prior_np, _, prior_checks = build_aligned_prior_matrix(project_root, dataset["feature_names"]["dynamic_features"])
    if not all(bool(row["passed"]) for row in prior_checks):
        raise RuntimeError("Prior alignment checks failed; refusing to run baseline smoke tests.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prior_matrix = torch.tensor(prior_np, dtype=torch.float32, device=device)

    all_metrics: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    usage_rows: list[dict[str, Any]] = []
    leakage_rows: list[dict[str, Any]] = []
    for baseline_name in baseline_names:
        baseline_cfg = baseline_configs.get(baseline_name, {})
        logger = setup_baseline_logger(project_root, baseline_name)
        try:
            metrics_rows, status_row, usage_row, leakage_row = run_one_baseline(
                project_root=project_root,
                baseline_name=baseline_name,
                baseline_cfg=baseline_cfg,
                dataset=dataset,
                fold_payload=fold_payload,
                train_arrays=train_arrays,
                val_arrays=val_arrays,
                prior_matrix=prior_matrix,
                device=device,
            )
            all_metrics.extend(metrics_rows)
            status_rows.append(status_row)
            usage_rows.append(usage_row)
            leakage_rows.append(leakage_row)
        except Exception as exc:
            logger.exception("baseline smoke test failed for %s", baseline_name)
            status_row, usage_row, leakage_row = failed_rows(baseline_name, baseline_cfg, exc)
            status_rows.append(status_row)
            usage_rows.append(usage_row)
            leakage_rows.append(leakage_row)

    tables_dir = project_root / "results" / "tables"
    write_csv(
        tables_dir / "baseline_smoke_test_metrics.csv",
        all_metrics,
        ["baseline_name", "fold", "epoch", "train_loss", "val_loss", "val_mae", "val_rmse", "n_val"],
    )
    write_csv(
        tables_dir / "baseline_implementation_status.csv",
        status_rows,
        [
            "baseline_name",
            "status",
            "reason",
            "failed_reason",
            "error_message",
            "official_code_used",
            "public_implementation_used",
            "faithful_reimplementation",
            "whether_official_code_found",
            "whether_public_implementation_found",
            "whether_adapter_completed",
            "next_fix_plan",
        ],
    )
    write_csv(
        tables_dir / "baseline_input_usage.csv",
        usage_rows,
        [
            "baseline_name",
            "patient_id",
            "static_features",
            "dynamic_features",
            "dynamic_mask",
            "delta_time",
            "treatment_features",
            "baseline_tbr_b",
            "endpoint_tbr_y",
            "endpoint_window",
            "endpoint_time",
            "prior_matrix",
            "usage_note",
        ],
    )
    write_csv(
        tables_dir / "baseline_leakage_check.csv",
        leakage_rows,
        [
            "baseline_name",
            "endpoint_tbr_in_features",
            "patient_level_leakage",
            "train_val_overlap",
            "train_test_overlap",
            "val_test_overlap",
            "leakage_blacklist_passed",
            "passed",
        ],
    )
    return {
        "baseline_count": len(baseline_names),
        "success_count": sum(1 for row in status_rows if row["status"] == "success"),
        "failed_count": sum(1 for row in status_rows if row["status"] == "failed"),
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Run fold_0 baseline smoke tests only.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260605)
    args = parser.parse_args(argv)
    return run_baseline_smoke_tests(Path(args.project_root), fold=args.fold, seed=args.seed)


if __name__ == "__main__":
    main()
