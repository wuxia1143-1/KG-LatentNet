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
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import apply_preprocess, ids_to_indices, load_dataset, load_fold  # noqa: E402
from src.data.prior_alignment import build_aligned_prior_matrix  # noqa: E402
from src.models.baselines.official_adapters import OFFICIAL_BASELINE_REGISTRY  # noqa: E402
from src.models.baselines.official_adapters.common import source_version  # noqa: E402
from src.models.modules.losses import EndpointMSELoss  # noqa: E402
from src.training.evaluate import evaluate_model  # noqa: E402
from src.training.metrics import regression_metrics  # noqa: E402


ORDER = ["hyperimts", "trans", "tgnn4i", "dhgas", "kedgn", "graphcare"]
DISPLAY = {"hyperimts": "HyperIMTS", "trans": "TRANS", "tgnn4i": "TGNN4I", "dhgas": "DHGAS", "kedgn": "KEDGN", "graphcare": "GraphCare"}
ARCHIVES = {"kedgn": "KEDGN-master.zip", "graphcare": "GraphCare-main.zip", "tgnn4i": "tgnn4i-main.zip", "dhgas": "DHGAS-main.zip"}
OBSERVED_COMMITS = {"kedgn": "c82bc004cd1d1305fc58ca7150369d78682fccad", "graphcare": "fe92aa67add80b62d4b5108d570b6483ee69a5d4"}


class OfficialTensorDataset(Dataset):
    def __init__(self, arrays: dict[str, Any]) -> None:
        self.arrays = arrays
        self.n = len(arrays["patient_id"])

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict[str, Any]:
        tensor_keys = ["static_features", "dynamic_features", "dynamic_mask", "delta_time", "treatment_features", "baseline_tbr_b", "endpoint_tbr_y"]
        tensors = {key: torch.tensor(self.arrays[key][idx], dtype=torch.float32) for key in tensor_keys}
        return {
            "tensors": tensors,
            "patient_id": str(self.arrays["patient_id"][idx]),
            "endpoint_window": int(self.arrays["endpoint_window"][idx]),
            "endpoint_time": str(self.arrays["endpoint_time"][idx]),
        }


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    keys = batch[0]["tensors"].keys()
    return {
        "tensors": {key: torch.stack([item["tensors"][key] for item in batch], dim=0) for key in keys},
        "patient_id": [item["patient_id"] for item in batch],
        "endpoint_window": torch.tensor([item["endpoint_window"] for item in batch], dtype=torch.long),
        "endpoint_time": [item["endpoint_time"] for item in batch],
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str], append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    mode = "a" if append else "w"
    with path.open(mode, newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if not append or not exists:
            writer.writeheader()
        writer.writerows(rows)


def logger_for(project_root: Path, baseline: str, suffix: str) -> logging.Logger:
    log_dir = project_root / "results" / "logs"
    if suffix in {"import_test", "forward_test"}:
        log_dir = project_root / "results" / "logs" / "official_baseline_check"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_name = f"{baseline}_{suffix}.log" if suffix in {"import_test", "forward_test"} else f"{baseline}_official_smoke_test.log"
    logger = logging.getLogger(f"{baseline}.{suffix}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_dir / log_name, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def leakage_detected(dataset: dict[str, Any]) -> bool:
    blacklist = [str(token).lower() for token in dataset.get("leakage_blacklist", [])]
    names = []
    for key in ["static_features", "dynamic_features", "treatment_features", "baseline_tbr_b"]:
        names.extend(dataset["feature_names"].get(key, []))
    return any(any(token and token in str(name).lower() for token in blacklist) for name in names)


def split_leakage(fold: dict[str, Any]) -> dict[str, int]:
    train, val, test = set(fold["train_patient_ids"]), set(fold["val_patient_ids"]), set(fold["test_patient_ids"])
    return {"train_val_overlap": len(train & val), "train_test_overlap": len(train & test), "val_test_overlap": len(val & test)}


def audit_base(project_root: Path, baseline: str, adapter: Any | None = None) -> dict[str, Any]:
    display = DISPLAY[baseline]
    repo = project_root / "remote_baselines" / display
    readme = next(repo.glob("README*"), None) if repo.exists() else None
    license_path = next(repo.glob("LICENSE*"), None) if repo.exists() else None
    deps = next(repo.glob("requirements*.txt"), None) if repo.exists() else None
    if display == "DHGAS":
        deps = repo / "setup.py"
    if display == "TRANS":
        deps = repo / "README.md"
    if display == "GraphCare":
        deps = repo / "README.md"
    version = source_version(project_root, display, ARCHIVES.get(baseline), OBSERVED_COMMITS.get(baseline))
    official_repo_url = getattr(adapter, "official_repo_url", "") if adapter is not None else ""
    official_model_class = getattr(adapter, "official_model_class_used", "") if adapter is not None else ""
    official_entry = getattr(adapter, "official_entry_script", "") if adapter is not None else ""
    adapter_file = getattr(adapter, "adapter_file", f"src/models/baselines/official_adapters/{baseline}_official_adapter.py")
    return {
        "baseline_name": baseline,
        "official_repo_url": official_repo_url,
        "repo_cloned_successfully": bool((repo / ".git").exists() or repo.exists()),
        "commit_hash": version,
        "license": license_path.name if license_path else "",
        "official_readme_found": bool(readme),
        "official_entry_script": official_entry,
        "official_dependencies_file": str(deps.relative_to(project_root)) if deps and deps.exists() else "",
        "official_demo_available": bool(readme),
        "official_demo_or_import_test_success": False,
        "adapter_file": adapter_file,
        "adapter_only_for_data_mapping": True,
        "official_model_class_used": official_model_class,
        "official_model_forward_used": False,
        "modified_official_code": False,
        "patch_file": "",
        "status": "failed",
        "failure_reason": "",
    }


def reset_outputs(project_root: Path) -> None:
    tables = project_root / "results" / "tables"
    for file_name in [
        "baseline_official_code_audit.csv",
        "official_baseline_smoke_test_metrics.csv",
        "official_baseline_leakage_check.csv",
        "official_baseline_input_usage.csv",
    ]:
        path = tables / file_name
        if path.exists():
            path.unlink()


def run_one(project_root: Path, baseline: str, fold_idx: int = 0, epochs: int = 3, seed: int = 20260605) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_dataset(project_root)
    fold = load_fold(project_root, fold_idx)
    with (project_root / "data" / "processed" / f"fold_{fold_idx}_preprocess.pkl").open("rb") as handle:
        preprocess = pickle.load(handle)
    train_arrays = apply_preprocess(dataset, preprocess, ids_to_indices(dataset, fold["train_patient_ids"]))
    val_arrays = apply_preprocess(dataset, preprocess, ids_to_indices(dataset, fold["val_patient_ids"]))
    prior_np, _, prior_checks = build_aligned_prior_matrix(project_root, dataset["feature_names"]["dynamic_features"])
    if not all(bool(row["passed"]) for row in prior_checks):
        raise RuntimeError("Prior alignment failed; official smoke refused.")
    prior_matrix = torch.tensor(prior_np, dtype=torch.float32, device=device)
    static_dim = int(dataset["static_features"].shape[1])
    dynamic_dim = int(dataset["dynamic_features"].shape[2])
    treatment_dim = int(dataset["treatment_features"].shape[2])

    import_logger = logger_for(project_root, baseline, "import_test")
    forward_logger = logger_for(project_root, baseline, "forward_test")
    smoke_logger = logger_for(project_root, baseline, "official_smoke_test")
    cls = OFFICIAL_BASELINE_REGISTRY[baseline]
    audit = audit_base(project_root, baseline)
    try:
        import_logger.info("baseline_name=%s", baseline)
        import_logger.info("official_repo_url=%s", cls.official_repo_url)
        import_logger.info("adapter_file=%s", f"src/models/baselines/official_adapters/{baseline}_official_adapter.py")
        model = cls(project_root, static_dim, dynamic_dim, treatment_dim, hidden_dim=32).to(device)
        audit = audit_base(project_root, baseline, model)
        import_logger.info("official_model_class=%s", model.official_model_class_used)
        import_logger.info("commit_hash=%s", audit["commit_hash"])
        import_logger.info("import_test_success=true")

        train_loader = DataLoader(OfficialTensorDataset(train_arrays), batch_size=8, shuffle=True, collate_fn=collate)
        val_loader = DataLoader(OfficialTensorDataset(val_arrays), batch_size=16, shuffle=False, collate_fn=collate)
        first_batch = next(iter(train_loader))
        tensor_batch = {key: value.to(device) for key, value in first_batch["tensors"].items()}
        forward_logger.info("baseline_name=%s", baseline)
        forward_logger.info("official_repo_url=%s", model.official_repo_url)
        forward_logger.info("commit_hash=%s", audit["commit_hash"])
        forward_logger.info("official_model_class=%s", model.official_model_class_used)
        forward_logger.info("adapter_file=%s", model.adapter_file)
        for key, value in tensor_batch.items():
            forward_logger.info("%s shape=%s", key, tuple(value.shape))
        y = tensor_batch["endpoint_tbr_y"]
        pred = model(tensor_batch, prior_matrix=prior_matrix)
        loss = EndpointMSELoss()(pred, y)
        loss.backward()
        forward_logger.info("y_true shape=%s", tuple(y.shape))
        forward_logger.info("y_pred shape=%s", tuple(pred.shape))
        forward_logger.info("one_batch_loss=%.6f", float(loss.item()))
        forward_logger.info("loss_is_nan=%s", str(bool(torch.isnan(loss) or torch.isinf(loss))).lower())
        forward_logger.info("forward_test_success=true")

        leak = leakage_detected(dataset)
        overlap = split_leakage(fold)
        patient_leak = any(value > 0 for value in overlap.values())
        smoke_logger.info("baseline_name=%s", baseline)
        smoke_logger.info("official_repo_url=%s", model.official_repo_url)
        smoke_logger.info("commit_hash=%s", audit["commit_hash"])
        smoke_logger.info("official_model_class=%s", model.official_model_class_used)
        smoke_logger.info("adapter_file=%s", model.adapter_file)
        smoke_logger.info("train_patient_count=%d val_patient_count=%d test_patient_count=%d", len(fold["train_patient_ids"]), len(fold["val_patient_ids"]), len(fold["test_patient_ids"]))
        smoke_logger.info("endpoint_tbr_in_features=%s", str(leak).lower())
        smoke_logger.info("patient_level_leakage=%d", int(patient_leak))
        if leak or patient_leak:
            raise RuntimeError("Leakage check failed before official smoke.")

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = EndpointMSELoss()
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
                if torch.isnan(loss) or torch.isinf(loss):
                    raise RuntimeError(f"NaN/Inf loss at epoch {epoch}")
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))
            val_loss, preds = evaluate_model(model, val_loader, criterion, device, prior_matrix=prior_matrix)
            metric = regression_metrics(preds)
            smoke_logger.info("epoch=%d train_loss=%.6f val_loss=%.6f val_mae=%.6f val_rmse=%.6f", epoch, float(np.mean(losses)), val_loss, metric["mae"], metric["rmse"])
            metrics_rows.append({"baseline_name": baseline, "fold": fold_idx, "epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val_loss, "val_mae": metric["mae"], "val_rmse": metric["rmse"], "n_val": int(metric["n"])})

        pred_nan = any(not math.isfinite(float(row["y_pred"])) for row in preds)
        preds_path = project_root / "results" / "predictions" / f"{baseline}_fold0_official_smoke_predictions.csv"
        write_csv(preds_path, preds, ["patient_id", "endpoint_window", "y_true", "y_pred", "absolute_error"])
        smoke_logger.info("prediction_is_nan=%s", str(pred_nan).lower())
        smoke_logger.info("prediction_file_saved_path=%s", preds_path)
        smoke_logger.info("loss_is_nan=false")
        audit["official_demo_or_import_test_success"] = True
        audit["official_model_forward_used"] = True
        audit["status"] = "success"
        audit["failure_reason"] = ""
        write_csv(project_root / "results" / "tables" / "official_baseline_smoke_test_metrics.csv", metrics_rows, ["baseline_name", "fold", "epoch", "train_loss", "val_loss", "val_mae", "val_rmse", "n_val"], append=True)
        write_csv(project_root / "results" / "tables" / "official_baseline_leakage_check.csv", [{"baseline_name": baseline, "endpoint_tbr_in_features": leak, "patient_level_leakage": patient_leak, **overlap, "passed": (not leak) and (not patient_leak)}], ["baseline_name", "endpoint_tbr_in_features", "patient_level_leakage", "train_val_overlap", "train_test_overlap", "val_test_overlap", "passed"], append=True)
        usage = model.input_usage()
        write_csv(project_root / "results" / "tables" / "official_baseline_input_usage.csv", [usage], ["baseline_name", "patient_id", "static_features", "dynamic_features", "dynamic_mask", "delta_time", "treatment_features", "baseline_tbr_b", "endpoint_tbr_y", "endpoint_window", "endpoint_time", "prior_matrix", "usage_note"], append=True)
    except Exception as exc:
        for log in [import_logger, forward_logger, smoke_logger]:
            log.error("official baseline failed: %s", exc)
            log.error(traceback.format_exc())
        audit["failure_reason"] = str(exc)
        audit["status"] = "failed"
    write_csv(project_root / "results" / "tables" / "baseline_official_code_audit.csv", [audit], ["baseline_name", "official_repo_url", "repo_cloned_successfully", "commit_hash", "license", "official_readme_found", "official_entry_script", "official_dependencies_file", "official_demo_available", "official_demo_or_import_test_success", "adapter_file", "adapter_only_for_data_mapping", "official_model_class_used", "official_model_forward_used", "modified_official_code", "patch_file", "status", "failure_reason"], append=True)
    return audit


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Run official baseline import, forward, backward, and fold_0 smoke test.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--baseline", choices=ORDER + ["all"], required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--reset-outputs", action="store_true")
    args = parser.parse_args(argv)
    project_root = Path(args.project_root).resolve()
    if args.reset_outputs:
        reset_outputs(project_root)
    baselines = ORDER if args.baseline == "all" else [args.baseline]
    rows = [run_one(project_root, baseline, args.fold, args.epochs) for baseline in baselines]
    return {"rows": rows}


if __name__ == "__main__":
    main()
