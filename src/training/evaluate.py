from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def evaluate_model(model, loader, criterion, device, prior_matrix=None):
    model.eval()
    losses = []
    rows = []
    for batch in loader:
        tensor_batch = {key: value.to(device) for key, value in batch["tensors"].items()}
        y = tensor_batch["endpoint_tbr_y"]
        pred = model(tensor_batch, prior_matrix=prior_matrix)
        loss = criterion(pred, y)
        losses.append(float(loss.item()))
        for patient_id, endpoint_window, y_true, y_pred in zip(
            batch["patient_id"],
            batch["endpoint_window"].cpu().numpy().tolist(),
            y.detach().cpu().numpy().reshape(-1).tolist(),
            pred.detach().cpu().numpy().reshape(-1).tolist(),
            strict=False,
        ):
            rows.append(
                {
                    "patient_id": patient_id,
                    "endpoint_window": int(endpoint_window),
                    "y_true": float(y_true),
                    "y_pred": float(y_pred),
                    "absolute_error": float(abs(y_true - y_pred)),
                }
            )
    return float(np.mean(losses)) if losses else float("nan"), rows


def evaluate_tabular_regression(model, payload: dict) -> list[dict]:
    x = payload["X"]
    y_true = np.asarray(payload["y"], dtype=np.float64).reshape(-1)
    y_pred = np.asarray(model.predict(x), dtype=np.float64).reshape(-1)
    rows = []
    for patient_id, endpoint_window, true_value, pred_value in zip(
        payload["patient_id"],
        payload["endpoint_window"],
        y_true,
        y_pred,
        strict=False,
    ):
        rows.append(
            {
                "patient_id": str(patient_id),
                "endpoint_window": int(endpoint_window),
                "y_true": float(true_value),
                "y_pred": float(pred_value),
                "absolute_error": float(abs(true_value - pred_value)),
            }
        )
    return rows
