from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    error = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return float(np.sqrt(np.mean(error**2)))


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float] = mae,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 20260605,
) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    rng = np.random.default_rng(seed)
    n = len(y_true)
    values = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        values.append(metric_fn(y_true[idx], y_pred[idx]))
    alpha = 1.0 - confidence
    return {
        "metric": metric_fn(y_true, y_pred),
        "ci_low": float(np.quantile(values, alpha / 2.0)),
        "ci_high": float(np.quantile(values, 1.0 - alpha / 2.0)),
        "n_bootstrap": int(n_bootstrap),
        "confidence": float(confidence),
        "seed": int(seed),
    }


def paired_bootstrap_difference_ci(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float] = mae,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 20260605,
) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred_a = np.asarray(y_pred_a, dtype=float).reshape(-1)
    y_pred_b = np.asarray(y_pred_b, dtype=float).reshape(-1)
    if not (y_true.shape == y_pred_a.shape == y_pred_b.shape):
        raise ValueError("All arrays must have the same shape.")
    rng = np.random.default_rng(seed)
    n = len(y_true)
    values = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        values.append(metric_fn(y_true[idx], y_pred_a[idx]) - metric_fn(y_true[idx], y_pred_b[idx]))
    alpha = 1.0 - confidence
    return {
        "metric_diff_a_minus_b": metric_fn(y_true, y_pred_a) - metric_fn(y_true, y_pred_b),
        "ci_low": float(np.quantile(values, alpha / 2.0)),
        "ci_high": float(np.quantile(values, 1.0 - alpha / 2.0)),
        "n_bootstrap": int(n_bootstrap),
        "confidence": float(confidence),
        "seed": int(seed),
    }
