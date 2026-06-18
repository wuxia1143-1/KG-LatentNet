from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats


def paired_absolute_error_wilcoxon(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    err_a = np.abs(np.asarray(y_pred_a, dtype=float).reshape(-1) - y_true)
    err_b = np.abs(np.asarray(y_pred_b, dtype=float).reshape(-1) - y_true)
    if err_a.shape != err_b.shape:
        raise ValueError("Predictions must have the same shape.")
    result = stats.wilcoxon(err_a, err_b, zero_method="wilcox", alternative="two-sided")
    return {
        "test": "wilcoxon_signed_rank_abs_error",
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "n": int(len(err_a)),
        "mean_abs_error_a": float(np.mean(err_a)),
        "mean_abs_error_b": float(np.mean(err_b)),
    }


def paired_permutation_mae_difference(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    n_permutations: int = 10000,
    seed: int = 20260605,
) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred_a = np.asarray(y_pred_a, dtype=float).reshape(-1)
    y_pred_b = np.asarray(y_pred_b, dtype=float).reshape(-1)
    if not (y_true.shape == y_pred_a.shape == y_pred_b.shape):
        raise ValueError("All arrays must have the same shape.")
    err_a = np.abs(y_pred_a - y_true)
    err_b = np.abs(y_pred_b - y_true)
    observed = float(np.mean(err_a) - np.mean(err_b))
    diffs = err_a - err_b
    rng = np.random.default_rng(seed)
    null_values = []
    for _ in range(n_permutations):
        signs = rng.choice([-1.0, 1.0], size=len(diffs))
        null_values.append(float(np.mean(diffs * signs)))
    p_value = float(np.mean(np.abs(null_values) >= abs(observed)))
    return {
        "test": "paired_sign_permutation_mae_diff",
        "observed_mae_diff_a_minus_b": observed,
        "p_value": p_value,
        "n": int(len(diffs)),
        "n_permutations": int(n_permutations),
        "seed": int(seed),
    }
