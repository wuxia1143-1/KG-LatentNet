from __future__ import annotations

import math
from typing import Any

import numpy as np


def regression_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {"mae": math.nan, "mse": math.nan, "rmse": math.nan, "n": 0.0}
    y_true = np.asarray([float(row["y_true"]) for row in rows], dtype=np.float64)
    y_pred = np.asarray([float(row["y_pred"]) for row in rows], dtype=np.float64)
    error = y_pred - y_true
    mse = float(np.mean(error**2))
    return {
        "mae": float(np.mean(np.abs(error))),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "n": float(len(rows)),
    }
