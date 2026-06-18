from __future__ import annotations

from typing import Any


def build_xgb_regressor(params: dict[str, Any] | None = None):
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:  # pragma: no cover - exercised on the remote environment
        raise ImportError("xgboost is required for the XGBRegressor baseline; do not substitute another model.") from exc

    config: dict[str, Any] = {
        "n_estimators": 100,
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "n_jobs": -1,
        "random_state": 20260605,
    }
    if params:
        config.update(params)
    return XGBRegressor(**config)
