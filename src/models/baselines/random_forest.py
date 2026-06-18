from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestRegressor


def build_random_forest_regressor(params: dict[str, Any] | None = None) -> RandomForestRegressor:
    config: dict[str, Any] = {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "bootstrap": True,
        "n_jobs": -1,
        "random_state": 20260605,
    }
    if params:
        config.update(params)
    return RandomForestRegressor(**config)
