"""Evaluation helpers that respect time order"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def metrics(actual: pd.Series, predicted: pd.Series, capacity: float | None = None) -> dict[str, float]:
    """ Calculates error metrics for one observed and predicted series """
    actual, predicted = pd.Series(actual).astype(float), pd.Series(predicted).astype(float)
    error = actual.to_numpy() - predicted.to_numpy()

    # Return raw and normalized errors together
    result = {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
    }
    nonzero = actual != 0
    result["mape"] = float(np.mean(np.abs(error[nonzero] / actual[nonzero]))) if nonzero.any() else np.nan

    if capacity:
        result["nmae_capacity"] = result["mae"] / capacity
        result["nrmse_capacity"] = result["rmse"] / capacity
    return result


def rolling_origins(
    series: pd.Series,
    horizon: int = 52,
    origins: int = 3,
    step: int = 52,
    minimum_history: int = 104,
) -> list[tuple[pd.Series, pd.Series]]:
    """ Creates train and test splits in time order """
    values = pd.Series(series).dropna().sort_index()
    latest_origin = len(values) - horizon
    result = []

    # Walk the origins from the oldest to the newest split
    for i in range(origins - 1, -1, -1):
        end = latest_origin - i * step
        if end >= minimum_history and end + horizon <= len(values):
            result.append((values.iloc[:end], values.iloc[end:end + horizon]))
    return result
