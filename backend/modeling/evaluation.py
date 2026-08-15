# Error metrics for a forecast

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def metrics(actual: pd.Series, predicted: pd.Series, capacity: float | None = None) -> dict[str, float]:
    """ Calculate usual and normalized forecast errors """

    actual = pd.Series(actual).astype(float)
    predicted = pd.Series(predicted).astype(float)
    error = actual.to_numpy() - predicted.to_numpy()

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
