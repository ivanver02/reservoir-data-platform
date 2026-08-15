"""Time-ordered evaluation helpers."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def metrics(actual: pd.Series, predicted: pd.Series, capacity: float | None = None) -> dict[str, float]:
    """Calculate raw and capacity-normalized forecast errors."""
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


def rolling_origins(
    series: pd.Series,
    horizon: int = 52,
    origins: int = 3,
    step: int = 52,
    minimum_history: int = 104,
) -> list[tuple[pd.Series, pd.Series]]:
    """Create train/test splits without looking into the future."""
    values = pd.Series(series).dropna().sort_index()
    latest_origin = len(values) - horizon
    splits = []
    for index in range(origins - 1, -1, -1):
        end = latest_origin - index * step
        if end >= minimum_history and end + horizon <= len(values):
            splits.append((values.iloc[:end], values.iloc[end:end + horizon]))
    return splits


def evaluate_model(
    series: pd.Series,
    forecast: Callable[[pd.Series, int], pd.Series],
    model_name: str,
    capacity: float | None = None,
    horizon: int = 52,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate one forecasting callable over the rolling origins."""
    rows = []
    predictions = []
    for origin, (train, test) in enumerate(rolling_origins(series, horizon=horizon)):
        predicted = pd.Series(np.asarray(forecast(train, horizon), dtype=float), index=test.index)
        row = metrics(test, predicted, capacity)
        row.update({"model": model_name, "origin": origin, "test_start": test.index[0]})
        rows.append(row)
        predictions.append(pd.DataFrame({
            "date": test.index,
            "actual": test.to_numpy(),
            "prediction": predicted.to_numpy(),
            "model": model_name,
            "origin": origin,
        }))
    return pd.DataFrame(rows), pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
