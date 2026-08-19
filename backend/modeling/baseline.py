"""Forecast baselines"""

from __future__ import annotations

import numpy as np
import pandas as pd


def seasonal_naive(series: pd.Series, horizon: int = 52, seasonal_lag: int = 52) -> pd.Series:
    """ Repeats the latest seasonal cycle or the last value """
    values = pd.Series(series).dropna().astype(float)
    if values.empty:
        raise ValueError("cannot forecast an empty series")
    if len(values) < seasonal_lag:
        forecast = np.repeat(values.iloc[-1], horizon)
    else:
        cycle = values.iloc[-seasonal_lag:].to_numpy()
        forecast = np.resize(cycle, horizon)

    # Start after the last observation
    start = pd.Timestamp(values.index[-1]) + pd.Timedelta(weeks=1)
    return pd.Series(forecast, index=pd.date_range(start, periods=horizon, freq="7D"))


def clip_storage(predictions: pd.Series, capacity: float) -> pd.Series:
    """ Keeps storage predictions between zero and the reservoir capacity """
    return pd.Series(predictions, index=predictions.index).astype(float).clip(0, capacity)
