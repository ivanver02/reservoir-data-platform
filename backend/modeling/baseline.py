""" The seasonal baseline """

from __future__ import annotations

import numpy as np
import pandas as pd


def seasonal_naive(series: pd.Series, horizon: int = 52, seasonal_lag: int = 52) -> pd.Series:
    """ Repeat the latest annual cycle """
    
    values = pd.Series(series).dropna().astype(float)
    if values.empty:
        raise ValueError("cannot forecast an empty series")
    cycle = values.iloc[-seasonal_lag:].to_numpy()
    forecast = np.resize(cycle, horizon)
    start = pd.Timestamp(values.index[-1]) + pd.Timedelta(weeks=1)
    return pd.Series(forecast, index=pd.date_range(start, periods=horizon, freq="7D"))
