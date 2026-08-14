""" Baseline forecast orchestration """

from __future__ import annotations

import pandas as pd

from backend.config.settings import FORECAST_SETTINGS
from backend.modeling.baseline import clip_storage, seasonal_naive


def forecast_one_year(series: pd.Series, capacity: float) -> pd.DataFrame:
    """ Build a table for one year using the seasonal baseline """
    prediction = clip_storage(
        seasonal_naive(series, FORECAST_SETTINGS["horizon_weeks"]),
        capacity,
    )
    return pd.DataFrame({"date": prediction.index, "seasonal_naive": prediction.values})
