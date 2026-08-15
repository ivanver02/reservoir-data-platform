"""Baseline rolling origin evaluation orchestration """

from __future__ import annotations

import pandas as pd

from backend.config.settings import FORECAST_SETTINGS
from backend.modeling.baseline import seasonal_naive
from backend.modeling.evaluation import evaluate_model


def evaluate_series(series: pd.Series, capacity: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ Evaluate the baseline over the rolling origins """
    return evaluate_model(
        series,
        lambda train, horizon: seasonal_naive(train, horizon),
        "seasonal_naive",
        capacity,
        FORECAST_SETTINGS["horizon_weeks"],
    )
