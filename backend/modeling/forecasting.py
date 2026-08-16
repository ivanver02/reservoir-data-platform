"""Forecast model orchestration"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from backend.config.settings import FORECAST_SETTINGS
from backend.modeling.baseline import clip_storage, seasonal_naive
from backend.modeling.ensemble import weighted_average
from backend.modeling.evaluation import metrics, rolling_origins
from backend.modeling.sarima_model import SARIMAModel


def _fit_predict(model, train: pd.Series, horizon: int) -> pd.Series:
    """ Fits one model and returns its forecast as a numeric series """
    model.fit(train)
    return pd.Series(model.predict(horizon), dtype=float)


def model_forecasts(
    capacity: float | None = None,
    models: tuple[str, ...] = ("seasonal_naive", "sarima", "prophet"),
) -> dict[str, Callable[[pd.Series, int], pd.Series]]:
    """ Returns the forecasting functions requested by the caller """
    def baseline(train, horizon):
        """ Forecasts with the seasonal baseline """
        return clip_storage(seasonal_naive(train, horizon), capacity) if capacity else seasonal_naive(train, horizon)

    def sarima(train, horizon):
        """ Forecasts with the SARIMA model """
        prediction = _fit_predict(SARIMAModel(), train, horizon)
        return clip_storage(prediction, capacity) if capacity else prediction

    def prophet(train, horizon):
        """ Forecasts with Prophet """
        # Import Prophet only when requested
        from backend.modeling.prophet_model import ProphetModel
        prediction = _fit_predict(ProphetModel(changepoint_prior_scale=0.0005), train, horizon)
        return clip_storage(prediction, capacity) if capacity else prediction

    available = {"seasonal_naive": baseline, "sarima": sarima, "prophet": prophet}
    return {name: available[name] for name in models if name in available}


def evaluate_origin(
    train: pd.Series,
    test: pd.Series,
    capacity: float,
    models: tuple[str, ...],
    origin: int,
    failures: list[dict[str, object]] | None = None,
    include_ensemble: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ Evaluates the requested models on one rolling origin """
    metric_rows = []
    prediction_frames = []
    valid_forecasts = {}
    forecasts = model_forecasts(capacity, models)

    for name, forecast in forecasts.items():
        try:
            predicted = pd.Series(
                np.asarray(forecast(train, len(test)), dtype=float),
                index=test.index,
            )
        except Exception as exc:
            if failures is not None:
                failures.append({
                    "model": name,
                    "origin": origin,
                    "error": f"{type(exc).__name__}: {exc}",
                })
            continue
        valid_forecasts[name] = predicted
        row = metrics(test, predicted, capacity)
        row.update({"model": name, "origin": origin, "test_start": test.index[0]})
        metric_rows.append(row)
        prediction_frames.append(pd.DataFrame({
            "date": test.index,
            "actual": test.to_numpy(),
            "prediction": predicted.to_numpy(),
            "model": name,
            "origin": origin,
        }))

    if valid_forecasts and include_ensemble:
        predicted = weighted_average(
            valid_forecasts,
            FORECAST_SETTINGS["ensemble_weights"],
            capacity,
        )
        predicted.index = test.index
        row = metrics(test, predicted, capacity)
        row.update({"model": "ensemble", "origin": origin, "test_start": test.index[0]})
        metric_rows.append(row)
        prediction_frames.append(pd.DataFrame({
            "date": test.index,
            "actual": test.to_numpy(),
            "prediction": predicted.to_numpy(),
            "model": "ensemble",
            "origin": origin,
        }))

    return (
        pd.DataFrame(metric_rows),
        pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame(),
    )


def evaluate_series(series: pd.Series, capacity: float, horizon: int = FORECAST_SETTINGS["horizon_weeks"],
    origins: int = FORECAST_SETTINGS["backtest_origins"],
    models: tuple[str, ...] = ("seasonal_naive", "sarima", "prophet"),
    failures: list[dict[str, object]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ Compares models using time splits """

    # Collect results for each origin
    metric_frames, prediction_frames = [], []
    for origin, (train, test) in enumerate(rolling_origins(
        series,
        horizon,
        origins,
        FORECAST_SETTINGS["backtest_step_weeks"],
        FORECAST_SETTINGS["minimum_history_weeks"],
    )):
        current_metrics, current_predictions = evaluate_origin(
            train,
            test,
            capacity,
            models,
            origin,
            failures,
        )
        metric_frames.append(current_metrics)
        prediction_frames.append(current_predictions)

    return (
        pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame(),
        pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame(),
    )


def forecast_one_year(
    series: pd.Series,
    capacity: float,
    models: tuple[str, ...] = ("seasonal_naive", "sarima", "prophet"),
) -> pd.DataFrame:
    """ Fits models and builds a yearly forecast table """

    # Skip models that fail
    forecasts = {}
    for name, forecast in model_forecasts(capacity, models).items():
        try:
            forecasts[name] = forecast(series, FORECAST_SETTINGS["horizon_weeks"])
        except Exception:
            continue
    if not forecasts:
        raise RuntimeError("no forecasting model produced a prediction")

    # Add the ensemble after the model columns
    frame = pd.DataFrame(forecasts)
    frame["ensemble"] = weighted_average(
        forecasts,
        FORECAST_SETTINGS["ensemble_weights"],
        capacity,
    )
    frame.index.name = "date"
    return frame.reset_index()
