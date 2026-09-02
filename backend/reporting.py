"""Report figures from saved outputs"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_forecast(history: pd.Series, forecast: pd.DataFrame, output: Path, title: str = "Reservoir forecast") -> None:
    """ Saves a figure with storage and forecasts """

    # Keep the recent history readable
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(10, 5))
    history = pd.Series(history)
    history.index = pd.to_datetime(history.index)
    history = history.sort_index()
    axis.plot(history.index[-104:], history.iloc[-104:], label="Observed", color="#12304a")

    # Draw the model forecasts
    forecast = forecast.copy()
    forecast["date"] = pd.to_datetime(forecast["date"])

    for column in ["seasonal_naive", "sarima", "prophet", "ensemble"]:
        if column in forecast:
            axis.plot(forecast["date"], forecast[column], label=column.replace("_", " ").title())
    axis.set(title=title, xlabel="Date", ylabel="Storage")
    axis.legend()
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_validation_comparison(
    history: pd.Series,
    predictions: pd.DataFrame,
    output: Path,
    reservoir_id: int,
) -> None:
    """ Saves validation predictions against recent observed storage """
    required = {"date", "actual", "prediction", "model", "origin"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"validation predictions missing columns: {sorted(missing)}")
    if predictions.empty:
        raise ValueError(f"no validation predictions for reservoir {reservoir_id}")

    predictions = predictions.copy()
    predictions["date"] = pd.to_datetime(predictions["date"])
    predictions = predictions.sort_values(["origin", "date", "model"])
    latest_origin = predictions["origin"].max()
    latest = predictions[predictions["origin"] == latest_origin]
    forecast_start = latest["date"].min()
    forecast_end = latest["date"].max()

    observed = pd.Series(history).copy()
    observed.index = pd.to_datetime(observed.index)
    observed = pd.to_numeric(observed, errors="coerce").dropna().sort_index()
    history_start = forecast_start - pd.DateOffset(years=4)
    observed = observed[(observed.index >= history_start) & (observed.index <= forecast_end)]

    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(13, 6.5))
    axis.plot(
        observed.index,
        observed,
        label="Ground truth",
        color="#17202a",
        linewidth=2.2,
        zorder=5,
    )

    colors = {
        "seasonal_naive": "#7f8c8d",
        "sarima": "#2980b9",
        "ensemble": "#8e44ad",
        "prophet": "#e67e22",
    }
    styles = {
        "seasonal_naive": "--",
        "sarima": "-.",
        "ensemble": ":",
        "prophet": "-",
    }
    for model in ("seasonal_naive", "sarima", "ensemble", "prophet"):
        model_rows = latest[latest["model"] == model]
        if model_rows.empty:
            continue
        axis.plot(
            model_rows["date"],
            model_rows["prediction"],
            label=model.replace("_", " ").title(),
            color=colors[model],
            linestyle=styles[model],
            linewidth=3.0 if model == "prophet" else 1.4,
            alpha=1.0 if model == "prophet" else 0.8,
            zorder=4 if model == "prophet" else 3,
        )

    axis.axvline(forecast_start, color="#95a5a6", linestyle="--", linewidth=1)
    axis.text(
        forecast_start,
        1.02,
        "Latest validation year",
        transform=axis.get_xaxis_transform(),
        ha="left",
        va="bottom",
        color="#566573",
    )
    axis.set(
        title=f"Reservoir {reservoir_id}: validation model comparison",
        xlabel="Date",
        ylabel="Storage",
    )
    axis.grid(alpha=0.2)
    axis.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
