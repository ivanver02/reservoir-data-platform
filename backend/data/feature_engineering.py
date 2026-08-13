"""First grouped temporal features for forecasting."""

from __future__ import annotations

import pandas as pd

from backend.data.validation import require_columns, validate_reservoirs, validate_water


def build_features(
    water: pd.DataFrame,
    reservoirs: pd.DataFrame,
    lags: tuple[int, ...] = (1, 2, 4, 52),
) -> pd.DataFrame:
    """Add calendar, capacity, and per-reservoir lag features."""
    require_columns(water, {"id", "date", "storage"}, "water")
    require_columns(reservoirs, {"id", "capacity"}, "reservoirs")
    validate_water(water)
    validate_reservoirs(reservoirs)
    frame = water.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["storage"] = pd.to_numeric(frame["storage"])
    frame = frame.sort_values(["id", "date"]).merge(
        reservoirs[["id", "capacity"]], on="id", validate="many_to_one"
    )
    frame["week_of_year"] = frame["date"].dt.isocalendar().week.astype(int)
    frame["storage_fraction"] = frame["storage"] / frame["capacity"]
    grouped = frame.groupby("id", sort=False)["storage"]
    for lag in lags:
        frame[f"storage_lag_{lag}w"] = grouped.shift(lag)
    return frame.reset_index(drop=True)
