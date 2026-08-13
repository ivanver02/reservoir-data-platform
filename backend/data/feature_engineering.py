"""Features for forecasts"""

from __future__ import annotations

from pathlib import Path
import os

import pandas as pd

from backend.data.validation import require_columns, validate_reservoirs, validate_water


def build_features(water: pd.DataFrame, reservoirs: pd.DataFrame,
    lags: tuple[int, ...] = (1, 2, 3, 4, 52),
    rolling_windows: tuple[int, ...] = (4, 13, 52)) -> pd.DataFrame:
    """ Builds lags, rolling values, and the next target """
    require_columns(water, {"id", "date", "storage"}, "water")
    require_columns(reservoirs, {"id", "capacity"}, "reservoirs")

    # Check keys before building features
    if water.duplicated(["id", "date"]).any():
        raise ValueError("water contains duplicated (id, date) observations")
    if reservoirs["id"].duplicated().any():
        raise ValueError("reservoirs contains duplicated ids")

    frame = water.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["storage"] = pd.to_numeric(frame["storage"], errors="coerce")
    reservoirs = reservoirs.copy()
    reservoirs["capacity"] = pd.to_numeric(reservoirs["capacity"], errors="coerce")

    # Check values before joining metadata
    if frame["date"].isna().any() or frame["storage"].isna().any():
        raise ValueError("water contains invalid dates or storage values")
    validate_water(frame)
    validate_reservoirs(reservoirs)

    # Sort each reservoir series
    frame = frame.sort_values(["id", "date"]).reset_index(drop=True)
    frame = frame.merge(
        reservoirs,
        on="id",
        how="left",
        suffixes=("", "_reservoir"),
        validate="many_to_one",
    )

    # Check capacity after the join
    if frame["capacity"].isna().any():
        raise ValueError("missing capacity after joining reservoir metadata")
    if (frame["capacity"] <= 0).any():
        raise ValueError("reservoirs contains non-positive capacities")
    if (frame["storage"] > frame["capacity"]).any():
        raise ValueError("water contains storage above capacity")
    if any(not isinstance(value, int) or value <= 0 for value in (*lags, *rolling_windows)):
        raise ValueError("lags and rolling_windows need positive integers")

    # Add calendar and capacity values
    frame["week_of_year"] = frame["date"].dt.isocalendar().week.astype(int)
    frame["year"] = frame["date"].dt.year
    frame["storage_fraction"] = frame["storage"] / frame["capacity"]
    grouped = frame.groupby("id", sort=False)["storage"]

    # Add storage lags
    for lag in lags:
        # Keep lags inside each reservoir
        frame[f"storage_lag_{lag}w"] = grouped.shift(lag)

    # Add rolling values
    for window in rolling_windows:
        # Align rolling values with the frame
        rolling = grouped.rolling(window, min_periods=window)
        frame[f"storage_mean_{window}w"] = rolling.mean().reset_index(level=0, drop=True)
        frame[f"storage_std_{window}w"] = rolling.std().reset_index(level=0, drop=True)

    # Use the next value as the target
    frame["target_storage"] = grouped.shift(-1)
    return frame


def write_features(water: pd.DataFrame, reservoirs: pd.DataFrame, path) -> pd.DataFrame:
    """ Builds features and saves them with a file replacement """
    features = build_features(water, reservoirs)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    features.to_parquet(temporary_path, index=False)
    os.replace(temporary_path, path)
    return features
