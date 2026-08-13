"""Basic validation at data-cleaning boundaries."""

from __future__ import annotations

import pandas as pd


def require_columns(frame: pd.DataFrame, columns: set[str], name: str = "dataset") -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def validate_water(frame: pd.DataFrame) -> None:
    require_columns(frame, {"id", "date", "storage"}, "water")
    if frame["id"].isna().any() or frame["date"].isna().any():
        raise ValueError("water contains null identifiers or dates")
    if (frame["storage"] < 0).any():
        raise ValueError("water contains negative storage values")


def validate_reservoirs(frame: pd.DataFrame) -> None:
    require_columns(frame, {"id", "capacity"}, "reservoirs")
    if frame["id"].duplicated().any():
        raise ValueError("reservoirs contains duplicated ids")
