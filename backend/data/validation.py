"""Validation helpers for pipeline boundaries"""

from __future__ import annotations

import pandas as pd


def require_columns(frame: pd.DataFrame, columns: set[str], name: str = "dataset") -> None:
    """ Checks that a DataFrame contains the required columns """
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def validate_water(frame: pd.DataFrame) -> None:
    """ Checks the basic rules for water observations """
    require_columns(frame, {"id", "date", "storage"}, "water")
    if frame["id"].isna().any() or frame["date"].isna().any():
        raise ValueError("water contains null identifiers or dates")
    if (frame["storage"] < 0).any():
        raise ValueError("water contains negative storage values")


def validate_reservoirs(frame: pd.DataFrame) -> None:
    """ Checks reservoir ids, capacities, and metadata structure """
    require_columns(frame, {"id", "capacity"}, "reservoirs")
    if frame["id"].duplicated().any():
        raise ValueError("reservoirs contains duplicated ids")
    if (frame["capacity"] <= 0).any():
        raise ValueError("reservoirs contains non-positive capacities")


def validate_curated(water: pd.DataFrame, reservoirs: pd.DataFrame) -> None:
    """ Checks that tables can be combined """
    validate_water(water)
    validate_reservoirs(reservoirs)
    unknown = set(water["id"].unique()).difference(reservoirs["id"])
    if unknown:
        raise ValueError(f"water contains ids absent from reservoirs: {sorted(unknown)}")
    merged = water.merge(reservoirs[["id", "capacity"]], on="id", how="left")
    if (merged["storage"] > merged["capacity"]).any():
        raise ValueError("curated water contains storage above capacity")
