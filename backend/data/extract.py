"""ETL input readers with an explicit pipeline-input boundary."""

from __future__ import annotations

import pandas as pd

from backend.config.settings import PATHS


def _read_csv(path, **kwargs):
    return pd.read_csv(path, **kwargs)


def extract_pipeline_inputs() -> dict[str, pd.DataFrame]:
    """Read all three raw tables used by the ETL orchestration."""
    return {
        "water": _read_csv(PATHS["raw"] / "water.csv"),
        "reservoirs": _read_csv(PATHS["raw"] / "reservoirs.csv"),
        "details": _read_csv(PATHS["raw"] / "UTF8list-3.tsv", sep="\t"),
    }


def extract_water_raw():
    return _read_csv(PATHS["raw"] / "water.csv")


def extract_reservoirs_raw():
    return _read_csv(PATHS["raw"] / "reservoirs.csv")
