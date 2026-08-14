"""Data pipeline commands"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from backend.config.settings import PATHS
from backend.data.extract import (
    extract_reservoirs_merged_definitive,
    extract_reservoirs_definitive,
    extract_water_definitive,
)
from backend.data.feature_engineering import write_features
from backend.data.transform.detailed_reservoir import etl_pipeline_detailed_reservoir
from backend.data.transform.merges import etl_pipeline_reservoirs_merged
from backend.data.transform.reservoir import etl_pipeline_reservoir
from backend.data.transform.water import etl_pipeline_water


def run_etl() -> dict[str, str]:
    """ Runs the transformations and records the ETL execution """
    etl_pipeline_water()
    etl_pipeline_reservoir()
    etl_pipeline_detailed_reservoir()
    etl_pipeline_reservoirs_merged()
    return write_manifest("etl")


def run_features() -> dict[str, str]:
    """ Builds features and records the stage """
    water = extract_water_definitive()
    reservoirs = extract_reservoirs_definitive()
    output = PATHS["curated"] / "water_features.parquet"
    write_features(water, reservoirs, output)
    return write_manifest("features")


def write_manifest(stage: str) -> dict[str, str]:
    """ Writes a manifest with the stage and execution time """
    PATHS["outputs"].mkdir(parents=True, exist_ok=True)
    manifest = {
        "stage": stage,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(PATHS["data_root"]),
    }
    path = PATHS["outputs"] / f"{stage}_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_curated() -> tuple[pd.DataFrame, pd.DataFrame]:
    """ Returns the tables used by the workflow """
    return extract_water_definitive(), extract_reservoirs_merged_definitive()
