"""First orchestration layer for the ETL workflow."""

from __future__ import annotations

from datetime import datetime, timezone
import json

from backend.config.settings import PATHS
from backend.data.transform.water import etl_pipeline_water


def run_etl() -> dict[str, str]:
    """Run the first water transformation and record its completion."""
    etl_pipeline_water()
    return write_manifest("etl")


def write_manifest(stage: str) -> dict[str, str]:
    """Record the stage and its execution time."""
    PATHS["outputs"].mkdir(parents=True, exist_ok=True)
    manifest = {
        "stage": stage,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (PATHS["outputs"] / f"{stage}_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest
