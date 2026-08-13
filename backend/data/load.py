"""Writers for cleaned and first curated feature outputs."""

from __future__ import annotations

from pathlib import Path

from backend.config.settings import PATHS


def _write(frame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def load_water_cleaned(frame) -> None:
    output = PATHS["intermediate"] / "post_cleaning" / "water_cleaned.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)


def load_reservoirs_cleaned(frame) -> None:
    output = PATHS["intermediate"] / "post_cleaning" / "reservoirs_cleaned.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)


def load_features(frame) -> None:
    _write(frame, PATHS["curated"] / "water_features.parquet")
