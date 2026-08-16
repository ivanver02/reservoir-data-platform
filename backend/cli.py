"""Command line entry points"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from backend.config.settings import PATHS, configure_data_root
from backend.data.pipeline import load_curated, run_etl, run_features
from backend.modeling.baseline import seasonal_naive


def command_forecast(_args) -> None:
    """Write a baseline forecast for every curated reservoir."""
    water, _ = load_curated()
    rows = []
    for reservoir_id, group in water.groupby("id"):
        prediction = seasonal_naive(group.set_index("date")["storage"])
        rows.append(pd.DataFrame({"id": reservoir_id, "date": prediction.index, "prediction": prediction.values}))
    output = pd.concat(rows, ignore_index=True)
    PATHS["outputs"].mkdir(parents=True, exist_ok=True)
    output.to_parquet(PATHS["outputs"] / "forecasts.parquet", index=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="reservoir-platform")
    parser.add_argument("--data-root", type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("etl").set_defaults(func=lambda _: print(run_etl()))
    commands.add_parser("features").set_defaults(func=lambda _: print(run_features()))
    commands.add_parser("forecast").set_defaults(func=command_forecast)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.data_root is not None:
        configure_data_root(args.data_root)
    args.func(args)
