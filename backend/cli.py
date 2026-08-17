"""Command line entry points"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from backend.config.settings import PATHS, configure_data_root
from backend.data.pipeline import run_etl, run_features, load_curated
from backend.modeling.forecasting import forecast_one_year


def _write(frame: pd.DataFrame, filename: str) -> Path:
    """ Writes one command output to the Parquet directory """
    PATHS["outputs"].mkdir(parents=True, exist_ok=True)
    path = PATHS["outputs"] / filename
    frame.to_parquet(path, index=False)
    return path


def command_forecast(args) -> None:
    """ Forecasts one year of storage for each reservoir with enough metadata """
    water, reservoirs = load_curated()
    rows = []
    for reservoir_id, group in water.groupby("id"):
        metadata = reservoirs[reservoirs["id"] == reservoir_id]
        if metadata.empty:
            continue
        prediction = forecast_one_year(
            group.set_index("date")["storage"],
            float(metadata.iloc[0]["capacity"]),
            tuple(args.models.split(",")),
        )
        prediction["id"] = reservoir_id
        rows.append(prediction)
    if not rows:
        raise RuntimeError("no reservoir produced a forecast")
    output = pd.concat(rows, ignore_index=True)

    _write(output, "forecasts.parquet")
    print(f"Wrote {len(output)} forecasts to {PATHS['outputs'] / 'forecasts.parquet'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="reservoir-platform")
    parser.add_argument("--data-root", type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("etl").set_defaults(func=lambda _: print(run_etl()))
    commands.add_parser("features").set_defaults(func=lambda _: print(run_features()))
    forecast = commands.add_parser("forecast")
    forecast.add_argument("--models", default="seasonal_naive,sarima")
    forecast.set_defaults(func=command_forecast)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.data_root is not None:
        configure_data_root(args.data_root)
    args.func(args)


if __name__ == "__main__":
    main()
