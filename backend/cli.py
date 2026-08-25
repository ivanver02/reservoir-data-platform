"""Command line entry points"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from backend.config.settings import EVALUATION_SETTINGS, FORECAST_SETTINGS, PATHS, configure_data_root
from backend.data.load import write_parquet_atomic
from backend.data.pipeline import load_curated, run_etl, run_features
from backend.modeling.forecasting import evaluate_series, forecast_one_year
from backend.modeling.evaluation_cache import run_cached_evaluation
from backend.modeling.evaluation_summary import format_summary, write_markdown_summary
from backend.modeling.model_selection import write_test_analysis, write_validation_decision
from backend.reporting import plot_forecast, plot_validation_comparison
from backend.transferring.planner import TransferConfig, plan_transfers, prepare_state, select_region


def _write(frame: pd.DataFrame, filename: str) -> Path:
    """ Writes one command output to the Parquet directory """
    path = PATHS["outputs"] / filename
    write_parquet_atomic(frame, path)
    return path


def _forecast_cache_path(
    reservoir_id: int,
    series: pd.Series,
    capacity: float,
    models: tuple[str, ...],
) -> Path:
    """ Builds a cache path from the reservoir data and model settings """
    ordered = series.sort_index()
    digest = hashlib.sha256()
    digest.update(pd.util.hash_pandas_object(ordered, index=True).to_numpy().tobytes())
    digest.update(f"{capacity}|{models}|{FORECAST_SETTINGS['horizon_weeks']}".encode())
    return PATHS["cache"] / "forecast_cache" / f"{reservoir_id}_{digest.hexdigest()}.parquet"


def _cached_forecast(
    reservoir_id: int,
    series: pd.Series,
    capacity: float,
    models: tuple[str, ...],
) -> tuple[pd.DataFrame, bool]:
    """ Loads a cached forecast or fits the requested models once """
    path = _forecast_cache_path(reservoir_id, series, capacity, models)
    if path.exists():
        return pd.read_parquet(path), True

    prediction = forecast_one_year(series, capacity, models)
    path.parent.mkdir(parents=True, exist_ok=True)
    prediction.to_parquet(path, index=False)
    return prediction, False


def command_forecast(args) -> None:
    """ Forecasts one year of storage for each reservoir with enough metadata """

    # Limit release forecasts to the evaluation population
    water, reservoirs = load_curated()
    community = reservoirs["autonomous_community"].astype("string").str.strip().str.casefold()
    reservoirs = reservoirs.loc[community == "andalucia"].copy()
    water = water[water["id"].isin(reservoirs["id"])]
    rows = []
    cached_count = 0

    for reservoir_id, group in water.groupby("id"):
        metadata = reservoirs[reservoirs["id"] == reservoir_id]
        if metadata.empty:
            continue
        capacity = float(metadata.iloc[0]["capacity"])

        try:
            prediction, cached = _cached_forecast(
                reservoir_id,
                group.set_index("date")["storage"],
                capacity,
                tuple(args.models.split(",")),
            )
            cached_count += int(cached)
        except Exception as exc:
            print(f"Skipping reservoir {reservoir_id}: {exc}")
            continue

        prediction["id"] = reservoir_id
        rows.append(prediction)

    if not rows:
        raise RuntimeError("no reservoir produced a forecast")
    output = pd.concat(rows, ignore_index=True)

    _write(output, "forecasts.parquet")
    print(
        f"Wrote {len(output)} forecasts to {PATHS['outputs'] / 'forecasts.parquet'} "
        f"({cached_count} loaded from cache)"
    )


def command_evaluate(args) -> None:
    """ Runs evaluation over time splits for the chosen reservoirs """

    # Collect metrics and predictions separately
    water, reservoirs = load_curated()
    metric_rows, prediction_rows = [], []

    for reservoir_number, (reservoir_id, group) in enumerate(water.groupby("id")):
        if args.limit and reservoir_number >= args.limit:
            break
        metadata = reservoirs[reservoirs["id"] == reservoir_id]
        if metadata.empty:
            continue

        current_metrics, predictions = evaluate_series(
            group.set_index("date")["storage"],
            float(metadata.iloc[0]["capacity"]),
            horizon=args.horizon,
            origins=args.origins,
            models=tuple(args.models.split(",")),
        )

        if not current_metrics.empty:
            current_metrics["id"] = reservoir_id
            metric_rows.append(current_metrics)
        if not predictions.empty:
            predictions["id"] = reservoir_id
            prediction_rows.append(predictions)

    # Write results
    if metric_rows:
        _write(pd.concat(metric_rows, ignore_index=True), "evaluation_metrics.parquet")
    if prediction_rows:
        _write(pd.concat(prediction_rows, ignore_index=True), "evaluation_predictions.parquet")
    print(f"Evaluation written to {PATHS['outputs']}")


def command_evaluate_split(args) -> None:
    """ Evaluates the next uncached reservoir in a fixed Andalusia split """
    models = tuple(model.strip() for model in args.models.split(",") if model.strip())
    supported = {"seasonal_naive", "sarima", "prophet"}
    unknown = set(models) - supported
    if unknown:
        raise ValueError(f"unsupported models: {sorted(unknown)}")
    if not models:
        raise ValueError("at least one model is required")
    if args.split == "test" and models != ("prophet",):
        raise ValueError("evaluate-test only supports the validation-selected model: prophet")

    result = run_cached_evaluation(
        *load_curated(),
        split=args.split,
        models=models,
        horizon=args.horizon,
        origins=args.origins,
        limit=args.limit,
        include_ensemble=args.split == "validation",
    )
    if result["processed"] == 0 and result["all_cached"]:
        print(f"All {result['requested']} {args.split} reservoirs are already cached")
    else:
        print(
            f"Processed {result['processed']} {args.split} reservoirs, "
            f"{result['remaining']} remaining, {result['cached']} cached"
        )
    print(f"Split metrics: {PATHS['outputs'] / f'evaluation_{args.split}_metrics.parquet'}")


def command_evaluation_summary(args) -> None:
    """ Prints provisional metrics from finished cache entries """
    print(format_summary(args.split))
    print(f"Markdown summary written to {write_markdown_summary(args.split)}")


def command_analyze_validation(args) -> None:
    """ Analyzes validation results and records the model decision """
    markdown_path, json_path = write_validation_decision()
    print(f"Validation decision written to {markdown_path}")
    print(f"Machine-readable decision written to {json_path}")


def command_analyze_test(args) -> None:
    """ Analyzes the locked model on the complete test split """
    markdown_path, json_path = write_test_analysis()
    print(f"Test analysis written to {markdown_path}")
    print(f"Machine-readable analysis written to {json_path}")


def command_plan(args) -> None:
    """ Builds forecasts and runs the transfer planner """

    # Filter before fitting forecasts
    water, reservoirs = load_curated()
    selected = select_region(reservoirs, args.community, args.latitude, args.longitude, args.radius_km)
    water = water[water["id"].isin(selected["id"])]
    forecasts = []
    cached_count = 0

    for reservoir_id, group in water.groupby("id"):
        metadata = selected[selected["id"] == reservoir_id]
        if metadata.empty:
            continue

        prediction, cached = _cached_forecast(
            reservoir_id,
            group.set_index("date")["storage"],
            float(metadata.iloc[0]["capacity"]),
            tuple(args.models.split(",")),
        )
        cached_count += int(cached)

        forecasts.append({
            "id": reservoir_id,
            "last_known_value": float(group.sort_values("date")["storage"].iloc[-1]),
            "low_forecasted_value": float(prediction["ensemble"].quantile(0.10)),
            "median_forecasted_value": float(prediction["ensemble"].median()),
        })

    # Build the planning state
    summary = pd.DataFrame(forecasts)
    state = prepare_state(selected, summary)

    final_state, log = plan_transfers(state, TransferConfig(max_distance_km=args.max_distance_km), args.max_iterations)

    # Write the planning outputs
    _write(final_state, "transfer_final_state.parquet")
    _write(log, "transfer_log.parquet")
    print(f"Planned {len(log)} transfers ({cached_count} forecasts loaded from cache)")


def command_make_sample(args) -> None:
    """ Creates a fixed sample dataset for local tests """

    # Keep sample data repeatable
    output = Path(args.output) if args.output is not None else PATHS["sample"]
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)

    # Create the sample dates
    dates = pd.date_range("2018-01-07", periods=260, freq="7D")
    reservoirs = pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["sample north", "sample south", "sample east"],
        "capacity": [1000, 1200, 900],
        "latitude": [37.2, 37.4, 37.1],
        "longitude": [-4.5, -4.2, -4.0],
        "autonomous_community": ["andalucia"] * 3,
    })

    # Give each sample reservoir a separate signal
    water = pd.concat([
        pd.DataFrame({"date": dates, "id": rid, "storage": np.clip(500 + 200 * np.sin(np.arange(len(dates)) / 26) + rng.normal(0, 15, len(dates)), 0, None)})
        for rid in reservoirs["id"]
    ], ignore_index=True)

    reservoirs.to_csv(output / "reservoirs.csv", index=False)
    water.to_csv(output / "water.csv", index=False)
    print(f"Sample data written to {output}")


def command_report(args) -> None:
    """ Generates a forecast plot from the saved command outputs """

    # Read observations and forecasts
    water, _ = load_curated()
    forecasts_path = PATHS["outputs"] / "forecasts.parquet"
    if not forecasts_path.exists():
        raise FileNotFoundError("run 'forecast' before generating a report")
    forecasts = pd.read_parquet(forecasts_path)
    reservoir_id = args.id if args.id is not None else forecasts["id"].iloc[0]

    history = water[water["id"] == reservoir_id].set_index("date")["storage"]
    prediction = forecasts[forecasts["id"] == reservoir_id]
    output = PATHS["outputs"] / f"forecast_{reservoir_id}.png"
    plot_forecast(history, prediction, output, f"Reservoir {reservoir_id} forecast")
    print(f"Report written to {output}")


def command_report_validation(args) -> None:
    """ Generates a validation comparison figure for one reservoir """
    predictions_path = PATHS["outputs"] / "evaluation_validation_predictions.parquet"
    if not predictions_path.exists():
        raise FileNotFoundError("run validation evaluation before generating a comparison report")

    predictions = pd.read_parquet(predictions_path)
    selected = predictions[predictions["id"] == args.id]
    if selected.empty:
        raise ValueError(f"no validation predictions for reservoir {args.id}")

    water, _ = load_curated()
    history = water[water["id"] == args.id].set_index("date")["storage"]
    output = PATHS["outputs"] / f"validation_comparison_{args.id}.png"
    plot_validation_comparison(history, selected, output, args.id)
    print(f"Validation comparison written to {output}")


def build_parser() -> argparse.ArgumentParser:
    """ Defines the command line interface """

    # Wire commands to their handlers
    parser = argparse.ArgumentParser(prog="reservoir-platform")
    parser.add_argument(
        "--data-root",
        type=Path,
        help="root directory for raw data, caches, intermediate files, and outputs",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Data commands
    sub.add_parser("etl", help="run extraction, cleaning and reconciliation").set_defaults(func=lambda a: print(run_etl()))
    sub.add_parser("features", help="build curated forecasting features").set_defaults(func=lambda a: print(run_features()))

    # Forecast commands
    forecast = sub.add_parser("forecast", help="fit models and forecast one year")
    forecast.add_argument("--models", default="seasonal_naive,sarima", help="comma-separated models")
    forecast.set_defaults(func=command_forecast)

    # Evaluation commands
    evaluate = sub.add_parser("evaluate", help="run rolling-origin evaluation")
    evaluate.add_argument("--horizon", type=int, default=FORECAST_SETTINGS["horizon_weeks"])
    evaluate.add_argument("--origins", type=int, default=FORECAST_SETTINGS["backtest_origins"])
    evaluate.add_argument("--models", default="seasonal_naive,sarima", help="comma-separated models")
    evaluate.add_argument("--limit", type=int, default=0, help="limit reservoirs, 0 means all")
    evaluate.set_defaults(func=command_evaluate)

    # Cached evaluation commands
    for split in ("validation", "test"):
        split_parser = sub.add_parser(
            f"evaluate-{split}",
            help=f"run cached {split} evaluation on the fixed Andalusia split",
        )
        split_parser.add_argument("--horizon", type=int, default=FORECAST_SETTINGS["horizon_weeks"])
        split_parser.add_argument("--origins", type=int, default=FORECAST_SETTINGS["backtest_origins"])
        if split == "validation":
            split_parser.set_defaults(models=",".join(EVALUATION_SETTINGS["validation_models"]))
        else:
            split_parser.set_defaults(models="prophet")
        split_parser.add_argument("--limit", type=int, default=0, help="number of next uncached reservoirs, 0 means all")
        split_parser.set_defaults(func=command_evaluate_split, split=split)

    # Read cached evaluation results
    summary = sub.add_parser("evaluation-summary", help="show provisional cached evaluation metrics")
    summary.add_argument("--split", choices=("validation", "test"), required=True)
    summary.set_defaults(func=command_evaluation_summary)

    analysis = sub.add_parser("analyze-validation", help="analyze validation metrics and choose a model")
    analysis.set_defaults(func=command_analyze_validation)

    test_analysis = sub.add_parser("analyze-test", help="analyze the locked model on test metrics")
    test_analysis.set_defaults(func=command_analyze_test)

    # Planning options
    plan = sub.add_parser("plan-transfers", help="run greedy transfer planning")
    plan.add_argument("--community")
    plan.add_argument("--latitude", type=float)
    plan.add_argument("--longitude", type=float)
    plan.add_argument("--radius-km", type=float)
    plan.add_argument("--max-distance-km", type=float)
    plan.add_argument("--max-iterations", type=int, default=10)
    plan.add_argument("--models", default="seasonal_naive,sarima", help="comma-separated models")
    plan.set_defaults(func=command_plan)

    # Sample and report commands
    sample = sub.add_parser("make-sample", help="generate a small synthetic example")
    sample.add_argument("--output", type=Path, help="output directory, defaults to data root sample")
    sample.set_defaults(func=command_make_sample)
    report = sub.add_parser("report", help="generate a forecast figure")
    report.add_argument("--id", type=int)
    report.set_defaults(func=command_report)
    validation_report = sub.add_parser(
        "report-validation",
        help="compare validation models with ground truth for one reservoir",
    )
    validation_report.add_argument("--id", type=int, required=True)
    validation_report.set_defaults(func=command_report_validation)

    return parser


def main() -> None:
    """ Parses command line arguments and runs the chosen command """
    args = build_parser().parse_args()

    # Set the data path before I/O
    if args.data_root is not None:
        configure_data_root(args.data_root)

    args.func(args)


if __name__ == "__main__":
    main()
