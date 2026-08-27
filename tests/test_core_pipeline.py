import numpy as np
import pandas as pd
import pytest

from backend.data.feature_engineering import build_features
from backend.data.cleaning import impute_nearest_neighbour
from backend.data.transform.detailed_reservoir import transform_detailed_reservoirs_raw
from backend.data.transform.reservoir import transform_reservoirs_raw
from backend.data.transform.water import transform_water_raw
from backend.modeling.baseline import seasonal_naive
from backend.modeling.ensemble import weighted_average
from backend.modeling.forecasting import evaluate_origin, evaluate_series
from backend.modeling.evaluation_cache import run_cached_evaluation
from backend.modeling.evaluation_summary import cached_summary, write_markdown_summary
from backend.modeling.model_selection import (
    analyze_test,
    analyze_validation,
    format_decision_markdown,
    format_test_analysis_markdown,
)
from backend.config.settings import PATHS, EVALUATION_SETTINGS, FORECAST_SETTINGS
from backend.cli import build_parser, command_plan
from backend.reporting import plot_forecast, plot_validation_comparison
from backend.transferring.planner import TransferConfig, plan_transfers


def test_features_are_grouped_by_reservoir():
    """ Checks that lags and fractions stay within each reservoir """
    dates = pd.date_range("2020-01-05", periods=60, freq="7D")
    water = pd.DataFrame({
        "id": np.repeat([1, 2], len(dates)),
        "date": list(dates) * 2,
        "storage": np.arange(len(dates) * 2, dtype=float),
    })
    reservoirs = pd.DataFrame({"id": [1, 2], "capacity": [200, 200]})

    # The first row has no lag
    result = build_features(water, reservoirs, lags=(1,), rolling_windows=(4,))
    first_each = result.groupby("id").head(1)
    assert first_each["storage_lag_1w"].isna().all()
    assert result["storage_fraction"].between(0, 1).all()


def test_feature_engineering_rejects_duplicate_observations():
    """ Checks that duplicate water observations are rejected """
    water = pd.DataFrame({
        "id": [1, 1],
        "date": pd.to_datetime(["2020-01-05", "2020-01-05"]),
        "storage": [10.0, 11.0],
    })
    reservoirs = pd.DataFrame({"id": [1], "capacity": [100.0]})
    with pytest.raises(ValueError, match="duplicated"):
        build_features(water, reservoirs)


def test_water_imputation_is_grouped():
    """ Checks that gaps use the next value from the same reservoir """
    water = pd.DataFrame({
        "date": ["01/01/2020", "15/01/2020", "01/01/2020"],
        "id": [1, 1, 2],
        "storage": [10.0, 20.0, 99.0],
    })

    # The gap uses the next value from reservoir 1
    result = transform_water_raw(water)
    reservoir_one = result[result["id"] == 1].set_index("date")
    assert reservoir_one.loc[pd.Timestamp("2020-01-08"), "storage"] == 20.0
    assert reservoir_one.loc[pd.Timestamp("2020-01-08"), "storage_imputed"] == 1
    assert result.loc[
        (result["id"] == 1) & (result["date"] == pd.Timestamp("2020-01-01")),
        "storage",
    ].iloc[0] == 10.0


def test_water_imputation_uses_next_observation():
    """ Checks that internal gaps use the next value in a reservoir """
    water = pd.DataFrame({
        "date": ["01/01/2020", "15/01/2020"],
        "id": [1, 1],
        "storage": [10.0, 20.0],
    })

    result = transform_water_raw(water)

    assert result.loc[result["date"] == pd.Timestamp("2020-01-08"), "storage"].iloc[0] == 20.0


def test_nearest_neighbour_records_imputation():
    """ Checks that nearby fills are recorded """
    data = pd.DataFrame({
        "longitude": [-4.0, -4.01],
        "latitude": [37.0, 37.01],
        "riverbed": ["guadalquivir", None],
    })
    result = impute_nearest_neighbour(data, "riverbed")
    assert result.loc[1, "riverbed"] == "guadalquivir"
    assert result.loc[1, "riverbed_imputed"]


def test_detailed_transform_keeps_imputed_crest_elevation():
    """ Checks that an imputed crest elevation survives detail cleaning """
    data = pd.DataFrame({
        "name": ["a", "b"],
        "reservoir": ["a", "b"],
        "basin": ["basin", "basin"],
        "province": ["cadiz", "cadiz"],
        "autonomous_community": ["andalucia", "andalucia"],
        "type": ["dam", "dam"],
        "longitude": ["-4.0", "-4.01"],
        "latitude": ["37.0", "37.01"],
        "riverbed": ["river", "river"],
        "crest_elevation": ["100,0", None],
        "dam_height": ["10,0", "9,0"],
    })

    # Reservoir b uses the nearest crest elevation
    result = transform_detailed_reservoirs_raw(data)
    assert result.loc[result["name"] == "b", "crest_elevation"].iloc[0] == 100.0


def test_reservoir_capacity_fallback_is_marked(monkeypatch):
    """ Checks that fallback capacities are marked in the output """
    import backend.data.transform.reservoir as reservoir_transform

    monkeypatch.setattr(
        reservoir_transform,
        "extract_water_cleaned",
        lambda: pd.DataFrame({"id": [1], "storage": [75.0]}),
    )
    result = reservoir_transform.transform_reservoirs_raw(
        pd.DataFrame({"id": [1], "name": ["sample"], "capacity": [None], "scope": ["scope"]})
    )
    assert result.loc[0, "capacity"] == 75.0
    assert result.loc[0, "capacity_imputed"]


def test_seasonal_naive_returns_one_year():
    """ Checks that the seasonal baseline returns 52 values """
    series = pd.Series(range(104), index=pd.date_range("2020-01-05", periods=104, freq="7D"))
    result = seasonal_naive(series)
    assert len(result) == 52
    assert result.iloc[0] == 52


def test_forecast_report_handles_string_dates(tmp_path):
    """ Checks that report plots accept string forecast dates """
    history = pd.Series(
        [10.0, 12.0],
        index=pd.date_range("2020-01-05", periods=2, freq="7D"),
    )
    forecast = pd.DataFrame({
        "date": ["2020-01-19", "2020-01-26"],
        "seasonal_naive": [13.0, 14.0],
    })
    output = tmp_path / "forecast.png"

    plot_forecast(history, forecast, output)

    assert output.exists()


def test_validation_comparison_report_includes_ground_truth(tmp_path):
    """ Checks that validation comparison reports accept model predictions """
    history_dates = pd.date_range("2020-01-05", periods=250, freq="7D")
    history = pd.Series(np.arange(250, dtype=float), index=history_dates)
    dates = pd.date_range("2024-01-07", periods=4, freq="7D")
    predictions = pd.concat([
        pd.DataFrame({
            "date": dates,
            "actual": [20.0, 21.0, 22.0, 23.0],
            "prediction": [19.0, 20.0, 21.0, 22.0],
            "model": model,
            "origin": 2,
        })
        for model in ("seasonal_naive", "sarima", "ensemble", "prophet")
    ], ignore_index=True)
    output = tmp_path / "validation_comparison.png"

    plot_validation_comparison(history, predictions, output, 1)

    assert output.exists()


def test_weighted_average_renormalizes_available_models():
    """ Checks that ensemble weights are renormalized for the models present """
    forecasts = {
        "seasonal_naive": pd.Series([10.0, 20.0]),
        "sarima": pd.Series([20.0, 40.0]),
    }
    result = weighted_average(forecasts, {"seasonal_naive": 0.10, "sarima": 0.25})
    assert result.tolist() == [17.142857142857142, 34.285714285714285]


def test_evaluation_includes_fixed_weight_ensemble():
    """ Checks that evaluation includes a bounded ensemble forecast """
    series = pd.Series(
        np.arange(208, dtype=float),
        index=pd.date_range("2018-01-07", periods=208, freq="7D"),
    )
    metrics, predictions = evaluate_series(
        series,
        capacity=500.0,
        horizon=52,
        origins=1,
        models=("seasonal_naive", "sarima"),
    )

    # The ensemble appears beside the models
    assert "ensemble" in metrics["model"].tolist()
    ensemble_predictions = predictions[predictions["model"] == "ensemble"]
    assert len(ensemble_predictions) == 52
    assert ensemble_predictions["prediction"].between(0, 500).all()


def test_test_evaluation_can_run_without_ensemble(monkeypatch):
    """ Checks that the locked test model can run without ensemble work """
    train = pd.Series([10.0, 20.0], index=pd.date_range("2020-01-05", periods=2, freq="7D"))
    test = pd.Series([20.0, 30.0], index=pd.date_range("2020-01-19", periods=2, freq="7D"))
    monkeypatch.setattr(
        "backend.modeling.forecasting.model_forecasts",
        lambda capacity, models: {"prophet": lambda series, steps: np.repeat(series.iloc[-1], steps)},
    )

    metrics, predictions = evaluate_origin(
        train,
        test,
        capacity=100.0,
        models=("prophet",),
        origin=0,
        include_ensemble=False,
    )

    assert metrics["model"].tolist() == ["prophet"]
    assert predictions["model"].unique().tolist() == ["prophet"]


def test_evaluation_cli_has_fixed_split_commands():
    """ Checks that validation and test commands are exposed """
    parser = build_parser()

    validation = parser.parse_args(["evaluate-validation", "--limit", "1"])
    assert validation.split == "validation"
    assert validation.models == ",".join(EVALUATION_SETTINGS["validation_models"])

    test = parser.parse_args(["evaluate-test", "--limit", "1"])
    assert test.split == "test"
    assert test.models == "prophet"

    summary = parser.parse_args(["evaluation-summary", "--split", "validation"])
    assert summary.split == "validation"


def test_cached_evaluation_processes_next_reservoir(monkeypatch, tmp_path):
    """ Checks that cached split runs advance one reservoir at a time """
    ids = EVALUATION_SETTINGS["validation_ids"]
    dates = pd.date_range("2020-01-05", periods=156, freq="7D")
    water = pd.concat([
        pd.DataFrame({"id": reservoir_id, "date": dates, "storage": np.arange(156, dtype=float)})
        for reservoir_id in ids
    ], ignore_index=True)
    reservoirs = pd.DataFrame({
        "id": ids,
        "capacity": [500.0] * len(ids),
        "autonomous_community": ["andalucia"] * len(ids),
    })

    def fake_evaluate_origin(train, test, capacity, models, origin, failures=None, include_ensemble=True):
        metrics = pd.DataFrame({"model": [models[0]], "origin": [0], "mae": [1.0]})
        predictions = pd.DataFrame({
            "date": test.index,
            "actual": test.to_numpy(),
            "prediction": test.to_numpy(),
            "model": models[0],
            "origin": 0,
        })
        return metrics, predictions

    monkeypatch.setattr("backend.modeling.evaluation_cache.evaluate_origin", fake_evaluate_origin)
    monkeypatch.setitem(PATHS, "outputs", tmp_path)

    first = run_cached_evaluation(water, reservoirs, "validation", ("seasonal_naive",), 52, 1, 1)
    second = run_cached_evaluation(water, reservoirs, "validation", ("seasonal_naive",), 52, 1, 1)

    assert first["processed"] == 1
    assert second["processed"] == 1
    assert second["cached"] == 2
    assert (tmp_path / "evaluation_validation_metrics.parquet").exists()


def test_evaluation_summary_ignores_incomplete_configurations(monkeypatch, tmp_path):
    """ Checks that provisional summaries ignore incomplete cache manifests """
    cache = tmp_path / "evaluation_cache" / "validation" / "old_configuration"
    cache.mkdir(parents=True)
    (cache / "reservoir_3.json").write_text(
        '{"status": "complete", "models": ["sarima"], "origins": 3}',
        encoding="utf-8",
    )
    monkeypatch.setitem(PATHS, "outputs", tmp_path)

    summary, counts = cached_summary("validation")

    assert summary.empty
    assert counts == {"configurations": 1, "ignored_configurations": 1}


def test_evaluation_summary_markdown_is_readable_and_idempotent(monkeypatch, tmp_path):
    """ Checks that the Markdown summary is rewritten without table separators """
    summary = pd.DataFrame([{
        "configuration": "h52_o3",
        "models": "sarima",
        "reservoirs": 2,
        "origins": 3,
        "horizon": 52,
        "model": "sarima",
        "metric_rows": 6,
        "reservoirs_with_metrics": 2,
        "mae_mean": 1.23456,
        "mae_median": 1.2,
        "rmse_mean": 2.34567,
        "rmse_median": 2.3,
        "mape_mean": 3.45678,
        "nmae_capacity_mean": 0.1,
        "nrmse_capacity_mean": 0.2,
    }])
    monkeypatch.setattr(
        "backend.modeling.evaluation_summary.cached_summary",
        lambda split: (summary, {"configurations": 1, "ignored_configurations": 0}),
    )
    monkeypatch.setitem(PATHS, "outputs", tmp_path)

    path = write_markdown_summary("validation")
    first = path.read_text(encoding="utf-8")
    write_markdown_summary("validation")

    assert path == tmp_path / "evaluation_summary_validation.md"
    assert path.read_text(encoding="utf-8") == first
    assert "|" not in first
    assert "Mae Mean: 1.2346" in first
    assert "Nrmse Capacity Mean: 0.2000" in first


def test_validation_model_selection_uses_complete_metrics(monkeypatch, tmp_path):
    """ Checks that validation analysis ranks models against the baseline """
    rows = []
    for reservoir_id in EVALUATION_SETTINGS["validation_ids"]:
        for origin in range(FORECAST_SETTINGS["backtest_origins"]):
            for model, error in (("seasonal_naive", 2.0), ("prophet", 1.0), ("sarima", 1.5), ("ensemble", 1.2)):
                rows.append({
                    "id": reservoir_id,
                    "origin": origin,
                    "model": model,
                    "mae": error,
                    "rmse": error,
                    "nmae_capacity": error,
                    "nrmse_capacity": error,
                })
    monkeypatch.setitem(PATHS, "outputs", tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(tmp_path / "evaluation_validation_metrics.parquet", index=False)

    decision = analyze_validation()

    assert decision["selected_model"] == "prophet"
    assert "Preferred individual model: prophet" in format_decision_markdown(decision)
    assert "|" not in format_decision_markdown(decision)


def test_test_analysis_checks_the_locked_model_scope(monkeypatch, tmp_path):
    """ Checks that test analysis accepts only complete Prophet metrics """
    rows = []
    for reservoir_id in EVALUATION_SETTINGS["test_ids"]:
        for origin in range(FORECAST_SETTINGS["backtest_origins"]):
            rows.append({
                "id": reservoir_id,
                "origin": origin,
                "model": "prophet",
                "mae": 1.0,
                "rmse": 2.0,
                "mape": 0.1,
                "nmae_capacity": 0.01,
                "nrmse_capacity": 0.02,
            })
    monkeypatch.setitem(PATHS, "outputs", tmp_path)
    pd.DataFrame(rows).to_parquet(tmp_path / "evaluation_test_metrics.parquet", index=False)

    analysis = analyze_test()

    assert analysis["reservoirs"] == len(EVALUATION_SETTINGS["test_ids"])
    assert analysis["metrics_rows"] == len(rows)
    assert "The locked Prophet model" in format_test_analysis_markdown(analysis)
    assert "|" not in format_test_analysis_markdown(analysis)


def test_planner_respects_transfer_limit():
    """ Checks that the planner respects the transfer fraction """
    state = pd.DataFrame({
        "id": [1, 2],
        "capacity": [1000.0, 1000.0],
        "latitude": [37.0, 37.01],
        "longitude": [-4.0, -4.01],
        "last_known_value": [900.0, 20.0],
        "low_forecasted_value": [800.0, 10.0],
        "median_forecasted_value": [900.0, 20.0],
        "critical": [False, True],
        "worrying": [False, True],
        "eligible_donor": [True, False],
    }).set_index("id")

    # One iteration and a ten percent donor limit cap the transfer at 100
    final, log = plan_transfers(state, TransferConfig(cost_threshold=1000, max_transfer_fraction=0.1), 1)
    assert len(log) <= 1
    if not log.empty:
        assert log.iloc[0]["volume"] <= 100
        assert final.loc[final["id"] == 1, "last_known_value"].iloc[0] >= 800


def test_transfer_command_forecasts_only_selected_reservoirs(monkeypatch, tmp_path):
    """ Checks that transfer forecasts are limited to the selected region """
    water = pd.DataFrame({
        "id": [1, 1, 2, 2],
        "date": pd.date_range("2020-01-05", periods=4, freq="7D"),
        "storage": [100.0, 110.0, 200.0, 210.0],
    })
    reservoirs = pd.DataFrame({
        "id": [1, 2],
        "capacity": [500.0, 500.0],
        "latitude": [37.0, 38.0],
        "longitude": [-4.0, -5.0],
        "autonomous_community": ["andalucia", "galicia"],
    })
    forecasted_ids = []
    planned_ids = []

    def fake_forecast(series, capacity, models):
        """ Returns a minimal forecast for the command test """
        forecasted_ids.append({100: 1, 200: 2}[int(series.iloc[0])])
        return pd.DataFrame({"date": [pd.Timestamp("2020-02-02")], "ensemble": [100.0]})

    def fake_prepare_state(selected, summary):
        """ Records the IDs passed to transfer planning """
        planned_ids.extend(selected["id"].tolist())
        return selected.assign(
            last_known_value=100.0,
            low_forecasted_value=10.0,
            median_forecasted_value=100.0,
        ).set_index("id")

    monkeypatch.setattr("backend.cli.load_curated", lambda: (water, reservoirs))
    monkeypatch.setattr("backend.cli.forecast_one_year", fake_forecast)
    monkeypatch.setattr("backend.cli.prepare_state", fake_prepare_state)
    monkeypatch.setattr(
        "backend.cli.plan_transfers",
        lambda state, config, max_iterations: (state.reset_index(), pd.DataFrame()),
    )
    monkeypatch.setitem(PATHS, "outputs", tmp_path)
    monkeypatch.setitem(PATHS, "cache", tmp_path / "cache")

    args = type("Args", (), {
        "community": "andalucia",
        "latitude": None,
        "longitude": None,
        "radius_km": None,
        "models": "prophet",
        "max_distance_km": None,
        "max_iterations": 10,
    })()
    command_plan(args)
    command_plan(args)

    assert forecasted_ids == [1]
    assert planned_ids == [1, 1]
