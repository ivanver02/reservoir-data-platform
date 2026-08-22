"""Validation analysis and model selection"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from backend.config.settings import EVALUATION_SETTINGS, FORECAST_SETTINGS, PATHS


DECISION_METRICS = (
    "mae",
    "rmse",
    "nmae_capacity",
    "nrmse_capacity",
)
BASELINE_MODEL = "seasonal_naive"
ENSEMBLE_MODEL = "ensemble"


def _metrics_path(split: str) -> Path:
    """ Returns the aggregated metrics path for one split """
    if split not in {"validation", "test"}:
        raise ValueError("split must be validation or test")
    return PATHS["outputs"] / f"evaluation_{split}_metrics.parquet"


def _validate_validation_metrics(metrics: pd.DataFrame) -> None:
    """ Checks that validation metrics cover the declared evaluation scope """
    required = set(DECISION_METRICS) | {"id", "model", "origin"}
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"validation metrics are missing columns: {sorted(missing)}")

    expected_ids = set(EVALUATION_SETTINGS["validation_ids"])
    actual_ids = set(metrics["id"].astype(int))
    if actual_ids != expected_ids:
        raise ValueError(
            f"validation metrics cover IDs {sorted(actual_ids)}, "
            f"expected {sorted(expected_ids)}"
        )

    expected_models = set(EVALUATION_SETTINGS["validation_models"]) | {ENSEMBLE_MODEL}
    actual_models = set(metrics["model"])
    if actual_models != expected_models:
        raise ValueError(
            f"validation metrics contain models {sorted(actual_models)}, "
            f"expected {sorted(expected_models)}"
        )

    expected_origins = set(range(FORECAST_SETTINGS["backtest_origins"]))
    actual_origins = set(metrics["origin"].astype(int))
    if actual_origins != expected_origins:
        raise ValueError(
            f"validation metrics contain origins {sorted(actual_origins)}, "
            f"expected {sorted(expected_origins)}"
        )

    counts = metrics.groupby(["id", "model"])["origin"].nunique()
    if not (counts == len(expected_origins)).all():
        raise ValueError("every validation reservoir and model needs every origin")


def _rank_models(metrics: pd.DataFrame) -> pd.DataFrame:
    """ Ranks models by their average error relative to the baseline """
    averages = metrics.groupby("model", as_index=True)[list(DECISION_METRICS)].mean()
    baseline = averages.loc[BASELINE_MODEL]
    ratios = averages[list(DECISION_METRICS)].div(baseline)
    ranking = averages.copy()
    ranking["composite_score"] = ratios.mean(axis=1)
    ranking["improvement_vs_baseline"] = 1.0 - ranking["composite_score"]
    return ranking.sort_values(
        ["composite_score", "nrmse_capacity", "rmse"],
        kind="mergesort",
    )


def analyze_validation() -> dict[str, object]:
    """ Analyzes complete validation metrics and records a model decision """
    path = _metrics_path("validation")
    if not path.exists():
        raise FileNotFoundError(f"validation metrics not found: {path}")

    metrics = pd.read_parquet(path)
    _validate_validation_metrics(metrics)
    ranking = _rank_models(metrics)
    selected_model = str(ranking.index[0])

    return {
        "split": "validation",
        "reservoirs": int(metrics["id"].nunique()),
        "origins": int(metrics["origin"].nunique()),
        "metrics_rows": len(metrics),
        "decision_metrics": list(DECISION_METRICS),
        "baseline_model": BASELINE_MODEL,
        "ranking": ranking.reset_index().to_dict(orient="records"),
        "selected_model": selected_model,
        "test_models": [selected_model],
        "ensemble_weights": dict(FORECAST_SETTINGS["ensemble_weights"]),
        "ensemble_decision": "not_run_on_test",
    }


def _metric_label(metric: str) -> str:
    """ Returns the readable label for one metric """
    return {
        "mae": "MAE mean",
        "rmse": "RMSE mean",
        "mape": "MAPE mean",
        "nmae_capacity": "NMAE capacity mean",
        "nrmse_capacity": "NRMSE capacity mean",
        "composite_score": "Composite score",
        "improvement_vs_baseline": "Improvement versus baseline",
    }.get(metric, metric)


def format_decision_markdown(decision: dict[str, object]) -> str:
    """ Formats the validation decision as readable Markdown """
    lines = [
        "# Validation model decision",
        "",
        f"Evaluation split: {decision['split']}",
        f"Reservoirs analyzed: {decision['reservoirs']}",
        f"Rolling origins analyzed: {decision['origins']}",
        f"Metric rows analyzed: {decision['metrics_rows']}",
        "",
        "## Decision",
        "",
        f"Preferred individual model: {decision['selected_model']}",
        "Reason: it has the lowest composite score across the four decision metrics",
        "Recommended test model argument: " + ",".join(decision["test_models"]),
        "Ensemble decision: do not run the ensemble on test",
        "Reason: test measures the locked model after validation and avoids repeating the full model comparison",
        "",
        "## Decision rule",
        "",
        "Baseline: seasonal_naive",
        "Each model score is its mean metric divided by the corresponding seasonal naive mean",
        "Composite score: mean of the MAE, RMSE, NMAE capacity and NRMSE capacity ratios",
        "Lower scores are better",
        "Capacity-normalized metrics are included so large reservoirs do not dominate the decision",
        "MAPE is reported in the general summary but is not used for selection because it is unstable near zero storage",
        "",
        "## Model results",
    ]

    for row in decision["ranking"]:
        lines.extend(["", f"### {row['model']}"])
        for metric in (*DECISION_METRICS, "composite_score", "improvement_vs_baseline"):
            value = row[metric]
            if metric == "improvement_vs_baseline":
                lines.append(f"{_metric_label(metric)}: {value:.4f}")
            else:
                lines.append(f"{_metric_label(metric)}: {value:.4f}")

    lines.extend([
        "",
        "## Why this does not open the test set",
        "",
        "The test set remains untouched by this analysis. The decision uses only the fixed validation reservoirs and their three rolling origins.",
        "The test command should be run with the recommended model argument after this report has been reviewed.",
        "",
        "## Limitations",
        "",
        "The decision summarizes ten reservoirs and three rolling origins, so it is not evidence of operational performance.",
        "The composite score gives equal weight to four metrics and does not optimize ensemble weights.",
        "The preferred model is a validation choice and must be checked again on the held-out test results.",
    ])
    return "\n".join(lines) + "\n"


def write_validation_decision() -> tuple[Path, Path]:
    """ Writes the validation decision as Markdown and JSON """
    decision = analyze_validation()
    PATHS["outputs"].mkdir(parents=True, exist_ok=True)
    markdown_path = PATHS["outputs"] / "validation_model_decision.md"
    json_path = PATHS["outputs"] / "validation_model_decision.json"
    markdown_path.write_text(format_decision_markdown(decision), encoding="utf-8")
    json_path.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    return markdown_path, json_path


def _validate_test_metrics(metrics: pd.DataFrame) -> None:
    """ Checks that test metrics cover the locked Prophet evaluation """
    required = set(DECISION_METRICS) | {"id", "model", "origin"}
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"test metrics are missing columns: {sorted(missing)}")

    expected_ids = set(EVALUATION_SETTINGS["test_ids"])
    actual_ids = set(metrics["id"].astype(int))
    if actual_ids != expected_ids:
        raise ValueError(
            f"test metrics cover {len(actual_ids)} IDs, expected {len(expected_ids)}"
        )

    actual_models = set(metrics["model"])
    if actual_models != {"prophet"}:
        raise ValueError(f"test metrics contain unexpected models: {sorted(actual_models)}")

    expected_origins = set(range(FORECAST_SETTINGS["backtest_origins"]))
    actual_origins = set(metrics["origin"].astype(int))
    if actual_origins != expected_origins:
        raise ValueError(
            f"test metrics contain origins {sorted(actual_origins)}, "
            f"expected {sorted(expected_origins)}"
        )

    counts = metrics.groupby("id")["origin"].nunique()
    if not (counts == len(expected_origins)).all():
        raise ValueError("every test reservoir needs every origin")

    if metrics[list(DECISION_METRICS)].isna().any().any():
        raise ValueError("test metrics contain missing values")


def _metric_statistics(metrics: pd.DataFrame) -> dict[str, dict[str, float]]:
    """ Computes distribution statistics for test metrics """
    statistics = {}
    for metric in (*DECISION_METRICS, "mape"):
        values = metrics[metric]
        statistics[metric] = {
            "mean": float(values.mean()),
            "median": float(values.median()),
            "std": float(values.std()),
            "p90": float(values.quantile(0.90)),
            "p95": float(values.quantile(0.95)),
            "maximum": float(values.max()),
        }
    return statistics


def _metric_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    """ Converts metric rows to JSON compatible records """
    records = []
    for record in frame.to_dict(orient="records"):
        records.append({
            key: float(value) if isinstance(value, float) else int(value) if hasattr(value, "item") else value
            for key, value in record.items()
        })
    return records


def analyze_test() -> dict[str, object]:
    """ Analyzes the locked Prophet model on the complete test split """
    path = _metrics_path("test")
    if not path.exists():
        raise FileNotFoundError(f"test metrics not found: {path}")

    metrics = pd.read_parquet(path)
    _validate_test_metrics(metrics)
    by_origin = metrics.groupby("origin")[list(DECISION_METRICS)].mean().reset_index()
    by_reservoir = (
        metrics.groupby("id")[list(DECISION_METRICS)]
        .mean()
        .sort_values("nrmse_capacity", ascending=False)
        .reset_index()
    )

    validation_path = _metrics_path("validation")
    validation_comparison = {}
    if validation_path.exists():
        validation = pd.read_parquet(validation_path)
        validation = validation[validation["model"] == "prophet"]
        if not validation.empty:
            test_means = metrics[list(DECISION_METRICS)].mean()
            validation_means = validation[list(DECISION_METRICS)].mean()
            validation_comparison = {
                metric: {
                    "validation_mean": float(validation_means[metric]),
                    "test_mean": float(test_means[metric]),
                    "test_to_validation_ratio": float(test_means[metric] / validation_means[metric]),
                }
                for metric in DECISION_METRICS
            }

    return {
        "split": "test",
        "model": "prophet",
        "reservoirs": int(metrics["id"].nunique()),
        "origins": int(metrics["origin"].nunique()),
        "metrics_rows": len(metrics),
        "statistics": _metric_statistics(metrics),
        "by_origin": _metric_records(by_origin),
        "worst_reservoirs_by_nrmse_capacity": _metric_records(by_reservoir.head(10)),
        "validation_comparison": validation_comparison,
    }


def format_test_analysis_markdown(analysis: dict[str, object]) -> str:
    """ Formats the test analysis as readable Markdown """
    lines = [
        "# Test analysis",
        "",
        f"Evaluation split: {analysis['split']}",
        f"Model evaluated: {analysis['model']}",
        f"Reservoirs analyzed: {analysis['reservoirs']}",
        f"Rolling origins analyzed: {analysis['origins']}",
        f"Metric rows analyzed: {analysis['metrics_rows']}",
        "",
        "## Conclusions",
        "",
        "The locked Prophet model was evaluated on every eligible test reservoir without fitting SARIMA, seasonal naive or the ensemble.",
        "The test measures generalization of the validation decision and does not select another model.",
        "The test errors are compared with validation errors to identify distribution shift and weak cases.",
        "",
        "## Metric distribution",
    ]

    for metric, values in analysis["statistics"].items():
        lines.extend([
            "",
            f"### {_metric_label(metric)}",
            f"Mean: {values['mean']:.4f}",
            f"Median: {values['median']:.4f}",
            f"Standard deviation: {values['std']:.4f}",
            f"90th percentile: {values['p90']:.4f}",
            f"95th percentile: {values['p95']:.4f}",
            f"Maximum: {values['maximum']:.4f}",
        ])

    lines.extend(["", "## Results by rolling origin"])
    for row in analysis["by_origin"]:
        lines.extend([
            "",
            f"### Origin {row['origin']}",
            f"MAE mean: {row['mae']:.4f}",
            f"RMSE mean: {row['rmse']:.4f}",
            f"NMAE capacity mean: {row['nmae_capacity']:.4f}",
            f"NRMSE capacity mean: {row['nrmse_capacity']:.4f}",
        ])

    lines.extend(["", "## Comparison with validation"])
    if analysis["validation_comparison"]:
        for metric, values in analysis["validation_comparison"].items():
            lines.extend([
                "",
                f"### {_metric_label(metric)}",
                f"Validation mean: {values['validation_mean']:.4f}",
                f"Test mean: {values['test_mean']:.4f}",
                f"Test to validation ratio: {values['test_to_validation_ratio']:.4f}",
            ])
    else:
        lines.extend(["", "Validation comparison: not available"])

    lines.extend(["", "## Worst reservoirs by NRMSE capacity"])
    for row in analysis["worst_reservoirs_by_nrmse_capacity"]:
        lines.extend([
            "",
            f"### Reservoir {row['id']}",
            f"MAE mean: {row['mae']:.4f}",
            f"RMSE mean: {row['rmse']:.4f}",
            f"NMAE capacity mean: {row['nmae_capacity']:.4f}",
            f"NRMSE capacity mean: {row['nrmse_capacity']:.4f}",
        ])

    lines.extend([
        "",
        "## Interpretation",
        "",
        "Prophet remains the locked model because this split evaluates a decision made before test execution.",
        "Higher test error than validation indicates that performance is not uniform across Andalusian reservoirs and should not be presented as a guaranteed operational accuracy.",
        "The origin and reservoir distributions identify where additional error analysis should focus, but they do not justify changing the model after seeing test results.",
        "The forecasts remain decision support and should be accompanied by their uncertainty and physical constraints.",
    ])
    return "\n".join(lines) + "\n"


def write_test_analysis() -> tuple[Path, Path]:
    """ Writes the test analysis as Markdown and JSON """
    analysis = analyze_test()
    PATHS["outputs"].mkdir(parents=True, exist_ok=True)
    markdown_path = PATHS["outputs"] / "test_model_analysis.md"
    json_path = PATHS["outputs"] / "test_model_analysis.json"
    markdown_path.write_text(format_test_analysis_markdown(analysis), encoding="utf-8")
    json_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    return markdown_path, json_path
