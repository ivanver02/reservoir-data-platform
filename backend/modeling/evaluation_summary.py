"""Summaries for validation and test evaluations"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from backend.config.settings import EVALUATION_SETTINGS, PATHS


def _cache_root(split: str) -> Path:
    """ Returns the cache root for an evaluation split """
    if split not in {"validation", "test"}:
        raise ValueError("split must be validation or test")
    return PATHS["outputs"] / EVALUATION_SETTINGS["cache_directory"] / split


def _read_valid_configuration(directory: Path) -> tuple[pd.DataFrame, dict[str, object]] | None:
    """ Reads one finished model setup from reservoir caches """
    metric_frames = []
    manifests = []
    for manifest_path in sorted(directory.glob("reservoir_*.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("status") != "complete"
            or manifest.get("failure_recorded") is not True
            or len(manifest.get("completed_origins", ())) != manifest.get("origins")
        ):
            continue
        manifests.append(manifest)
        for origin in manifest["completed_origins"]:
            path = directory / f"reservoir_{manifest['reservoir_id']}_origin_{origin}_metrics.parquet"
            if path.exists():
                metric_frames.append(pd.read_parquet(path))

    if not metric_frames:
        return None
    metrics = pd.concat(metric_frames, ignore_index=True)
    metadata = {
        "configuration": directory.name,
        "models": ",".join(manifests[0]["models"]),
        "reservoirs": len(manifests),
        "origins": manifests[0]["origins"],
        "horizon": manifests[0]["horizon"],
    }
    return metrics, metadata


def cached_summary(split: str) -> tuple[pd.DataFrame, dict[str, int]]:
    """ Builds metric summaries from finished reservoir caches """
    rows = []
    root = _cache_root(split)
    ignored = 0
    configurations = 0
    if not root.exists():
        return pd.DataFrame(), {"configurations": 0, "ignored_configurations": 0}

    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        configurations += 1
        result = _read_valid_configuration(directory)
        if result is None:
            ignored += 1
            continue
        metrics, metadata = result
        for model, group in metrics.groupby("model"):
            rows.append({
                **metadata,
                "model": model,
                "metric_rows": len(group),
                "reservoirs_with_metrics": group["id"].nunique(),
                "mae_mean": group["mae"].mean(),
                "mae_median": group["mae"].median(),
                "rmse_mean": group["rmse"].mean(),
                "rmse_median": group["rmse"].median(),
                "mape_mean": group["mape"].mean(),
                "nmae_capacity_mean": group["nmae_capacity"].mean(),
                "nrmse_capacity_mean": group["nrmse_capacity"].mean(),
            })
    return pd.DataFrame(rows), {
        "configurations": configurations,
        "ignored_configurations": ignored,
    }


def format_summary(split: str) -> str:
    """ Formats a metric summary for terminal output """
    summary, counts = cached_summary(split)
    lines = [f"Cached {split} evaluation summary"]
    lines.append(f"Configurations found: {counts['configurations']}")
    lines.append(f"Configurations with valid completed caches: {counts['configurations'] - counts['ignored_configurations']}")
    if summary.empty:
        lines.append("No complete cached metrics are available")
        return "\n".join(lines)

    display = summary.copy()
    numeric = [
        "mae_mean", "mae_median", "rmse_mean", "rmse_median", "mape_mean",
        "nmae_capacity_mean", "nrmse_capacity_mean",
    ]
    display[numeric] = display[numeric].round(4)
    lines.append(display.to_string(index=False))
    return "\n".join(lines)


def format_markdown_summary(split: str) -> str:
    """ Formats a metric summary as readable Markdown lines """
    summary, counts = cached_summary(split)
    lines = [
        f"# Cached {split} evaluation summary",
        "",
        f"Configurations found: {counts['configurations']}",
        f"Configurations with valid completed caches: {counts['configurations'] - counts['ignored_configurations']}",
    ]

    if summary.empty:
        lines.extend(["", "No complete cached metrics are available"])
        return "\n".join(lines) + "\n"

    numeric = [
        "mae_mean", "mae_median", "rmse_mean", "rmse_median", "mape_mean",
        "nmae_capacity_mean", "nrmse_capacity_mean",
    ]

    for _, row in summary.iterrows():
        lines.extend([
            "",
            f"## {row['configuration']} - {row['model']}",
            f"Configuration: {row['configuration']}",
            f"Models: {row['models']}",
            f"Reservoirs: {row['reservoirs']}",
            f"Origins: {row['origins']}",
            f"Horizon: {row['horizon']}",
            f"Metric rows: {row['metric_rows']}",
            f"Reservoirs with metrics: {row['reservoirs_with_metrics']}",
        ])
        for metric in numeric:
            lines.append(f"{metric.replace('_', ' ').title()}: {row[metric]:.4f}")

    return "\n".join(lines) + "\n"


def write_markdown_summary(split: str) -> Path:
    """ Writes a readable Markdown summary and returns its path """
    output = PATHS["outputs"] / f"evaluation_summary_{split}.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(format_markdown_summary(split), encoding="utf-8")
    return output
