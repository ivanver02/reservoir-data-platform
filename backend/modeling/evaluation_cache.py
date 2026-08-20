"""Evaluation runs for the Andalusia splits"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from backend.config.settings import EVALUATION_SETTINGS, PATHS
from backend.modeling.evaluation import rolling_origins
from backend.modeling.forecasting import evaluate_origin


def evaluation_split_ids(
    water: pd.DataFrame,
    reservoirs: pd.DataFrame,
    split: str,
    horizon: int,
    origins: int,
) -> list[int]:
    """ Returns eligible IDs for one Andalusia split """
    if split not in {"validation", "test"}:
        raise ValueError("split must be validation or test")

    configured_ids = EVALUATION_SETTINGS[f"{split}_ids"]
    community = EVALUATION_SETTINGS["community"]
    community_ids = set(
        reservoirs.loc[
            reservoirs["autonomous_community"].astype(str).str.casefold() == community,
            "id",
        ].astype(int)
    )
    water_groups = water.sort_values("date").groupby("id")["storage"]
    available_ids = {
        int(reservoir_id)
        for reservoir_id, series in water_groups
        if len(rolling_origins(series, horizon=horizon, origins=origins)) > 0
    }

    missing_community = set(configured_ids) - community_ids
    missing_water = set(configured_ids) - available_ids
    if missing_community or missing_water:
        raise ValueError(
            f"fixed {split} split is invalid: "
            f"community={sorted(missing_community)}, "
            f"water={sorted(missing_water)}"
        )
    return [int(reservoir_id) for reservoir_id in configured_ids]


def _cache_directory(
    split: str,
    models: tuple[str, ...],
    horizon: int,
    origins: int,
    include_ensemble: bool = True,
) -> Path:
    """ Returns the cache directory for one evaluation setup """
    model_key = "_".join(models)
    ensemble_suffix = "" if include_ensemble else "_without_ensemble"
    return PATHS["outputs"] / EVALUATION_SETTINGS["cache_directory"] / split / f"{model_key}_h{horizon}_o{origins}{ensemble_suffix}"


def _cache_paths(directory: Path, reservoir_id: int) -> tuple[Path, Path, Path]:
    """ Returns paths for one reservoir """
    stem = f"reservoir_{reservoir_id}"
    return directory / f"{stem}_metrics.parquet", directory / f"{stem}_predictions.parquet", directory / f"{stem}.json"


def _origin_cache_paths(directory: Path, reservoir_id: int, origin: int) -> tuple[Path, Path]:
    """ Returns metric and prediction paths for one rolling origin """
    stem = f"reservoir_{reservoir_id}_origin_{origin}"
    return directory / f"{stem}_metrics.parquet", directory / f"{stem}_predictions.parquet"


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    """ Writes a Parquet file through a temporary path """
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _is_cached(
    manifest_path: Path,
    models: tuple[str, ...],
    horizon: int,
    origins: int,
    include_ensemble: bool = True,
) -> bool:
    """ Checks one matching cache entry """
    if not manifest_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return (
        manifest.get("status") == "complete"
        and manifest.get("failure_recorded") is True
        and tuple(manifest.get("models", ())) == models
        and manifest.get("horizon") == horizon
        and manifest.get("origins") == origins
        and manifest.get("include_ensemble", True) is include_ensemble
        and len(manifest.get("completed_origins", ())) == origins
    )


def _aggregate_cache(directory: Path, split: str) -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
    """ Combines finished reservoir caches into split outputs """
    metric_frames = []
    prediction_frames = []
    completed_ids = []
    for manifest_path in sorted(directory.glob("reservoir_*.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            continue
        reservoir_id = int(manifest["reservoir_id"])
        for origin in manifest.get("completed_origins", []):
            metrics_path, predictions_path = _origin_cache_paths(directory, reservoir_id, origin)
            if not metrics_path.exists() or not predictions_path.exists():
                continue
            metric_frames.append(pd.read_parquet(metrics_path))
            prediction_frames.append(pd.read_parquet(predictions_path))
        completed_ids.append(reservoir_id)

    metrics = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    output_prefix = PATHS["outputs"] / f"evaluation_{split}"
    if not metrics.empty:
        _write_parquet(metrics, output_prefix.with_name(f"{output_prefix.name}_metrics.parquet"))
    if not predictions.empty:
        _write_parquet(predictions, output_prefix.with_name(f"{output_prefix.name}_predictions.parquet"))
    if completed_ids:
        output_prefix.with_name(f"{output_prefix.name}_manifest.json").write_text(
            json.dumps({
                "split": split,
                "completed_reservoir_ids": completed_ids,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            }, indent=2),
            encoding="utf-8",
        )
    return metrics, predictions, completed_ids


def run_cached_evaluation(
    water: pd.DataFrame,
    reservoirs: pd.DataFrame,
    split: str,
    models: tuple[str, ...],
    horizon: int,
    origins: int,
    limit: int,
    include_ensemble: bool = True,
) -> dict[str, object]:
    """ Evaluates the next reservoirs and refreshes split outputs """
    reservoir_ids = evaluation_split_ids(water, reservoirs, split, horizon, origins)
    directory = _cache_directory(split, models, horizon, origins, include_ensemble)
    directory.mkdir(parents=True, exist_ok=True)
    pending = []
    for reservoir_id in reservoir_ids:
        _, _, manifest_path = _cache_paths(directory, reservoir_id)
        if not _is_cached(manifest_path, models, horizon, origins, include_ensemble):
            pending.append(reservoir_id)

    selected_ids = pending if limit == 0 else pending[:limit]
    for reservoir_id in selected_ids:
        group = water[water["id"] == reservoir_id].sort_values("date")
        metadata = reservoirs[reservoirs["id"] == reservoir_id]
        series = group.set_index("date")["storage"]
        rolling_splits = rolling_origins(series, horizon=horizon, origins=origins)
        _, _, manifest_path = _cache_paths(directory, reservoir_id)
        manifest = {
            "status": "in_progress",
            "split": split,
            "reservoir_id": reservoir_id,
            "models": list(models),
            "horizon": horizon,
            "origins": origins,
            "include_ensemble": include_ensemble,
            "completed_origins": [],
            "failures": [],
            "failure_recorded": True,
        }
        if manifest_path.exists():
            old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                tuple(old_manifest.get("models", ())) == models
                and old_manifest.get("horizon") == horizon
                and old_manifest.get("origins") == origins
                and old_manifest.get("include_ensemble", True) is include_ensemble
            ):
                manifest.update(old_manifest)

        for origin, (train, test) in enumerate(rolling_splits):
            if origin in manifest["completed_origins"]:
                continue
            failures: list[dict[str, object]] = []
            current_metrics, predictions = evaluate_origin(
                train,
                test,
                float(metadata.iloc[0]["capacity"]),
                models,
                origin,
                failures,
                include_ensemble,
            )
            current_metrics["id"] = reservoir_id
            current_metrics["split"] = split
            predictions["id"] = reservoir_id
            predictions["split"] = split
            metrics_path, predictions_path = _origin_cache_paths(directory, reservoir_id, origin)
            _write_parquet(current_metrics, metrics_path)
            _write_parquet(predictions, predictions_path)
            manifest["completed_origins"].append(origin)
            manifest["failures"].extend(failures)
            manifest["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        manifest["status"] = "complete"
        manifest["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    metrics, predictions, completed_ids = _aggregate_cache(directory, split)
    return {
        "split": split,
        "requested": len(reservoir_ids),
        "processed": len(selected_ids),
        "remaining": len(pending) - len(selected_ids),
        "cached": len(completed_ids),
        "metrics_rows": len(metrics),
        "prediction_rows": len(predictions),
        "all_cached": not pending or len(selected_ids) == len(pending),
    }
