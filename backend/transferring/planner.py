"""Greedy transfer planning"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import haversine_distances

from backend.config.settings import EARTH_RADIUS_KM, TRANSFER_OPTIMIZATION_THRESHOLDS, TRANSFER_PHYSICAL_LIMITS


@dataclass(frozen=True)
class TransferConfig:
    critical: float = TRANSFER_OPTIMIZATION_THRESHOLDS["critical"]
    worrying: float = TRANSFER_OPTIMIZATION_THRESHOLDS["worrying"]
    donor_minimum: float = TRANSFER_OPTIMIZATION_THRESHOLDS["could_give_if_critical"]
    cost_threshold: float = TRANSFER_OPTIMIZATION_THRESHOLDS["cost_threshold"]
    max_distance_km: float | None = TRANSFER_PHYSICAL_LIMITS["max_distance_km"]
    max_transfer_fraction: float = TRANSFER_PHYSICAL_LIMITS["max_transfer_fraction_of_donor_capacity"]
    max_transfer_volume: float | None = TRANSFER_PHYSICAL_LIMITS["max_transfer_volume"]


def haversine_km(frame: pd.DataFrame) -> np.ndarray:
    """ Returns the distance matrix for reservoir coordinates """
    coordinates = np.radians(frame[["latitude", "longitude"]].to_numpy(float))
    return haversine_distances(coordinates, coordinates) * EARTH_RADIUS_KM


def prepare_state(
    reservoirs: pd.DataFrame,
    forecast_summary: pd.DataFrame,
    config: TransferConfig = TransferConfig(),
) -> pd.DataFrame:
    """ Builds the planning state from metadata and forecast summaries """
    required = {"id", "capacity", "latitude", "longitude"}
    missing = required.difference(reservoirs.columns)
    if missing:
        raise ValueError(f"missing reservoir columns: {sorted(missing)}")

    # Join forecasts before deriving status
    state = reservoirs.merge(forecast_summary, on="id", how="inner").copy().set_index("id")
    state["need_rate"] = state["low_forecasted_value"] / state["capacity"]
    state["donor_rate"] = state["median_forecasted_value"] / state["capacity"]
    state["critical"] = state["need_rate"] < config.critical
    state["worrying"] = state["need_rate"] < config.worrying
    state["eligible_donor"] = state["donor_rate"] > config.donor_minimum
    return state


def _transfer_volume(state: pd.DataFrame, donor, receiver, config: TransferConfig) -> float:
    """ Calculates the volume for one donor and receiver """
    donor_capacity = float(state.loc[donor, "capacity"])
    donor_storage = float(state.loc[donor, "last_known_value"])
    receiver_capacity = float(state.loc[receiver, "capacity"])
    receiver_storage = float(state.loc[receiver, "last_known_value"])

    # Balance the pair before applying limits
    target_fraction = (donor_storage + receiver_storage) / (donor_capacity + receiver_capacity)
    desired = max(0.0, donor_storage - target_fraction * donor_capacity)
    limit = donor_capacity * config.max_transfer_fraction
    if config.max_transfer_volume is not None:
        limit = min(limit, config.max_transfer_volume)
    return min(desired, limit, receiver_capacity - receiver_storage)


def plan_transfers(
    state: pd.DataFrame,
    config: TransferConfig = TransferConfig(),
    max_iterations: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ Applies the transfer rule and records each decision """

    # Keep distances while storage changes
    state = state.copy()
    distance = haversine_km(state)
    ids = list(state.index)
    positions = {reservoir_id: index for index, reservoir_id in enumerate(ids)}
    logs = []

    for iteration in range(max_iterations):
        receivers = state.index[state["critical"]].tolist() or state.index[state["worrying"]].tolist()
        candidates = []

        # Build feasible donor and receiver options
        for donor in state.index[state["eligible_donor"]]:

            for receiver in receivers:
                if donor == receiver:
                    continue
                d = float(distance[positions[donor], positions[receiver]])
                if config.max_distance_km is not None and d > config.max_distance_km:
                    continue
                volume = _transfer_volume(state, donor, receiver, config)

                if volume <= 0:
                    continue
                cost = d / max(volume, 1e-9)
                if cost < config.cost_threshold:
                    candidates.append((cost, donor, receiver, volume, d))

        if not candidates:
            break
        cost, donor, receiver, volume, distance_km = min(candidates, key=lambda item: (item[0], item[1], item[2]))

        # Apply the transfer to storage and forecasts
        state.loc[donor, ["last_known_value", "low_forecasted_value", "median_forecasted_value"]] -= volume
        state.loc[receiver, ["last_known_value", "low_forecasted_value", "median_forecasted_value"]] += volume

        # Refresh the two affected statuses
        for rid in (donor, receiver):
            state.loc[rid, "need_rate"] = state.loc[rid, "low_forecasted_value"] / state.loc[rid, "capacity"]
            state.loc[rid, "donor_rate"] = state.loc[rid, "median_forecasted_value"] / state.loc[rid, "capacity"]
            state.loc[rid, "critical"] = state.loc[rid, "need_rate"] < config.critical
            state.loc[rid, "worrying"] = state.loc[rid, "need_rate"] < config.worrying
            state.loc[rid, "eligible_donor"] = state.loc[rid, "donor_rate"] > config.donor_minimum

        # Record the decision
        logs.append({
            "iteration": iteration,
            "donor": donor,
            "receiver": receiver,
            "volume": volume,
            "distance_km": distance_km,
            "cost": cost,
            "receiver_critical_after": bool(state.loc[receiver, "critical"]),
            "receiver_worrying_after": bool(state.loc[receiver, "worrying"]),
        })
    return state.reset_index(), pd.DataFrame(logs)


def select_region(
    reservoirs: pd.DataFrame,
    community: str | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    radius_km: float | None = None,
) -> pd.DataFrame:
    """ Filters reservoirs by community and geographic radius """
    selected = reservoirs.copy()
    if community is not None:
        selected = selected[selected["autonomous_community"].str.casefold() == community.casefold()]
    if latitude is not None and longitude is not None and radius_km is not None:
        point = np.radians([[latitude, longitude]])
        coords = np.radians(selected[["latitude", "longitude"]].to_numpy(float))
        distances = haversine_distances(point, coords)[0] * EARTH_RADIUS_KM
        selected = selected.loc[distances <= radius_km]
    return selected
