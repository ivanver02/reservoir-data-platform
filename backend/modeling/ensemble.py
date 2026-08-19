"""Forecast combination"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from backend.modeling.baseline import clip_storage


def weighted_average(
    forecasts: Mapping[str, pd.Series],
    weights: Mapping[str, float],
    capacity: float | None = None,
) -> pd.Series:
    """ Combines forecasts and rescales their weights """
    available = [name for name in forecasts if name in weights]
    if not available:
        raise ValueError("no forecast has a configured ensemble weight")

    total_weight = sum(weights[name] for name in available)
    result = sum(forecasts[name] * weights[name] for name in available) / total_weight
    return clip_storage(result, capacity) if capacity is not None else pd.Series(result, dtype=float)
