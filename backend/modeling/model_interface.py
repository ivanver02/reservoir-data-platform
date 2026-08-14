""" Interface shared by the forecasting models """

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class ForecastModel(ABC):
    """ Contract introduced for diverse forecast implementations"""

    @abstractmethod
    def fit(self, series: pd.Series) -> "ForecastModel":
        raise NotImplementedError

    @abstractmethod
    def predict(self, horizon: int) -> pd.Series:
        raise NotImplementedError
