"""Base interface shared by the forecasting models"""

from abc import ABC, abstractmethod

import pandas as pd


class BaseModel(ABC):
    """ Interface for forecasting models """

    def __init__(self, model_name: str):
        """ Stores the state used by a forecasting model """
        self.model = None
        self.model_name = model_name
        self.is_fitted = False
        self.last_storage = None

    @abstractmethod
    def fit(self, train_data):
        """ Trains the model on the supplied series """
        pass

    @abstractmethod
    def predict(self, steps: int):
        """ Generates the requested number of values """
        pass

    def adjust_negatives(self, series):
        """ Shifts negative forecast values upward without changing their shape """
        values = series.to_numpy(dtype=float).copy()
        correction = 0.0

        # Carry each lift over to the rest of the forecast
        for i in range(len(values)):
            values[i] += correction
            if values[i] < 0:
                correction += -values[i]
                values[i] = 0.0
        return pd.Series(values, index=series.index)
