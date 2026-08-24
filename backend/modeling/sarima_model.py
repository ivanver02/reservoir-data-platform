from typing import Tuple
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from .model_interface import BaseModel


class SARIMAModel(BaseModel):
    """ SARIMA model implementation """
    
    def __init__(self, 
                 order: Tuple[int, int, int] = (1, 1, 1),
                 seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 52)):
        """ Stores the SARIMA orders used when fitting the model """
        super().__init__("SARIMA")
        self.order = order
        self.seasonal_order = seasonal_order
        self.fitted_model = None
    
    def fit(self, train_data: pd.Series):
        """ Fits SARIMA to one reservoir storage series """
        # Let seasonal differencing handle the trend
        self.model = SARIMAX(
            train_data,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        self.fitted_model = self.model.fit(disp=False)
        self.is_fitted = True
        self.last_storage = train_data.iloc[-1]

        return self

    def predict(self, steps: int):
        """ Produces a forecast after the model has been fitted """
        if not self.is_fitted:
            raise ValueError("Model needs fitting before making predictions")
        
        forecast = self.fitted_model.get_forecast(steps=steps)

        # Anchor the forecast at the last observed value
        forecast.predicted_mean += (self.last_storage - forecast.predicted_mean.iloc[0])

        return self.adjust_negatives(forecast.predicted_mean)
