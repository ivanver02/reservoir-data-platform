from typing import Tuple
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from .model_interface import BaseModel


class SARIMAModel(BaseModel):
    """ SARIMA model implementation """
    
    def __init__(self, 
                 order: Tuple[int, int, int] = (1, 1, 1),
                 seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 52)):
        """
        Initialize SARIMA model
        
        Args:
            order: (p, d, q) parameters for ARIMA
            seasonal_order: (P, D, Q, s) parameters for seasonal ARIMA
        """
        super().__init__("SARIMA")
        self.order = order
        self.seasonal_order = seasonal_order
        self.fitted_model = None
    
    def fit(self, train_data: pd.Series):
        """
        Train the SARIMA model on the provided data
        
        Args:
            train_data: Time series data for training
            
        Returns:
            Self for method chaining
        """

        # Create and fit SARIMAX model
        self.model = SARIMAX(
            train_data,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        # Fit the model
        self.fitted_model = self.model.fit(disp=False)
        self.is_fitted = True
        self.last_storage = train_data.iloc[-1]

        return self
    
    def adjust_negatives(self, series):
        """ Shifts negative forecast values without changing their shape """
        series = series.copy()
        for i in range(len(series)):
            if series.iloc[i] < 0:
                correction = -series.iloc[i]
                series.iloc[i:] += correction
        return series

    def predict(self, steps: int):
        """
        Generate forecasts for specified number of steps
        
        Args:
            steps: Number of time steps to forecast
            
        Returns:
            Array of predicted values
        """
        if not self.is_fitted:
            raise ValueError("Model needs fitting before making predictions")
        
        forecast = self.fitted_model.get_forecast(steps=steps)

        # Anchor the forecast at the last observed value
        forecast.predicted_mean += (self.last_storage - forecast.predicted_mean.iloc[0])

        return self.adjust_negatives(forecast.predicted_mean)
