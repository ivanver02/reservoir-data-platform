import pandas as pd
import numpy as np
from prophet import Prophet
from .model_interface import BaseModel


class ProphetModel(BaseModel):
    """ Prophet model implementation """
    
    def __init__(self, 
                 yearly_seasonality: bool = True,
                 seasonality_mode: str = 'additive',
                 changepoint_prior_scale: float = 0.05):
        """ Stores the Prophet options used when fitting the model """
        super().__init__("Prophet")
        self.yearly_seasonality = yearly_seasonality
        self.seasonality_mode = seasonality_mode
        self.changepoint_prior_scale = changepoint_prior_scale
        self.fitted_model = None
        self.is_fitted = False
    
    def fit(self, train_data: pd.Series):
        """ Fits Prophet after converting the series to its expected columns """
        # Use Prophet's date and value columns
        df_prophet = pd.DataFrame({
            'ds': train_data.index,
            'y': train_data.values
        })
        
        self.model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            seasonality_mode=self.seasonality_mode,
            changepoint_prior_scale=self.changepoint_prior_scale
        )
        
        self.fitted_model = self.model.fit(df_prophet)
        self.is_fitted = True
        self.last_storage = train_data.iloc[-1]

        return self
    
    def adjust_negatives(self, series):
        """ Shifts negative forecast values upward without changing their shape """
        series = series.copy()
        for i in range(len(series)):
            if series.iloc[i] < 0:
                correction = -series.iloc[i]
                series.iloc[i:] += correction
                
        return series

    def predict(self, steps: int):
        """ Produces a Prophet forecast after the model has been fitted """
        
        if not self.is_fitted:
            raise ValueError("Model needs fitting before making predictions")
        
        # Build the forecast dates
        future = self.fitted_model.make_future_dataframe(periods=steps, freq='W')
        
        forecast = self.fitted_model.predict(future)
        
        predictions = pd.Series(
            forecast['yhat'].iloc[-steps:].to_numpy(),
            index=pd.to_datetime(forecast['ds'].iloc[-steps:]),
            dtype=float,
        )

        # Start after the last observed date
        predictions += (self.last_storage - predictions.iloc[0])

        return self.adjust_negatives(predictions)
