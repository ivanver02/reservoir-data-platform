import pandas as pd
import numpy as np
from prophet import Prophet
from .model_interface import BaseModel


class ProphetModel(BaseModel):
    """Prophet model implementation following the BaseModel interface"""
    
    def __init__(self, 
                 yearly_seasonality: bool = True,
                 seasonality_mode: str = 'additive',
                 changepoint_prior_scale: float = 0.05):
        """
        Initialize Prophet model
        
        Args:
            yearly_seasonality: Whether to include yearly seasonality
            seasonality_mode: 'additive' or 'multiplicative'
            changepoint_prior_scale: Controls flexibility of trend changes
        """
        super().__init__("Prophet")
        self.yearly_seasonality = yearly_seasonality
        self.seasonality_mode = seasonality_mode
        self.changepoint_prior_scale = changepoint_prior_scale
        self.fitted_model = None
        self.is_fitted = False
    
    def fit(self, train_data: pd.Series):
        """
        Train the Prophet model on the provided data
        
        Args:
            train_data: Time series data for training
            
        Returns:
            Self for method chaining
        """
        # Convert to Prophet format (ds: date, y: value)
        df_prophet = pd.DataFrame({
            'ds': train_data.index,
            'y': train_data.values
        })
        
        # Create and fit Prophet model
        self.model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            seasonality_mode=self.seasonality_mode,
            changepoint_prior_scale=self.changepoint_prior_scale
        )
        
        # Fit the model
        self.fitted_model = self.model.fit(df_prophet)
        self.is_fitted = True
        self.last_storage = train_data.iloc[-1]

        return self
    
    def adjust_negatives(self, series):
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
            Series of predicted values
        """
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Create future dataframe for prediction
        future = self.fitted_model.make_future_dataframe(periods=steps, freq='W')
        
        # Generate forecast
        forecast = self.fitted_model.predict(future)
        
        # Return only the forecasted values (last 'steps' predictions)
        predictions = forecast['yhat'].iloc[-steps:]

        # Make it so that I calculate the difference between the first prediction and the last known value, and add it to the whole prediction
        predictions += (self.last_storage - predictions.iloc[0])

        return self.adjust_negatives(predictions)
