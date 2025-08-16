from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from .model_interface import BaseModel


class XGBoostModel(BaseModel):
    """XGBoost model implementation following the BaseModel interface"""
    
    def __init__(self, 
                 n_estimators: int = 100,
                 max_depth: int = 6,
                 learning_rate: float = 0.1,
                 random_state: int = 27):
        """
        Initialize XGBoost model
        
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            random_state: Random state for reproducibility
        """
        super().__init__("XGBoost")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.fitted_model = None
        self.is_fitted = False
        self.training_data = None
    
    def fit(self, train_data):
        """
        Train the XGBoost model on the provided data
        
        Args:
            train_data: DataFrame containing features and target
            
        Returns:
            Self for method chaining
        """
        # Separate features and target
        X = train_data.drop(columns=['next_storage_value'])
        y = train_data['next_storage_value'].values
        
        # Store training data for predictions
        # self.training_data = train_data.copy()
        
        # Create and fit XGBoost model
        self.fitted_model = XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state
        )
        
        self.fitted_model.fit(X, y)
        self.is_fitted = True
        
        return self

    def _create_next_row(self, prediction_value):
        """
        Create next row with updated lag features
        
        Args:
            prediction_value: The predicted storage value
            
        Returns:
            Dictionary representing the next row
        """
        last_row = self.training_data.iloc[-1]
        
        # Update time features (increment week)
        new_year = last_row['year']
        new_month = last_row['month'] 
        new_week_idx = (last_row['week_idx'] * 52 + 1) / 52  # Increment week and normalize
        
        # Handle year transition
        if new_week_idx >= 1.0:
            new_week_idx = 0.0
            new_year += 1/100  # Small increment for year normalization
        
        # Check if we need to increment month (every ~4 weeks)
        current_week_number = int(new_week_idx * 52)
        
        # If last 4 weeks are from same month, increment month
        if current_week_number % 4 == 0 and current_week_number > 0:
            new_month += 1/12
            
            # Handle month wraparound: if month becomes 1 and week < 5, reset to 0
            if new_month >= 1.0:
                if current_week_number < 5:
                    new_month = 0.0
            
        # Create new row with updated lag features
        new_row = {
            'year': new_year,
            'month': new_month,
            'week_idx': new_week_idx
        }
        
        # Update all lag features: shift everything by one position
        # storage_lag_1 becomes the current prediction
        new_row['storage_lag_1'] = prediction_value
        
        # All other lags shift: lag_2 becomes old lag_1, lag_3 becomes old lag_2, etc.
        for lag in range(2, 157):  # 2 to 156
            old_lag_col = f'storage_lag_{lag-1}'
            new_lag_col = f'storage_lag_{lag}'
            if old_lag_col in last_row:
                new_row[new_lag_col] = last_row[old_lag_col]
            else:
                new_row[new_lag_col] = prediction_value  # Fallback
        
        return new_row

    def predict(self, X):
        """
        Generate forecasts for specified number of steps
        
        Args:
            steps: Number of time steps to forecast
            
        Returns:
            Array of predicted values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Make prediction
        pred = self.fitted_model.predict(X)

        return pred
        '''
        # Create next row with updated features
        next_row = self._create_next_row(pred)
        next_row['next_storage_value'] = np.nan  # Use np.nan instead of None
        
        # Add to training data - ensure consistent data types
        new_df = pd.DataFrame([next_row])
        # Ensure new_df has the same column order as training_data
        new_df = new_df.reindex(columns=self.training_data.columns)
        self.training_data = pd.concat([self.training_data, new_df], ignore_index=True)
        '''
        
        # return np.array(predictions)
    
    def get_params(self) -> dict:
        """Get model parameters"""
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'random_state': self.random_state
        }
