from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error


class BaseModel(ABC):
    """Base interface for all forecasting models"""

    def __init__(self, model_name: str):
        self.model = None
        self.model_name = model_name

    @abstractmethod
    def fit(self, train_data):
        """Train the model on the provided data"""
        pass
    
    @abstractmethod
    def predict(self, steps: int):
        """Generate forecasts for specified number of steps"""
        pass

    def evaluate(self, test_data: pd.Series, predictions: pd.Series, show_metrics: bool = True):
        """
        Evaluate model performance using common metrics
        
        Args:
            test_data: Actual values for testing
            predictions: Predicted values
            
        Returns:
            Dictionary with evaluation metrics
        """
        if len(test_data) != len(predictions):
            raise ValueError("Test data and predictions must have the same length")
        
        metrics = {
            'mae': mean_absolute_error(test_data, predictions),
            'mse': mean_squared_error(test_data, predictions),
            'mape': mean_absolute_percentage_error(test_data, predictions)
        }

        if show_metrics:
            for key, value in metrics.items():
                print(f"{key}: {value:.4f}")

        return metrics
    
    def plot_prediction(self, 
                       train_data: pd.Series, 
                       test_data: pd.Series, 
                       predictions: pd.Series,
                       reservoir_id: int):
        """
        Plot training data, test data, and predictions
        
        Args:
            train_data: Training time series data
            test_data: Test time series data
            predictions: Model predictions
            reservoir_id: Reservoir ID for the title
        """
        plt.figure(figsize=(12, 6))
        
        # Plot training and test data
        plt.plot(train_data.index, train_data, label='Train')
        plt.plot(test_data.index, test_data, label='Test')
        plt.plot(test_data.index, predictions, label='Forecast')
        
        # Add train/test split line
        plt.axvline(test_data.index[0], color='gray', linestyle='--', label='Train/Test Split')
        
        # Formatting
        plt.legend()
        title = f'{self.model_name} Forecast for Reservoir {reservoir_id}'
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Storage')
        plt.tight_layout()
        plt.show()
