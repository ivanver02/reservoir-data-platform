from abc import ABC, abstractmethod
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error


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

    def evaluate(self, test_data: pd.Series, predictions: pd.Series, show_metrics: bool = True):
        """ Calculates errors for a test series and predictions """
        if len(test_data) != len(predictions):
            raise ValueError("Test data and predictions need the same length")
        
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
        """ Draws reservoir data and forecasts """
        plt.figure(figsize=(12, 6))
        
        # Plot training and test data
        plt.plot(train_data.index, train_data, label='Train')
        plt.plot(test_data.index, test_data, label='Test')
        plt.plot(test_data.index, predictions, label='Forecast')
        
        # Mark the split
        plt.axvline(test_data.index[0], color='gray', linestyle='--', label='Train/Test Split')
        
        # Set figure labels
        plt.legend()
        title = f'{self.model_name} Forecast for Reservoir {reservoir_id}'
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Storage')
        plt.tight_layout()
        plt.show()
