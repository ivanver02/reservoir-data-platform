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
