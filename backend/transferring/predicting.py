import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import numpy as np
import threading

# Setting up the path to include the parent directory
sys.path.append(str(Path.cwd().parent.parent))
from backend.config.settings import PATHS
from backend.modeling.sarima_model import SARIMAModel
from backend.modeling.prophet_model import ProphetModel
from backend.modeling.xgboost_model import XGBoostModel
from sklearn.linear_model import LinearRegression


def prepare_data(df, reservoir_id):
    df_res = df[df['id'] == reservoir_id].copy()
    df_res = df_res.sort_values('date')

    capacity = df[df['id'] == reservoir_id]['capacity'].values[0]

    # Set date as index
    series = df_res.set_index('date')['storage']
    length = len(series)
    years = min(13, length // 52) - 1 # At most we get 10 years of data

    # Train/test split: last 52 weeks as test
    train_series = series.iloc[-(52*(years+1)):-52] # 'years' years

    train_stacking_index = train_series[-(52*(years-4)):].index
    # Create a dataframe with train_index as index
    train_stacking = pd.DataFrame(index=train_stacking_index)
    train_stacking['storage'] = train_series[-(52*(years-4)):] # 'years'-4 years, as the first 4 are for training sarima and prophet
    train_stacking['sarima_prediction'] = np.nan
    train_stacking['prophet_prediction'] = np.nan

    test_stacking = pd.DataFrame(index=series.iloc[-52:].index)
    test_stacking['storage'] = series.iloc[-52:] # 1 year
    test_stacking['sarima_prediction'] = np.nan
    test_stacking['prophet_prediction'] = np.nan

    return train_series, train_stacking, test_stacking, capacity, years

def cv_sarima(train_sarima_prophet, train, test, years, capacity):
    for i in range(years-4):
        sarima_model = SARIMAModel(order=(1,1,1), seasonal_order=(1,1,1,52))
        sarima_model.fit(train_sarima_prophet.iloc[(52*i):(52*(i+4))])
        sarima_test_pred = sarima_model.predict(steps=52)
        mask = sarima_test_pred > capacity
        sarima_test_pred[mask] = capacity
        train.iloc[(52*i):(52*(i+1)), train.columns.get_loc('sarima_prediction')] = sarima_test_pred

    # Predicting SARIMA for test data
    sarima_model = SARIMAModel(order=(1,1,1), seasonal_order=(1,1,1,52))
    sarima_model.fit(train_sarima_prophet[-(52*4):])
    sarima_test_pred = sarima_model.predict(steps=52)
    mask = sarima_test_pred > capacity
    sarima_test_pred[mask] = capacity
    sarima_test_pred.index = test.index

    test['sarima_prediction'] = sarima_test_pred

    return train, test


def cv_prophet(train_sarima_prophet, train, test, years, capacity):
    for i in range(years-4):
        prophet_model = ProphetModel(yearly_seasonality=True, changepoint_prior_scale=0.0005)
        prophet_model.fit(train_sarima_prophet.iloc[(52*i):(52*(i+4))])
        prophet_test_pred = prophet_model.predict(steps=52)
        mask = prophet_test_pred > capacity
        prophet_test_pred[mask] = capacity
        train.iloc[(52*i):(52*(i+1)), train.columns.get_loc('prophet_prediction')] = prophet_test_pred

    # Predicting prophet for test data
    prophet_model = ProphetModel(yearly_seasonality=True, changepoint_prior_scale=0.0005)
    prophet_model.fit(train_sarima_prophet[-(52*4):])
    prophet_test_pred = prophet_model.predict(steps=52)
    mask = prophet_test_pred > capacity
    prophet_test_pred[mask] = capacity
    prophet_test_pred.index = test.index

    test['prophet_prediction'] = prophet_test_pred

    return train, test


def cv_xgboost(train, test, train_series_xgboost, years_training, years_training_xgboost, capacity):
    def predict_next_year(model, X, years_training_xgboost):
        predictions = []
        for week in range(1, 53):
            X_step = X.copy()[-(52*years_training_xgboost):]
            next_year_predictions = pd.concat([X_step, test], axis=0)
            X_step['sarima_prediction'] = next_year_predictions['sarima_prediction'].shift(-week).iloc[:52*years_training_xgboost]
            X_step['prophet_prediction'] = next_year_predictions['prophet_prediction'].shift(-week).iloc[:52*years_training_xgboost]
            storage_one_year_ago = X_step['storage'].iloc[-52]
            difference = storage_one_year_ago - X_step['storage'].iloc[-1]
            last_row = X_step.iloc[-1]
            X_step['next_storage_value'] = X_step['storage'].shift(-week)
            X_step = X_step.iloc[:-52]
            
            last_row['storage'] = last_row['storage'] + difference
            last_row['sarima_prediction'] = last_row['sarima_prediction'] + difference
            last_row['prophet_prediction'] = last_row['prophet_prediction'] + difference
            
            model.fit(X_step)
            
            # Generate forecasts for 52 steps
            prediction = model.predict(last_row.to_frame().T)
            predictions.append(prediction[0]-difference)

        return pd.Series(predictions, dtype=float)
    
    # Create and train XGBoost model
    model = XGBoostModel(n_estimators=100, max_depth=6, learning_rate=0.1)

    for i in range(years_training):
        xgboost_test_pred = predict_next_year(model, train_series_xgboost.iloc[i*52:(i+years_training_xgboost)*52], years_training_xgboost)
        mask = xgboost_test_pred > capacity
        xgboost_test_pred[mask] = capacity
        train.iloc[(52*i):(52*(i+1)), train.columns.get_loc('xgboost_prediction')] = xgboost_test_pred.values.astype('float64')

    xgboost_test_pred = predict_next_year(model, train_series_xgboost.iloc[-(52*years_training_xgboost):], years_training_xgboost)
    mask = xgboost_test_pred > capacity
    xgboost_test_pred[mask] = capacity
    xgboost_test_pred.index = test.index

    test['xgboost_prediction'] = xgboost_test_pred.astype('float64')
    train['xgboost_prediction'] = train['xgboost_prediction'].astype(float)

    return train, test

def predict_one_year(reservoir_id, years_training, plotting=True):

    next_year_predictions_path = PATHS['processed_data'] / 'next_year_predictions.parquet'
    next_year_predictions = pd.read_parquet(next_year_predictions_path)

    if reservoir_id in next_year_predictions.columns:
        return next_year_predictions[reservoir_id]

    else:

        next_year_lock = threading.Lock()

        water_engineered_path = PATHS['engineered_data'] / 'water_engineered.parquet'
        df = pd.read_parquet(water_engineered_path)
        reservoirs_2024 = df[df['date'] == '2024-09-24']['id'].unique()
        df = df[df['id'].isin(reservoirs_2024)]

        # Reuse existing preparation and training
        train_sarima_prophet, train, test, capacity, years = prepare_data(df, reservoir_id)
        train, test = cv_sarima(train_sarima_prophet, train, test, years, capacity)
        train, test = cv_prophet(train_sarima_prophet, train, test, years, capacity)

        # Prepare data for XGBoost
        train_series_xgboost = train.copy()
        train = train[-(52*years_training):]
        train.loc[:, 'storage'] = train['storage'].astype(float)
        train.loc[:, 'sarima_prediction'] = train['sarima_prediction'].astype(float)
        train.loc[:, 'prophet_prediction'] = train['prophet_prediction'].astype(float)
        train['xgboost_prediction'] = np.nan

        test.loc[:, 'storage'] = test['storage'].astype(float)
        test.loc[:, 'sarima_prediction'] = test['sarima_prediction'].astype(float)
        test.loc[:, 'prophet_prediction'] = test['prophet_prediction'].astype(float)
        test['xgboost_prediction'] = np.nan

        years_training_xgboost = (len(train_series_xgboost) // 52) - years_training
        train, test = cv_xgboost(train, test, train_series_xgboost, years_training, years_training_xgboost, capacity)

        # Train the linear regression stacking model
        X_train = train[['sarima_prediction', 'prophet_prediction', 'xgboost_prediction']].copy()
        y_train = train['storage'].copy()
        stacking_model = LinearRegression()
        stacking_model.fit(X_train, y_train)



        # Create next year index
        last_week = test.iloc[-1].name
        last_capacity = test['storage'].iloc[-1]
        start = last_week
        weekly = pd.date_range(start=start, periods=53, freq='7D')
        next_year_index = weekly[1:]
        next_year = pd.DataFrame(index=next_year_index)



        # 1. SARIMA
        full_series = df[df['id'] == reservoir_id].set_index('date')['storage'].sort_index()
        sarima_model_next = SARIMAModel(order=(1,1,1), seasonal_order=(1,1,1,52))
        sarima_model_next.fit(full_series[-(52*4):])  # Last 4 years
        sarima_next_pred = sarima_model_next.predict(steps=52)
        mask = sarima_next_pred > capacity
        sarima_next_pred[mask] = capacity
        sarima_next_pred.index = next_year_index
        next_year['sarima_prediction'] = sarima_next_pred

        # 2. PROPHET
        prophet_model_next = ProphetModel(yearly_seasonality=True, changepoint_prior_scale=0.0005)
        prophet_model_next.fit(full_series[-(52*4):])  # Last 4 years
        prophet_next_pred = prophet_model_next.predict(steps=52)
        mask = prophet_next_pred > capacity
        prophet_next_pred[mask] = capacity
        prophet_next_pred.index = next_year_index
        next_year['prophet_prediction'] = prophet_next_pred

        # 3. XGBOOST
        def predict_next_year_xgboost(model, historical_data, years_training_xgboost, next_year_index):
            predictions = []
            # Use the full historical data including test predictions as features
            full_historical = pd.concat([historical_data, test], axis=0)
            
            for week in range(1, 53):
                # Get the training window
                X_step = full_historical.copy()[-(52*years_training_xgboost):]
                
                # For future predictions, we need to shift our features appropriately
                # We'll use the predicted values from SARIMA and Prophet for this future week
                last_row = X_step.iloc[-1].copy()
                
                # Set the features for this prediction step
                last_row['storage'] = full_historical['storage'].iloc[-1]  # Last known storage
                last_row['sarima_prediction'] = sarima_next_pred.iloc[week-1]  # SARIMA prediction for this week
                last_row['prophet_prediction'] = prophet_next_pred.iloc[week-1]  # Prophet prediction for this week
                
                # Create target variable for training (shift by the week we're predicting)
                X_step['next_storage_value'] = X_step['storage'].shift(-1)
                X_step = X_step.iloc[:-1]  # Remove last row since it has no target
                
                # Train model on this window
                model.fit(X_step)
                
                # Predict
                prediction = model.predict(last_row.to_frame().T)
                predictions.append(prediction[0])
                
                # Update the historical data with this prediction for next iteration
                new_row = pd.DataFrame({
                    'storage': [prediction[0]], 
                    'sarima_prediction': [sarima_next_pred.iloc[week-1]], 
                    'prophet_prediction': [prophet_next_pred.iloc[week-1]]
                }, index=[next_year_index[week-1]])
                full_historical = pd.concat([full_historical, new_row])
            
            return pd.Series(predictions, index=next_year_index, dtype=float)

        # Create XGBoost model and predict
        xgb_model_next = XGBoostModel(n_estimators=100, max_depth=6, learning_rate=0.1)
        xgb_next_pred = predict_next_year_xgboost(xgb_model_next, train_series_xgboost, years_training_xgboost, next_year_index)
        mask = xgb_next_pred > capacity
        xgb_next_pred[mask] = capacity
        next_year['xgboost_prediction'] = xgb_next_pred

        # 4. LINEAR REGRESSION
        X_next = next_year[['sarima_prediction', 'prophet_prediction', 'xgboost_prediction']].copy()
        lr_next_pred = stacking_model.predict(X_next)
        next_year['LR_prediction'] = lr_next_pred

        # 5. WEIGHTED AVERAGE
        next_year['full_weighted_average'] = next_year['prophet_prediction'] * 0.15 + next_year['sarima_prediction'] * 0.25 + next_year['xgboost_prediction'] * 0.25 + next_year['LR_prediction'] * 0.35
        next_year['full_weighted_average'] = next_year['full_weighted_average'] + last_capacity - next_year['full_weighted_average'].iloc[0]  # Adjust to last known capacity
        mask = next_year['full_weighted_average'] > capacity
        next_year.loc[mask, 'full_weighted_average'] = capacity

        def adjust_negatives(series):
            series = series.copy()
            for i in range(len(series)):
                if series.iloc[i] < 0:
                    correction = -series.iloc[i]
                    series.iloc[i:] += correction
            return series

        next_year.loc[:, 'full_weighted_average'] = adjust_negatives(next_year['full_weighted_average'])

        # Define a lock to access the next_year DataFrame
        next_year_lock = threading.Lock()

        with next_year_lock:
            path = PATHS['processed_data'] / 'next_year_predictions.parquet'
            next_year_predictions = pd.read_parquet(next_year_predictions_path)
            next_year_predictions[reservoir_id] = next_year['full_weighted_average']
            next_year_predictions.to_parquet(path, index=True)

        # COMPREHENSIVE PLOT
        def plot_historical_and_next_year(train, test, next_year, reservoir_id):
            plt.figure(figsize=(20, 10))

            train = pd.concat([train, test], axis=0)  # Combine train and test for plotting
            
            # Plot historical training data
            plt.plot(train.index, train['storage'], label='Known evolution', color='blue', linewidth=2, alpha=0.8)
            
            # Plot next year predictions
            plt.plot(next_year.index, next_year['full_weighted_average'], label='Next Year Prediction', 
                    color='black', linestyle='-', linewidth=3, alpha=0.9)
            
            # Add vertical lines for transitions

            plt.axvline(next_year.index[0], color='red', linestyle='--', alpha=0.6, label='Known / Prediction Split')

            # Formatting
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.title(f'Reservoir {reservoir_id}', fontsize=14)
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Storage', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

        if plotting:
            # Call the plotting function
            plot_historical_and_next_year(train, test, next_year, reservoir_id)

        return next_year_predictions[reservoir_id]