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

next_year_lock = threading.Lock()

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

def sarima_cv(train_series_src, train_df, test_df, yrs, cap, df, reservoir_id):
        # Cross-validation style historical predictions
        for i in range(yrs-4):
            model = SARIMAModel(order=(1,1,1), seasonal_order=(1,1,1,52))
            model.fit(train_series_src.iloc[(52*i):(52*(i+4))])
            preds = model.predict(steps=52)
            preds[preds > cap] = cap
            train_df.iloc[(52*i):(52*(i+1)), train_df.columns.get_loc('sarima_prediction')] = preds
        # Test year prediction
        model = SARIMAModel(order=(1,1,1), seasonal_order=(1,1,1,52))
        model.fit(train_series_src[-(52*4):])
        test_preds = model.predict(steps=52)
        test_preds[test_preds > cap] = cap
        test_preds.index = test_df.index
        test_df['sarima_prediction'] = test_preds
        # Future year prediction (after test)
        full_series = df[df['id'] == reservoir_id].set_index('date')['storage'].sort_index()
        future_model = SARIMAModel(order=(1,1,1), seasonal_order=(1,1,1,52))
        future_model.fit(full_series[-(52*4):])
        future_preds = future_model.predict(steps=52)
        future_preds[future_preds > cap] = cap
        return train_df, test_df, future_preds


def prophet_cv(train_series_src, train_df, test_df, yrs, cap, df, reservoir_id):
    for i in range(yrs-4):
        model = ProphetModel(yearly_seasonality=True, changepoint_prior_scale=0.0005)
        model.fit(train_series_src.iloc[(52*i):(52*(i+4))])
        preds = model.predict(steps=52)
        preds[preds > cap] = cap
        train_df.iloc[(52*i):(52*(i+1)), train_df.columns.get_loc('prophet_prediction')] = preds
    model = ProphetModel(yearly_seasonality=True, changepoint_prior_scale=0.0005)
    model.fit(train_series_src[-(52*4):])
    test_preds = model.predict(steps=52)
    test_preds[test_preds > cap] = cap
    test_preds.index = test_df.index
    test_df['prophet_prediction'] = test_preds
    full_series = df[df['id'] == reservoir_id].set_index('date')['storage'].sort_index()
    future_model = ProphetModel(yearly_seasonality=True, changepoint_prior_scale=0.0005)
    future_model.fit(full_series[-(52*4):])
    future_preds = future_model.predict(steps=52)
    future_preds[future_preds > cap] = cap
    return train_df, test_df, future_preds

def predict_next_year(model, X_slice, yrs_train_xgb, test_df):
    preds_local = []
    for week in range(1, 53):
        X_step = X_slice.copy()[-(52*yrs_train_xgb):]
        next_year_predictions = pd.concat([X_step, test_df], axis=0)
        X_step['sarima_prediction'] = next_year_predictions['sarima_prediction'].shift(-week).iloc[:52*yrs_train_xgb]
        X_step['prophet_prediction'] = next_year_predictions['prophet_prediction'].shift(-week).iloc[:52*yrs_train_xgb]
        storage_one_year_ago = X_step['storage'].iloc[-52]
        diff = storage_one_year_ago - X_step['storage'].iloc[-1]
        last_row = X_step.iloc[-1]
        X_step['next_storage_value'] = X_step['storage'].shift(-week)
        X_step = X_step.iloc[:-52]
        last_row['storage'] += diff
        last_row['sarima_prediction'] += diff
        last_row['prophet_prediction'] += diff
        model.fit(X_step)
        pred = model.predict(last_row.to_frame().T)[0]
        preds_local.append(pred - diff)
    return pd.Series(preds_local, dtype=float)

def xgboost_cv(train_trunc, test_df, train_series_full, yrs_train, cap, future_sarima, future_prophet):
    train_series_xgboost = train_series_full.copy()
    years_training_xgboost = (len(train_series_xgboost)//52) - yrs_train
    # Initialize columns
    train_trunc['xgboost_prediction'] = np.nan
    test_df['xgboost_prediction'] = np.nan

    model_hist = XGBoostModel(n_estimators=100, max_depth=6, learning_rate=0.1)
    for i in range(yrs_train):
        slice_ = train_series_xgboost.iloc[i*52:(i+years_training_xgboost)*52]
        hist_pred = predict_next_year(model_hist, slice_, years_training_xgboost, test_df)
        hist_pred[hist_pred > cap] = cap
        train_trunc.iloc[(52*i):(52*(i+1)), train_trunc.columns.get_loc('xgboost_prediction')] = hist_pred.values.astype('float64')

    # Test period predictions
    slice_ = train_series_xgboost.iloc[-(52*years_training_xgboost):]
    test_pred = predict_next_year(model_hist, slice_, years_training_xgboost, test_df)
    test_pred[test_pred > cap] = cap
    test_pred.index = test_df.index
    test_df['xgboost_prediction'] = test_pred.astype('float64')
    train_trunc['xgboost_prediction'] = train_trunc['xgboost_prediction'].astype(float)

    # Future rolling prediction using sarima/prophet future values
    future_model = XGBoostModel(n_estimators=100, max_depth=6, learning_rate=0.1)
    full_hist = pd.concat([train_series_xgboost, test_df], axis=0)
    future_index_local = pd.date_range(start=test_df.index[-1], periods=53, freq='7D')[1:]
    future_preds_list = []
    for w in range(52):
        window = full_hist.copy()[-(52*years_training_xgboost):]
        if 'xgboost_prediction' not in window.columns:
            window['xgboost_prediction'] = np.nan
        window['next_storage_value'] = window['storage'].shift(-1)
        window = window.iloc[:-1]
        if len(window) == 0:
            future_preds_list.append(full_hist['storage'].iloc[-1])
            continue
        future_model.fit(window)
        last_row = window.iloc[-1].copy()
        last_row['storage'] = full_hist['storage'].iloc[-1]
        last_row['sarima_prediction'] = future_sarima.iloc[w]
        last_row['prophet_prediction'] = future_prophet.iloc[w]
        X_pred = last_row.to_frame().T.drop(columns=[c for c in ['next_storage_value'] if c in last_row.index], errors='ignore')
        feature_names = getattr(future_model.fitted_model, 'feature_names_in_', None)
        if feature_names is not None:
            for col in feature_names:
                if col not in X_pred.columns:
                    X_pred[col] = np.nan
            X_pred = X_pred[list(feature_names)]
        pred_val = future_model.predict(X_pred)[0]
        future_preds_list.append(pred_val)
        new_row = pd.DataFrame({
            'storage':[pred_val],
            'sarima_prediction':[future_sarima.iloc[w]],
            'prophet_prediction':[future_prophet.iloc[w]]
        }, index=[future_index_local[w]])
        full_hist = pd.concat([full_hist, new_row])
    future_xgb = pd.Series(future_preds_list, index=future_index_local, dtype=float)
    future_xgb[future_xgb > cap] = cap
    return train_trunc, test_df, future_xgb, future_index_local

def adjust_neg(series):
    s = series.copy()
    for i in range(len(s)):
        if s.iloc[i] < 0:
            s.iloc[i:] += -s.iloc[i]
    return s

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
    
def predict_one_year(reservoir_id, years_training, plotting=True):
    '''
    - Prophet and Sarima are trained first
    - Some of that data is used as training data for the XGBoost model, and XGBoost predicts the rest too
    - From the dates that has predictions from all three models, a linear regression model is trained in the first years, and predicts the rest
    - The predictions from all models are then combined using a full weighted average approach, where the weights are determined based on the performance of each model
    '''
    
    next_year_predictions_path = PATHS['processed_data_notebooks'] / 'next_year_predictions.parquet'
    next_year_predictions = pd.read_parquet(next_year_predictions_path)

    if reservoir_id in next_year_predictions.columns:
        if plotting:
            water_definitive_path = PATHS['definitive_notebooks'] / 'water_definitive.parquet'
            water = pd.read_parquet(water_definitive_path)
            water = water[water['id'] == reservoir_id]
            train = water.iloc[-(52*(years_training)):-52].set_index('date')
            test = water.iloc[-52:].set_index('date')
            next_year = pd.DataFrame(index=next_year_predictions.index)
            next_year['full_weighted_average'] = next_year_predictions[reservoir_id]
            plot_historical_and_next_year(train, test, next_year, reservoir_id)
        return next_year_predictions[reservoir_id]

    water_engineered_path = PATHS['engineered_data_notebooks'] / 'water_engineered.parquet'
    df = pd.read_parquet(water_engineered_path)
    reservoirs_2024 = df[df['date'] == '2024-09-24']['id'].unique()
    df = df[df['id'].isin(reservoirs_2024)]

    # Base preparation
    train_sarima_prophet, train, test, capacity, years = prepare_data(df, reservoir_id)

    train, test, sarima_future = sarima_cv(train_sarima_prophet, train, test, years, capacity, df, reservoir_id)
    train, test, prophet_future = prophet_cv(train_sarima_prophet, train, test, years, capacity, df, reservoir_id)

    # Restrict train for stacking to last years_training years
    train_series_full = train.copy()
    train = train[-(52*years_training):]
    for col in ['storage','sarima_prediction','prophet_prediction']:
        train[col] = train[col].astype(float)
    test[['storage','sarima_prediction','prophet_prediction']] = test[['storage','sarima_prediction','prophet_prediction']].astype(float)

    train, test, xgb_future, future_index = xgboost_cv(train, test, train_series_full, years_training, capacity, sarima_future, prophet_future)

    # Stacking model
    stacking_model = LinearRegression().fit(train[['sarima_prediction','prophet_prediction','xgboost_prediction']], train['storage'])

    # Assemble next year df
    next_year = pd.DataFrame({
        'sarima_prediction': sarima_future.values,
        'prophet_prediction': prophet_future.values,
        'xgboost_prediction': xgb_future.values
    }, index=future_index)
    next_year['LR_prediction'] = stacking_model.predict(next_year[['sarima_prediction','prophet_prediction','xgboost_prediction']])

    last_capacity = test['storage'].iloc[-1]
    next_year['full_weighted_average'] = (
        next_year['prophet_prediction'] * 0.15 +
        next_year['sarima_prediction'] * 0.25 +
        next_year['xgboost_prediction'] * 0.25 +
        next_year['LR_prediction'] * 0.35
    )
    next_year['full_weighted_average'] = next_year['full_weighted_average'] + last_capacity - next_year['full_weighted_average'].iloc[0]
    next_year.loc[next_year['full_weighted_average'] > capacity, 'full_weighted_average'] = capacity

    next_year['full_weighted_average'] = adjust_neg(next_year['full_weighted_average'])

    with next_year_lock:
        path = PATHS['processed_data_notebooks'] / 'next_year_predictions.parquet'
        nxt = pd.read_parquet(path)
        nxt[reservoir_id] = next_year['full_weighted_average']
        nxt.to_parquet(path, index=True)

    if plotting:
        plot_historical_and_next_year(train, test, next_year, reservoir_id)

    return next_year['full_weighted_average']


def predict_one_year_sarima(reservoir_id):
    next_year_predictions_path = PATHS['processed_data_notebooks'] / 'next_year_predictions.parquet'
    next_year_predictions = pd.read_parquet(next_year_predictions_path)

    if reservoir_id in next_year_predictions.columns:
        return next_year_predictions[reservoir_id]

    else:
        water_engineered_path = PATHS['engineered_data_notebooks'] / 'water_engineered.parquet'
        df = pd.read_parquet(water_engineered_path)
        df = df[df['id'] == reservoir_id]
        full_series = df.set_index('date')['storage'].sort_index()

        sarima_model_next = SARIMAModel(order=(1,1,1), seasonal_order=(1,1,1,52))
        sarima_model_next.fit(full_series)  # All available data
        sarima_next_pred = sarima_model_next.predict(steps=52)
        
        next_year_index = pd.date_range(start=full_series.index[-1] + pd.Timedelta(weeks=1), periods=52, freq='7D')
        sarima_next_pred.index = next_year_index
        
        with next_year_lock:
            path = PATHS['processed_data_notebooks'] / 'next_year_predictions.parquet'
            next_year_predictions = pd.read_parquet(next_year_predictions_path)
            next_year_predictions[reservoir_id] = sarima_next_pred
            next_year_predictions.to_parquet(path, index=True)

        return sarima_next_pred