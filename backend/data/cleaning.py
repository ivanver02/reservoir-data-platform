import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import haversine_distances

from backend.config.settings import CLEANING_SETTINGS

REPLACEMENT_MAPPING = CLEANING_SETTINGS['REPLACEMENT_MAPPING']
ARTICLES_MAPPING = CLEANING_SETTINGS['ARTICLES_MAPPING']

WEEKLY_FREQ = CLEANING_SETTINGS['WEEKLY_FREQ']

EARTH_RADIUS_KM = CLEANING_SETTINGS['EARTH_RADIUS_KM']


def clean_string_series(series):
    """ Normalizes names for comparison """
    series = series.astype("string").str.lower()
    
    # Normalize accents and symbols
    series = series.replace(REPLACEMENT_MAPPING, regex=True)
    
    # Separate articles from names
    series = ' ' + series + ' '
    
    # Remove common articles
    series = series.replace(ARTICLES_MAPPING, regex=True)

    # Trim repeated spaces
    series = series.str.strip()
    
    return series

def reindex_weekly(group, group_id=None):  # Reindex one reservoir
    """ Fills missing dates for one reservoir """
    group = group.copy().sort_values('date')
    group_id = group.name if group_id is None else group_id
    if group['date'].duplicated().any():
        raise ValueError(f"reservoir {group_id} contains duplicated dates")
    
    # Create the weekly index
    full_range = pd.date_range(start=group['date'].min(), end=group['date'].max(), freq=WEEKLY_FREQ)
    group = group.set_index('date').reindex(full_range)
    group['id'] = group_id
    group = group.reset_index().rename(columns={'index': 'date'})
    return group

def impute_nearest_neighbour(detailed_reservoirs_data, column_name):
    """ Fills a column from the nearest geographic neighbour """
    data = detailed_reservoirs_data.copy()
    imputed_column = f"{column_name}_imputed"
    data[imputed_column] = False
    missing = data[column_name].isna()
    if not missing.any():
        return data

    # Keep rows with coordinates
    coordinates = data[['longitude', 'latitude']].apply(pd.to_numeric, errors='coerce')
    coordinates = coordinates.where(np.isfinite(coordinates), np.nan)
    candidates = (~missing) & coordinates.notna().all(axis=1)
    targets = missing & coordinates.notna().all(axis=1)
    if not candidates.any():
        raise ValueError(f"cannot impute {column_name}: no valid candidate coordinates")
    if not targets.any():
        return data

    # Calculate distances to nearby rows
    target_radians = np.radians(coordinates.loc[targets].to_numpy(dtype=float))
    candidate_radians = np.radians(coordinates.loc[candidates].to_numpy(dtype=float))
    distances = haversine_distances(target_radians, candidate_radians) * EARTH_RADIUS_KM
    closest_indices = distances.argmin(axis=1)
    candidate_values = data.loc[candidates, column_name].iloc[closest_indices].to_numpy()
    data.loc[targets, column_name] = candidate_values
    data.loc[targets, imputed_column] = True
    return data

