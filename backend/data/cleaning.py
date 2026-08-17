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
    nan_reservoirs = detailed_reservoirs_data[detailed_reservoirs_data.loc[:, column_name].isna()]
    non_nan_reservoirs = detailed_reservoirs_data[detailed_reservoirs_data.loc[:, column_name].notna()]

    nan_coord_rads = np.radians(nan_reservoirs[['longitude', 'latitude']].astype(float).to_numpy()).reshape(-1, 2)
    non_nan_coord_rads = np.radians(non_nan_reservoirs[['longitude', 'latitude']].astype(float).to_numpy()).reshape(-1, 2)

    distances = haversine_distances(nan_coord_rads, non_nan_coord_rads) * EARTH_RADIUS_KM
    closest_indices = distances.argmin(axis=1)

    # Create a mapping from indices to reservoir values
    reservoir_mapping = non_nan_reservoirs[column_name].iloc[closest_indices]
    detailed_reservoirs_data.loc[nan_reservoirs.index, column_name] = reservoir_mapping.values
    return detailed_reservoirs_data
