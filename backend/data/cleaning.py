import sys
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.metrics.pairwise import haversine_distances

sys.path.append(str(Path.cwd().parent.parent))
from backend.config.settings import CLEANING_SETTINGS

REPLACEMENT_MAPPING = CLEANING_SETTINGS['REPLACEMENT_MAPPING']
ARTICLES_MAPPING = CLEANING_SETTINGS['ARTICLES_MAPPING']

WEEKLY_FREQ = CLEANING_SETTINGS['WEEKLY_FREQ']

EARTH_RADIUS_KM = CLEANING_SETTINGS['EARTH_RADIUS_KM']


def clean_string_series(series):
    """
    Cleans a pandas Series by lowercasing, replacing accents, removing articles, and stripping whitespace.
    """
    # Lowercasing
    series = series.str.lower()
    
    # Replacing accents and special characters
    series = series.replace(REPLACEMENT_MAPPING, regex=True)
    
    # Adding spaces around the strings and articles to avoid removing parts of words when removing articles
    series = ' ' + series + ' '
    
    # Removing articles
    series = series.replace(ARTICLES_MAPPING, regex=True)

    # Stripping whitespace
    series = series.str.strip()
    
    return series

def reindex_weekly(group):  # This function is applied to every reservoir
    group_id = group.name
    # Create a complete weekly date range for this id
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

