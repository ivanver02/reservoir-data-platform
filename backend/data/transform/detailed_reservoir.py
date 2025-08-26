# Importing modules
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from sklearn.metrics.pairwise import haversine_distances

# Setting up the path to include the parent directory
sys.path.append(str(Path.cwd().parent.parent.parent))
from backend.data.extract import extract_detailed_reservoirs_raw
from backend.data.load import load_detailed_reservoirs_cleaned
from backend.data.cleaning import clean_string_series

columns_dict = {'x': 'longitude', 'y': 'latitude'}

def impute_nearest_neighbour(detailed_reservoirs_data, column_name):
    nan_reservoirs = detailed_reservoirs_data[detailed_reservoirs_data.loc[:, column_name].isna()]
    non_nan_reservoirs = detailed_reservoirs_data[detailed_reservoirs_data.loc[:, column_name].notna()]

    nan_coord_rads = np.radians(nan_reservoirs[['longitude', 'latitude']].astype(float).to_numpy()).reshape(-1, 2)
    non_nan_coord_rads = np.radians(non_nan_reservoirs[['longitude', 'latitude']].astype(float).to_numpy()).reshape(-1, 2)

    distances = haversine_distances(nan_coord_rads, non_nan_coord_rads) * 6371.0 
    closest_indices = distances.argmin(axis=1)

    # Create a mapping from indices to reservoir values
    reservoir_mapping = non_nan_reservoirs[column_name].iloc[closest_indices]
    detailed_reservoirs_data.loc[nan_reservoirs.index, column_name] = reservoir_mapping.values
    return detailed_reservoirs_data

def handle_missing_values(detailed_reservoirs_data):
    # Fill missing values in other columns
    detailed_reservoirs_data = impute_nearest_neighbour(detailed_reservoirs_data, 'riverbed')
    detailed_reservoirs_data = impute_nearest_neighbour(detailed_reservoirs_data, 'crest_elevation')
    return detailed_reservoirs_data

def clean_string_columns(detailed_reservoirs_data, columns):
    for column in columns:
        detailed_reservoirs_data.loc[:, column] = clean_string_series(detailed_reservoirs_data[column])

def transform_detailed_reservoirs_raw(detailed_reservoirs_data):
    detailed_reservoirs_data.columns = detailed_reservoirs_data.columns.str.lower()
    detailed_reservoirs_renamed = detailed_reservoirs_data.rename(columns=columns_dict)
    detailed_reservoirs_renamed = detailed_reservoirs_renamed.drop(columns=['code'])

    clean_string_columns(detailed_reservoirs_renamed, ['name', 'reservoir', 'basin', 'province', 'autonomous_community', 'type'])

    # As reservoir has unsignificant words in their names, we will go for name (visualization at detailed_reservoir.ipynb)
    detailed_reservoirs_renamed = detailed_reservoirs_renamed.drop(columns=['reservoir'])

    # These next columns were object type
    detailed_reservoirs_renamed.loc[:, 'longitude'] = detailed_reservoirs_renamed['longitude'].str.replace(',', '.').astype(float)
    detailed_reservoirs_renamed.loc[:, 'latitude'] = detailed_reservoirs_renamed['latitude'].str.replace(',', '.').astype(float)

    detailed_reservoirs_imputed = handle_missing_values(detailed_reservoirs_renamed)

    detailed_reservoirs_imputed.loc[:, 'riverbed'] = clean_string_columns(detailed_reservoirs_imputed, ['riverbed'])

    detailed_reservoirs_imputed.loc[:, 'crest_elevation'] = detailed_reservoirs_renamed['crest_elevation'].str.replace(',', '.')
    detailed_reservoirs_imputed.loc[:, 'crest_elevation'] = detailed_reservoirs_imputed['crest_elevation'].astype(float)
    
    mask = detailed_reservoirs_imputed['dam_height'].notna()
    detailed_reservoirs_imputed.loc[mask, 'dam_height'] = detailed_reservoirs_imputed.loc[mask, 'dam_height'].str.replace(',', '.').astype(float)

    # Handling duplicates
    # We well keep the first one that has the highest dam height
    detailed_reservoirs_imputed = detailed_reservoirs_imputed.sort_values('dam_height', ascending=False).drop_duplicates('name')
    
    return detailed_reservoirs_imputed

def etl_pipeline_detailed_reservoirs():
    detailed_reservoirs_data = extract_detailed_reservoirs_raw()
    transformed_detailed_reservoirs = transform_detailed_reservoirs_raw(detailed_reservoirs_data)
    load_detailed_reservoirs_cleaned(transformed_detailed_reservoirs)

if __name__ == "__main__":
    etl_pipeline_detailed_reservoirs()
