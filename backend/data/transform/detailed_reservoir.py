import pandas as pd
import numpy as np

from backend.config.settings import TRANSFORM_SETTINGS
from backend.data.extract import extract_detailed_reservoirs_raw
from backend.data.load import load_detailed_reservoirs_cleaned
from backend.data.cleaning import clean_string_series, impute_nearest_neighbour

COLUMNS_DICT = TRANSFORM_SETTINGS['DETAILED_RESERVOIRS_TRANSFORM']['COLUMNS_DICT']

def handle_missing_values(detailed_reservoirs_data):
    """ Fills detail fields from nearby reservoirs """
    detailed_reservoirs_data = impute_nearest_neighbour(detailed_reservoirs_data, 'riverbed')
    detailed_reservoirs_data = impute_nearest_neighbour(detailed_reservoirs_data, 'crest_elevation')
    return detailed_reservoirs_data

def clean_string_columns(detailed_reservoirs_data, columns):
    """ Applies name normalisation to chosen columns """
    detailed_reservoirs_data = detailed_reservoirs_data.copy()
    for column in columns:
        detailed_reservoirs_data.loc[:, column] = clean_string_series(detailed_reservoirs_data[column])
    return detailed_reservoirs_data

def transform_detailed_reservoirs_raw(detailed_reservoirs_data):
    """ Cleans and deduplicates reservoir details """
    detailed_reservoirs_data = detailed_reservoirs_data.copy()
    detailed_reservoirs_data.columns = detailed_reservoirs_data.columns.str.lower()
    detailed_reservoirs_renamed = detailed_reservoirs_data.rename(columns=COLUMNS_DICT)
    detailed_reservoirs_renamed = detailed_reservoirs_renamed.drop(columns=['code'], errors='ignore')

    detailed_reservoirs_renamed = clean_string_columns(
        detailed_reservoirs_renamed,
        ['name', 'reservoir', 'basin', 'province', 'autonomous_community', 'type'],
    )

    # Use the normalized name as the key
    detailed_reservoirs_renamed = detailed_reservoirs_renamed.drop(columns=['reservoir'])

    # Convert coordinates and decimal commas
    for column in ['longitude', 'latitude']:
        detailed_reservoirs_renamed[column] = pd.to_numeric(
            detailed_reservoirs_renamed[column].astype('string').str.replace(',', '.', regex=False),
            errors='coerce',
        )

    mask = detailed_reservoirs_renamed['riverbed'].astype('string').str.lower() == 'sin nombre'
    detailed_reservoirs_renamed.loc[mask, 'riverbed'] = np.nan

    detailed_reservoirs_imputed = handle_missing_values(detailed_reservoirs_renamed)

    detailed_reservoirs_imputed['riverbed'] = clean_string_series(detailed_reservoirs_imputed['riverbed'])

    # Convert values after imputation
    detailed_reservoirs_imputed['crest_elevation'] = pd.to_numeric(
        detailed_reservoirs_imputed['crest_elevation'].astype('string').str.replace(',', '.', regex=False),
        errors='coerce',
    )

    mask = detailed_reservoirs_imputed['dam_height'].notna()
    detailed_reservoirs_imputed.loc[mask, 'dam_height'] = pd.to_numeric(
        detailed_reservoirs_imputed.loc[mask, 'dam_height'].astype('string').str.replace(',', '.', regex=False),
        errors='coerce',
    )

    # Keep the row with the largest dam
    detailed_reservoirs_imputed = (
        detailed_reservoirs_imputed
        .sort_values(['name', 'dam_height'], ascending=[True, False], na_position='last', kind='mergesort')
        .drop_duplicates('name', keep='first')
    )
    
    return detailed_reservoirs_imputed

def etl_pipeline_detailed_reservoir():
    """ Runs the detailed reservoir extraction and load steps """
    detailed_reservoirs_data = extract_detailed_reservoirs_raw()
    transformed_detailed_reservoirs = transform_detailed_reservoirs_raw(detailed_reservoirs_data)
    load_detailed_reservoirs_cleaned(transformed_detailed_reservoirs)

if __name__ == "__main__":
    etl_pipeline_detailed_reservoir()
