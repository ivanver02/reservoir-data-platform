import pandas as pd

from backend.config.settings import TRANSFORM_SETTINGS
from backend.data.extract import extract_reservoirs_raw, extract_water_cleaned
from backend.data.load import load_reservoirs_cleaned
from backend.data.cleaning import clean_string_series
from backend.data.validation import require_columns, validate_reservoirs

COLUMNS_DICT = TRANSFORM_SETTINGS['RESERVOIR_TRANSFORM']['COLUMNS_DICT']

def transform_reservoirs_raw(reservoirs_data):
    """ Cleans reservoir metadata and fills missing capacities """
    reservoirs_renamed = reservoirs_data.rename(columns=COLUMNS_DICT).copy()
    require_columns(reservoirs_renamed, {'id', 'name', 'capacity'}, 'reservoirs')
    if reservoirs_renamed['id'].duplicated().any():
        raise ValueError('reservoirs contains duplicated ids')

    reservoirs_renamed['capacity'] = pd.to_numeric(reservoirs_renamed['capacity'], errors='coerce')
    reservoirs_renamed['capacity_imputed'] = reservoirs_renamed['capacity'].isna()
    water_df = extract_water_cleaned()

    # Use maximum storage when capacity is absent
    id_to_current_water = water_df.groupby('id')['storage'].max()
    mask = reservoirs_renamed['capacity'].isna()
    reservoirs_renamed.loc[mask, 'capacity'] = reservoirs_renamed.loc[mask, 'id'].map(id_to_current_water)
    reservoirs_renamed['capacity'] = reservoirs_renamed['capacity'].astype(float)

    # Apply the same name rules to names and scopes
    reservoirs_renamed['name'] = clean_string_series(reservoirs_renamed['name'])
    reservoirs_renamed['scope'] = clean_string_series(reservoirs_renamed['scope'])
    validate_reservoirs(reservoirs_renamed)
    return reservoirs_renamed

def etl_pipeline_reservoir():
    """ Runs the reservoir metadata extraction and load steps """
    reservoirs_data = extract_reservoirs_raw()
    transformed_reservoirs = transform_reservoirs_raw(reservoirs_data)
    load_reservoirs_cleaned(transformed_reservoirs)

if __name__ == "__main__":
    etl_pipeline_reservoir()
