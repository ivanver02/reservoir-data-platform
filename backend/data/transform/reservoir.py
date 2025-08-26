# Importing modules
import pandas as pd
from pathlib import Path
import sys

# Setting up the path to include the parent directory
sys.path.append(str(Path.cwd().parent.parent.parent))
from backend.data.extract import extract_reservoirs_raw, extract_water_cleaned
from backend.data.load import load_reservoirs_cleaned
from backend.data.cleaning import clean_string_series

columns_dict = {'ID': 'id', 'SCOPE_NAME': 'scope', 'RESERVOIR_NAME': 'name', 'TOTAL_WATER':'capacity', 'ELECTRIC_FLAG': 'electric_flag'}

def transform_reservoirs_raw(reservoirs_data):
    reservoirs_renamed = reservoirs_data.rename(columns=columns_dict)

    # Missing values: only one capacity, so it will be filled with it's maximum storage registered
    water_df = extract_water_cleaned()
    id_to_current_water = water_df.groupby('id')['storage'].max() # Creates a series with 'ID' as index
    mask = reservoirs_renamed['capacity'].isna()
    reservoirs_renamed.loc[mask, 'capacity'] = reservoirs_renamed.loc[mask, 'id'].map(id_to_current_water)
    reservoirs_renamed.loc[:, 'capacity'] = reservoirs_renamed['capacity'].astype(int)

    # Cleaning string columns
    reservoirs_renamed.loc[:, 'name'] = clean_string_series(reservoirs_renamed['name'])
    reservoirs_renamed.loc[:, 'scope'] = clean_string_series(reservoirs_renamed['scope'])
    return reservoirs_renamed

def etl_pipeline_reservoirs():
    reservoirs_data = extract_reservoirs_raw()
    transformed_reservoirs = transform_reservoirs_raw(reservoirs_data)
    load_reservoirs_cleaned(transformed_reservoirs)

if __name__ == "__main__":
    etl_pipeline_reservoirs()
