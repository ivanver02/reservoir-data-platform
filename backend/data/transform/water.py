# Importing modules
import pandas as pd
from pathlib import Path
import sys

# Setting up the path to include the parent directory
sys.path.append(str(Path.cwd().parent.parent.parent))
from backend.config.settings import TRANSFORM_SETTINGS
from backend.data.extract import extract_water_raw
from backend.data.load import load_water_cleaned
from backend.data.cleaning import reindex_weekly

COLUMNS_MAP = TRANSFORM_SETTINGS['WATER_TRANSFORM']['COLUMNS_MAP']
DATE_FORMAT = TRANSFORM_SETTINGS['WATER_TRANSFORM']['DATE_FORMAT']
YEAR_CUTOFF = TRANSFORM_SETTINGS['WATER_TRANSFORM']['YEAR_CUTOFF']

def date_column_to_datetime(water_data):
    water_pd_split = water_data['date'].str.rsplit(pat='/', n=1, expand=True)
    two_digit_years = water_pd_split[1].str.split(' ').str[0].astype(int)
    complete_years = two_digit_years.apply(lambda x: x + 1900 if x >= YEAR_CUTOFF else x + 2000)
    definite_date_column = water_pd_split[0] + '/' + complete_years.astype(str)
    water_data['date'] = pd.to_datetime(definite_date_column, format=DATE_FORMAT)
    return water_data

def transform_water_raw(water_data):
    water_renamed = water_data.rename(columns=COLUMNS_MAP)

    # As there are only two rows with missing storage
    water_imputed = water_renamed.dropna()
    water_with_datetime = date_column_to_datetime(water_imputed)

    # Filtering out rows before 2005 for reservoir 396 and before 2012 for reservoir 376
    filter_mask = ((water_with_datetime['id'] == 396) & (water_with_datetime['date'].dt.year < 2005)) | ((water_with_datetime['id'] == 376) & (water_with_datetime['date'].dt.year < 2012))
    water_filtered = water_with_datetime[filter_mask == False]

    # Reindex data to have a complete weekly date range
    water_filtered_full = water_filtered.drop(columns=['id']).groupby(water_filtered['id'], group_keys=False).apply(reindex_weekly)

    # Add an imputed feature to be able to know in the future if the storage was filled
    water_filtered_full['storage_imputed'] = water_filtered_full['storage'].isna().astype(int)

    # Fill backward the imputed values
    water_filtered_full['storage'] = water_filtered_full['storage'].fillna(method='bfill')

    water_filtered_full['storage'] = water_filtered_full['storage'].astype(int)
    return water_filtered_full

def etl_pipeline_water():
    water_data = extract_water_raw()
    transformed_water = transform_water_raw(water_data)
    load_water_cleaned(transformed_water)

if __name__ == "__main__":
    etl_pipeline_water()
