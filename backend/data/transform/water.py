# Importing modules
import pandas as pd
from pathlib import Path
import sys

# Setting up the path to include the parent directory
sys.path.append(str(Path.cwd().parent.parent.parent))
from backend.data.extract import extract_water_raw
from backend.data.load import load_water_cleaned

COLUMNS_MAP = {'DATE': 'date', 'CURRENT_WATER': 'storage', 'ID': 'id'}
DATE_FORMAT = '%d/%m/%Y'
YEAR_CUTOFF = 85

def date_column_to_datetime(water_data):
    water_pd_split = water_data['date'].str.rsplit(pat='/', n=1, expand=True)
    two_digit_years = water_pd_split[1].str.split(' ').str[0].astype(int)
    complete_years = two_digit_years.apply(lambda x: x + 1900 if x >= YEAR_CUTOFF else x + 2000)
    definite_date_column = water_pd_split[0] + '/' + complete_years.astype(str)
    water_data.loc[:,'date'] = pd.to_datetime(definite_date_column, format=DATE_FORMAT)
    return water_data

def transform_water_raw(water_data):
    water_renamed = water_data.rename(columns=COLUMNS_MAP)

    # As there are only two rows with missing storage
    water_imputed = water_renamed.dropna()
    water_with_datetime = date_column_to_datetime(water_imputed)
    water_with_datetime.loc[:, 'storage'] = water_with_datetime['storage'].astype(int)
    return water_with_datetime

def etl_pipeline_water():
    water_data = extract_water_raw()
    transformed_water = transform_water_raw(water_data)
    load_water_cleaned(transformed_water)

if __name__ == "__main__":
    etl_pipeline_water()
