import pandas as pd

from backend.config.settings import TRANSFORM_SETTINGS
from backend.data.extract import extract_water_raw
from backend.data.load import load_water_cleaned
from backend.data.cleaning import reindex_weekly
from backend.data.validation import require_columns, validate_water

COLUMNS_MAP = TRANSFORM_SETTINGS['WATER_TRANSFORM']['COLUMNS_MAP']
DATE_FORMAT = TRANSFORM_SETTINGS['WATER_TRANSFORM']['DATE_FORMAT']

def date_column_to_datetime(water_data):
    """ Converts the date column into timestamps """
    data = water_data.copy()
    raw_dates = data['date'].astype('string').str.strip()
    parsed = pd.to_datetime(raw_dates, format=DATE_FORMAT, errors='coerce', dayfirst=True)
    parsed = parsed.fillna(pd.to_datetime(raw_dates, errors='coerce', dayfirst=True))

    if parsed.isna().any():
        examples = raw_dates[parsed.isna()].head(3).tolist()
        raise ValueError(f"water contains invalid dates, examples: {examples}")
    data['date'] = parsed
    return data

def transform_water_raw(water_data):
    """ Cleans water data and rebuilds observations """
    water_renamed = water_data.rename(columns=COLUMNS_MAP).copy()
    require_columns(water_renamed, {'id', 'date', 'storage'}, 'water')
    water_renamed['storage'] = pd.to_numeric(water_renamed['storage'], errors='coerce')
    water_renamed = date_column_to_datetime(water_renamed)
    water_renamed = water_renamed.dropna(subset=['id', 'date'])

    if water_renamed.duplicated(['id', 'date']).any():
        raise ValueError('water contains duplicated (id, date) observations')

    # Apply corrections before rebuilding gaps
    filter_mask = ((water_renamed['id'] == 396) & (water_renamed['date'].dt.year < 2005)) | ((water_renamed['id'] == 376) & (water_renamed['date'].dt.year < 2012))
    water_filtered = water_renamed.loc[~filter_mask].copy()

    reindexed_groups = [
        reindex_weekly(group.drop(columns=['id']), reservoir_id)
        for reservoir_id, group in water_filtered.groupby('id', sort=False)
    ]
    
    if not reindexed_groups:
        raise ValueError('water contains no observations after source filters')
    water_filtered_full = pd.concat(reindexed_groups, ignore_index=True)
    water_filtered_full = water_filtered_full.reset_index(drop=True)

    # Mark gaps before filling
    water_filtered_full['storage_imputed'] = water_filtered_full['storage'].isna().astype(int)
    water_filtered_full['storage'] = water_filtered_full.groupby('id')['storage'].bfill()
    water_filtered_full = water_filtered_full.dropna(subset=['storage']).copy()
    water_filtered_full['storage'] = water_filtered_full['storage'].astype(float)
    water_filtered_full = water_filtered_full.sort_values(['id', 'date']).reset_index(drop=True)
    validate_water(water_filtered_full)
    return water_filtered_full

def etl_pipeline_water():
    """ Runs the water extraction, transformation, and load steps """
    water_data = extract_water_raw()
    transformed_water = transform_water_raw(water_data)
    load_water_cleaned(transformed_water)

if __name__ == "__main__":
    etl_pipeline_water()
