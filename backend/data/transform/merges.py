# Importing modules
import pandas as pd
from pathlib import Path
import sys

# Setting up path
sys.path.append(str(Path.cwd().parent.parent.parent))

from backend.config.settings import PATHS, TRANSFORM_SETTINGS
from backend.data.extract import extract_water_cleaned, extract_reservoirs_cleaned, extract_detailed_reservoirs_cleaned
from backend.data.load import load_water_definitive, load_reservoirs_definitive, load_detailed_reservoirs_definitive, load_reservoirs_merged_definitive
from backend.scraping.scrape_reservoirs import get_reservoir_province_reusing_data
from backend.scraping.coordinates_api import get_coordinates_reusing_data
from backend.scraping.real_names_api import get_real_names_nominatim
from backend.data.cleaning import impute_nearest_neighbour

REPEATED_IN_DETAIL = TRANSFORM_SETTINGS['MERGES_TRANSFORM']['REPEATED_IN_DETAIL']
REPEATED_IN_RESERVOIRS = TRANSFORM_SETTINGS['MERGES_TRANSFORM']['REPEATED_IN_RESERVOIRS']
MANUAL_PROVINCE_NAMES = TRANSFORM_SETTINGS['MERGES_TRANSFORM']['MANUAL_PROVINCE_NAMES']
MANUAL_PROVINCES = TRANSFORM_SETTINGS['MERGES_TRANSFORM']['MANUAL_PROVINCES']
UNIFIED_PROVINCES = TRANSFORM_SETTINGS['MERGES_TRANSFORM']['UNIFIED_PROVINCES']
PROVINCE_TO_COMMUNITY = TRANSFORM_SETTINGS['MERGES_TRANSFORM']['PROVINCE_TO_COMMUNITY']
MANUAL_COORDINATES_ROWS = TRANSFORM_SETTINGS['MERGES_TRANSFORM']['MANUAL_COORDINATES_ROWS']
ACCENTS = TRANSFORM_SETTINGS['MERGES_TRANSFORM']['ACCENTS']
STOPWORDS = TRANSFORM_SETTINGS['MERGES_TRANSFORM']['STOPWORDS']
MISSING_REAL_NAMES = TRANSFORM_SETTINGS['MERGES_TRANSFORM']['MISSING_REAL_NAMES']

def get_real_names_nominatim_reusing_data(final_merged_df):
    path = PATHS['raw_data_notebooks'] / 'real_names.csv'
    try:
        reservoirs = pd.read_csv(path)
        reservoirs = pd.Series(reservoirs['real_name'].values, index=final_merged_df.index)
        final_merged_df['real_name'] = reservoirs
    except:
        final_merged_df = get_real_names_nominatim(final_merged_df)
        final_merged_df['real_name'].to_csv(path, index=False)
    return final_merged_df


def contains_variations(candidate, real_name):
    real_name_cleaned = real_name.lower()
    for accented_char, unaccented_char in ACCENTS.items():
        candidate = candidate.replace(accented_char, unaccented_char)
        real_name_cleaned = real_name_cleaned.replace(accented_char, unaccented_char)

    return (candidate in real_name_cleaned), real_name

def select_between_commas(names_series):
    definitive = pd.DataFrame(columns=['name', 'real_name'])
    for _, row in names_series.iterrows():
        name = row['name']
        real_name = row['real_name']
        found = False
        name_split = name.split(' ')
        real_name_split = real_name.split(',')
        name_word = 0
        while not found and name_word < len(name_split):
            word = name_split[name_word]
            real_name_word = 0
            while not found and real_name_word < len(real_name_split):
                real_word = real_name_split[real_name_word]
                found, result = contains_variations(word, real_word)
                real_name_word += 1
            name_word += 1
        if found:
            definitive.loc[len(definitive)] = {'name': name, 'real_name': result.strip()}
    return definitive


def remove_stopwords(df):
    for index, name in df.iterrows():
        for stopword in STOPWORDS:
            if stopword in name['real_name']:
                df.at[index, 'real_name'] = df['real_name'].str.split(stopword).str[-1].values[index]
                break
    return df

def transform_manually(df):
    df.loc[164, 'real_name'] = 'Pico de Urdiceto'
    df.loc[300, 'real_name'] = 'Gasset'
    df.loc[202, 'real_name'] = 'Canal del Taibilla'
    df.loc[234, 'real_name'] = 'La Cierva'
    df.loc[258, 'real_name'] = 'La Toba'
    df.loc[291, 'real_name'] = 'Llosa de Cavall'
    df.loc[319, 'real_name'] = 'Las Yeguas'
    df.loc[326, 'real_name'] = 'Juan Benet'
    return df

def transform_all(water_non_merged_pd: pd.DataFrame,
                  reservoirs_pd: pd.DataFrame,
                  detailed_reservoirs_pd: pd.DataFrame):

    # Name correspondence mapping
    series_repeated = pd.Series(REPEATED_IN_DETAIL, index=REPEATED_IN_RESERVOIRS)
    reservoirs_pd['name'] = reservoirs_pd['name'].map(series_repeated).fillna(reservoirs_pd['name'])

    final_merged_df = pd.merge(reservoirs_pd, detailed_reservoirs_pd, how='left', on='name')
    
    # Province scraping reuse
    final_merged_df['province'] = final_merged_df['province'].where(
        final_merged_df['province'].notna(),
        final_merged_df['name'].apply(get_reservoir_province_reusing_data)
    )
    
    
    if final_merged_df['province'].isna().any():
        series_manual = pd.Series(MANUAL_PROVINCES, index=MANUAL_PROVINCE_NAMES)
        mask_manual = final_merged_df['province'].isna()
        final_merged_df.loc[mask_manual, 'province'] = final_merged_df.loc[mask_manual, 'name'].map(series_manual)

    # Province normalization & autonomous community
    final_merged_df['province'] = final_merged_df['province'].map(UNIFIED_PROVINCES).fillna(final_merged_df['province'])
    final_merged_df['autonomous_community'] = final_merged_df['province'].map(PROVINCE_TO_COMMUNITY)

    
    # Merge water with selected reservoir cols for capacity clipping
    water_pd = pd.merge(
        water_non_merged_pd,
        final_merged_df[['id', 'capacity', 'crest_elevation', 'province', 'autonomous_community']],
        on='id', how='left'
    )

    # Clip storage to capacity
    mask_storage = water_pd['storage'] > water_pd['capacity']
    if mask_storage.any():
        water_pd.loc[mask_storage, 'storage'] = water_pd.loc[mask_storage, 'capacity']
        water_non_merged_pd.loc[mask_storage, 'storage'] = water_pd.loc[mask_storage, 'capacity']


    # Filter datasets to reservoirs current at max date
    max_date = water_pd['date'].max()
    list_final_reservoirs = water_pd[water_pd['date'] == max_date]['id'].values
    water_non_merged_pd = water_non_merged_pd[water_non_merged_pd['id'].isin(list_final_reservoirs)]
    reservoirs_pd = reservoirs_pd[reservoirs_pd['id'].isin(list_final_reservoirs)]
    detailed_reservoirs_pd = detailed_reservoirs_pd[detailed_reservoirs_pd['id'].isin(list_final_reservoirs)] if 'id' in detailed_reservoirs_pd.columns else detailed_reservoirs_pd
    final_merged_df = final_merged_df[final_merged_df['id'].isin(list_final_reservoirs)]

    # Coordinate imputation
    reservoirs_without_coordinates = final_merged_df[final_merged_df['latitude'].isna()]
    if not reservoirs_without_coordinates.empty:
        coordinates = get_coordinates_reusing_data(reservoirs_without_coordinates['name'].values)
        
        coordinates_df = coordinates.copy()
        coords_df = pd.concat([pd.DataFrame(MANUAL_COORDINATES_ROWS), coordinates_df], ignore_index=True)
        coords_df = coords_df[coords_df['latitude'].notna()]
        coords_df.set_index('name', inplace=True)
        final_merged_df.loc[final_merged_df['id'] == 342, 'name'] = 'agavanzal'
        name_to_lat = coords_df['latitude'].to_dict(); name_to_lon = coords_df['longitude'].to_dict()
        mask_coords = final_merged_df['name'].isin(coords_df.index)
        final_merged_df.loc[mask_coords, 'latitude'] = final_merged_df.loc[mask_coords, 'name'].map(name_to_lat)
        final_merged_df.loc[mask_coords, 'longitude'] = final_merged_df.loc[mask_coords, 'name'].map(name_to_lon)

    # Nearest neighbour imputations (crest_elevation, riverbed, basin)
    for col in ['crest_elevation', 'riverbed', 'basin']:
        if col in final_merged_df.columns:
            final_merged_df = impute_nearest_neighbour(final_merged_df, col)

    reservoirs_merged = get_real_names_nominatim_reusing_data(final_merged_df)
    reservoirs_merged = reservoirs_merged.drop_duplicates(subset='name', keep='first')
    names_series = reservoirs_merged[['name', 'real_name']]
    definitive = select_between_commas(names_series)
    df = definitive.copy()
    df = remove_stopwords(df)
    df = transform_manually(df)
    df = pd.concat([df, pd.DataFrame(MISSING_REAL_NAMES)], ignore_index=True)
    reservoirs_merged.loc[:, 'real_name'] = reservoirs_merged['name'].map(df.set_index('name')['real_name'])

    return water_non_merged_pd, reservoirs_pd, detailed_reservoirs_pd, reservoirs_merged

def etl_pipeline_reservoirs_merged():
    # Extract
    water_non_merged_pd = extract_water_cleaned()
    reservoirs_pd = extract_reservoirs_cleaned()
    detailed_reservoirs_pd = extract_detailed_reservoirs_cleaned()

    # Single transform replicating notebook
    water_final, reservoirs_final, detailed_final, reservoirs_merged = transform_all(
        water_non_merged_pd, reservoirs_pd, detailed_reservoirs_pd
    )

    # Load
    load_water_definitive(water_final)
    load_reservoirs_merged_definitive(reservoirs_merged)
    load_reservoirs_definitive(reservoirs_final)
    load_detailed_reservoirs_definitive(detailed_final)


if __name__ == "__main__":
    etl_pipeline_reservoirs_merged()
