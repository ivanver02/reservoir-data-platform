import pandas as pd
import re

from backend.config.settings import PATHS, TRANSFORM_SETTINGS
from backend.data.extract import extract_water_cleaned, extract_reservoirs_cleaned, extract_detailed_reservoirs_cleaned
from backend.data.load import load_water_definitive, load_reservoirs_definitive, load_detailed_reservoirs_definitive, load_reservoirs_merged_definitive
from backend.scraping.scrape_reservoirs import get_reservoir_province_reusing_data
from backend.scraping.coordinates_api import get_coordinates_reusing_data
from backend.scraping.real_names_api import get_real_names_nominatim
from backend.data.cleaning import impute_nearest_neighbour

REPEATED_IN_DETAIL = TRANSFORM_SETTINGS['MERGES_TRANSFORM']['REPEATED_IN_DETAIL']
REPEATED_IN_RESERVOIRS = TRANSFORM_SETTINGS['MERGES_TRANSFORM']['REPEATED_IN_RESERVOIRS']
NAME_ALIASES = TRANSFORM_SETTINGS['MERGES_TRANSFORM']['NAME_ALIASES']
MANUAL_PROVINCE_NAMES = TRANSFORM_SETTINGS['MERGES_TRANSFORM']['MANUAL_PROVINCE_NAMES']
MANUAL_PROVINCES = TRANSFORM_SETTINGS['MERGES_TRANSFORM']['MANUAL_PROVINCES']
UNIFIED_PROVINCES = TRANSFORM_SETTINGS['MERGES_TRANSFORM']['UNIFIED_PROVINCES']
PROVINCE_TO_COMMUNITY = TRANSFORM_SETTINGS['MERGES_TRANSFORM']['PROVINCE_TO_COMMUNITY']
MANUAL_COORDINATES_ROWS = TRANSFORM_SETTINGS['MERGES_TRANSFORM']['MANUAL_COORDINATES_ROWS']
ACCENTS = TRANSFORM_SETTINGS['MERGES_TRANSFORM']['ACCENTS']
STOPWORDS = TRANSFORM_SETTINGS['MERGES_TRANSFORM']['STOPWORDS']
MISSING_REAL_NAMES = TRANSFORM_SETTINGS['MERGES_TRANSFORM']['MISSING_REAL_NAMES']
MANUAL_REAL_NAMES = TRANSFORM_SETTINGS['MERGES_TRANSFORM']['MANUAL_REAL_NAMES']

def get_real_names_nominatim_reusing_data(final_merged_df):
    """ Reuses cached real names before calling the external lookup """
    final_merged_df = final_merged_df.copy()
    path = PATHS['cache'] / 'real_names.csv'

    # Reject an invalid cache before joining data
    if path.exists():
        cached = pd.read_csv(path)
        required = {'name', 'real_name'}

        if not required.issubset(cached.columns) or cached['name'].duplicated().any():
            raise ValueError('real_names.csv has an invalid schema or duplicate names')
        final_merged_df = final_merged_df.merge(
            cached[['name', 'real_name']],
            on='name',
            how='left',
            validate='many_to_one',
        )

    if 'real_name' not in final_merged_df or final_merged_df['real_name'].isna().any():
        final_merged_df = get_real_names_nominatim(final_merged_df)
        path.parent.mkdir(parents=True, exist_ok=True)
        final_merged_df[['name', 'real_name']].drop_duplicates('name').to_csv(path, index=False)
    return final_merged_df


def contains_variations(candidate, real_name):
    """ Checks whether a candidate name appears inside a real name """
    if pd.isna(candidate) or pd.isna(real_name):
        return False, real_name
    candidate = str(candidate).strip().lower()
    real_name_cleaned = real_name.lower()
    for accented_char, unaccented_char in ACCENTS.items():
        candidate = candidate.replace(accented_char, unaccented_char)
        real_name_cleaned = real_name_cleaned.replace(accented_char, unaccented_char)

    return (candidate in real_name_cleaned), real_name

def select_between_commas(names_series):
    """ Selects the useful part of a lookup result """
    definitive = pd.DataFrame(columns=['name', 'real_name'])
    for _, row in names_series.iterrows():
        name = row['name']
        real_name = row['real_name']
        if pd.isna(name) or pd.isna(real_name):
            continue
        found = False
        name_split = name.split(' ')
        real_name_split = real_name.split(',')
        name_word = 0

        # Match words against the lookup parts
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
    """ Removes prefixes from resolved names """
    result = df.copy()
    for stopword in STOPWORDS:
        mask = result['real_name'].notna() & result['real_name'].astype(str).str.contains(
            re.escape(stopword), regex=True,
        )
        result.loc[mask, 'real_name'] = (
            result.loc[mask, 'real_name'].astype(str).str.split(stopword, n=1).str[-1].str.strip()
        )
    return result


def transform_manually(df):
    """ Applies reviewed name corrections """
    result = df.copy()
    replacements = result['name'].map(MANUAL_REAL_NAMES)
    result.loc[replacements.notna(), 'real_name'] = replacements.dropna()
    return result

def transform_all(water_non_merged_pd: pd.DataFrame, reservoirs_pd: pd.DataFrame,
                  detailed_reservoirs_pd: pd.DataFrame):
    """ Joins three tables into reservoir tables """

    # Copy inputs before building the outputs
    water_non_merged_pd = water_non_merged_pd.copy()
    reservoirs_pd = reservoirs_pd.copy()
    detailed_reservoirs_pd = detailed_reservoirs_pd.copy()
    water_non_merged_pd['date'] = pd.to_datetime(water_non_merged_pd['date'])

    # Reject duplicate keys before merging
    if water_non_merged_pd.duplicated(['id', 'date']).any():
        raise ValueError('water contains duplicated (id, date) observations')

    if reservoirs_pd['id'].duplicated().any():
        raise ValueError('reservoirs contains duplicated ids')

    if detailed_reservoirs_pd['name'].duplicated().any():
        raise ValueError('detailed reservoirs contains duplicated names')

    # Align names before joining detail fields
    series_repeated = pd.Series(REPEATED_IN_DETAIL, index=REPEATED_IN_RESERVOIRS)
    reservoirs_pd['name'] = reservoirs_pd['name'].map(series_repeated).fillna(reservoirs_pd['name'])
    reservoirs_for_merge = reservoirs_pd.drop_duplicates('name', keep='first').copy()
    detailed_for_merge = detailed_reservoirs_pd.copy()
    reservoirs_for_merge['_join_name'] = reservoirs_for_merge['name'].replace(NAME_ALIASES)
    detailed_for_merge['_join_name'] = detailed_for_merge['name'].replace(NAME_ALIASES)

    # Add detail fields to the main table
    final_merged_df = pd.merge(
        reservoirs_for_merge,
        detailed_for_merge,
        how='left',
        on='_join_name',
        validate='one_to_one',
    )
    final_merged_df['name'] = final_merged_df['name_x'].replace(NAME_ALIASES)
    final_merged_df = final_merged_df.drop(columns=['name_x', 'name_y', '_join_name'])

    # Keep old reservoirs out of metadata lookups
    max_date = water_non_merged_pd['date'].max()
    freshness_limit = max_date - pd.Timedelta(
        weeks=TRANSFORM_SETTINGS['WATER_TRANSFORM']['MAX_STALENESS_WEEKS']
    )
    latest_dates = water_non_merged_pd.groupby('id')['date'].max()
    fresh_ids = latest_dates[latest_dates >= freshness_limit].index
    final_merged_df = final_merged_df[final_merged_df['id'].isin(fresh_ids)]
    
    # Reuse province lookups before network calls
    missing_province = final_merged_df['province'].isna()
    if missing_province.any():
        final_merged_df.loc[missing_province, 'province'] = (
            final_merged_df.loc[missing_province, 'name']
            .apply(get_reservoir_province_reusing_data)
        )
    
    
    # Fill names that the lookup misses
    if final_merged_df['province'].isna().any():
        series_manual = pd.Series(MANUAL_PROVINCES, index=MANUAL_PROVINCE_NAMES)
        mask_manual = final_merged_df['province'].isna()
        final_merged_df.loc[mask_manual, 'province'] = final_merged_df.loc[mask_manual, 'name'].map(series_manual)

    # Normalize provinces before deriving the community
    final_merged_df['province'] = final_merged_df['province'].map(UNIFIED_PROVINCES).fillna(final_merged_df['province'])
    final_merged_df['autonomous_community'] = final_merged_df['province'].map(PROVINCE_TO_COMMUNITY)

    
    # Cap storage at reservoir capacity
    capacity_by_id = reservoirs_pd.set_index('id')['capacity']
    water_capacity = water_non_merged_pd['id'].map(capacity_by_id)
    if water_capacity.isna().any():
        raise ValueError('water contains ids without reservoir capacity')
    mask_storage = water_non_merged_pd['storage'] > water_capacity
    if mask_storage.any():
        water_non_merged_pd.loc[mask_storage, 'storage'] = water_capacity[mask_storage]


    # Keep reservoirs with observations
    max_date = water_non_merged_pd['date'].max()
    freshness_limit = max_date - pd.Timedelta(
        weeks=TRANSFORM_SETTINGS['WATER_TRANSFORM']['MAX_STALENESS_WEEKS']
    )
    latest_dates = water_non_merged_pd.groupby('id')['date'].max()
    list_final_reservoirs = latest_dates[latest_dates >= freshness_limit].index
    water_non_merged_pd = water_non_merged_pd[water_non_merged_pd['id'].isin(list_final_reservoirs)]
    reservoirs_pd = reservoirs_pd[reservoirs_pd['id'].isin(list_final_reservoirs)]
    detailed_reservoirs_pd = detailed_reservoirs_pd[detailed_reservoirs_pd['id'].isin(list_final_reservoirs)] if 'id' in detailed_reservoirs_pd.columns else detailed_reservoirs_pd
    final_merged_df = final_merged_df[final_merged_df['id'].isin(list_final_reservoirs)]

    # Fill coordinates from cache and fixed values
    reservoirs_without_coordinates = final_merged_df[
        final_merged_df[['latitude', 'longitude']].isna().any(axis=1)
    ]
    if not reservoirs_without_coordinates.empty:
        coordinates = get_coordinates_reusing_data(
            reservoirs_without_coordinates['name'].values
        )

        # Prefer fixed coordinates over service results
        coordinates_df = coordinates.copy()
        coords_df = pd.concat([pd.DataFrame(MANUAL_COORDINATES_ROWS), coordinates_df], ignore_index=True)
        coords_df = coords_df[coords_df[['latitude', 'longitude']].notna().all(axis=1)]
        coords_df.set_index('name', inplace=True)
        name_to_lat = coords_df['latitude'].to_dict()
        name_to_lon = coords_df['longitude'].to_dict()
        mask_coords = final_merged_df['name'].isin(coords_df.index)
        final_merged_df.loc[mask_coords, 'latitude'] = final_merged_df.loc[mask_coords, 'name'].map(name_to_lat)
        final_merged_df.loc[mask_coords, 'longitude'] = final_merged_df.loc[mask_coords, 'name'].map(name_to_lon)

    # Check coordinates before spatial imputation
    valid_latitude = final_merged_df['latitude'].dropna().between(-90, 90).all()
    valid_longitude = final_merged_df['longitude'].dropna().between(-180, 180).all()
    if not valid_latitude or not valid_longitude:
        raise ValueError('merged reservoirs contains invalid coordinates')

    # Use nearby rows to fill these fields
    for col in ['crest_elevation', 'riverbed', 'basin']:
        if col in final_merged_df.columns:
            final_merged_df = impute_nearest_neighbour(final_merged_df, col)

    # Resolve names after filtering reservoirs
    reservoirs_merged = get_real_names_nominatim_reusing_data(final_merged_df)
    if reservoirs_merged['name'].duplicated().any():
        raise ValueError('merged reservoirs contains duplicated names')
    # Clean lookup results and apply reviewed exceptions
    names_series = reservoirs_merged[['name', 'real_name']]
    definitive = select_between_commas(names_series)
    df = definitive.copy()
    df = remove_stopwords(df)
    df = transform_manually(df)
    df = pd.concat([df, pd.DataFrame(MISSING_REAL_NAMES)], ignore_index=True)
    df = df.drop_duplicates('name', keep='last')
    reservoirs_merged.loc[:, 'real_name'] = reservoirs_merged['name'].map(df.set_index('name')['real_name'])

    reservoirs_merged = reservoirs_merged[
        [
            'id', 'scope', 'name', 'capacity', 'electric_flag', 'longitude',
            'latitude', 'basin', 'riverbed', 'google', 'openstreetmap',
            'wikidata', 'province', 'autonomous_community', 'type',
            'crest_elevation', 'dam_height', 'report', 'real_name',
        ]
    ]

    reservoirs_pd = reservoirs_pd[['id', 'scope', 'name', 'capacity', 'electric_flag']]
    detailed_reservoirs_pd = detailed_reservoirs_pd[
        [
            'name', 'longitude', 'latitude', 'basin', 'riverbed', 'google',
            'openstreetmap', 'wikidata', 'province', 'autonomous_community',
            'type', 'crest_elevation', 'dam_height', 'report',
        ]
    ]

    return water_non_merged_pd, reservoirs_pd, detailed_reservoirs_pd, reservoirs_merged

def etl_pipeline_reservoirs_merged():
    """ Joins the tables and writes the output tables """
    # Read inputs for the merge
    water_non_merged_pd = extract_water_cleaned()
    reservoirs_pd = extract_reservoirs_cleaned()
    detailed_reservoirs_pd = extract_detailed_reservoirs_cleaned()

    # Build all outputs in one step
    water_final, reservoirs_final, detailed_final, reservoirs_merged = transform_all(
        water_non_merged_pd, reservoirs_pd, detailed_reservoirs_pd
    )

    # Write each output to its path
    load_water_definitive(water_final)
    load_reservoirs_merged_definitive(reservoirs_merged)
    load_reservoirs_definitive(reservoirs_final)
    load_detailed_reservoirs_definitive(detailed_final)


if __name__ == "__main__":
    etl_pipeline_reservoirs_merged()
