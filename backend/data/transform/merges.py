# Importing modules
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Setting up path
sys.path.append(str(Path.cwd().parent.parent.parent))

from backend.config.settings import PATHS
from backend.data.extract import (
    extract_water_cleaned,
    extract_reservoirs_cleaned,
    extract_detailed_reservoirs_cleaned,
)
from backend.data.load import (
    load_water_definitive,
    load_reservoirs_definitive,
    load_detailed_reservoirs_definitive,
    load_reservoirs_merged_definitive
)
from backend.scraping.scrape_reservoirs import get_reservoir_province_reusing_data
from backend.scraping.coordinates_api import get_coordinates_reusing_data
from backend.data.cleaning import impute_nearest_neighbour

REPEATED_IN_DETAIL = ['agrio', 'aguilar campoo', 'algeciras  rambla', 'arcos', 'bachimana alto', 'banos montemayor', 'burguillo', 'castrelo mino', 'chandreja', 'grado i', 'guadalteba', 'ibon ip', 'jerte', 'malpasillo jauja', 'molinos matachel', 'monteagudo vicarias', 'lago negro', 'peares', 'puentes iv', 'sant ponc', 'torre abraham', 'tous', 'vilagudin', 'zahara', 'juan benet']
REPEATED_IN_RESERVOIRS = ['agrio (aznalcollar)', 'aguilar', 'algeciras', 'arcos frontera', 'bachimana (lago)', 'banos', 'burguillo   puente nuevo', 'castrelo', 'chandrexa', 'grado', 'guadalhorce guadalteba', 'ip', 'jerte   plasencia', 'malpasillo ( jauja )', 'molinos', 'vicarias', 'negro (lago)', 'peares  os', 'puentes', 'sant pons', 'torre abrahan', 'tous   ribera', 'villagudin', 'zahara gastor', 'porma (juan benet)']

MANUAL_PROVINCE_NAMES = ['certescans', 'vellon (pedrezuela)', 'cornalbo', 'lechago']
MANUAL_PROVINCES = ['lleida', 'madrid', 'badajoz', 'teruel']

UNIFIED_PROVINCES = {
    'alacant alicante': 'alicante',
    'araba alava': 'alava',
    'castello castellon': 'castellon',
    'gipuzkoa guipuzcoa': 'guipuzcoa',
    'ourense': 'orense',
    'valència valencia': 'valencia',
    'a coruna': 'la coruna',
    'rioja': 'la rioja'
}

PROVINCE_TO_COMMUNITY = {
    'alava': 'pais vasco', 'albacete': 'castilla   mancha', 'alicante': 'comunitat valenciana',
    'almeria': 'andalucia', 'asturias': 'principado asturias', 'avila': 'castilla y leon',
    'badajoz': 'extremadura', 'barcelona': 'cataluna', 'burgos': 'castilla y leon',
    'caceres': 'extremadura', 'cadiz': 'andalucia', 'cantabria': 'cantabria',
    'castellon': 'comunitat valenciana', 'ciudad real': 'castilla   mancha', 'cordoba': 'andalucia',
    'cuenca': 'castilla   mancha', 'girona': 'cataluna', 'granada': 'andalucia',
    'guadalajara': 'castilla   mancha', 'guipuzcoa': 'pais vasco', 'huelva': 'andalucia',
    'huesca': 'aragon', 'jaen': 'andalucia', 'la coruna': 'galicia', 'la rioja': 'rioja',
    'las palmas': 'canarias', 'leon': 'castilla y leon', 'lleida': 'cataluna', 'lugo': 'galicia',
    'madrid': 'comunidad madrid', 'malaga': 'andalucia', 'murcia': 'region murcia',
    'navarra': 'comunidad foral navarra', 'orense': 'galicia', 'palencia': 'castilla y leon',
    'pontevedra': 'galicia', 'salamanca': 'castilla y leon', 'santa cruz de tenerife': 'canarias',
    'segovia': 'castilla y leon', 'sevilla': 'andalucia', 'soria': 'castilla y leon',
    'tarragona': 'cataluna', 'teruel': 'aragon', 'toledo': 'castilla   mancha',
    'valencia': 'comunitat valenciana', 'valladolid': 'castilla y leon', 'vizcaya': 'pais vasco',
    'zamora': 'castilla y leon', 'ceuta': 'ceuta', 'melilla': 'melilla', 'balears': 'islas baleares',
    'zaragoza': 'aragon'
}

MANUAL_COORDINATES_ROWS = [
    {'name': 'sistema lagos espot', 'latitude': 42.581771, 'longitude': 1.003018},
    {'name': 'agavanzal', 'latitude': 41.979600, 'longitude': -6.200336},
    {'name': 'sistema aguas limpias', 'latitude': 42.793868, 'longitude': -0.329262},
    {'name': 'sistema alto caldares', 'latitude': 42.427684, 'longitude': -0.227609},
    {'name': 'tremp o talarn', 'latitude': 42.204670, 'longitude': 0.949327},
]


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

    # 5. Province normalization & autonomous community
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

    return water_non_merged_pd, reservoirs_pd, detailed_reservoirs_pd, final_merged_df

def etl_pipeline():
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
    etl_pipeline()
