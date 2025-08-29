from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import pandas as pd
import time
import requests
import sys
from pathlib import Path

sys.path.append(str(Path.cwd().parent.parent))
from backend.config.settings import PATHS, COORDINATES_SETTINGS

# Nominatim configuration
NOMINATIM_USER_AGENT = COORDINATES_SETTINGS["NOMINATIM_USER_AGENT"]
NOMINATIM_TIMEOUT = COORDINATES_SETTINGS["NOMINATIM_TIMEOUT"]
NOMINATIM_SEARCH_TEMPLATES = COORDINATES_SETTINGS["NOMINATIM_SEARCH_TEMPLATES"]

# Photon configuration
PHOTON_URL = COORDINATES_SETTINGS["PHOTON_URL"]
PHOTON_LIMIT = COORDINATES_SETTINGS["PHOTON_LIMIT"]
PHOTON_TIMEOUT = COORDINATES_SETTINGS["PHOTON_TIMEOUT"]
PHOTON_QUERY_TEMPLATE = COORDINATES_SETTINGS["PHOTON_QUERY_TEMPLATE"]

# Generic rate limiting between external calls (seconds)
RATE_LIMIT_SECONDS = COORDINATES_SETTINGS["RATE_LIMIT_SECONDS"]

# Column names for output (centralized in case of future renames)
COL_NAME = COORDINATES_SETTINGS["COL_NAME"]
COL_LAT = COORDINATES_SETTINGS["COL_LAT"]
COL_LON = COORDINATES_SETTINGS["COL_LON"]

def get_coordinates_nominatim(reservoir_names):
    """Return coordinates for each reservoir name using Nominatim.

    Tries several query templates until one resolves. Respects a 1 second
    delay between attempts to stay polite with the public API.
    """
    geolocator_osm = Nominatim(user_agent=NOMINATIM_USER_AGENT)
    results = []
    for name in reservoir_names:
        try:
            location = None
            for template in NOMINATIM_SEARCH_TEMPLATES:
                term = template.format(name=name)
                try:
                    location = geolocator_osm.geocode(term, timeout=NOMINATIM_TIMEOUT)
                    if location:
                        break
                except GeocoderTimedOut:
                    continue
            if location:
                results.append({COL_NAME: name, COL_LAT: location.latitude, COL_LON: location.longitude})
            else:
                results.append({COL_NAME: name, COL_LAT: None, COL_LON: None})
        except Exception:
            results.append({COL_NAME: name, COL_LAT: None, COL_LON: None})
        time.sleep(RATE_LIMIT_SECONDS)
    return pd.DataFrame(results)

def get_coordinates_photon(reservoir_list):
    """Return coordinates using Photon for each provided reservoir name."""
    results = []
    for name in reservoir_list:
        try:
            params = {
                'q': PHOTON_QUERY_TEMPLATE.format(name=name),
                'limit': PHOTON_LIMIT,
            }
            response = requests.get(PHOTON_URL, params=params, timeout=PHOTON_TIMEOUT)
            data = response.json()
            features = data.get('features', [])
            if features:
                coords = features[0]['geometry']['coordinates']  # [lon, lat]
                results.append({COL_NAME: name, COL_LAT: coords[1], COL_LON: coords[0]})
            else:
                results.append({COL_NAME: name, COL_LAT: None, COL_LON: None})
        except Exception:
            results.append({COL_NAME: name, COL_LAT: None, COL_LON: None})
        time.sleep(RATE_LIMIT_SECONDS)
    return pd.DataFrame(results)

def get_coordinates_reusing_data(reservoir_list):
    """Reuse previously stored coordinates unless there are still missing values."""
    saved_df_path = PATHS['definitive_notebooks'] / 'reservoirs_merged.parquet'
    saved_df = pd.read_parquet(saved_df_path)
    if saved_df[COL_LAT].isna().any():
        return get_coordinates_photon(reservoir_list)
    path = PATHS['raw_data_notebooks'] / 'coordinates.csv'
    return pd.read_csv(path)

if __name__ == "__main__":  # Simple manual test
    sample = ['burdalo', 'cabezuela']
    print(get_coordinates_photon(sample))