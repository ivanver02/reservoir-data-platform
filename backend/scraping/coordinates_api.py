from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import pandas as pd
import time
import requests
from backend.config.settings import PATHS, COORDINATES_SETTINGS

# Nominatim settings
NOMINATIM_USER_AGENT = COORDINATES_SETTINGS["NOMINATIM_USER_AGENT"]
NOMINATIM_TIMEOUT = COORDINATES_SETTINGS["NOMINATIM_TIMEOUT"]
NOMINATIM_SEARCH_TEMPLATES = COORDINATES_SETTINGS["NOMINATIM_SEARCH_TEMPLATES"]

# Photon settings
PHOTON_URL = COORDINATES_SETTINGS["PHOTON_URL"]
PHOTON_LIMIT = COORDINATES_SETTINGS["PHOTON_LIMIT"]
PHOTON_TIMEOUT = COORDINATES_SETTINGS["PHOTON_TIMEOUT"]
PHOTON_QUERY_TEMPLATE = COORDINATES_SETTINGS["PHOTON_QUERY_TEMPLATE"]

# Request delay
RATE_LIMIT_SECONDS = COORDINATES_SETTINGS["RATE_LIMIT_SECONDS"]

# Output columns
COL_NAME = COORDINATES_SETTINGS["COL_NAME"]
COL_LAT = COORDINATES_SETTINGS["COL_LAT"]
COL_LON = COORDINATES_SETTINGS["COL_LON"]

def get_coordinates_nominatim(reservoir_names):
    """ Looks up coordinates with Nominatim and keeps failed names as nulls """
    geolocator_osm = Nominatim(user_agent=NOMINATIM_USER_AGENT)
    results = []

    # Try each query for each reservoir
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
    """ Looks up coordinates with Photon for each reservoir name """
    results = []

    # Photon returns longitude before latitude
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
                coords = features[0]['geometry']['coordinates']  # Read longitude and latitude
                results.append({COL_NAME: name, COL_LAT: coords[1], COL_LON: coords[0]})
            else:
                results.append({COL_NAME: name, COL_LAT: None, COL_LON: None})

        except Exception:
            results.append({COL_NAME: name, COL_LAT: None, COL_LON: None})
        time.sleep(RATE_LIMIT_SECONDS)
    return pd.DataFrame(results)

def get_coordinates_reusing_data(reservoir_list):
    """ Reuses the coordinate cache before making Photon requests """
    path = PATHS['cache'] / 'coordinates.csv'
    if path.exists():
        return pd.read_csv(path)
    result = get_coordinates_photon(reservoir_list)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False)
    return result

if __name__ == "__main__":  # Run a local check
    sample = ['burdalo', 'cabezuela']
    print(get_coordinates_photon(sample))
