from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import pandas as pd
import time
import requests

from backend.config.settings import PATHS

def get_coordinates_nominatim(reservoir_names):
    # Initialize geocoders
    geolocator_osm = Nominatim(user_agent="ReservoirProject")
    
    results = []
    
    for name in reservoir_names:
        try:
            # Try with different search terms
            search_terms = [
                f"{name} embalse España",
                f"{name} embalse",
                f"Embalse de {name}",
                f"{name} pantano",
            ]
            
            location = None
            for term in search_terms:
                try:
                    location = geolocator_osm.geocode(term, timeout=10)
                    if location:
                        break
                except GeocoderTimedOut:
                    continue
            
            if location:
                results.append({
                    'name': name,
                    'latitude': location.latitude,
                    'longitude': location.longitude
                })
            else:
                results.append({
                    'name': name,
                    'latitude': None,
                    'longitude': None
                })
                
        except Exception as e:
            results.append({'name': name, 'latitude': None, 'longitude': None})
        
        time.sleep(1)  # Rate limiting
    
    return pd.DataFrame(results)

def get_coordinates_photon(reservoir_list):
    results = []
    for name in reservoir_list:
        try:
            url = "https://photon.komoot.io/api/"
            params = {
                'q': f"{name} embalse España",
                'limit': 1,
            }
            response = requests.get(url, params=params)
            data = response.json()

            if data['features']:
                coords = data['features'][0]['geometry']['coordinates']
                results.append({'name': name, 'latitude': coords[1], 'longitude': coords[0]})
            else:
                results.append({'name': name, 'latitude': None, 'longitude': None})

        except Exception as e:
            results.append({'name': name, 'latitude': None, 'longitude': None})

        time.sleep(1)  # Rate limiting

    return pd.DataFrame(results)

def get_coordinates_reusing_data(reservoir_list):
    saved_df_path = PATHS['definitive_notebooks'] / 'reservoirs_merged.parquet'
    saved_df = pd.read_parquet(saved_df_path)

    if saved_df['latitude'].isna().any():
        return get_coordinates_photon(reservoir_list)
    else:
        path = PATHS['raw_data_notebooks'] / 'coordinates.csv'
        coordinates = pd.read_csv(path)
        return coordinates

if __name__ == "__main__":
    # Test the geocoding function
    reservoir_name = ['burdalo', 'cabezuela']
    df = get_coordinates_photon(reservoir_name)
    print(df)