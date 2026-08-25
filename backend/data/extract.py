"""Read the source and pipeline tables from their data areas"""

import pandas as pd

from backend.config.settings import PATHS


def _read_csv(path, **kwargs):
    """ Reads a CSV with the options passed by the extraction stage """
    return pd.read_csv(path, **kwargs)


# Raw tables, kept outside the repository
def extract_water_raw():
    """ Loads water observations """
    return _read_csv(PATHS['raw'] / 'water.csv')

def extract_reservoirs_raw():
    """ Loads reservoir metadata """
    return _read_csv(PATHS['raw'] / 'reservoirs.csv')

def extract_detailed_reservoirs_raw():
    """ Loads the reservoir detail file """
    return _read_csv(PATHS['raw'] / 'UTF8list-3.tsv', sep='\t')


# Cleaned tables, written by the transform steps
def extract_water_cleaned():
    """ Loads water observations after cleaning """
    return _read_csv(PATHS['cleaned'] / 'water_cleaned.csv')

def extract_reservoirs_cleaned():
    """ Loads reservoir metadata after cleaning """
    return _read_csv(PATHS['cleaned'] / 'reservoirs_cleaned.csv')

def extract_detailed_reservoirs_cleaned():
    """ Loads reservoir details after cleaning """
    return _read_csv(PATHS['cleaned'] / 'detailed_reservoirs_cleaned.csv')


# Curated tables, ready for modelling
def extract_water_definitive():
    """ Loads the water observations prepared for modelling """
    return _read_csv(PATHS['curated'] / 'water_definitive.csv')

def extract_reservoirs_definitive():
    """ Loads reservoir metadata for modelling """
    return _read_csv(PATHS['curated'] / 'reservoirs_definitive.csv')

def extract_detailed_reservoirs_definitive():
    """ Loads reservoir details for modelling """
    return _read_csv(PATHS['curated'] / 'detailed_reservoirs_definitive.csv')

def extract_reservoirs_merged_definitive():
    """ Loads the metadata table """
    return _read_csv(PATHS['curated'] / 'reservoirs_merged_definitive.csv')


if __name__ == "__main__":
    water_data = extract_water_raw()
    print(water_data.head())
