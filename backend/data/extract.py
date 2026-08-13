import pandas as pd

from backend.config.settings import PATHS


def _read_csv(path, **kwargs):
    """ Reads a CSV with the options passed by the extraction stage """
    return pd.read_csv(path, **kwargs)


def _read_parquet(path):
    """Read a curated table through one shared helper."""
    return pd.read_parquet(path)

def extract_water_raw():
    """ Loads water observations """
    file_path = PATHS['raw'] / 'water.csv'
    water_data = pd.read_csv(file_path)
    return water_data

def extract_water_cleaned():
    """ Loads water observations after cleaning """
    water_data = _read_csv(PATHS['cleaned'] / 'water_cleaned.csv')
    return water_data

def extract_reservoirs_raw():
    """ Loads reservoir metadata """
    file_path = PATHS['raw'] / 'reservoirs.csv'
    reservoirs_data = pd.read_csv(file_path)
    return reservoirs_data

def extract_reservoirs_cleaned():
    """ Loads reservoir metadata after cleaning """
    reservoirs_data = _read_csv(PATHS['cleaned'] / 'reservoirs_cleaned.csv')
    return reservoirs_data


def extract_features_curated():
    """Load the first persisted feature table when it exists."""
    return _read_parquet(PATHS['curated'] / 'water_features.parquet')

def extract_detailed_reservoirs_raw():
    """ Loads the reservoir detail file """
    file_path = PATHS['raw'] / 'UTF8list-3.tsv'
    detailed_reservoirs_data = pd.read_csv(file_path, sep='\t')
    return detailed_reservoirs_data

def extract_detailed_reservoirs_cleaned():
    """ Loads reservoir details after cleaning """
    detailed_reservoirs_data = _read_csv(PATHS['cleaned'] / 'detailed_reservoirs_cleaned.csv')
    return detailed_reservoirs_data

def extract_water_definitive():
    """ Loads the water observations prepared for modelling """
    water_data = _read_csv(PATHS['curated'] / 'water_definitive.csv')
    return water_data

def extract_reservoirs_definitive():
    """ Loads reservoir metadata for modelling """
    reservoirs_data = _read_csv(PATHS['curated'] / 'reservoirs_definitive.csv')
    return reservoirs_data

def extract_detailed_reservoirs_definitive():
    """ Loads reservoir details for modelling """
    detailed_reservoirs_data = _read_csv(PATHS['curated'] / 'detailed_reservoirs_definitive.csv')
    return detailed_reservoirs_data

def extract_reservoirs_merged_definitive():
    """ Loads the metadata table """
    reservoirs_merged_data = _read_csv(PATHS['curated'] / 'reservoirs_merged_definitive.csv')
    return reservoirs_merged_data

if __name__ == "__main__":
    water_data = extract_water_raw()
    print(water_data.head())
