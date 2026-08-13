from backend.config.settings import PATHS


def _write_parquet(df, path):
    """Write a curated table after creating its parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)

def load_water_cleaned(df):
    """ Saves water observations in the data area """
    cleaned_water_path = PATHS['cleaned'] / 'water_cleaned.csv'
    cleaned_water_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cleaned_water_path, index=False)

def load_reservoirs_cleaned(df):
    """ Saves reservoir metadata in the data area """
    cleaned_reservoirs_path = PATHS['cleaned'] / 'reservoirs_cleaned.csv'
    cleaned_reservoirs_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cleaned_reservoirs_path, index=False)

def load_detailed_reservoirs_cleaned(df):
    """ Saves reservoir details in the data area """
    cleaned_detailed_reservoirs_path = PATHS['cleaned'] / 'detailed_reservoirs_cleaned.csv'
    cleaned_detailed_reservoirs_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cleaned_detailed_reservoirs_path, index=False)

def load_water_definitive(df):
    """ Saves water observations """
    definitive_water_path = PATHS['curated'] / 'water_definitive.csv'
    definitive_water_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(definitive_water_path, index=False)

def load_reservoirs_definitive(df):
    """ Saves main metadata """
    definitive_reservoirs_path = PATHS['curated'] / 'reservoirs_definitive.csv'
    definitive_reservoirs_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(definitive_reservoirs_path, index=False)

def load_detailed_reservoirs_definitive(df):
    """ Saves reservoir details """
    definitive_detailed_reservoirs_path = PATHS['curated'] / 'detailed_reservoirs_definitive.csv'
    definitive_detailed_reservoirs_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(definitive_detailed_reservoirs_path, index=False)

def load_reservoirs_merged_definitive(df):
    """ Saves the combined metadata table """
    merged_reservoirs_path = PATHS['curated'] / 'reservoirs_merged_definitive.csv'
    merged_reservoirs_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(merged_reservoirs_path, index=False)


def load_features_curated(df):
    """Persist the first feature table in the curated area."""
    _write_parquet(df, PATHS['curated'] / 'water_features.parquet')
