"""Save the pipeline tables in their data areas"""

import pandas as pd

from backend.config.settings import PATHS


def save_table(df: pd.DataFrame, path) -> None:
    """ Writes one table as CSV and creates its folder when needed """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_parquet_atomic(df: pd.DataFrame, path) -> None:
    """ Writes a Parquet file through a temporary path """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    df.to_parquet(temporary, index=False)
    temporary.replace(path)


# Cleaned tables, written by the transform steps
def load_water_cleaned(df):
    """ Saves water observations in the data area """
    save_table(df, PATHS['cleaned'] / 'water_cleaned.csv')

def load_reservoirs_cleaned(df):
    """ Saves reservoir metadata in the data area """
    save_table(df, PATHS['cleaned'] / 'reservoirs_cleaned.csv')

def load_detailed_reservoirs_cleaned(df):
    """ Saves reservoir details in the data area """
    save_table(df, PATHS['cleaned'] / 'detailed_reservoirs_cleaned.csv')


# Curated tables, written once the merge is done
def load_water_definitive(df):
    """ Saves water observations """
    save_table(df, PATHS['curated'] / 'water_definitive.csv')

def load_reservoirs_definitive(df):
    """ Saves main metadata """
    save_table(df, PATHS['curated'] / 'reservoirs_definitive.csv')

def load_detailed_reservoirs_definitive(df):
    """ Saves reservoir details """
    save_table(df, PATHS['curated'] / 'detailed_reservoirs_definitive.csv')

def load_reservoirs_merged_definitive(df):
    """ Saves the combined metadata table """
    save_table(df, PATHS['curated'] / 'reservoirs_merged_definitive.csv')
