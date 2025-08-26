from pathlib import Path
import sys

# Setting up the path to include the parent directory
sys.path.append(str(Path.cwd().parent.parent))
from backend.config.settings import PATHS

def load_water_cleaned(df):
    cleaned_water_path = PATHS['cleaned_data_production'] / 'water_cleaned.csv'
    cleaned_water_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cleaned_water_path, index=False)

def load_reservoirs_cleaned(df):
    cleaned_reservoirs_path = PATHS['cleaned_data_production'] / 'reservoirs_cleaned.csv'
    cleaned_reservoirs_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cleaned_reservoirs_path, index=False)

def load_detailed_reservoirs_cleaned(df):
    cleaned_detailed_reservoirs_path = PATHS['cleaned_data_production'] / 'detailed_reservoirs_cleaned.csv'
    cleaned_detailed_reservoirs_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cleaned_detailed_reservoirs_path, index=False)
