import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Setting up the path to include the parent directory
sys.path.append(str(Path.cwd().parent.parent))
from backend.config.settings import PATHS

def extract_water_raw():
    file_path = PATHS['raw_data_production'] / 'water.csv'
    water_data = pd.read_csv(file_path)
    return water_data

def extract_water_cleaned():
    file_path = PATHS['cleaned_data_production'] / 'water_cleaned.csv'
    water_data = pd.read_csv(file_path)
    return water_data

def extract_reservoirs_raw():
    file_path = PATHS['raw_data_production'] / 'reservoirs.csv'
    reservoirs_data = pd.read_csv(file_path)
    return reservoirs_data

def extract_reservoirs_cleaned():
    file_path = PATHS['cleaned_data_production'] / 'reservoirs_cleaned.csv'
    reservoirs_data = pd.read_csv(file_path)
    return reservoirs_data

def extract_detailed_reservoirs_raw():
    file_path = PATHS['raw_data_production'] / 'UTF8list-3.tsv'
    detailed_reservoirs_data = pd.read_csv(file_path, sep='\t')
    return detailed_reservoirs_data

if __name__ == "__main__":
    water_data = extract_water_raw()
    print(water_data.head())