# Importing modules
from backend.data.extract import extract_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from backend.config.settings import PATHS, RAW_FILES


def main():
    """
    Main function to run the ETL pipeline.
    """
    # Extract data
    dataframes = extract_data(PATHS['raw_data'], RAW_FILES)
    water_df = dataframes['water.csv']
    reservoirs_df = dataframes['reservoirs.csv']
    detailed_reservoirs_df = dataframes['UTF8list-3.tsv']



if __name__ == "__main__":
    main()