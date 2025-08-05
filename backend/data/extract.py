import pandas as pd

def extract_data(raw_data_path, files):
    """
    Extracts raw data from raw data files and returns them as pandas DataFrames.
    """
    dataframes = {}

    for file in files:
        if file.endswith('.csv'):
            df = pd.read_csv(raw_data_path / file)
        elif file.endswith('.tsv'):
            df = pd.read_csv(raw_data_path / file, sep='\t')
        dataframes[file] = df

    return dataframes