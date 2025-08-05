def transform_water(df):
    # To be implemented
    return df

def transform_reservoirs(df):
    # To be implemented
    return df

def transform_UTF8list(df):
    # To be implemented
    return df

def transform_default(df):
    # Default transformation if no specific function is defined
    return df

# Defined here as it uses functions from this module
TRANSFORMATIONS = {
    'water.csv': transform_water,
    'reservoirs.csv': transform_reservoirs,
    'UTF8list-3.tsv': transform_UTF8list,
}

def transform_data(dataframes: dict) -> dict:
    transformed = {}
    for filename, df in dataframes.items():
        func = TRANSFORMATIONS.get(filename, transform_default)
        transformed[filename] = func(df)
    return transformed

