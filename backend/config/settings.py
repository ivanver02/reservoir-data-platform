# Main settings file for the Reservoir project
from pathlib import Path

PROJECT = {
    'name': 'Reservoir',
    'author': 'ivanver02',
    'description': 'Time series forecasting and optimization for Andalusian reservoirs'
}

BASE_PATH = Path(__file__).parent.parent.parent.resolve()

PATHS = {
    'base': BASE_PATH,
    'raw_data_notebooks': BASE_PATH / 'data' / 'for_notebooks' / 'raw',
    'raw_data_production': BASE_PATH / 'data' / 'for_production' / 'raw',
    'processed_data_notebooks': BASE_PATH / 'data' / 'for_notebooks' / 'processed',
    'processed_data_production': BASE_PATH / 'data' / 'for_production' / 'processed',
    'cleaned_data_notebooks': BASE_PATH / 'data' / 'for_notebooks' / 'processed' / 'post_cleaning',
    'cleaned_data_production': BASE_PATH / 'data' / 'for_production' / 'processed' / 'post_cleaning',
    'pre_EDA': BASE_PATH / 'data' / 'for_notebooks' / 'processed' / 'pre_EDA',
    'engineered_data_notebooks': BASE_PATH / 'data' / 'for_notebooks' / 'processed' / 'post_feature_engineering',
    'engineered_data_production': BASE_PATH / 'data' / 'for_production' / 'processed' / 'post_feature_engineering',
    'definitive_notebooks': BASE_PATH / 'data' / 'for_notebooks' / 'definitive',
    'definitive_production': BASE_PATH / 'data' / 'for_production' / 'definitive',
    'backend': BASE_PATH / 'backend',
    'config': BASE_PATH / 'backend' / 'config'
}

RAW_FILES = ['water.csv', 'reservoirs.csv', 'UTF8list-3.tsv']


