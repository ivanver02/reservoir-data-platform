# Main settings file for the Reservoir project
from pathlib import Path

from backend.data.transform import transform_water, transform_reservoirs, transform_UTF8list

PROJECT = {
    'name': 'Reservoir',
    'author': 'ivanver02',
    'description': 'Time series forecasting and optimization for Andalusian reservoirs'
}

BASE_PATH = Path(__file__).parent.parent.parent.resolve()

PATHS = {
    'base': BASE_PATH,
    'raw_data': BASE_PATH / 'data' / 'raw',
    'processed_data': BASE_PATH / 'data' / 'processed',
    'cleaned_data': BASE_PATH / 'data' / 'processed' / 'post_cleaning',
    'notebooks': BASE_PATH / 'notebooks',
    'backend': BASE_PATH / 'backend',
    'config': BASE_PATH / 'backend' / 'config'
}

RAW_FILES = ['water.csv', 'reservoirs.csv', 'UTF8list-3.tsv']


