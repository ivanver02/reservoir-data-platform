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

CLEANING_SETTINGS = {
    'REPLACEMENT_MAPPING': {
        'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
        'ñ': 'n', 'ü': 'u', 'ç': 'c', ',': ' ', '-': ' ', '/': ' '
    },
    'ARTICLES_MAPPING': {article: ' ' for article in [
        ' el ', ' la ', ' los ', ' las ', ' un ', ' una ', ' unos ', ' unas ',
        ' del ', ' de '
    ]},
    'WEEKLY_FREQ': '7D',
    'EARTH_RADIUS_KM': 6371.0  # Mean Earth radius used for haversine distance (km)
}

COORDINATES_SETTINGS = {
    "NOMINATIM_USER_AGENT": "ReservoirProject/coordinates",
    "NOMINATIM_TIMEOUT": 10,  # seconds per request
    "NOMINATIM_SEARCH_TEMPLATES": [
        "{name} embalse España",
        "{name} embalse",
        "Embalse de {name}",
        "{name} pantano",
    ],
    "PHOTON_URL": "https://photon.komoot.io/api/",
    "PHOTON_LIMIT": 1,
    "PHOTON_TIMEOUT": 10,  # seconds (requests default if not set, but explicit)
    "PHOTON_QUERY_TEMPLATE": "{name} embalse España",
    "RATE_LIMIT_SECONDS": 1,
    "COL_NAME": "name",
    "COL_LAT": "latitude",
    "COL_LON": "longitude"
}


REAL_NAME_SETTINGS = {
    "PHOTON_URL": "https://photon.komoot.io/api/",
    "NOMINATIM_URL": "https://nominatim.openstreetmap.org/search",
    "PHOTON_TIMEOUT": 8,
    "NOMINATIM_TIMEOUT": 10,
    "DEFAULT_USER_AGENT": "ReservoirProject/real-names",
    "RATE_LIMIT_SECONDS": 1,
    "QUERY_VARIANT_TEMPLATES": [
        "Embalse de {name}",
        "Presa de {name}",
        "Pantano de {name}",
        "{name} embalse",
        "{name}",
    ],
    "LOWER_WORDS": {"de", "del", "la", "las", "los", "y", "el"}
}


SCRAPING_SETTINGS = {
    "SEARCH_URL": "https://www.embalses.net/buscar.php",
    "BASE_URL": "https://www.embalses.net",
    "HEADERS": {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
}


TRANSFER_OPTIMIZATION_THRESHOLDS = {
    'critical': 0.075,
    'worrying': 0.15,
    'could_give_if_critical': 0.25,
    'could_give_if_worrying': 0.35,
    'able_to_donate': 0.6,
    'cost_threshold': 1.5,
    'epsilon': 1e-6
}

TRANSFORM_SETTINGS = {
    'WATER_TRANSFORM': {
        'COLUMNS_MAP': {'DATE': 'date', 'CURRENT_WATER': 'storage', 'ID': 'id'},
        'DATE_FORMAT': '%d/%m/%Y',
        'YEAR_CUTOFF': 85
    },
    'RESERVOIR_TRANSFORM': {
        'COLUMNS_DICT': {'ID': 'id', 'SCOPE_NAME': 'scope', 'RESERVOIR_NAME': 'name', 'TOTAL_WATER':'capacity', 'ELECTRIC_FLAG': 'electric_flag'}
    },
    'MERGES_TRANSFORM': {
        'REPEATED_IN_DETAIL' : ['agrio', 'aguilar campoo', 'algeciras  rambla', 'arcos', 'bachimana alto', 'banos montemayor', 'burguillo', 'castrelo mino', 'chandreja', 'grado i', 'guadalteba', 'ibon ip', 'jerte', 'malpasillo jauja', 'molinos matachel', 'monteagudo vicarias', 'lago negro', 'peares', 'puentes iv', 'sant ponc', 'torre abraham', 'tous', 'vilagudin', 'zahara', 'juan benet'],
        'REPEATED_IN_RESERVOIRS' : ['agrio (aznalcollar)', 'aguilar', 'algeciras', 'arcos frontera', 'bachimana (lago)', 'banos', 'burguillo   puente nuevo', 'castrelo', 'chandrexa', 'grado', 'guadalhorce guadalteba', 'ip', 'jerte   plasencia', 'malpasillo ( jauja )', 'molinos', 'vicarias', 'negro (lago)', 'peares  os', 'puentes', 'sant pons', 'torre abrahan', 'tous   ribera', 'villagudin', 'zahara gastor', 'porma (juan benet)'],
        'MANUAL_PROVINCE_NAMES' : ['certescans', 'vellon (pedrezuela)', 'cornalbo', 'lechago'],
        'MANUAL_PROVINCES' : ['lleida', 'madrid', 'badajoz', 'teruel'],
        'UNIFIED_PROVINCES' : {
            'alacant alicante': 'alicante',
            'araba alava': 'alava',
            'castello castellon': 'castellon',
            'gipuzkoa guipuzcoa': 'guipuzcoa',
            'ourense': 'orense',
            'valència valencia': 'valencia',
            'a coruna': 'la coruna',
            'rioja': 'la rioja'
        },
        'PROVINCE_TO_COMMUNITY': {
            'alava': 'pais vasco', 'albacete': 'castilla   mancha', 'alicante': 'comunitat valenciana',
            'almeria': 'andalucia', 'asturias': 'principado asturias', 'avila': 'castilla y leon',
            'badajoz': 'extremadura', 'barcelona': 'cataluna', 'burgos': 'castilla y leon',
            'caceres': 'extremadura', 'cadiz': 'andalucia', 'cantabria': 'cantabria',
            'castellon': 'comunitat valenciana', 'ciudad real': 'castilla   mancha', 'cordoba': 'andalucia',
            'cuenca': 'castilla   mancha', 'girona': 'cataluna', 'granada': 'andalucia',
            'guadalajara': 'castilla   mancha', 'guipuzcoa': 'pais vasco', 'huelva': 'andalucia',
            'huesca': 'aragon', 'jaen': 'andalucia', 'la coruna': 'galicia', 'la rioja': 'rioja',
            'las palmas': 'canarias', 'leon': 'castilla y leon', 'lleida': 'cataluna', 'lugo': 'galicia',
            'madrid': 'comunidad madrid', 'malaga': 'andalucia', 'murcia': 'region murcia',
            'navarra': 'comunidad foral navarra', 'orense': 'galicia', 'palencia': 'castilla y leon',
            'pontevedra': 'galicia', 'salamanca': 'castilla y leon', 'santa cruz de tenerife': 'canarias',
            'segovia': 'castilla y leon', 'sevilla': 'andalucia', 'soria': 'castilla y leon',
            'tarragona': 'cataluna', 'teruel': 'aragon', 'toledo': 'castilla   mancha',
            'valencia': 'comunitat valenciana', 'valladolid': 'castilla y leon', 'vizcaya': 'pais vasco',
            'zamora': 'castilla y leon', 'ceuta': 'ceuta', 'melilla': 'melilla', 'balears': 'islas baleares',
            'zaragoza': 'aragon'
        },
        'MANUAL_COORDINATES_ROWS': [
            {'name': 'sistema lagos espot', 'latitude': 42.581771, 'longitude': 1.003018},
            {'name': 'agavanzal', 'latitude': 41.979600, 'longitude': -6.200336},
            {'name': 'sistema aguas limpias', 'latitude': 42.793868, 'longitude': -0.329262},
            {'name': 'sistema alto caldares', 'latitude': 42.427684, 'longitude': -0.227609},
            {'name': 'tremp o talarn', 'latitude': 42.204670, 'longitude': 0.949327},
        ],
        'ACCENTS': {'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u', 'ü': 'u', 'ñ': 'n'},
        'STOPWORDS': ['Embalse de la ', 'Embalse del ', 'Embalse de ', 'Embalse ', 'Embassament de la ', 'Embassament de ','Pantano del ', 'Pantano de ', 'Pantano ', 'Presa del ', 'Presa de ', 'Presa das ','Presa ', 'Municipio del ', 'Municipio de ', 'Municipio ', 'Lago del ', 'Lago de ', 'Lago ', 'Camino del ', 'Camino de ', 'Camino ', 'Club Náutico ','Encoro del ', 'Encoro de ', 'Encoro ', 'Carretera del ', 'Carretera de ', 'Bassa de ', 'Estrada dos ', 'Calle ', 'Represa del ', 'toma del ', 'embalse de ', 'Iglesia de ', 'Laguna de ', 'presa de ', 'Pantà de '],
        'MISSING_REAL_NAMES': [
            {'name': 'chandreja', 'real_name': 'Chandreja'},
            {'name': 'ullivarri', 'real_name': 'Ullíbarri-Gamboa'},
            {'name': 'montijo', 'real_name': 'Montijo'},
            {'name': 'portas', 'real_name': 'Portas'},
            {'name': 'llerena', 'real_name': 'Llerena'},
            {'name': 'villar rey', 'real_name': 'Villar del Rey'},
            {'name': 'alcollarin', 'real_name': 'Alcollarín'},
            {'name': 'certescans', 'real_name': 'Lago Certascan'},
            {'name': 'zujar', 'real_name': 'Zújar'},
            {'name': 'pias (san agustin)', 'real_name': 'Pías (San Agustín)'},
            {'name': 'urkulu', 'real_name': 'Urkulu'},
            {'name': 'olivargas', 'real_name': 'Olivargas'},
            {'name': 'puentes viejas', 'real_name': 'Puentes Viejas'},
            {'name': 'parras (las)', 'real_name': 'Las Parras'},
            {'name': 'villagonzalo', 'real_name': 'Villagonzalo'},
            {'name': 'eiras', 'real_name': 'Eiras'},
            {'name': 'ribarroja', 'real_name': 'Ribarroja'}
        ]
    },
    'DETAILED_RESERVOIRS_TRANSFORM': {
        'COLUMNS_DICT': {'x': 'longitude', 'y': 'latitude'}
    }
}
