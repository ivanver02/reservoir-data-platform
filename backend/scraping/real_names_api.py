import requests
import pandas as pd
import time
from backend.config.settings import REAL_NAME_SETTINGS

# Service endpoints
PHOTON_URL = REAL_NAME_SETTINGS["PHOTON_URL"]
NOMINATIM_URL = REAL_NAME_SETTINGS["NOMINATIM_URL"]

# Network settings
PHOTON_TIMEOUT = REAL_NAME_SETTINGS["PHOTON_TIMEOUT"]
NOMINATIM_TIMEOUT = REAL_NAME_SETTINGS["NOMINATIM_TIMEOUT"]
DEFAULT_USER_AGENT = REAL_NAME_SETTINGS["DEFAULT_USER_AGENT"]

# Request delay
RATE_LIMIT_SECONDS = REAL_NAME_SETTINGS["RATE_LIMIT_SECONDS"]

QUERY_VARIANT_TEMPLATES = REAL_NAME_SETTINGS["QUERY_VARIANT_TEMPLATES"]

LOWER_WORDS = REAL_NAME_SETTINGS["LOWER_WORDS"]

def get_real_names_photon(df, name_col='name', id_col='id'):
    """ Resolves reservoir names through Photon """

    if name_col not in df.columns:
        raise ValueError(f"Column '{name_col}' not found in DataFrame")
    if id_col not in df.columns:
        raise ValueError(f"Column '{id_col}' not found in DataFrame")

    real_names = []

    # Try variants for abbreviated names
    for _, row in df[[id_col, name_col]].iterrows():
        cleaned = str(row[name_col]).strip()
        best = None
        variants = [tpl.format(name=cleaned) for tpl in QUERY_VARIANT_TEMPLATES]

        # Stop at the first usable response
        for query in variants:
            try:
                response = requests.get(
                    PHOTON_URL,
                    params={"q": query, "limit": 1, "lang": "es"},
                    timeout=PHOTON_TIMEOUT,
                )

                if response.status_code != 200:
                    continue
                features = response.json().get("features", [])
                if features:
                    properties = features[0].get("properties", {})
                    best = properties.get("name") or properties.get("osm_value")

                    if best:
                        break
            except Exception:
                continue
            finally:
                time.sleep(RATE_LIMIT_SECONDS)

        real_names.append(best or _simple_title(cleaned))

    # Keep input rows and add names
    result = df.copy()
    result["real_name"] = pd.Series(real_names, index=result.index)
    return result


def _simple_title(s):
    """ Formats a fallback name with short words in lower case """
    parts = s.split()
    formatted = []
    for i, part in enumerate(parts):
        if part.lower() in LOWER_WORDS and 0 < i < len(parts) - 1:
            formatted.append(part.lower())
        else:
            formatted.append(part.capitalize())
    return " ".join(formatted)


def get_real_names_nominatim(df, name_col='name', id_col='id', user_agent=DEFAULT_USER_AGENT):
    """ Resolves reservoir names through Nominatim """

    if name_col not in df.columns:
        raise ValueError(f"Column '{name_col}' not found in DataFrame")
    if id_col not in df.columns:
        raise ValueError(f"Column '{id_col}' not found in DataFrame")

    headers = {"User-Agent": user_agent}
    real_names = []

    # Try Photon before Nominatim
    for _, row in df[[id_col, name_col]].iterrows():
        cleaned = str(row[name_col]).strip()
        best = None
        variants = [tpl.format(name=cleaned) for tpl in QUERY_VARIANT_TEMPLATES]

        for query in variants:
            try:
                response = requests.get(
                    NOMINATIM_URL,
                    params={"q": query, "format": "json", "limit": 1, "accept-language": "es"},
                    headers=headers,
                    timeout=NOMINATIM_TIMEOUT,
                )

                if response.status_code != 200:
                    continue
                results = response.json()
                if results:
                    best = results[0].get("display_name") or results[0].get("name")

                    if best:
                        break
            except Exception:
                continue
            finally:
                time.sleep(RATE_LIMIT_SECONDS)

        real_names.append(best or _simple_title(cleaned))

    # Return rows with names
    result = df.copy()
    result["real_name"] = pd.Series(real_names, index=result.index)
    return result


if __name__ == "__main__":
	df = extract_reservoirs_merged_definitive()
	sample = pd.DataFrame({
		'id': [1, 2, 4, 5],
		'name': ['fernandina', 'puebla cazalla', 'vinuela', 'rules']
	})
	print('Nominatim:')
	print(get_real_names_nominatim(df.head()))

