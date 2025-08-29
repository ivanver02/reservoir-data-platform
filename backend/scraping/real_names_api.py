import requests
import pandas as pd
import time
from pathlib import Path
import sys

# Setting up the path to include the parent directory
sys.path.append(str(Path.cwd().parent.parent))
from backend.data.extract import extract_reservoirs_merged_definitive

def get_real_names_photon(df, name_col='name', id_col='id'):
	"""
	Given a DataFrame with a cleaned reservoir name column, try to fetch a more
	official / accented name using the free Photon (Komoot) geocoding API.

	Approach (very simple):
	- For each name, query Photon with a few Spanish reservoir variants.
	- Take the first result's 'name' property when available.
	- Fallback: basic title-casing of the original cleaned name.

	Returns a new DataFrame with a 'real_name' column.
	"""
	if name_col not in df.columns:
		raise ValueError(f"Column '{name_col}' not found in DataFrame")
	if id_col not in df.columns:
		raise ValueError(f"Column '{id_col}' not found in DataFrame")

	names = []
	real_names = []

	for _, row in df[[id_col, name_col]].iterrows():
		rid = row[id_col]
		cleaned = str(row[name_col]).strip()
		best = None
		# Query variants (ordered by likelihood)
		variants = [
			f"Embalse de {cleaned}",
			f"Presa de {cleaned}",
			f"Pantano de {cleaned}",
			f"{cleaned} embalse",
			cleaned,
		]

		for q in variants:
			try:
				resp = requests.get(
					"https://photon.komoot.io/api/",
					params={"q": q, "limit": 1, "lang": "es"},
					timeout=8,
				)
				if resp.status_code != 200:
					continue
				data = resp.json()
				feats = data.get('features', [])
				if feats:
					props = feats[0].get('properties', {})
					candidate = props.get('name') or props.get('osm_value')
					if candidate:
						best = candidate
						break
			except Exception:
				continue
			finally:
				time.sleep(1)  # polite rate limiting

		if not best:
			best = _simple_title(cleaned)

		names.append(rid)
		real_names.append(best)

	out = df.copy()
	out['real_name'] = pd.Series(real_names, index=out.index)
	return out


def _simple_title(s):
	# Keep common Spanish articles/prepositions lowercase inside phrase
	lower_words = {"de", "del", "la", "las", "los", "y", "el"}
	parts = s.split()
	formatted = []
	for i, p in enumerate(parts):
		if p.lower() in lower_words and 0 < i < len(parts)-1:
			formatted.append(p.lower())
		else:
			formatted.append(p.capitalize())
	return " ".join(formatted)


def get_real_names_nominatim(df, name_col='name', id_col='id', user_agent='ReservoirProject/real-names'):
	"""Simplified Nominatim variant to obtain a nicer reservoir name.

	For each cleaned name it tries a few query templates; takes the first result's
	'display_name' or 'name'. Falls back to title-cased cleaned version.

	Respect: 1 request / second.
	"""
	if name_col not in df.columns:
		raise ValueError(f"Column '{name_col}' not found in DataFrame")
	if id_col not in df.columns:
		raise ValueError(f"Column '{id_col}' not found in DataFrame")

	headers = {"User-Agent": user_agent}
	names = []
	real_names = []

	for _, row in df[[id_col, name_col]].iterrows():
		rid = row[id_col]
		cleaned = str(row[name_col]).strip()
		best = None
		variants = [
			f"Embalse de {cleaned}",
			f"Presa de {cleaned}",
			f"Pantano de {cleaned}",
			f"{cleaned} embalse",
			cleaned,
		]
		for q in variants:
			try:
				resp = requests.get(
					"https://nominatim.openstreetmap.org/search",
					params={"q": q, "format": "json", "limit": 1, "accept-language": "es"},
					headers=headers,
					timeout=10,
				)
				if resp.status_code != 200:
					continue
				js = resp.json()
				if js:
					cand = js[0].get('display_name') or js[0].get('name')
					if cand:
						best = cand
						break
			except Exception:
				continue
			finally:
				time.sleep(1)

		if not best:
			best = _simple_title(cleaned)
		names.append(rid)
		real_names.append(best)

	out = df.copy()
	out['real_name'] = pd.Series(real_names, index=out.index)
	return out


if __name__ == "__main__":
	df = extract_reservoirs_merged_definitive()
	sample = pd.DataFrame({
		'id': [1, 2, 4, 5],
		'name': ['fernandina', 'puebla cazalla', 'vinuela', 'rules']
	})
	# print('Photon:')
	# print(get_real_names_photon(sample))
	print('Nominatim:')
	print(get_real_names_nominatim(df.head()))

