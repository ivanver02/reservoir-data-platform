import time
import re
import threading
from itertools import combinations as word_combinations

import pandas as pd
import unicodedata
import requests
from requests.adapters import HTTPAdapter, Retry
from bs4 import BeautifulSoup

from backend.config.settings import PATHS, SCRAPING_SETTINGS

# Serialize province lookups so requests stay one at a time
_reservoir_lock = threading.Lock()

SEARCH_URL = SCRAPING_SETTINGS["SEARCH_URL"]
BASE_URL = SCRAPING_SETTINGS["BASE_URL"]
HEADERS = SCRAPING_SETTINGS["HEADERS"]


def get_session():
    """ Creates an HTTP session with retries and headers """
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=0.5)
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update(HEADERS)
    return session


def find_reservoir_link(soup, name):
    """ Finds a link whose displayed name matches the query """
    name_lower = name.lower().strip()
    for row in soup.find_all("tr", class_=re.compile("ResultadoCampo")):
        link = row.find("a", href=True)
        if not link:
            continue
        
        text = link.get_text().lower().strip()
        href = link["href"].strip()
        
        if name_lower in text or text in name_lower:
            if href.startswith("/"):
                href = BASE_URL + href
            return href
    return None


def generate_n_variants(name):
    """ Generates spelling variants that replace n with ñ """
    positions = [i for i, c in enumerate(name) if c.lower() == 'n']
    if not positions:
        return []
    
    variants = []
    for i in range(1, 2**len(positions)):
        chars = list(name)
        for j, pos in enumerate(positions):
            if i & (1 << j):
                chars[pos] = 'Ñ' if chars[pos].isupper() else 'ñ'
        variants.append(''.join(chars))
    
    return list(dict.fromkeys(variants))  # Remove repeated variants


def fix_encoding(text):
    """ Repairs a text decoding error """
    if 'Ã' in text:
        try:
            return text.encode('latin1').decode('utf-8')
        except Exception:
            # Keep the original text when the repair does not work
            pass
    return text


def clean_text(text):
    """ Normalizes scraped text for storage and comparisons """
    text = fix_encoding(text)
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(c for c in text if not unicodedata.combining(c))
    return re.sub(r'\s+', ' ', text.strip().lower())


def get_reservoir_url(name):
    """ Searches the reservoir site and returns a matching URL """
    session = get_session()
    
    def search(query):
        """ Sends one request and extracts a matching link """
        resp = session.post(SEARCH_URL, data={"TxtBusqueda": query, "BtSus": "Buscar"})
        resp.encoding = 'utf-8'
        soup = BeautifulSoup(resp.text, "html.parser")
        return find_reservoir_link(soup, query)
    
    # Try the base name
    url = search(name)
    if url:
        time.sleep(0.5)
        return url
    
    # Try spelling variants
    for variant in generate_n_variants(name):
        time.sleep(1.0)
        url = search(variant)
        if url:
            time.sleep(0.5)
            return url
    
    raise ValueError(f"Reservoir '{name}' not found")


def get_reservoir_html(name):
    """ Downloads the detail page for a reservoir name """
    url = get_reservoir_url(name)
    session = get_session()
    resp = session.get(url)
    resp.encoding = 'utf-8'
    return resp.text


def generate_word_combinations(name):
    """ Builds query combinations for long reservoir names """
    words = name.split()
    if len(words) <= 1:
        return []

    combinations = []

    # Try longer single words first and then word groups
    combinations.extend(sorted(words, key=len, reverse=True))
    for combo_len in range(2, len(words) + 1):
        for combo in word_combinations(range(len(words)), combo_len):
            combinations.append(" ".join(words[i] for i in combo))

    return combinations


def get_reservoir_province(name):
    """ Looks up and normalizes a reservoir province """
    def try_get_province(query_name):
        """ Reads a province from one query """
        try:
            html = get_reservoir_html(query_name)
            soup = BeautifulSoup(html, "html.parser")
            
            for row in soup.find_all("div", class_="FilaSeccion"):
                label = row.find("div", class_="CampoInf")
                value = row.find("div", class_="ResultadoInf")
                
                if not label or not value:
                    continue
                    
                if label.get_text().strip().lower().rstrip(':') == "provincia":
                    text = value.get_text().strip()
                    text = re.sub(r"^\W+", "", text)  # Remove leading symbols
                    if text:
                        return clean_text(text)
            return None
        except Exception:
            # A failed query just moves the search to the next combination
            return None
       
    # Try the base name
    result = try_get_province(name)
    if result:
        return result
    
    # Try word combinations
    for combination in generate_word_combinations(name):
        result = try_get_province(combination)
        if result:
            return result
    
    return None

def get_reservoir_province_rate_limited(name):
    """ Serializes province lookups and waits between requests """
    with _reservoir_lock:
        result = get_reservoir_province(name)
        time.sleep(1)  # Wait before releasing the lock
        return result

def get_reservoir_province_reusing_data(reservoir_name):
    """ Reads a cached province or fetches and stores it """

    # Read the cache before making a request
    saved_df_path = PATHS['cache'] / 'provinces.csv'
    if saved_df_path.exists():
        saved_df = pd.read_csv(saved_df_path)
        match = saved_df.loc[saved_df['name'] == reservoir_name, 'province']
        if match.notna().any():
            return match.iloc[0]
    province = get_reservoir_province_rate_limited(reservoir_name)

    # Replace this name in the cache
    saved_df_path.parent.mkdir(parents=True, exist_ok=True)
    cached = pd.read_csv(saved_df_path) if saved_df_path.exists() else pd.DataFrame(columns=['name', 'province'])
    cached = cached[cached['name'] != reservoir_name]
    cached = pd.concat([cached, pd.DataFrame([{'name': reservoir_name, 'province': province}])], ignore_index=True)
    cached.to_csv(saved_df_path, index=False)
    return province


if __name__ == "__main__":
    name = "pedro marin"
    province = get_reservoir_province_reusing_data(name)
    print(f"Province for '{name}': {province}")
