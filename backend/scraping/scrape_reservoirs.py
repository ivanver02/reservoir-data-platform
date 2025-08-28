import sys
import time
import re
import pandas as pd
import unicodedata
import requests
from requests.adapters import HTTPAdapter, Retry
from bs4 import BeautifulSoup

from backend.config.settings import PATHS
import threading



_reservoir_lock = threading.Lock()

SEARCH_URL = "https://www.embalses.net/buscar.php"
BASE_URL = "https://www.embalses.net"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}


def get_session():
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=0.5)
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update(HEADERS)
    return session


def find_reservoir_link(soup, name):
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
    
    return list(dict.fromkeys(variants))  # remove duplicates


def fix_encoding(text):
    if 'Ã' in text:
        try:
            return text.encode('latin1').decode('utf-8')
        except:
            pass
    return text


def clean_text(text):
    text = fix_encoding(text)
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(c for c in text if not unicodedata.combining(c))
    return re.sub(r'\s+', ' ', text.strip().lower())


def get_reservoir_url(name):
    session = get_session()
    
    def search(query):
        resp = session.post(SEARCH_URL, data={"TxtBusqueda": query, "BtSus": "Buscar"})
        resp.encoding = 'utf-8'
        soup = BeautifulSoup(resp.text, "html.parser")
        return find_reservoir_link(soup, query)
    
    # Try original name first
    url = search(name)
    if url:
        time.sleep(0.5)
        return url
    
    # Try variants with ñ
    for variant in generate_n_variants(name):
        time.sleep(1.0)
        url = search(variant)
        if url:
            time.sleep(0.5)
            return url
    
    raise ValueError(f"Reservoir '{name}' not found")


def get_reservoir_html(name):
    url = get_reservoir_url(name)
    session = get_session()
    resp = session.get(url)
    resp.encoding = 'utf-8'
    return resp.text


def generate_word_combinations(name):
    words = name.split()
    if len(words) <= 1:
        return []
    
    combinations = []

    # We try with the longest words first
    sorted_words = sorted(words, key=len, reverse=True)
    combinations.extend(sorted_words)
    
    for i in range(len(words)):
        for j in range(i+1, len(words)):
            combinations.append(f"{words[i]} {words[j]}")
    
    # Three word combinations and beyond
    for combo_len in range(3, len(words) + 1):
        from itertools import combinations as iter_combinations
        for combo in iter_combinations(range(len(words)), combo_len):
            combo_words = [words[i] for i in combo]
            combinations.append(" ".join(combo_words))
    
    return combinations


def get_reservoir_province(name):
    def try_get_province(query_name):
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
                    text = re.sub(r"^\W+", "", text)  # remove leading symbols
                    if text:
                        return clean_text(text)
            return None
        except:
            return None
    
    # Try original name first
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
    with _reservoir_lock:
        result = get_reservoir_province(name)
        time.sleep(1)  # Wait 1 second before releasing the lock
        return result

def get_reservoir_province_reusing_data(reservoir_name):

    saved_df_path = PATHS['pre_EDA'] / 'merges_for_EDA.csv'
    saved_df = pd.read_csv(saved_df_path)

    if saved_df.loc[saved_df['name'] == reservoir_name, 'province'].notna().any():
        return saved_df.loc[saved_df['name'] == reservoir_name, 'province'].values[0]
    return get_reservoir_province_rate_limited(reservoir_name)


if __name__ == "__main__":
    name = "pedro marin"
    province = get_reservoir_province_reusing_data(name)
    print(f"Province for '{name}': {province}")
