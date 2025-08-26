REPLACEMENT_MAPPING = {'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u', 'ñ': 'n', 'ü': 'u', 'ç': 'c', ',': ' ', '-': ' ', '/': ' '}
ARTICLES = [' el ', ' la ', ' los ', ' las ', ' un ', ' una ', ' unos ', ' unas ', ' del ', ' de ']
ARTICLES_MAPPING = {article: ' ' for article in ARTICLES}


def clean_string_series(series):
    """
    Cleans a pandas Series by lowercasing, replacing accents, removing articles, and stripping whitespace.
    """
    # Lowercasing
    series = series.str.lower()
    
    # Replacing accents and special characters
    series = series.replace(REPLACEMENT_MAPPING, regex=True)
    
    # Adding spaces around the strings and articles to avoid removing parts of words when removing articles
    series = ' ' + series + ' '
    
    # Removing articles
    series = series.replace(ARTICLES_MAPPING, regex=True)

    # Stripping whitespace
    series = series.str.strip()
    
    return series

