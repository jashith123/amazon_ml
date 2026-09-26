import argparse
import pandas as pd
import numpy as np
import re
import unicodedata
from anyascii import anyascii

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common.schema import NORMALIZED_COLS

LEGAL_SUFFIXES = [
    'pvt ltd', 'private limited', 'pvt limited', 'private ltd',
    'ltd', 'limited', 'corp', 'corporation', 'inc', 'incorporated',
    'llc', 'gmbh', 'sarl', 'sas', 'sci', 'eurl', 'co', 'company', 'plc'
]

LEGAL_SUFFIXES_SORTED = sorted(LEGAL_SUFFIXES, key=len, reverse=True)
LEGAL_SUFFIX_REGEX = re.compile(r'\b(' + '|'.join(LEGAL_SUFFIXES_SORTED) + r')\b$')

ABBREVIATIONS = {
    r'\brd\b': 'road',
    r'\bst\b': 'street',
    r'\bblvd\b': 'boulevard',
    r'\bave\b': 'avenue',
    r'\bdr\b': 'drive',
    r'\bln\b': 'lane',
    r'\bpl\b': 'place',
    r'\bct\b': 'court',
    r'\bpt\b': 'point'
}

FRENCH_LIGATURES = {
    'œ': 'oe', 'æ': 'ae', 'ß': 'ss', 'ô': 'o', 'ñ': 'n', 
    'é': 'e', 'è': 'e', 'ê': 'e', 'ç': 'c', 'à': 'a', 'â': 'a'
}

def remove_punctuation(text):
    if pd.isna(text): return ''
    return re.sub(r'[^\w\s]', ' ', text)

def collapse_whitespace(text):
    if pd.isna(text): return ''
    return re.sub(r'\s+', ' ', text).strip()

def extract_legal_suffix(name):
    if not name:
        return name, ''
    match = LEGAL_SUFFIX_REGEX.search(name)
    if match:
        suffix = match.group(1)
        core = name[:match.start()].strip()
        return core, suffix
    return name, ''

def get_tokens(text):
    if not text: return ''
    tokens = set(text.split())
    return '|'.join(sorted(list(tokens)))

def extract_numbers(text):
    if not text: return ''
    tokens = text.split()
    nums = [t for t in tokens if re.search(r'\d', t)]
    return '|'.join(nums)

def extract_pin(text):
    if not text: return ''
    matches = re.findall(r'\b\d{5,6}\b', text)
    if matches:
        return matches[-1]
    return ''

def is_non_latin(text):
    if not text: return False
    try:
        text.encode('ascii')
        return False
    except UnicodeEncodeError:
        return True

def normalize_source(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out['entity_id'] = df['entity_id'].astype(str)
    out['country'] = df['country'].astype(str)
    out['name_raw'] = df['business_name'].fillna('').astype(str)
    out['address_raw'] = df['business_address'].fillna('').astype(str)
    
    out['is_non_latin'] = out['name_raw'].apply(is_non_latin)
    
    def norm_text(t):
        if not t: return ''
        # Transliterate Indic text and drop accents BEFORE stripping punctuation
        if not t.isascii():
            t = anyascii(t)
        t = t.lower()
        t = remove_punctuation(t)
        t = collapse_whitespace(t)
        return t
        
    out['name_norm'] = out['name_raw'].apply(norm_text)
    
    cores = []
    suffixes = []
    for n in out['name_norm']:
        c, s = extract_legal_suffix(n)
        cores.append(c)
        suffixes.append(s)
        
    out['name_core'] = cores
    out['legal_suffix'] = suffixes
    out['name_tokens'] = out['name_core'].apply(get_tokens)
    
    # name_transliterated logic now redundant as name_norm does anyascii, 
    # but to follow contract we can just copy name_norm or apply French fold on raw
    def transliterate_and_fold(raw, norm, non_latin):
        if not raw: return ''
        # If it was non_latin, norm already transliterated it via anyascii
        # But let's apply French ligatures fold explicitly on the transliterated result or raw
        t = anyascii(raw).lower() if non_latin else raw.lower()
        for k, v in FRENCH_LIGATURES.items():
            t = t.replace(k, v)
        t = remove_punctuation(t)
        t = collapse_whitespace(t)
        return t
        
    out['name_transliterated'] = out.apply(lambda row: transliterate_and_fold(row['name_raw'], row['name_norm'], row['is_non_latin']), axis=1)
    
    def norm_addr(t):
        if not t: return ''
        if not t.isascii():
            t = anyascii(t)
        t = t.lower()
        t = remove_punctuation(t)
        t = collapse_whitespace(t)
        for patt, repl in ABBREVIATIONS.items():
            t = re.sub(patt, repl, t)
        return t
        
    out['address_norm'] = out['address_raw'].apply(norm_addr)
    out['address_numbers'] = out['address_norm'].apply(extract_numbers)
    out['address_pin'] = out['address_norm'].apply(extract_pin)
    
    out = out[NORMALIZED_COLS]
    return out

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input TSV file')
    parser.add_argument('--output', required=True, help='Output Parquet file')
    args = parser.parse_args()
    
    df = pd.read_csv(args.input, sep='\t', dtype=str)
    df_norm = normalize_source(df)
    df_norm.to_parquet(args.output, index=False)
