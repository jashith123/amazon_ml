import argparse
import pandas as pd
import numpy as np
import re
import unicodedata
from indic_transliteration import sanscript
from indic_transliteration.detect import detect

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common.schema import NORMALIZED_COLUMNS

LEGAL_SUFFIXES = [
    'pvt ltd', 'private limited', 'pvt limited', 'private ltd',
    'ltd', 'limited', 'corp', 'corporation', 'inc', 'incorporated',
    'llc', 'gmbh', 'sarl', 'co', 'company', 'plc'
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

def transliterate_and_fold(text):
    if not text: return ''
    
    if is_non_latin(text):
        scheme = detect(text)
        if scheme:
            # Prevent detecting ascii as ITRANS if it somehow still happens, though is_non_latin prevents this
            text = sanscript.transliterate(text, scheme, sanscript.ITRANS).lower()
            
    for k, v in FRENCH_LIGATURES.items():
        text = text.replace(k, v)
        
    return text

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
    
    def norm_text(t):
        if not t: return ''
        t = unicodedata.normalize('NFKD', t)
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
    
    out['is_non_latin'] = out['name_raw'].apply(is_non_latin)
    out['name_transliterated'] = out['name_norm'].apply(transliterate_and_fold)
    
    def norm_addr(t):
        if not t: return ''
        t = unicodedata.normalize('NFKD', t).lower()
        t = remove_punctuation(t)
        t = collapse_whitespace(t)
        for patt, repl in ABBREVIATIONS.items():
            t = re.sub(patt, repl, t)
        return t
        
    out['address_norm'] = out['address_raw'].apply(norm_addr)
    out['address_numbers'] = out['address_norm'].apply(extract_numbers)
    out['address_pin'] = out['address_norm'].apply(extract_pin)
    
    out = out[NORMALIZED_COLUMNS]
    return out

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input TSV file')
    parser.add_argument('--output', required=True, help='Output Parquet file')
    args = parser.parse_args()
    
    df = pd.read_csv(args.input, sep='\t', dtype=str)
    df_norm = normalize_source(df)
    df_norm.to_parquet(args.output, index=False)
