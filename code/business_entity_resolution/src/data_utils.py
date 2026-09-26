"""Data loading and multi-representation normalisation.

Every record gets several normalised views of its name and address so that the
blocking legs and the pairwise features can each use the representation that
suits them (see README).  Nothing here touches any external resource: all
normalisation is rule based or learned from the provided training data.
"""
from __future__ import annotations

import re
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from anyascii import anyascii

COLS = ["entity_id", "business_name", "business_address", "country"]

# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #

def read_tsv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    missing = [c for c in COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}")
    return df[COLS]


def parse_ground_truth(path: str | Path) -> dict[str, set[str]]:
    gt = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    out: dict[str, set[str]] = {}
    for s1, m in zip(gt["source1_entity_id"], gt["matched_entity_ids"]):
        out[s1] = {x for x in m.split(",") if x} if m else set()
    return out


# --------------------------------------------------------------------------- #
# Normalisation dictionaries (rule based, no external lookups)
# --------------------------------------------------------------------------- #

LEGAL_SUFFIX = {
    "inc": "inc", "incorporated": "inc", "llc": "llc", "ltd": "ltd", "limited": "ltd",
    "pvt": "pvt", "private": "pvt", "corp": "corp", "corporation": "corp", "co": "co",
    "company": "co", "llp": "llp", "lp": "lp", "plc": "plc", "pllc": "pllc", "pc": "pc",
    "sarl": "sarl", "sas": "sas", "sasu": "sas", "eurl": "eurl", "sa": "sa", "sci": "sci",
    "snc": "snc", "scp": "scp", "gmbh": "gmbh", "ag": "ag", "bv": "bv", "nv": "nv",
    "pte": "pte", "pty": "pty", "opc": "opc",
}

NOISE_TOKENS = {"null", "none", "unknown", "nil", "nan", "na"}

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md", "massachusetts": "ma",
    "michigan": "mi", "minnesota": "mn", "mississippi": "ms", "missouri": "mo", "montana": "mt",
    "nebraska": "ne", "nevada": "nv", "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm",
    "new york": "ny", "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
    "district of columbia": "dc", "puerto rico": "pr",
}

IN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br",
    "chhattisgarh": "cg", "chattisgarh": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr",
    "himachal pradesh": "hp", "jharkhand": "jh", "karnataka": "ka", "kerala": "kl",
    "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mn", "meghalaya": "ml", "mizoram": "mz",
    "nagaland": "nl", "odisha": "od", "orissa": "od", "punjab": "pb", "rajasthan": "rj",
    "sikkim": "sk", "tamil nadu": "tn", "tamilnadu": "tn", "telangana": "tg", "tripura": "tr",
    "uttar pradesh": "up", "uttarakhand": "uk", "uttaranchal": "uk", "west bengal": "wb",
    "delhi": "dl", "jammu and kashmir": "jk", "jammu kashmir": "jk", "chandigarh": "ch",
    "puducherry": "py", "pondicherry": "py", "andaman and nicobar islands": "an",
    "dadra and nagar haveli": "dn", "daman and diu": "dd", "lakshadweep": "ld", "ladakh": "la",
}

_STATE_MAP = {**US_STATES, **IN_STATES}
_STATE_RE = re.compile(
    r"\b(" + "|".join(sorted((re.escape(k) for k in _STATE_MAP), key=len, reverse=True)) + r")\b"
)

ADDR_ABBR = {
    "rd": "road", "st": "street", "str": "street", "dr": "drive", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "bd": "boulevard", "hwy": "highway", "ln": "lane", "ct": "court",
    "pl": "place", "cir": "circle", "pkwy": "parkway", "sq": "square", "ste": "suite",
    "apt": "apartment", "bldg": "building", "fl": "floor", "flr": "floor", "rte": "route",
    "trl": "trail", "ter": "terrace", "terr": "terrace", "hts": "heights", "mt": "mount",
    "ft": "fort", "pt": "point", "n": "north", "s": "south", "e": "east", "w": "west",
    "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest",
    "nr": "near", "opp": "opposite", "num": "number", "r": "rue", "all": "allee",
    "imp": "impasse", "ch": "chemin", "chem": "chemin", "sec": "sector", "sect": "sector",
    "ph": "phase", "clny": "colony", "mkt": "market", "nagr": "nagar", "cplx": "complex",
}

_ALIAS_RE = re.compile(
    r"\b(?:a\s*/\s*k\s*/\s*a|f\s*/\s*k\s*/\s*a|d\s*/\s*b\s*/\s*a|aka|fka|dba|"
    r"formerly known as|formerly|now known as|trading as|t\s*/\s*a)\b"
)
_DOMAIN_RE = re.compile(
    r"^\s*(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9\-]*)\.(?:com|in|co\.in|org|net|co|io|fr|biz|info|us|eu|ltd|inc)\s*$"
)
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_ORDINAL_RE = re.compile(r"^(\d+)(st|nd|rd|th)$")
_SPLIT_RE = re.compile(r"[\s,;|/\\()\[\]{}]+")
_NA_RE = re.compile(r"\bn\s*/\s*a\b")
_DIGITS_RE = re.compile(r"\d+")


def fold(text: str) -> str:
    """Lowercase ASCII fold; transliterates any non-Latin script (Indic, accents)."""
    if not text:
        return ""
    if text.isascii():
        return text.lower()
    return anyascii(text).lower()


def _nonascii_frac(text: str) -> float:
    if not text or text.isascii():
        return 0.0
    return sum(ord(c) > 127 for c in text) / len(text)


def _tokens(text: str) -> list[str]:
    text = text.replace("'", "").replace("&", " and ")
    text = _NA_RE.sub(" ", text)
    return [t for t in _NON_ALNUM_RE.sub(" ", text).split() if t not in NOISE_TOKENS]


# --------------------------------------------------------------------------- #
# Name normalisation
# --------------------------------------------------------------------------- #

def normalize_name(raw: str) -> dict:
    f = fold(raw)
    is_domain = 0
    alt = ""
    m = _DOMAIN_RE.match(f)
    if m:
        is_domain = 1
        core_tokens = [m.group(1).replace("-", "")]
        suffix: list[str] = []
        all_tokens = core_tokens
    else:
        parts = _ALIAS_RE.split(f, maxsplit=1)
        if len(parts) == 2 and parts[0].strip() and parts[1].strip():
            primary, alt_raw = parts[1], parts[0]
            alt = " ".join(t for t in _tokens(alt_raw) if t not in LEGAL_SUFFIX)
        else:
            primary = f
        all_tokens = _tokens(primary)
        suffix = sorted({LEGAL_SUFFIX[t] for t in all_tokens if t in LEGAL_SUFFIX})
        core_tokens = [t for t in all_tokens if t not in LEGAL_SUFFIX]
        if not core_tokens:
            core_tokens = all_tokens
    core = " ".join(core_tokens)
    return {
        "name_core": core,
        "name_norm": " ".join(all_tokens),
        "name_suffix": " ".join(suffix),
        "name_nospace": "".join(core_tokens),
        "name_sorted": " ".join(sorted(core_tokens)),
        "name_alt": alt,
        "name_is_domain": is_domain,
        "name_nonascii": _nonascii_frac(raw),
    }


# --------------------------------------------------------------------------- #
# Address normalisation
# --------------------------------------------------------------------------- #

def normalize_address(raw: str) -> dict:
    if not raw or not raw.strip():
        return {
            "addr_norm": "", "addr_ascii_only": "", "house_no": "", "postcode": "",
            "addr_nums": "", "street": "", "addr_nonascii": 0.0, "addr_ntok": 0,
        }
    ascii_parts: list[str] = []
    other_parts: list[str] = []
    for tok in _SPLIT_RE.split(raw):
        if not tok:
            continue
        if tok.isascii():
            ascii_parts.append(tok.lower())
        else:
            other_parts.append(anyascii(tok).lower())
    ascii_text = _STATE_RE.sub(lambda m: _STATE_MAP[m.group(1)], " ".join(ascii_parts))
    tokens = _tokens(ascii_text)
    if not tokens:  # entire address in a non-Latin script: fall back to transliteration
        tokens = _tokens(_STATE_RE.sub(lambda m: _STATE_MAP[m.group(1)], " ".join(other_parts)))
    out_tokens: list[str] = []
    for t in tokens:
        om = _ORDINAL_RE.match(t)
        if om:
            t = om.group(1)
        out_tokens.append(ADDR_ABBR.get(t, t))
    tokens = out_tokens

    numeric_positions = [i for i, t in enumerate(tokens) if t[0].isdigit()]
    postcode = ""
    house_no = ""
    for j, i in enumerate(numeric_positions):
        t = tokens[i]
        if t.isdigit() and (len(t) == 6 or (len(t) == 5 and j > 0)):
            postcode = t
            break
    for i in numeric_positions:
        t = tokens[i]
        if t == postcode:
            continue
        house_no = _DIGITS_RE.match(t).group(0)
        break
    street = ""
    if numeric_positions:
        start = numeric_positions[0] + 1
        for t in tokens[start:]:
            if t.isalpha() and len(t) >= 3:
                street = t
                break
    nums = sorted({d for t in tokens for d in _DIGITS_RE.findall(t)})
    return {
        "addr_norm": " ".join(tokens),
        "addr_ascii_only": " ".join(tokens),
        "house_no": house_no,
        "postcode": postcode,
        "addr_nums": " ".join(nums),
        "street": street,
        "addr_nonascii": _nonascii_frac(raw),
        "addr_ntok": len(tokens),
    }


def _normalize_rows(rows: list[tuple[str, str]]) -> dict[str, list]:
    cols: dict[str, list] = {}
    for name, addr in rows:
        rec = normalize_name(name)
        rec.update(normalize_address(addr))
        for k, v in rec.items():
            cols.setdefault(k, []).append(v)
    return cols


def normalize_frame(df: pd.DataFrame, workers: int = 8, chunk_rows: int = 50_000) -> pd.DataFrame:
    """Return df with the normalised representation columns appended."""
    pairs = list(zip(df["business_name"].tolist(), df["business_address"].tolist()))
    chunks = [pairs[i:i + chunk_rows] for i in range(0, len(pairs), chunk_rows)]
    if workers > 1 and len(chunks) > 1:
        with Pool(workers) as pool:
            parts = pool.map(_normalize_rows, chunks, chunksize=1)
    else:
        parts = [_normalize_rows(c) for c in chunks]
    merged: dict[str, list] = {}
    for p in parts:
        for k, v in p.items():
            merged.setdefault(k, []).extend(v)
    norm = pd.DataFrame(merged, index=df.index)
    out = pd.concat([df.reset_index(drop=True), norm.reset_index(drop=True)], axis=1)
    for c in ("name_is_domain", "addr_ntok"):
        out[c] = out[c].astype(np.int16)
    for c in ("name_nonascii", "addr_nonascii"):
        out[c] = out[c].astype(np.float32)
    out.drop(columns=["addr_ascii_only"], inplace=True)
    return out
