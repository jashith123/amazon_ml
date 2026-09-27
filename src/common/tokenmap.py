"""Learned token normalisation (gallery token -> Source 1 token), see scripts/learn_token_map.py.

Applied to Source 2/3 names in blocking AND features so train and test are treated alike.
Active when BLOCK_TOKEN_MAP points to a JSON file or models/token_map.json exists;
BLOCK_TOKEN_MAP=0 disables it.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from . import schema as S

_MAP: dict[str, str] | None = None
_CHECKED = False


def load_token_map() -> dict[str, str]:
    global _MAP, _CHECKED
    if not _CHECKED:
        _CHECKED = True
        _MAP = {}
        env = os.environ.get("BLOCK_TOKEN_MAP", "")
        if env == "0":
            return _MAP
        path = Path(env) if env else S.MODELS_DIR / "token_map.json"
        if path.exists():
            _MAP = json.loads(path.read_text(encoding="utf-8"))
            print(f"    token map loaded: {len(_MAP):,} mappings from {path}", flush=True)
    return _MAP or {}


def apply_map(names: pd.Series, mapping: dict[str, str]) -> pd.Series:
    if not mapping:
        return names
    return names.fillna("").astype(str).map(lambda s: " ".join(mapping.get(t, t) for t in s.split()) if s else "")


def apply_to_gallery(df: pd.DataFrame) -> pd.DataFrame:
    """Rewrite name_norm / name_core / name_tokens of a Contract-1 gallery frame in place."""
    mapping = load_token_map()
    if not mapping or df.empty:
        return df
    df = df.copy()
    if "name_norm" in df.columns:
        df["name_norm"] = apply_map(df["name_norm"], mapping)
    if "name_core" in df.columns:
        df["name_core"] = apply_map(df["name_core"], mapping)
        if "name_tokens" in df.columns:
            df["name_tokens"] = df["name_core"].map(lambda s: "|".join(sorted(set(s.split()))) if s else "")
    return df
