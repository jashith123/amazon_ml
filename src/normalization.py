"""Stage 1-2 (Person A): raw TSV -> Contract 1 normalized parquet.

Contract: see src/common/schema.py (NORMALIZED_COLS) and team_task_split.md.

CLI:
  python src/normalization.py --input dataset/train/train_source1.tsv \
      --output data/normalized/source1_normalized.parquet

A complete working reference implementation (multi-representation normalisation with
Indic transliteration, legal-suffix extraction, address abbreviation expansion, house
number / PIN extraction) lives in src/reference/normalization_ref.py. You may build on it
or write your own; only the output schema is fixed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import schema as S  # noqa: E402


def normalize_source(df: pd.DataFrame) -> pd.DataFrame:
    """Raw frame (entity_id, business_name, business_address, country) -> Contract 1 frame.

    TODO (Person A): implement. Must return exactly S.NORMALIZED_COLS in that order,
    same row count as the input, no dropped entity_ids.
    """
    raise NotImplementedError("Person A: implement normalize_source (see docstring / reference)")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Stage 1-2: normalisation")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args(argv)
    df = pd.read_csv(args.input, sep="\t", dtype=str, keep_default_na=False)
    out = normalize_source(df)
    assert list(out.columns) == S.NORMALIZED_COLS, "Contract 1 column mismatch"
    assert len(out) == len(df), "row count changed"
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)
    print(f"wrote {args.output}: {len(out):,} rows")


if __name__ == "__main__":
    main()
