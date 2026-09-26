"""Stage 6-7 (Person C): Contract 1 + Contract 2 -> Contract 3 feature matrix + entity splits.

Contract: see src/common/schema.py (FEATURE_PREFIXES, LABEL_COL, FEAT_* names that Person D
relies on by exact name) and team_task_split.md. Hard requirements:
  * every feature column prefixed name__ / addr__ / cross__ / sem__ / comp__ / block__
  * comp__ features are rank / percentile / gap only - NEVER raw candidate counts
  * no NaN anywhere; sem__* = 0.0 when embeddings were skipped
  * splits written at the S1-entity level (train / val / val_us_only / val_india_only)

CLI:
  python src/features.py --split train
  python src/features.py --split test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import schema as S  # noqa: E402


def build_features(pairs: pd.DataFrame, s1: pd.DataFrame, gallery: pd.DataFrame) -> pd.DataFrame:
    """Contract 2 pairs + Contract 1 records -> Contract 3 frame (label kept if present).

    TODO (Person C): implement the six feature groups (rapidfuzz.process.cpdist is the fast
    element-wise pairwise scorer: process.cpdist(list_a, list_b, scorer=fuzz.ratio, workers=-1)).
    """
    raise NotImplementedError("Person C: implement build_features (see docstring)")


def write_splits(feats: pd.DataFrame, splits_dir: Path = S.SPLITS_DIR, val_frac: float = 0.2) -> None:
    """Entity-level 80/20 split plus per-country validation lists (open country set)."""
    splits_dir.mkdir(parents=True, exist_ok=True)
    ents = feats.drop_duplicates("source1_entity_id")[["source1_entity_id", "country"]]
    rng = np.random.RandomState(S.RANDOM_SEED)
    ids = ents["source1_entity_id"].to_numpy()
    perm = rng.permutation(len(ids))
    n_val = int(val_frac * len(ids))
    val_ids, train_ids = ids[perm[:n_val]], ids[perm[n_val:]]
    (splits_dir / S.SPLIT_TRAIN_IDS.name).write_text("\n".join(train_ids))
    (splits_dir / S.SPLIT_VAL_IDS.name).write_text("\n".join(val_ids))
    country = ents.set_index("source1_entity_id")["country"]
    val_country = country.reindex(val_ids)
    (splits_dir / S.SPLIT_VAL_US_IDS.name).write_text("\n".join(val_ids[(val_country == "US").to_numpy()]))
    (splits_dir / S.SPLIT_VAL_INDIA_IDS.name).write_text("\n".join(val_ids[(val_country == "India").to_numpy()]))
    for c in sorted(set(val_country.unique()) - {"US", "India"}):
        (splits_dir / f"val_{str(c).lower()}_only_ids.txt").write_text("\n".join(val_ids[(val_country == c).to_numpy()]))


def validate_contract3(feats: pd.DataFrame, train: bool) -> None:
    cols = S.feature_columns(feats.columns)
    assert cols, "no prefixed feature columns"
    S.assert_no_raw_counts(cols)
    assert feats[cols].isna().sum().sum() == 0, "NaNs in feature matrix"
    for c in (S.FEAT_HOUSE_CONFLICT, S.FEAT_ADDR_SIM, S.FEAT_NAME_SIM, S.FEAT_PIN_EXACT):
        assert c in feats.columns, f"Person D needs column {c} by exact name"
    assert (S.LABEL_COL in feats.columns) == train


def main(argv=None):
    ap = argparse.ArgumentParser(description="Stage 6-7: features + splits")
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--normalized-dir", default=str(S.NORMALIZED_DIR))
    args = ap.parse_args(argv)
    suffix = "" if args.split == "train" else "_test"
    nd = Path(args.normalized_dir)
    s1 = pd.read_parquet(nd / S.NORMALIZED_FILE.format(n=1, suffix=suffix))
    gal = pd.concat([pd.read_parquet(nd / S.NORMALIZED_FILE.format(n=n, suffix=suffix)) for n in (2, 3)], ignore_index=True)
    pairs = pd.read_parquet(S.CANDIDATES_TRAIN if args.split == "train" else S.CANDIDATES_TEST)
    if args.split == "train":
        pairs = pairs.rename(columns={"is_true_match": S.LABEL_COL})
        pairs[S.LABEL_COL] = pairs[S.LABEL_COL].astype(int)
    feats = build_features(pairs, s1, gal)
    validate_contract3(feats, train=args.split == "train")
    path = S.FEATURES_TRAIN if args.split == "train" else S.FEATURES_TEST
    path.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(path, index=False)
    if args.split == "train":
        write_splits(feats)
    print(f"wrote {path}: {len(feats):,} rows, {len(S.feature_columns(feats.columns))} features")


if __name__ == "__main__":
    main()
