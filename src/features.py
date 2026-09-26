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
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import rapidfuzz
import rapidfuzz.distance
import rapidfuzz.process
import jellyfish
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import paired_cosine_distances

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import schema as S  # noqa: E402


def _get_base_metrics(a: pd.Series, b: pd.Series, prefix: str, vec: TfidfVectorizer) -> pd.DataFrame:
    a_list = a.fillna("").astype(str).tolist()
    b_list = b.fillna("").astype(str).tolist()
    
    # Fast metrics with cpdist
    lev = rapidfuzz.process.cpdist(a_list, b_list, scorer=rapidfuzz.fuzz.ratio, workers=-1) / 100.0
    jw = rapidfuzz.process.cpdist(a_list, b_list, scorer=rapidfuzz.distance.JaroWinkler.normalized_similarity, workers=-1)
    tsr = rapidfuzz.process.cpdist(a_list, b_list, scorer=rapidfuzz.fuzz.token_sort_ratio, workers=-1) / 100.0
    pr = rapidfuzz.process.cpdist(a_list, b_list, scorer=rapidfuzz.fuzz.partial_ratio, workers=-1) / 100.0
    tset = rapidfuzz.process.cpdist(a_list, b_list, scorer=rapidfuzz.fuzz.token_set_ratio, workers=-1) / 100.0
    
    # 3. char_tfidf_cosine
    if len(a_list) > 0:
        tfidf_a = vec.transform(a_list)
        tfidf_b = vec.transform(b_list)
        tfidf_cos = (1.0 - paired_cosine_distances(tfidf_a, tfidf_b)).tolist()
    else:
        tfidf_cos = []
    
    # 4. token_jaccard
    def tok_jac(x, y):
        sx, sy = set(x.split()), set(y.split())
        if not sx and not sy: return 0.0
        return len(sx & sy) / len(sx | sy)
    t_jac = [tok_jac(x, y) for x, y in zip(a_list, b_list)]
    
    # 5. token_overlap
    def tok_ov(x, y):
        sx, sy = set(x.split()), set(y.split())
        if not sx or not sy: return 0.0
        return len(sx & sy) / min(len(sx), len(sy))
    t_ov = [tok_ov(x, y) for x, y in zip(a_list, b_list)]
    
    # 7. trigram_jaccard
    def tri_jac(x, y):
        sx = set([x[i:i+3] for i in range(len(x)-2)]) if len(x) >= 3 else set([x])
        sy = set([y[i:i+3] for i in range(len(y)-2)]) if len(y) >= 3 else set([y])
        sx.discard("")
        sy.discard("")
        if not sx and not sy: return 0.0
        if not sx or not sy: return 0.0
        return len(sx & sy) / len(sx | sy)
    tr_jac = [tri_jac(x, y) for x, y in zip(a_list, b_list)]
    
    # 8. prefix_overlap
    pref_ov = [len(os.path.commonprefix([x, y])) / max(len(x), len(y)) if max(len(x), len(y)) > 0 else 0.0 for x, y in zip(a_list, b_list)]
    
    # 9. length_diff
    l_diff = [float(abs(len(x) - len(y))) for x, y in zip(a_list, b_list)]
    
    # 12. length_ratio
    l_rat = [min(len(x), len(y)) / max(len(x), len(y)) if max(len(x), len(y)) > 0 else 1.0 for x, y in zip(a_list, b_list)]
    
    return pd.DataFrame({
        f"{prefix}lev_ratio": lev,
        f"{prefix}jaro_winkler": jw,
        f"{prefix}char_tfidf_cosine": tfidf_cos,
        f"{prefix}token_jaccard": t_jac,
        f"{prefix}token_overlap": t_ov,
        f"{prefix}token_sort_ratio": tsr,
        f"{prefix}trigram_jaccard": tr_jac,
        f"{prefix}prefix_overlap": pref_ov,
        f"{prefix}length_diff": l_diff,
        f"{prefix}partial_ratio": pr,
        f"{prefix}token_set_ratio": tset,
        f"{prefix}length_ratio": l_rat,
    }, index=a.index)


def build_features(pairs: pd.DataFrame, s1: pd.DataFrame, gallery: pd.DataFrame, name_vec: TfidfVectorizer, addr_vec: TfidfVectorizer) -> pd.DataFrame:
    """Contract 2 pairs + Contract 1 records -> Contract 3 frame (label kept if present)."""
    df = pairs.merge(s1.add_prefix("s1_"), left_on="source1_entity_id", right_on="s1_entity_id", how="left")
    df = df.merge(gallery.add_prefix("cand_"), left_on="candidate_entity_id", right_on="cand_entity_id", how="left")
    
    name_base = _get_base_metrics(df["s1_name_norm"], df["cand_name_norm"], "name__", name_vec)
    addr_base = _get_base_metrics(df["s1_address_norm"], df["cand_address_norm"], "addr__", addr_vec)
    
    a_norm, b_norm = df['s1_name_norm'].fillna(""), df['cand_name_norm'].fillna("")
    a_core, b_core = df['s1_name_core'].fillna(""), df['cand_name_core'].fillna("")
    a_suff, b_suff = df['s1_legal_suffix'].fillna(""), df['cand_legal_suffix'].fillna("")
    
    exact_norm = (a_norm == b_norm).astype(float)
    exact_core = (a_core == b_core).astype(float)
    
    a_phon = a_core.apply(lambda x: jellyfish.soundex(x) if x else "")
    b_phon = b_core.apply(lambda x: jellyfish.soundex(x) if x else "")
    phonetic_match = ((a_phon == b_phon) & (a_phon != "")).astype(float)
    
    suff_match = ((a_suff == b_suff) & (a_suff != "")).astype(float)
    suff_conf = ((a_suff != b_suff) & (a_suff != "") & (b_suff != "")).astype(float)
    exact_raw = (df['s1_name_raw'].fillna("") == df['cand_name_raw'].fillna("")).astype(float)
    
    name_spec = pd.DataFrame({
        "name__exact_norm_match": exact_norm,
        "name__exact_core_match": exact_core,
        "name__phonetic_match": phonetic_match,
        "name__suffix_match": suff_match,
        "name__suffix_conflict": suff_conf,
        "name__exact_raw_match": exact_raw,
    }, index=df.index)
    
    a_hn_first = df['s1_address_numbers'].fillna("").str.split("|").str[0]
    b_hn_first = df['cand_address_numbers'].fillna("").str.split("|").str[0]
    a_pin, b_pin = df['s1_address_pin'].fillna(""), df['cand_address_pin'].fillna("")
    a_raw, b_raw = df['s1_address_raw'].fillna(""), df['cand_address_raw'].fillna("")
    
    hn_exact = ((a_hn_first == b_hn_first) & (a_hn_first != "")).astype(float)
    hn_conf = ((a_hn_first != b_hn_first) & (a_hn_first != "") & (b_hn_first != "")).astype(float)
    pin_exact = ((a_pin == b_pin) & (a_pin != "")).astype(float)
    both_have = ((a_raw != "") & (b_raw != "")).astype(float)
    
    addr_spec = pd.DataFrame({
        "addr__house_number_exact": hn_exact,
        "addr__house_number_conflict": hn_conf,
        "addr__pin_exact": pin_exact,
        "addr__both_have_address": both_have,
    }, index=df.index)
    
    feats = pd.concat([name_base, name_spec, addr_base, addr_spec], axis=1)
    
    name_sim = feats["name__lev_ratio"]
    addr_sim = feats["addr__lev_ratio"]
    
    feats["cross__name_addr_product"] = name_sim * addr_sim
    feats["cross__name_addr_sum"] = name_sim + addr_sim
    feats["cross__name_addr_min"] = np.minimum(name_sim, addr_sim)
    feats["cross__name_addr_max"] = np.maximum(name_sim, addr_sim)
    feats["cross__name_high_addr_low"] = ((name_sim > 0.8) & (addr_sim < 0.5)).astype(float)
    feats["cross__name_low_addr_high"] = ((name_sim < 0.5) & (addr_sim > 0.8)).astype(float)
    feats["cross__name_addr_diff"] = np.abs(name_sim - addr_sim)
    feats["cross__name_addr_harmonic"] = (2 * name_sim * addr_sim) / (name_sim + addr_sim + 1e-6)
    
    feats["cross__country_exact_match"] = (df['country'] == df['cand_country']).astype(float)
    
    def extract_nums(s): return set(re.findall(r'\d+', str(s)))
    s1_nums = (df['s1_name_raw'].fillna("") + " " + df['s1_address_raw'].fillna("")).apply(extract_nums)
    s2_nums = (df['cand_name_raw'].fillna("") + " " + df['cand_address_raw'].fillna("")).apply(extract_nums)
    feats["cross__numeric_conflict"] = [float(bool(n1 and n2 and not (n1 & n2))) for n1, n2 in zip(s1_nums, s2_nums)]
    
    for c in S.BLOCK_FLAG_COLS:
        feats[f"block__{c}"] = df[c].fillna(0).astype(float)
    feats["block__num_legs_retrieved"] = df["num_legs_retrieved"].astype(float)
    
    grp = df.groupby("source1_entity_id")["blocking_score"]
    feats["comp__rank"] = grp.rank(ascending=False, method="first")
    feats["comp__score_percentile"] = grp.rank(pct=True)
    feats["comp__score_gap_top1"] = grp.transform("max") - df["blocking_score"]
    
    feats["sem__embedding_cosine"] = 0.0
    feats["sem__embedding_rank"] = 0.0
    feats["sem__embedding_percentile"] = 0.0
    feats["sem__embedding_gap"] = 0.0
    
    feats["source1_entity_id"] = df["source1_entity_id"]
    feats["candidate_entity_id"] = df["candidate_entity_id"]
    feats["candidate_source"] = df["candidate_source"]
    feats["country"] = df["country"]
    if S.LABEL_COL in df.columns:
        feats[S.LABEL_COL] = df[S.LABEL_COL]
        
    for c in feats.columns:
        if c.startswith(S.FEATURE_PREFIXES):
            feats[c] = feats[c].fillna(0.0)
            
    return feats


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
        
    print("Fitting TF-IDF vectorizers...")
    name_corpus = pd.concat([s1["name_norm"], gal["name_norm"]]).dropna().unique()
    if len(name_corpus) > 200_000:
        name_corpus = np.random.RandomState(S.RANDOM_SEED).choice(name_corpus, 200_000, replace=False)
    name_vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2,3)).fit(name_corpus)
    
    addr_corpus = pd.concat([s1["address_norm"], gal["address_norm"]]).dropna().unique()
    if len(addr_corpus) > 200_000:
        addr_corpus = np.random.RandomState(S.RANDOM_SEED).choice(addr_corpus, 200_000, replace=False)
    addr_vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2,3)).fit(addr_corpus)

    print("Building features in chunks...")
    s1_ids = pairs["source1_entity_id"].unique()
    chunk_size = 100_000
    feats_chunks = []
    
    for i in range(0, len(s1_ids), chunk_size):
        chunk_s1 = set(s1_ids[i:i+chunk_size])
        chunk_pairs = pairs[pairs["source1_entity_id"].isin(chunk_s1)]
        fc = build_features(chunk_pairs, s1, gal, name_vec, addr_vec)
        feats_chunks.append(fc)
        
    feats = pd.concat(feats_chunks, ignore_index=True)
    
    validate_contract3(feats, train=args.split == "train")
    path = S.FEATURES_TRAIN if args.split == "train" else S.FEATURES_TEST
    path.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(path, index=False)
    if args.split == "train":
        write_splits(feats)
    print(f"wrote {path}: {len(feats):,} rows, {len(S.feature_columns(feats.columns))} features")


if __name__ == "__main__":
    main()
