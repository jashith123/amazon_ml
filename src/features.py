"""Stage 6-7 (Person C): Contract 1 + Contract 2 -> Contract 3 feature matrix + entity splits.

Contract: see src/common/schema.py (FEATURE_PREFIXES, LABEL_COL, FEAT_* names that Person D
relies on by exact name) and team_task_split.md. Hard requirements:
  * every feature column prefixed name__ / addr__ / cross__ / sem__ / comp__ / block__
  * comp__ features are rank / percentile / gap only - NEVER raw candidate counts
  * no NaN anywhere; sem__* = 0.0 when embeddings were skipped
  * splits written at the S1-entity level (train / val / val_us_only / val_india_only)

Speed design (same 60 columns as before):
  * every record is vectorised ONCE per country (char 2-3-gram TF-IDF, binary token set,
    binary numeric-token set); pair metrics are sparse row-wise products by index
  * the five edit-distance metrics use rapidfuzz.process.cpdist (multithreaded C++)
  * pairs are processed in chunks of S1 entities and streamed to parquet as float32

CLI:
  python src/features.py --split train [--max-s1 300000]
  python src/features.py --split test
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import jellyfish
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.sparse as sp
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import schema as S  # noqa: E402

CHAR_VEC_FIT_SAMPLE = 200_000
BASE_METRICS = ["lev_ratio", "jaro_winkler", "char_tfidf_cosine", "token_jaccard", "token_overlap",
                "token_sort_ratio", "trigram_jaccard", "prefix_overlap", "length_diff", "partial_ratio",
                "token_set_ratio", "length_ratio"]


# --------------------------------------------------------------------------- #
# sparse helpers
# --------------------------------------------------------------------------- #

def _rowwise_dot(A: sp.csr_matrix, B: sp.csr_matrix, ia: np.ndarray, ib: np.ndarray,
                 chunk: int = 500_000) -> np.ndarray:
    out = np.empty(len(ia), dtype=np.float32)
    for s in range(0, len(ia), chunk):
        a = A[ia[s:s + chunk]]
        b = B[ib[s:s + chunk]]
        out[s:s + chunk] = np.asarray(a.multiply(b).sum(axis=1)).ravel()
    return out


def _nnz_per_row(M: sp.csr_matrix) -> np.ndarray:
    return np.diff(M.indptr).astype(np.float32)


def _safe_div(num, den, fill=0.0):
    den = np.asarray(den, dtype=np.float32)
    out = np.full(len(den), fill, dtype=np.float32)
    ok = den > 0
    out[ok] = np.asarray(num, dtype=np.float32)[ok] / den[ok]
    return out


def _common_prefix_len(a: list[str], b: list[str]) -> np.ndarray:
    out = np.empty(len(a), dtype=np.float32)
    for i, (x, y) in enumerate(zip(a, b)):
        n = min(len(x), len(y))
        j = 0
        while j < n and x[j] == y[j]:
            j += 1
        out[i] = j
    return out


# --------------------------------------------------------------------------- #
# per-record representations (computed once per country)
# --------------------------------------------------------------------------- #

def fit_char_vectorizers(s1: pd.DataFrame, gallery: pd.DataFrame, seed: int = S.RANDOM_SEED):
    """char_wb 2-3-gram TF-IDF for names and addresses, fitted on <= 200k unique strings."""
    rng = np.random.RandomState(seed)
    vecs = {}
    for field in ("name_norm", "address_norm"):
        corpus = pd.concat([s1[field], gallery[field]]).fillna("").astype(str)
        corpus = corpus[corpus != ""].unique()
        if len(corpus) > CHAR_VEC_FIT_SAMPLE:
            corpus = rng.choice(corpus, CHAR_VEC_FIT_SAMPLE, replace=False)
        vecs[field] = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), lowercase=False,
                                      dtype=np.float32).fit(corpus)
    return vecs["name_norm"], vecs["address_norm"]


class RecordReps:
    """Vectorised views of every record (S1 + gallery) of one country, indexed by entity_id."""

    def __init__(self, recs: pd.DataFrame, name_vec: TfidfVectorizer, addr_vec: TfidfVectorizer):
        recs = recs.reset_index(drop=True)
        self.ids = pd.Index(recs["entity_id"].astype(str))
        g = lambda c: recs[c].fillna("").astype(str)  # noqa: E731
        self.name_norm = g("name_norm").to_numpy(dtype=object)
        self.name_core = g("name_core").to_numpy(dtype=object)
        self.name_raw = g("name_raw").to_numpy(dtype=object)
        self.suffix = g("legal_suffix").to_numpy(dtype=object)
        self.addr_norm = g("address_norm").to_numpy(dtype=object)
        self.addr_raw = g("address_raw").to_numpy(dtype=object)
        self.pin = g("address_pin").to_numpy(dtype=object)
        self.first_num = g("address_numbers").str.split("|").str[0].fillna("").to_numpy(dtype=object)
        self.country = g("country").to_numpy(dtype=object)
        self.name_len = g("name_norm").str.len().to_numpy(np.float32)
        self.addr_len = g("address_norm").str.len().to_numpy(np.float32)
        # soundex of name_core, computed once per unique value
        core_series = g("name_core")
        uniq = core_series.unique()
        sdx = {u: (jellyfish.soundex(u) if u else "") for u in uniq}
        self.soundex = core_series.map(sdx).to_numpy(dtype=object)
        # char TF-IDF (l2-normed -> dot = cosine) and its binary presence for n-gram jaccard
        self.name_tfidf = name_vec.transform(self.name_norm).tocsr()
        self.addr_tfidf = addr_vec.transform(self.addr_norm).tocsr()
        self.name_bin = self.name_tfidf.sign().tocsr()
        self.addr_bin = self.addr_tfidf.sign().tocsr()
        # binary token sets
        tok_vec = CountVectorizer(token_pattern=r"\S+", lowercase=False, binary=True, dtype=np.float32)
        self.name_tok = tok_vec.fit_transform(self.name_norm).tocsr()
        tok_vec_a = CountVectorizer(token_pattern=r"\S+", lowercase=False, binary=True, dtype=np.float32)
        self.addr_tok = tok_vec_a.fit_transform(self.addr_norm).tocsr()
        # binary numeric tokens from raw name + raw address (cross__numeric_conflict)
        num_vec = CountVectorizer(token_pattern=r"\d+", lowercase=False, binary=True, dtype=np.float32)
        joined = [f"{a} {b}" for a, b in zip(self.name_raw, self.addr_raw)]
        self.nums = num_vec.fit_transform(joined).tocsr()

    def index_of(self, ids) -> np.ndarray:
        idx = self.ids.get_indexer(pd.Index(ids).astype(str))
        if (idx < 0).any():
            raise KeyError(f"{(idx < 0).sum()} candidate/S1 ids not found in normalized records")
        return idx


# --------------------------------------------------------------------------- #
# pairwise features
# --------------------------------------------------------------------------- #

def _base_metrics(prefix: str, strs: np.ndarray, tfidf: sp.csr_matrix, binm: sp.csr_matrix,
                  tok: sp.csr_matrix, lens: np.ndarray, ia: np.ndarray, ib: np.ndarray) -> dict[str, np.ndarray]:
    a = strs[ia].tolist()
    b = strs[ib].tolist()
    f = {}
    f[f"{prefix}lev_ratio"] = cpdist(a, b, scorer=fuzz.ratio, workers=-1, dtype=np.float32) / 100.0
    f[f"{prefix}jaro_winkler"] = cpdist(a, b, scorer=JaroWinkler.normalized_similarity, workers=-1, dtype=np.float32)
    f[f"{prefix}token_sort_ratio"] = cpdist(a, b, scorer=fuzz.token_sort_ratio, workers=-1, dtype=np.float32) / 100.0
    f[f"{prefix}partial_ratio"] = cpdist(a, b, scorer=fuzz.partial_ratio, workers=-1, dtype=np.float32) / 100.0
    f[f"{prefix}token_set_ratio"] = cpdist(a, b, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32) / 100.0
    f[f"{prefix}char_tfidf_cosine"] = _rowwise_dot(tfidf, tfidf, ia, ib)
    nbin = _nnz_per_row(binm)
    inter = _rowwise_dot(binm, binm, ia, ib)
    f[f"{prefix}trigram_jaccard"] = _safe_div(inter, nbin[ia] + nbin[ib] - inter)
    ntok = _nnz_per_row(tok)
    tinter = _rowwise_dot(tok, tok, ia, ib)
    f[f"{prefix}token_jaccard"] = _safe_div(tinter, ntok[ia] + ntok[ib] - tinter)
    f[f"{prefix}token_overlap"] = _safe_div(tinter, np.minimum(ntok[ia], ntok[ib]))
    la, lb = lens[ia], lens[ib]
    mx = np.maximum(la, lb)
    f[f"{prefix}prefix_overlap"] = _safe_div(_common_prefix_len(a, b), mx)
    f[f"{prefix}length_diff"] = np.abs(la - lb).astype(np.float32)
    f[f"{prefix}length_ratio"] = _safe_div(np.minimum(la, lb), mx, fill=1.0)
    return f


def build_features(pairs: pd.DataFrame, s1: pd.DataFrame | None = None, gallery: pd.DataFrame | None = None,
                   reps: RecordReps | None = None, name_vec=None, addr_vec=None) -> pd.DataFrame:
    """Contract 2 pairs + Contract 1 records -> Contract 3 frame (label kept if present).

    Pass `reps` (precomputed once per country) for speed; otherwise it is built from s1+gallery.
    """
    if reps is None:
        if name_vec is None or addr_vec is None:
            name_vec, addr_vec = fit_char_vectorizers(s1, gallery)
        reps = RecordReps(pd.concat([s1, gallery], ignore_index=True), name_vec, addr_vec)
    pairs = pairs.reset_index(drop=True)
    ia = reps.index_of(pairs["source1_entity_id"])
    ib = reps.index_of(pairs["candidate_entity_id"])

    f: dict[str, np.ndarray] = {}
    f.update(_base_metrics("name__", reps.name_norm, reps.name_tfidf, reps.name_bin, reps.name_tok, reps.name_len, ia, ib))
    f["name__exact_norm_match"] = (reps.name_norm[ia] == reps.name_norm[ib]).astype(np.float32)
    f["name__exact_core_match"] = (reps.name_core[ia] == reps.name_core[ib]).astype(np.float32)
    sa, sb = reps.soundex[ia], reps.soundex[ib]
    f["name__phonetic_match"] = ((sa == sb) & (sa != "")).astype(np.float32)
    xa, xb = reps.suffix[ia], reps.suffix[ib]
    f["name__suffix_match"] = ((xa == xb) & (xa != "")).astype(np.float32)
    f["name__suffix_conflict"] = ((xa != xb) & (xa != "") & (xb != "")).astype(np.float32)
    f["name__exact_raw_match"] = (reps.name_raw[ia] == reps.name_raw[ib]).astype(np.float32)

    f.update(_base_metrics("addr__", reps.addr_norm, reps.addr_tfidf, reps.addr_bin, reps.addr_tok, reps.addr_len, ia, ib))
    ha, hb = reps.first_num[ia], reps.first_num[ib]
    f["addr__house_number_exact"] = ((ha == hb) & (ha != "")).astype(np.float32)
    f["addr__house_number_conflict"] = ((ha != hb) & (ha != "") & (hb != "")).astype(np.float32)
    pa_, pb_ = reps.pin[ia], reps.pin[ib]
    f["addr__pin_exact"] = ((pa_ == pb_) & (pa_ != "")).astype(np.float32)
    f["addr__both_have_address"] = ((reps.addr_raw[ia] != "") & (reps.addr_raw[ib] != "")).astype(np.float32)

    name_sim = f["name__lev_ratio"]
    addr_sim = f["addr__lev_ratio"]
    f["cross__name_addr_product"] = name_sim * addr_sim
    f["cross__name_addr_sum"] = name_sim + addr_sim
    f["cross__name_addr_min"] = np.minimum(name_sim, addr_sim)
    f["cross__name_addr_max"] = np.maximum(name_sim, addr_sim)
    f["cross__name_high_addr_low"] = ((name_sim > 0.8) & (addr_sim < 0.5)).astype(np.float32)
    f["cross__name_low_addr_high"] = ((name_sim < 0.5) & (addr_sim > 0.8)).astype(np.float32)
    f["cross__name_addr_diff"] = np.abs(name_sim - addr_sim)
    f["cross__name_addr_harmonic"] = (2 * name_sim * addr_sim) / (name_sim + addr_sim + 1e-6)
    f["cross__country_exact_match"] = (reps.country[ia] == reps.country[ib]).astype(np.float32)
    nn = _nnz_per_row(reps.nums)
    ninter = _rowwise_dot(reps.nums, reps.nums, ia, ib)
    f["cross__numeric_conflict"] = ((nn[ia] > 0) & (nn[ib] > 0) & (ninter == 0)).astype(np.float32)

    for c in S.BLOCK_FLAG_COLS:
        f[f"block__{c}"] = pairs[c].fillna(0).to_numpy().astype(np.float32)
    f["block__num_legs_retrieved"] = pairs["num_legs_retrieved"].to_numpy().astype(np.float32)

    grp = pairs.groupby("source1_entity_id", sort=False)["blocking_score"]
    f["comp__rank"] = grp.rank(ascending=False, method="first").to_numpy(np.float32)
    f["comp__score_percentile"] = grp.rank(pct=True).to_numpy(np.float32)
    f["comp__score_gap_top1"] = (grp.transform("max") - pairs["blocking_score"]).to_numpy(np.float32)
    # gallery-side competition (global, computed over the whole candidate file before chunking):
    # one S2/S3 record belongs to at most one S1, so "is this S1 the best-scoring S1 for this
    # candidate" is a direct false-positive detector. Ratio and rank only: density-safe.
    if "g_max_score" in pairs.columns:
        gmax = pairs["g_max_score"].to_numpy(np.float32)
        bs = pairs["blocking_score"].to_numpy(np.float32)
        f["comp__g_best_ratio"] = np.where(gmax > 0, bs / np.maximum(gmax, 1e-6), 1.0).astype(np.float32)
        f["comp__g_rank"] = pairs["g_rank"].to_numpy(np.float32)
    else:
        f["comp__g_best_ratio"] = np.ones(len(pairs), dtype=np.float32)
        f["comp__g_rank"] = np.ones(len(pairs), dtype=np.float32)

    for c in ("sem__embedding_cosine", "sem__embedding_rank", "sem__embedding_percentile", "sem__embedding_gap"):
        f[c] = np.zeros(len(pairs), dtype=np.float32)

    feats = pd.DataFrame(f)
    for c in S.feature_columns(feats.columns):
        feats[c] = np.nan_to_num(feats[c].to_numpy(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    feats.insert(0, "country", pairs["country"].to_numpy())
    feats.insert(0, "candidate_source", pairs["candidate_source"].to_numpy())
    feats.insert(0, "candidate_entity_id", pairs["candidate_entity_id"].to_numpy())
    feats.insert(0, "source1_entity_id", pairs["source1_entity_id"].to_numpy())
    if S.LABEL_COL in pairs.columns:
        feats[S.LABEL_COL] = pairs[S.LABEL_COL].to_numpy().astype(np.int8)
    return feats


# --------------------------------------------------------------------------- #
# splits + validation
# --------------------------------------------------------------------------- #

def write_splits(entities: pd.DataFrame, splits_dir: Path = S.SPLITS_DIR, val_frac: float = 0.2) -> None:
    """Entity-level 80/20 split plus per-country validation lists (open country set).

    `entities`: frame with unique source1_entity_id + country.
    """
    splits_dir.mkdir(parents=True, exist_ok=True)
    ents = entities.drop_duplicates("source1_entity_id")[["source1_entity_id", "country"]]
    rng = np.random.RandomState(S.RANDOM_SEED)
    ids = ents["source1_entity_id"].to_numpy(dtype=object)
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


# --------------------------------------------------------------------------- #
# CLI: per-country, chunked, streamed to parquet
# --------------------------------------------------------------------------- #

def _load(nd: Path, suffix: str):
    s1 = pd.read_parquet(nd / S.NORMALIZED_FILE.format(n=1, suffix=suffix))
    gal = pd.concat([pd.read_parquet(nd / S.NORMALIZED_FILE.format(n=n, suffix=suffix)) for n in (2, 3)], ignore_index=True)
    return s1, gal


def run(split: str, normalized_dir: Path, out_path: Path, max_s1: int | None = None,
        chunk_s1: int = 100_000, cand_path: Path | None = None) -> None:
    t0 = time.time()
    suffix = "" if split == "train" else "_test"
    s1, gal = _load(normalized_dir, suffix)
    from common.tokenmap import apply_to_gallery  # same learned normalisation as blocking
    gal = apply_to_gallery(gal)
    pairs = pd.read_parquet(cand_path or (S.CANDIDATES_TRAIN if split == "train" else S.CANDIDATES_TEST))
    # gallery-side competition over the FULL candidate file (before any train sampling)
    gcol = pairs["candidate_entity_id"]
    pairs["g_max_score"] = pairs.groupby(gcol, sort=False)["blocking_score"].transform("max").astype(np.float32)
    pairs["g_rank"] = pairs.groupby(gcol, sort=False)["blocking_score"].rank(ascending=False, method="min").astype(np.float32)
    print(f"gallery-side competition computed over {len(pairs):,} pairs", flush=True)
    if split == "train":
        pairs = pairs.rename(columns={"is_true_match": S.LABEL_COL})
        pairs[S.LABEL_COL] = pairs[S.LABEL_COL].astype(int)
        if max_s1 and pairs["source1_entity_id"].nunique() > max_s1:
            ents = pairs.drop_duplicates("source1_entity_id")[["source1_entity_id", "country"]]
            keep = ents.groupby("country", group_keys=False).sample(frac=max_s1 / len(ents), random_state=S.RANDOM_SEED)
            pairs = pairs[pairs["source1_entity_id"].isin(keep["source1_entity_id"])].reset_index(drop=True)
            print(f"train: sampled {pairs['source1_entity_id'].nunique():,} S1 entities -> {len(pairs):,} pairs")
    print(f"loaded: {len(s1):,} S1, {len(gal):,} gallery, {len(pairs):,} pairs in {time.time()-t0:.0f}s", flush=True)

    name_vec, addr_vec = fit_char_vectorizers(s1, gal)
    print(f"char vectorizers fitted ({time.time()-t0:.0f}s)", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    n_rows = 0
    entities = []
    for country in pairs["country"].unique():
        pc = pairs[pairs["country"] == country]
        used_ids = set(pc["source1_entity_id"]) | set(pc["candidate_entity_id"])
        recs = pd.concat([s1[s1["entity_id"].isin(used_ids)], gal[gal["entity_id"].isin(used_ids)]], ignore_index=True)
        reps = RecordReps(recs, name_vec, addr_vec)
        del recs
        print(f"[{country}] {len(pc):,} pairs, {len(reps.ids):,} records vectorised ({time.time()-t0:.0f}s)", flush=True)
        s1_ids = pc["source1_entity_id"].unique()
        for i in range(0, len(s1_ids), chunk_s1):
            chunk = pc[pc["source1_entity_id"].isin(set(s1_ids[i:i + chunk_s1]))]
            feats = build_features(chunk, reps=reps)
            if writer is None:
                validate_contract3(feats, train=split == "train")
                writer = pq.ParquetWriter(str(out_path), pa.Table.from_pandas(feats.head(1), preserve_index=False).schema)
            writer.write_table(pa.Table.from_pandas(feats, preserve_index=False))
            n_rows += len(feats)
            entities.append(feats[["source1_entity_id", "country"]].drop_duplicates())
            print(f"  [{country}] {min(i + chunk_s1, len(s1_ids)):,}/{len(s1_ids):,} S1 -> {n_rows:,} rows ({time.time()-t0:.0f}s)", flush=True)
        del reps
    if writer is not None:
        writer.close()
    if split == "train" and entities:
        write_splits(pd.concat(entities, ignore_index=True))
    n_feat = len(S.feature_columns(pq.read_schema(str(out_path)).names)) if n_rows else 0
    print(f"wrote {out_path}: {n_rows:,} rows, {n_feat} features in {time.time()-t0:.0f}s")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Stage 6-7: features + splits")
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--normalized-dir", default=str(S.NORMALIZED_DIR))
    ap.add_argument("--candidates", default=None, help="override Contract 2 input path")
    ap.add_argument("--out", default=None, help="override Contract 3 output path")
    ap.add_argument("--max-s1", type=int, default=None, help="train only: sample this many S1 entities (stratified by country)")
    ap.add_argument("--chunk-s1", type=int, default=100_000)
    args = ap.parse_args(argv)
    out = Path(args.out) if args.out else (S.FEATURES_TRAIN if args.split == "train" else S.FEATURES_TEST)
    run(args.split, Path(args.normalized_dir), out, max_s1=args.max_s1, chunk_s1=args.chunk_s1,
        cand_path=Path(args.candidates) if args.candidates else None)


if __name__ == "__main__":
    main()
