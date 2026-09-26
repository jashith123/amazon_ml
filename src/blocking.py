"""Stage 3-5 (Person B): Contract 1 normalized records -> Contract 2 candidate pairs.

Contract: see src/common/schema.py (CANDIDATE_COLS / CANDIDATE_TRAIN_COLS) and
team_task_split.md. Hard requirements:
  * country partitioning on every leg (open-set string label, never hard-coded)
  * <= S.MAX_CANDIDATES_PER_S1 candidates per Source 1 entity after adaptive-K pruning
  * blocking recall diagnostic against train_ground_truth.tsv printed before hand-off
  * Legs A, B, C, D, E, F implemented; Leg G skipped (embedding_block_hit = False)

CLI:
  python src/blocking.py --split train   # writes data/candidates/candidate_pairs_train.parquet
  python src/blocking.py --split test    # writes data/candidates/candidate_pairs_test.parquet
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import jellyfish
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import schema as S  # noqa: E402

# Leg bitmask flags matching S.BLOCK_FLAG_COLS order
LEG_NAME_TFIDF = 1         # name_tfidf_block_hit (Leg A)
LEG_CHAR_NGRAM = 2         # char_ngram_block_hit (Leg B)
LEG_ADDRESS    = 4         # address_block_hit (Address exact / number keys)
LEG_EXACT_KEY  = 8         # exact_key_hit (Leg D: core name, tokens, acronym)
LEG_PIN        = 16        # pin_block_hit (Leg E: PIN / ZIP)
LEG_PHONETIC   = 32        # phonetic_block_hit (Leg F: Soundex / Metaphone)
LEG_EMBEDDING  = 64        # embedding_block_hit (Leg G: skipped -> False)
LEG_BIDIR      = 128       # bidirectional_block_hit (Leg C)

# Contract 1 columns actually used by blocking (the CLI loads only these)
BLOCKING_COLS = ["entity_id", "country", "name_norm", "name_core", "name_tokens",
                 "address_norm", "address_numbers", "address_pin"]


def adaptive_k_prune(candidates, scores, k_min=5, gap=0.1, k_max=S.MAX_CANDIDATES_PER_S1):
    """Keep rank < k_min OR score >= top1 - gap, cap at k_max (pipeline doc section 10)."""
    if not candidates:
        return []
    top1 = max(scores)
    threshold = top1 - gap
    kept = [(c, s) for c, s in zip(candidates, scores) if s >= threshold]
    kept.sort(key=lambda x: -x[1])
    full = sorted(zip(candidates, scores), key=lambda x: -x[1])
    if len(kept) < k_min:
        kept = full[:k_min]
    return [c for c, _ in kept[:k_max]]


def load_ground_truth(gt_path: Path) -> dict[str, set[str]]:
    """Load ground truth TSV into a mapping: source1_entity_id -> set of matched entity IDs."""
    gt = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False)
    return {
        s: ({x for x in m.split(",") if x} if m else set())
        for s, m in zip(gt[S.GT_COLS[0]], gt[S.GT_COLS[1]])
    }


def rowwise_dot(A: sp.csr_matrix, B: sp.csr_matrix, ia: np.ndarray, ib: np.ndarray, chunk: int = 500_000) -> np.ndarray:
    """Row-wise sparse dot product between A[ia] and B[ib]."""
    if len(ia) == 0:
        return np.empty(0, dtype=np.float32)
    out = np.empty(len(ia), dtype=np.float32)
    for s in range(0, len(ia), chunk):
        e = s + chunk
        a = A[ia[s:e]]
        b = B[ib[s:e]]
        out[s:e] = np.asarray(a.multiply(b).sum(axis=1)).ravel()
    return out


def _hash_keys(ks: pd.Series) -> np.ndarray:
    """64-bit hash of key strings: key tables of 30M+ rows fit in RAM (strings do not)."""
    return pd.util.hash_pandas_object(ks.astype(str), index=False).to_numpy(dtype=np.uint64)


def _get_acronyms(tok_series: pd.Series) -> pd.Series:
    """Generate first-letter acronyms from pipe-joined token strings."""
    uniq = tok_series.dropna().unique()
    acr_map = {}
    for t in uniq:
        if t:
            parts = [p[0] for p in t.split("|") if p]
            if len(parts) >= 2:
                acr_map[t] = "".join(parts)
    return tok_series.map(acr_map).fillna("")


def _extract_keys_by_leg(df: pd.DataFrame) -> dict[int, list[pd.Series]]:
    """Extract blocking key series grouped by leg bit.
    
    Empty strings represent 'no key'.
    """
    core = df["name_core"].fillna("").astype(str).str.strip()
    numbers = df["address_numbers"].fillna("").astype(str)
    first_num = numbers.str.split("|").str[0].fillna("")
    tokens = df["name_tokens"].fillna("").astype(str)
    addr = df["address_norm"].fillna("").astype(str).str.strip()
    pin = df["address_pin"].fillna("").astype(str).str.strip()

    # Leg D: Exact / near-exact name keys -> LEG_EXACT_KEY
    k_core = core.where(core.str.len() >= 3, "")
    k_core_num = (core + "|" + first_num).where((first_num != "") & (core.str.len() >= 3), "")
    k_tokens = tokens.where(tokens.str.len() >= 3, "")
    k_acronym = _get_acronyms(tokens)

    # Address exact keys -> LEG_ADDRESS
    k_addr_full = addr.where(addr.str.len() >= 12, "")
    addr_words = addr.str.split().str[:2].str.join(" ")
    k_num_words = (first_num + "|" + addr_words).where((first_num != "") & (addr_words != ""), "")

    # Leg E: PIN / ZIP -> LEG_PIN
    k_pin = pin.where(pin.str.match(r"^\d{5,6}$"), "")

    # Leg F: Soundex / Metaphone on first token of name_core -> LEG_PHONETIC
    tok0 = core.str.split().str[0].fillna("")
    uniq_toks = [t for t in tok0.unique() if len(t) >= 3]
    sdx_map = {t: jellyfish.soundex(t) for t in uniq_toks}
    meta_map = {t: jellyfish.metaphone(t) for t in uniq_toks}
    k_soundex = tok0.map(sdx_map).fillna("")
    k_metaphone = tok0.map(meta_map).fillna("")

    return {
        LEG_EXACT_KEY: [k_core, k_core_num, k_tokens, k_acronym],
        LEG_ADDRESS: [k_addr_full, k_num_words],
        LEG_PIN: [k_pin],
        LEG_PHONETIC: [k_soundex, k_metaphone],
    }


def _build_key_tables(gal: pd.DataFrame, bucket_cap: int = 200, top_k_bucket: int = 30) -> pd.DataFrame:
    """Build key tables for gallery records with bucket cap."""
    keys_by_leg = _extract_keys_by_leg(gal)
    tables = []
    n_gal = len(gal)
    g_indices = np.arange(n_gal, dtype=np.int32)
    for leg_bit, key_list in keys_by_leg.items():
        for ks in key_list:
            arr = ks.to_numpy()
            valid = arr != ""
            if not np.any(valid):
                continue
            t = pd.DataFrame({
                "key": _hash_keys(ks[valid]),
                "g": g_indices[valid],
                "leg": np.int32(leg_bit),
            })
            # Bucket cap: drop oversized buckets (> 200)
            counts = t.groupby("key")["g"].transform("count")
            t = t[counts <= bucket_cap]
            # Cap at top_k_bucket per bucket
            t = t.groupby("key").head(top_k_bucket)
            tables.append(t)
    if tables:
        return pd.concat(tables, ignore_index=True)
    return pd.DataFrame(columns=["key", "g", "leg"])


def _query_key_tables(q: pd.DataFrame, gal_key_table: pd.DataFrame, top_k_per_query: int = 30) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Query key tables for a set of S1 records."""
    if gal_key_table.empty:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int32)
    q_keys_by_leg = _extract_keys_by_leg(q)
    q_tables = []
    n_q = len(q)
    s1_indices = np.arange(n_q, dtype=np.int32)
    for leg_bit, key_list in q_keys_by_leg.items():
        for ks in key_list:
            arr = ks.to_numpy()
            valid = arr != ""
            if not np.any(valid):
                continue
            qt = pd.DataFrame({
                "key": _hash_keys(ks[valid]),
                "s1": s1_indices[valid],
                "leg": np.int32(leg_bit),
            })
            q_tables.append(qt)
    if not q_tables:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int32)
    q_key_df = pd.concat(q_tables, ignore_index=True)
    m = q_key_df.merge(gal_key_table, on=["key", "leg"], how="inner")
    if m.empty:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int32)
    # Cap candidate count per (s1, leg)
    m = m.groupby(["s1", "leg"]).head(top_k_per_query)
    return m["s1"].to_numpy(np.int64), m["g"].to_numpy(np.int64), m["leg"].to_numpy(np.int32)


# Terms present in more than this many gallery records are dropped from the TF-IDF legs.
# A fractional cap alone makes the retrieval cost grow quadratically with the gallery
# (each query touches a fixed *fraction* of the index); an absolute cap bounds it.
MAX_DF_ABS = int(os.environ.get("BLOCK_MAX_DF_ABS", "10000"))
# Phonetic leg (word TF-IDF over Metaphone codes of name tokens); 0 disables the leg.
PHONETIC_TOPN = int(os.environ.get("BLOCK_PHONETIC_TOPN", "0"))
PHONETIC_THR = float(os.environ.get("BLOCK_PHONETIC_THR", "0.5"))
PHONETIC_WEIGHT = float(os.environ.get("BLOCK_PHONETIC_WEIGHT", "0.0"))


def _phonetic_docs(name_core: pd.Series) -> pd.Series:
    """Metaphone code per token, space-joined; codes computed once per unique token."""
    s = name_core.fillna("").astype(str)
    uniq = pd.unique(pd.Series(" ".join(s.tolist()).split()))
    code = {}
    for t in uniq:
        if t.isdigit():
            code[t] = t
        else:
            try:
                code[t] = jellyfish.metaphone(t) or t
            except Exception:  # noqa: BLE001
                code[t] = t
    return s.map(lambda x: " ".join(code.get(t, t) for t in x.split()) if x else "")


def _max_df(n_gal: int, frac: float):
    if n_gal <= 1000:
        return 1.0
    return max(2, min(int(frac * n_gal), MAX_DF_ABS))


# Learned pruner (stage 5 of the plan): a LightGBM ranker over cheap pair features re-orders the
# union (top UNION_K by heuristic) into the final top-K. Active when the model file exists.
UNION_K = int(os.environ.get("BLOCK_UNION_K", "80"))
RETURN_PRUNER_FEATURES = os.environ.get("BLOCK_RETURN_PRUNER_FEATURES") == "1"
PRUNER_PATH = Path(os.environ.get("BLOCK_PRUNER", str(S.MODELS_DIR / "pruner.txt")))
PRUNER_FEATURES = ["word_cos", "char_cos", "n_key_legs", "f_exact", "f_addr", "f_pin", "f_bidir", "f_word", "f_char",
                   "name_tsr", "addr_tsr", "name_ratio", "house_eq", "house_conflict", "name_ntok_s1", "name_ntok_g",
                   "addr_ntok_g", "g_best_ratio", "g_rank"]
_PRUNER = None
_PRUNER_CHECKED = False


def _get_pruner():
    global _PRUNER, _PRUNER_CHECKED
    if not _PRUNER_CHECKED:
        _PRUNER_CHECKED = True
        if os.environ.get("BLOCK_NO_PRUNER") != "1" and PRUNER_PATH.exists():
            import lightgbm as lgb
            _PRUNER = lgb.Booster(model_file=str(PRUNER_PATH))
            print(f"    pruner loaded from {PRUNER_PATH}", flush=True)
    return _PRUNER


def _pair_cheap_features(P, s1_chunk, s1_u, g_u, legs_u, word_cos, char_cos):
    """Cheap features for union pairs (arrays aligned with s1_u/g_u)."""
    from rapidfuzz import fuzz
    from rapidfuzz.process import cpdist
    q_name = s1_chunk["name_norm"].fillna("").astype(str).to_numpy(dtype=object)
    q_addr = s1_chunk["address_norm"].fillna("").astype(str).to_numpy(dtype=object)
    q_first = s1_chunk["address_numbers"].fillna("").astype(str).str.split("|").str[0].fillna("").to_numpy(dtype=object)
    a_name = q_name[s1_u].tolist(); b_name = P.gal_name[g_u].tolist()
    a_addr = q_addr[s1_u].tolist(); b_addr = P.gal_addr[g_u].tolist()
    f = {}
    f["word_cos"] = word_cos.astype(np.float32)
    f["char_cos"] = char_cos.astype(np.float32)
    n_key = np.zeros(len(legs_u), dtype=np.float32)
    for bit in (LEG_EXACT_KEY, LEG_ADDRESS, LEG_PIN):
        n_key += (legs_u & bit) > 0
    f["n_key_legs"] = n_key
    f["f_exact"] = ((legs_u & LEG_EXACT_KEY) > 0).astype(np.float32)
    f["f_addr"] = ((legs_u & LEG_ADDRESS) > 0).astype(np.float32)
    f["f_pin"] = ((legs_u & LEG_PIN) > 0).astype(np.float32)
    f["f_bidir"] = ((legs_u & LEG_BIDIR) > 0).astype(np.float32)
    f["f_word"] = ((legs_u & LEG_NAME_TFIDF) > 0).astype(np.float32)
    f["f_char"] = ((legs_u & LEG_CHAR_NGRAM) > 0).astype(np.float32)
    f["name_tsr"] = cpdist(a_name, b_name, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32) / 100.0
    f["addr_tsr"] = cpdist(a_addr, b_addr, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32) / 100.0
    f["name_ratio"] = cpdist(a_name, b_name, scorer=fuzz.ratio, workers=-1, dtype=np.float32) / 100.0
    ha = q_first[s1_u]; hb = P.gal_first[g_u]
    f["house_eq"] = ((ha == hb) & (ha != "")).astype(np.float32)
    f["house_conflict"] = ((ha != hb) & (ha != "") & (hb != "")).astype(np.float32)
    q_ntok = s1_chunk["name_norm"].fillna("").astype(str).str.count(" ").to_numpy(np.float32) + 1
    f["name_ntok_s1"] = q_ntok[s1_u]
    f["name_ntok_g"] = P.gal_name_ntok[g_u]
    f["addr_ntok_g"] = P.gal_addr_ntok[g_u]
    # chunk-local gallery-side competition: this pair's heuristic vs the best S1 for the same candidate
    heur = word_cos + char_cos + 0.25 * np.minimum(n_key, 3)
    df = pd.DataFrame({"g": g_u, "h": heur})
    gmax = df.groupby("g")["h"].transform("max").to_numpy(np.float32)
    f["g_best_ratio"] = np.where(gmax > 0, heur / np.maximum(gmax, 1e-6), 1.0).astype(np.float32)
    f["g_rank"] = df.groupby("g")["h"].rank(ascending=False, method="min").to_numpy(np.float32)
    return f


class _PartitionIndex:
    """Gallery-side structures for one country, fitted once and queried in S1 chunks."""

    def __init__(self, gal_part: pd.DataFrame, threads: int):
        n_gal = len(gal_part)
        gal_word_doc = (gal_part["name_norm"].fillna("").astype(str) + " " + gal_part["address_norm"].fillna("").astype(str)).str.strip()
        gal_char_doc = gal_part["name_norm"].fillna("").astype(str).str.replace(" ", "", regex=False)
        # Leg A: word TF-IDF
        self.word_vec = TfidfVectorizer(token_pattern=r"\S+", lowercase=False, sublinear_tf=True, min_df=1,
                                        max_df=_max_df(n_gal, 0.02), dtype=np.float32)
        self.G_word = self.word_vec.fit_transform(gal_word_doc).tocsr()
        self.G_word_T = self.G_word.T.tocsr()
        # Leg B: char 3-4-gram TF-IDF on space-less name
        self.char_vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 4), lowercase=False, sublinear_tf=True,
                                        min_df=2 if n_gal > 100 else 1, max_df=_max_df(n_gal, 0.01), dtype=np.float32)
        self.G_char = self.char_vec.fit_transform(gal_char_doc).tocsr()
        self.G_char_T = self.G_char.T.tocsr()
        # Leg F (phonetic): word TF-IDF over the Metaphone code of every name token. Catches
        # transliteration spellings ("sauth phuds praivet" ~ "south foods private") that no
        # character n-gram or exact key survives.
        self.phon_vec = None
        if PHONETIC_TOPN > 0:
            self.phon_vec = TfidfVectorizer(token_pattern=r"\S+", lowercase=False, sublinear_tf=True, min_df=1,
                                            max_df=_max_df(n_gal, 0.02), dtype=np.float32)
            self.G_phon = self.phon_vec.fit_transform(_phonetic_docs(gal_part["name_core"])).tocsr()
            self.G_phon_T = self.G_phon.T.tocsr()
            self.phon_vec.stop_words_ = None
        # sklearn keeps every pruned term (min_df/max_df) in stop_words_: millions of strings here
        self.word_vec.stop_words_ = None
        self.char_vec.stop_words_ = None
        # Legs D, E, F: key tables
        self.key_tables = _build_key_tables(gal_part, bucket_cap=200, top_k_bucket=30)
        self.gal_ids = gal_part["entity_id"].to_numpy()
        self.gal_name = gal_part["name_norm"].fillna("").astype(str).to_numpy(dtype=object)
        self.gal_addr = gal_part["address_norm"].fillna("").astype(str).to_numpy(dtype=object)
        self.gal_first = gal_part["address_numbers"].fillna("").astype(str).str.split("|").str[0].fillna("").to_numpy(dtype=object)
        self.gal_name_ntok = gal_part["name_norm"].fillna("").astype(str).str.count(" ").to_numpy(np.float32) + 1
        self.gal_addr_ntok = gal_part["address_norm"].fillna("").astype(str).str.count(" ").to_numpy(np.float32) + 1
        self.n_gal = n_gal
        self.threads = threads

    def transform(self, s1_part: pd.DataFrame):
        word_doc = (s1_part["name_norm"].fillna("").astype(str) + " " + s1_part["address_norm"].fillna("").astype(str)).str.strip()
        char_doc = s1_part["name_norm"].fillna("").astype(str).str.replace(" ", "", regex=False)
        Q_phon = self.phon_vec.transform(_phonetic_docs(s1_part["name_core"])).tocsr() if self.phon_vec is not None else None
        return self.word_vec.transform(word_doc).tocsr(), self.char_vec.transform(char_doc).tocsr(), Q_phon

    def bidirectional(self, Q_word_all):
        """Leg C over the whole partition: gallery records as queries against the S1 index."""
        C = sp_matmul_topn(self.G_word, Q_word_all.T.tocsr(), top_n=3, threshold=0.30, sort=True, n_threads=self.threads).tocoo()
        order = np.argsort(C.col, kind="stable")
        return C.col.astype(np.int64)[order], C.row.astype(np.int64)[order]


def _query_chunk(P: _PartitionIndex, s1_chunk: pd.DataFrame, offset: int, Q_word, Q_char,
                 bidir_s1: np.ndarray, bidir_g: np.ndarray, country: str, Q_phon=None) -> pd.DataFrame:
    """All legs for one chunk of S1 rows (partition-local indices offset..offset+len)."""
    n_gal = P.n_gal
    nq = len(s1_chunk)
    parts_s1, parts_g, parts_legs = [], [], []

    C_word = sp_matmul_topn(Q_word, P.G_word_T, top_n=40, threshold=0.15, sort=True, n_threads=P.threads).tocoo()
    parts_s1.append(C_word.row.astype(np.int64)); parts_g.append(C_word.col.astype(np.int64))
    parts_legs.append(np.full(len(C_word.row), LEG_NAME_TFIDF, dtype=np.int32))

    C_char = sp_matmul_topn(Q_char, P.G_char_T, top_n=40, threshold=0.30, sort=True, n_threads=P.threads).tocoo()
    parts_s1.append(C_char.row.astype(np.int64)); parts_g.append(C_char.col.astype(np.int64))
    parts_legs.append(np.full(len(C_char.row), LEG_CHAR_NGRAM, dtype=np.int32))

    if Q_phon is not None and PHONETIC_TOPN > 0:
        C_phon = sp_matmul_topn(Q_phon, P.G_phon_T, top_n=PHONETIC_TOPN, threshold=PHONETIC_THR, sort=True,
                                n_threads=P.threads).tocoo()
        parts_s1.append(C_phon.row.astype(np.int64)); parts_g.append(C_phon.col.astype(np.int64))
        parts_legs.append(np.full(len(C_phon.row), LEG_PHONETIC, dtype=np.int32))

    lo, hi = np.searchsorted(bidir_s1, offset), np.searchsorted(bidir_s1, offset + nq)
    if hi > lo:
        parts_s1.append(bidir_s1[lo:hi] - offset); parts_g.append(bidir_g[lo:hi])
        parts_legs.append(np.full(hi - lo, LEG_BIDIR, dtype=np.int32))

    key_s1, key_g, key_bits = _query_key_tables(s1_chunk, P.key_tables, top_k_per_query=30)
    if len(key_s1) > 0:
        parts_s1.append(key_s1); parts_g.append(key_g); parts_legs.append(key_bits)

    all_s1 = np.concatenate(parts_s1); all_g = np.concatenate(parts_g); all_legs = np.concatenate(parts_legs)
    if len(all_s1) == 0:
        return pd.DataFrame()

    pair_keys = all_s1 * n_gal + all_g
    order = np.argsort(pair_keys, kind="stable")
    pair_keys = pair_keys[order]; all_legs = all_legs[order]
    uniq_keys, start_indices = np.unique(pair_keys, return_index=True)
    legs_u = np.bitwise_or.reduceat(all_legs, start_indices)
    s1_u = (uniq_keys // n_gal).astype(np.int32)
    g_u = (uniq_keys % n_gal).astype(np.int32)

    word_cos = rowwise_dot(Q_word, P.G_word, s1_u, g_u)
    char_cos = rowwise_dot(Q_char, P.G_char, s1_u, g_u)
    phon_cos = rowwise_dot(Q_phon, P.G_phon, s1_u, g_u) if Q_phon is not None else np.zeros(len(s1_u), np.float32)

    # Ranking score for EVERY pair from the real cosines plus a small bonus per key leg and
    # for a bidirectional hit. (A fixed 0.5 placeholder for key-only hits cost 8 points of recall.)
    n_key_legs = np.zeros(len(legs_u), dtype=np.float32)
    for bit in (LEG_EXACT_KEY, LEG_ADDRESS, LEG_PIN):
        n_key_legs += (legs_u & bit) > 0
    blocking_score = (word_cos + char_cos + PHONETIC_WEIGHT * phon_cos + 0.25 * np.minimum(n_key_legs, 3)
                      + 0.15 * ((legs_u & LEG_BIDIR) > 0)).astype(np.float64)

    def _topk(score, k):
        order = np.lexsort((-score, s1_u))
        first = np.r_[0, np.flatnonzero(np.diff(s1_u[order])) + 1]
        sizes = np.diff(np.r_[first, len(order)])
        rank = np.arange(len(order)) - np.repeat(first, sizes)
        return order[rank < k]

    pruner = _get_pruner()
    want_feats = pruner is not None or RETURN_PRUNER_FEATURES
    feats = None
    if want_feats:
        # union top UNION_K by heuristic, then cheap features on those pairs only
        keep = _topk(blocking_score, max(UNION_K, S.MAX_CANDIDATES_PER_S1))
        s1_u, g_u, legs_u, blocking_score = s1_u[keep], g_u[keep], legs_u[keep], blocking_score[keep]
        word_cos, char_cos = word_cos[keep], char_cos[keep]
        feats = _pair_cheap_features(P, s1_chunk, s1_u, g_u, legs_u, word_cos, char_cos)
        if pruner is not None:
            X = np.column_stack([feats[c] for c in PRUNER_FEATURES]).astype(np.float32)
            blocking_score = pruner.predict(X).astype(np.float64)
    # Final top-K cut per S1 BEFORE materialising string ids (keeps memory flat at full scale)
    keep = _topk(blocking_score, S.MAX_CANDIDATES_PER_S1)
    s1_u, g_u, legs_u, blocking_score = s1_u[keep], g_u[keep], legs_u[keep], blocking_score[keep]
    if feats is not None:
        feats = {c: v[keep] for c, v in feats.items()}

    s1_ids = s1_chunk["entity_id"].to_numpy()[s1_u]
    cand_ids = P.gal_ids[g_u]
    extra = {f"pf_{c}": v for c, v in feats.items()} if (feats is not None and RETURN_PRUNER_FEATURES) else {}
    return pd.DataFrame({
        "source1_entity_id": s1_ids,
        "candidate_entity_id": cand_ids,
        "candidate_source": pd.Series(cand_ids).str[:2].to_numpy(),
        "country": country,
        "name_tfidf_block_hit": (legs_u & LEG_NAME_TFIDF) > 0,
        "char_ngram_block_hit": (legs_u & LEG_CHAR_NGRAM) > 0,
        "address_block_hit": (legs_u & LEG_ADDRESS) > 0,
        "exact_key_hit": (legs_u & LEG_EXACT_KEY) > 0,
        "pin_block_hit": (legs_u & LEG_PIN) > 0,
        "phonetic_block_hit": (legs_u & LEG_PHONETIC) > 0,
        "embedding_block_hit": False,
        "bidirectional_block_hit": (legs_u & LEG_BIDIR) > 0,
        "num_legs_retrieved": np.unpackbits(legs_u.astype(np.uint8)[:, None], axis=1).sum(axis=1).astype(np.int64),
        "blocking_score": blocking_score,
        **extra,
    })


def generate_candidates_for_partition(
    s1_part: pd.DataFrame,
    gal_part: pd.DataFrame,
    country: str,
    threads: int = 8,
    chunk_size: int = 100_000,
) -> pd.DataFrame:
    """Generate candidate pairs for a single country partition across all legs.

    The gallery index is fitted once; S1 rows are queried in chunks of `chunk_size` so memory
    stays flat for partitions with >1M Source 1 records.
    """
    n_s1 = len(s1_part)
    if n_s1 == 0 or len(gal_part) == 0:
        return pd.DataFrame()
    P = _PartitionIndex(gal_part, threads)
    Q_word_all, Q_char_all, Q_phon_all = P.transform(s1_part)
    bidir_s1, bidir_g = P.bidirectional(Q_word_all)
    parts = []
    for offset in range(0, n_s1, chunk_size):
        chunk = s1_part.iloc[offset:offset + chunk_size].reset_index(drop=True)
        parts.append(_query_chunk(P, chunk, offset, Q_word_all[offset:offset + chunk_size],
                                  Q_char_all[offset:offset + chunk_size], bidir_s1, bidir_g, country,
                                  Q_phon=None if Q_phon_all is None else Q_phon_all[offset:offset + chunk_size]))
        print(f"  [{country}] blocked {min(offset + chunk_size, n_s1):,}/{n_s1:,} S1 rows", flush=True)
    parts = [x for x in parts if not x.empty]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def generate_candidates(s1: pd.DataFrame, gallery: pd.DataFrame, ground_truth: dict[str, set[str]] | None = None) -> pd.DataFrame:
    """Contract 1 frames in -> Contract 2 frame out (is_true_match only when ground_truth given)."""
    countries = s1["country"].dropna().unique()
    partitions = []
    for c in countries:
        cols = [x for x in BLOCKING_COLS if x in s1.columns]
        s1_c = s1.loc[s1["country"] == c, cols].reset_index(drop=True)
        gal_c = gallery.loc[gallery["country"] == c, cols].reset_index(drop=True)
        if s1_c.empty or gal_c.empty:
            continue
        part_candidates = generate_candidates_for_partition(s1_c, gal_c, str(c))
        if not part_candidates.empty:
            partitions.append(part_candidates)

    if partitions:
        out = pd.concat(partitions, ignore_index=True)
    else:
        cols = S.CANDIDATE_TRAIN_COLS if ground_truth is not None else S.CANDIDATE_COLS
        return pd.DataFrame(columns=cols)

    # Deduplicate in case of any overlap
    out = out.drop_duplicates(["source1_entity_id", "candidate_entity_id"]).reset_index(drop=True)

    if ground_truth is not None:
        # Join ground truth labels for train
        is_true = [
            cand in ground_truth.get(s1_id, set())
            for s1_id, cand in zip(out["source1_entity_id"], out["candidate_entity_id"])
        ]
        out["is_true_match"] = is_true
        run_recall_diagnostic(out, ground_truth, s1)
        expected_cols = S.CANDIDATE_TRAIN_COLS
    else:
        expected_cols = S.CANDIDATE_COLS

    return out[expected_cols]


def run_recall_diagnostic(candidates_df: pd.DataFrame, ground_truth: dict[str, set[str]], s1_df: pd.DataFrame) -> None:
    """Compute and print blocking recall report per country and overall."""
    s1_entities = set(s1_df["entity_id"].dropna().unique())
    total_truth_pairs = sum(len(matches) for s1_id, matches in ground_truth.items() if s1_id in s1_entities)
    
    if total_truth_pairs == 0:
        print("Diagnostic: Ground truth contains 0 true matches for this split.")
        return

    # Vectorised (a Python set of all candidate pairs does not fit in RAM at full scale)
    truth_df = pd.DataFrame(
        [(s, m) for s, ms in ground_truth.items() if s in s1_entities for m in ms],
        columns=["source1_entity_id", "candidate_entity_id"],
    )
    hit_df = truth_df.merge(candidates_df[["source1_entity_id", "candidate_entity_id"]], how="inner")
    recalled_matches = len(hit_df)

    overall_recall = recalled_matches / total_truth_pairs
    avg_cands = len(candidates_df) / max(1, len(s1_entities))
    max_cands = candidates_df.groupby("source1_entity_id").size().max() if not candidates_df.empty else 0

    print("\n" + "=" * 64)
    print("           BLOCKING RECALL DIAGNOSTIC (STAGE 4)")
    print("=" * 64)
    print(f"Total S1 entities           : {len(s1_entities):,}")
    print(f"Total Ground Truth pairs    : {total_truth_pairs:,}")
    print(f"Candidate pairs generated   : {len(candidates_df):,} (avg {avg_cands:.2f}, max {max_cands} per S1)")
    print(f"Recalled Ground Truth pairs : {recalled_matches:,}")
    print(f"Overall Blocking Recall     : {overall_recall:.2%}")
    print("-" * 64)
    print("Per-Country Breakdown:")

    for country in sorted(s1_df["country"].dropna().unique()):
        c_s1 = set(s1_df[s1_df["country"] == country]["entity_id"])
        c_truth = sum(len(ground_truth.get(s, set())) for s in c_s1)
        c_pairs = candidates_df[candidates_df["country"] == country]
        c_recalled = len(hit_df[hit_df["source1_entity_id"].isin(c_s1)])
        c_recall = (c_recalled / c_truth) if c_truth > 0 else 1.0
        c_avg = len(c_pairs) / max(1, len(c_s1))
        print(f"  {country:10s} | Truth: {c_truth:6,d} | Recalled: {c_recalled:6,d} | Recall: {c_recall:6.2%} | Avg Cands: {c_avg:5.2f}")
    print("=" * 64 + "\n", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Stage 3-5: blocking / candidate generation")
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--normalized-dir", default=str(S.NORMALIZED_DIR))
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    suffix = "" if args.split == "train" else "_test"
    nd = Path(args.normalized_dir)
    s1_path = nd / S.NORMALIZED_FILE.format(n=1, suffix=suffix)
    if not s1_path.exists():
        raise FileNotFoundError(f"Source 1 normalized file not found: {s1_path}")
    s1 = pd.read_parquet(s1_path, columns=BLOCKING_COLS)

    gal_dfs = []
    for n in (2, 3):
        p = nd / S.NORMALIZED_FILE.format(n=n, suffix=suffix)
        if p.exists():
            gal_dfs.append(pd.read_parquet(p, columns=BLOCKING_COLS))
    gal = pd.concat(gal_dfs, ignore_index=True) if gal_dfs else pd.DataFrame(columns=S.NORMALIZED_COLS)

    gt = None
    if args.split == "train":
        gt_path = S.dataset_dir() / "train" / "train_ground_truth.tsv"
        if not gt_path.exists():
            sample_gt = S.SAMPLES_DIR / "train_ground_truth.tsv"
            if sample_gt.exists():
                gt_path = sample_gt
        if gt_path.exists():
            gt = load_ground_truth(gt_path)

    out = generate_candidates(s1, gal, gt)
    expected = S.CANDIDATE_TRAIN_COLS if args.split == "train" else S.CANDIDATE_COLS
    assert list(out.columns) == expected, f"Contract 2 column mismatch: {list(out.columns)} vs {expected}"
    if not out.empty:
        assert out.groupby("source1_entity_id").size().max() <= S.MAX_CANDIDATES_PER_S1
        assert not out.duplicated(["source1_entity_id", "candidate_entity_id"]).any()

    path = Path(args.out) if args.out else (S.CANDIDATES_TRAIN if args.split == "train" else S.CANDIDATES_TEST)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    print(f"wrote {path}: {len(out):,} pairs")


if __name__ == "__main__":
    main()
