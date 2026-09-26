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
                "key": arr[valid],
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
                "key": arr[valid],
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


def generate_candidates_for_partition(
    s1_part: pd.DataFrame,
    gal_part: pd.DataFrame,
    country: str,
    threads: int = 4,
) -> pd.DataFrame:
    """Generate candidate pairs for a single country partition across all 7 legs."""
    n_s1 = len(s1_part)
    n_gal = len(gal_part)
    if n_s1 == 0 or n_gal == 0:
        return pd.DataFrame()

    # Documents for TF-IDF
    s1_word_doc = (s1_part["name_norm"].fillna("").astype(str) + " " + s1_part["address_norm"].fillna("").astype(str)).str.strip()
    gal_word_doc = (gal_part["name_norm"].fillna("").astype(str) + " " + gal_part["address_norm"].fillna("").astype(str)).str.strip()

    s1_char_doc = s1_part["name_norm"].fillna("").astype(str).str.replace(" ", "", regex=False)
    gal_char_doc = gal_part["name_norm"].fillna("").astype(str).str.replace(" ", "", regex=False)

    # Leg A: Word TF-IDF Vectorizer
    word_max_df = 0.02 if n_gal > 1000 else 1.0
    word_vec = TfidfVectorizer(
        token_pattern=r"\S+",
        lowercase=False,
        sublinear_tf=True,
        min_df=1,
        max_df=word_max_df,
        dtype=np.float32,
    )
    G_word = word_vec.fit_transform(gal_word_doc).tocsr()
    G_word_T = G_word.T.tocsr()
    Q_word = word_vec.transform(s1_word_doc).tocsr()

    # Leg B: Char 4-gram TF-IDF Vectorizer
    char_max_df = 0.01 if n_gal > 1000 else 1.0
    char_min_df = 2 if n_gal > 100 else 1
    char_vec = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 4),
        lowercase=False,
        sublinear_tf=True,
        min_df=char_min_df,
        max_df=char_max_df,
        dtype=np.float32,
    )
    G_char = char_vec.fit_transform(gal_char_doc).tocsr()
    G_char_T = G_char.T.tocsr()
    Q_char = char_vec.transform(s1_char_doc).tocsr()

    # Query Leg A: word TF-IDF top-40
    C_word = sp_matmul_topn(Q_word, G_word_T, top_n=40, threshold=0.15, sort=True, n_threads=threads).tocoo()
    leg_a_s1 = C_word.row.astype(np.int64)
    leg_a_g = C_word.col.astype(np.int64)
    leg_a_bits = np.full(len(leg_a_s1), LEG_NAME_TFIDF, dtype=np.int32)

    # Query Leg B: char 4-gram TF-IDF top-40
    C_char = sp_matmul_topn(Q_char, G_char_T, top_n=40, threshold=0.30, sort=True, n_threads=threads).tocoo()
    leg_b_s1 = C_char.row.astype(np.int64)
    leg_b_g = C_char.col.astype(np.int64)
    leg_b_bits = np.full(len(leg_b_s1), LEG_CHAR_NGRAM, dtype=np.int32)

    # Query Leg C: Bidirectional TF-IDF (gallery queries against S1 index)
    C_bidir = sp_matmul_topn(G_word, Q_word.T.tocsr(), top_n=3, threshold=0.30, sort=True, n_threads=threads).tocoo()
    leg_c_s1 = C_bidir.col.astype(np.int64)
    leg_c_g = C_bidir.row.astype(np.int64)
    leg_c_bits = np.full(len(leg_c_s1), LEG_BIDIR, dtype=np.int32)

    # Legs D, E, F: Exact keys, Address keys, PIN, Phonetic keys
    gal_key_tables = _build_key_tables(gal_part, bucket_cap=200, top_k_bucket=30)
    key_s1, key_g, key_bits = _query_key_tables(s1_part, gal_key_tables, top_k_per_query=30)

    # Union all retrieved legs
    parts_s1 = [leg_a_s1, leg_b_s1, leg_c_s1]
    parts_g = [leg_a_g, leg_b_g, leg_c_g]
    parts_legs = [leg_a_bits, leg_b_bits, leg_c_bits]
    if len(key_s1) > 0:
        parts_s1.append(key_s1)
        parts_g.append(key_g)
        parts_legs.append(key_bits)

    all_s1 = np.concatenate(parts_s1)
    all_g = np.concatenate(parts_g)
    all_legs = np.concatenate(parts_legs)

    if len(all_s1) == 0:
        return pd.DataFrame()

    # Deduplicate (s1, g) pairs and merge leg bitmasks
    pair_keys = all_s1 * n_gal + all_g
    order = np.argsort(pair_keys, kind="stable")
    pair_keys = pair_keys[order]
    all_legs = all_legs[order]

    uniq_keys, start_indices = np.unique(pair_keys, return_index=True)
    legs_u = np.bitwise_or.reduceat(all_legs, start_indices)
    s1_u = (uniq_keys // n_gal).astype(np.int32)
    g_u = (uniq_keys % n_gal).astype(np.int32)

    # Compute similarity cosines for blocking score
    word_cos = rowwise_dot(Q_word, G_word, s1_u, g_u)
    char_cos = rowwise_dot(Q_char, G_char, s1_u, g_u)

    has_tfidf = (legs_u & (LEG_NAME_TFIDF | LEG_CHAR_NGRAM | LEG_BIDIR)) > 0
    max_tfidf_cos = np.maximum(
        np.where((legs_u & (LEG_NAME_TFIDF | LEG_BIDIR)) > 0, word_cos, 0.0),
        np.where((legs_u & LEG_CHAR_NGRAM) > 0, char_cos, 0.0),
    )
    # Per prompt: max TF-IDF cosine if TF-IDF hit with real score; placeholder 0.5 for exact/phonetic hits
    blocking_score = np.where(has_tfidf & (max_tfidf_cos > 0.0), max_tfidf_cos, 0.5)
    is_key_hit = (legs_u & (LEG_EXACT_KEY | LEG_ADDRESS | LEG_PIN | LEG_PHONETIC)) > 0
    blocking_score = np.where(is_key_hit, np.maximum(blocking_score, 0.5), blocking_score)

    s1_ids = s1_part["entity_id"].iloc[s1_u].to_numpy()
    cand_ids = gal_part["entity_id"].iloc[g_u].to_numpy()

    part_df = pd.DataFrame({
        "source1_entity_id": s1_ids,
        "candidate_entity_id": cand_ids,
        "candidate_source": [cid[:2] for cid in cand_ids],
        "country": country,
        "name_tfidf_block_hit": (legs_u & LEG_NAME_TFIDF) > 0,
        "char_ngram_block_hit": (legs_u & LEG_CHAR_NGRAM) > 0,
        "address_block_hit": (legs_u & LEG_ADDRESS) > 0,
        "exact_key_hit": (legs_u & LEG_EXACT_KEY) > 0,
        "pin_block_hit": (legs_u & LEG_PIN) > 0,
        "phonetic_block_hit": (legs_u & LEG_PHONETIC) > 0,
        "embedding_block_hit": False,
        "bidirectional_block_hit": (legs_u & LEG_BIDIR) > 0,
        "num_legs_retrieved": [int(bin(int(l)).count("1")) for l in legs_u],
        "blocking_score": blocking_score.astype(np.float64),
    })

    # Adaptive-K prune to <= S.MAX_CANDIDATES_PER_S1
    part_df.sort_values(["source1_entity_id", "blocking_score"], ascending=[True, False], inplace=True, ignore_index=True)
    grp = part_df.groupby("source1_entity_id", sort=False)
    rank = grp.cumcount() + 1
    top1 = grp["blocking_score"].transform("max")
    score_gap = top1 - part_df["blocking_score"]
    keep = (rank <= S.MAX_CANDIDATES_PER_S1) & ((rank <= 5) | (score_gap <= 0.1))
    return part_df[keep].reset_index(drop=True)


def generate_candidates(s1: pd.DataFrame, gallery: pd.DataFrame, ground_truth: dict[str, set[str]] | None = None) -> pd.DataFrame:
    """Contract 1 frames in -> Contract 2 frame out (is_true_match only when ground_truth given)."""
    countries = s1["country"].dropna().unique()
    partitions = []
    for c in countries:
        s1_c = s1[s1["country"] == c].reset_index(drop=True)
        gal_c = gallery[gallery["country"] == c].reset_index(drop=True)
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

    cand_pairs_set = set(zip(candidates_df["source1_entity_id"], candidates_df["candidate_entity_id"]))
    recalled_matches = 0
    for s1_id, matches in ground_truth.items():
        if s1_id in s1_entities:
            for m in matches:
                if (s1_id, m) in cand_pairs_set:
                    recalled_matches += 1

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
        c_recalled = sum(1 for s, m in zip(c_pairs["source1_entity_id"], c_pairs["candidate_entity_id"]) if m in ground_truth.get(s, set()))
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
    s1 = pd.read_parquet(s1_path)

    gal_dfs = []
    for n in (2, 3):
        p = nd / S.NORMALIZED_FILE.format(n=n, suffix=suffix)
        if p.exists():
            gal_dfs.append(pd.read_parquet(p))
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
