"""Shared schema contract for the whole team (see team_task_split.md).

Everyone imports column names from here so nobody typo's a column.
Nothing in this file may hard-code a country set: `country` is an open string label.
"""
from __future__ import annotations

from pathlib import Path

RANDOM_SEED = 42

# --------------------------------------------------------------------------- #
# Paths (relative to repo root)
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parents[2]


def dataset_dir() -> Path:
    """`dataset/` at repo root, or the unzipped student_resource layout as fallback."""
    for cand in (REPO_ROOT / "dataset", REPO_ROOT / "student_resource" / "student_resource" / "dataset"):
        if (cand / "train").is_dir():
            return cand
    return REPO_ROOT / "dataset"


DATA_DIR = REPO_ROOT / "data"
NORMALIZED_DIR = DATA_DIR / "normalized"      # Person A writes here
CANDIDATES_DIR = DATA_DIR / "candidates"      # Person B writes here
FEATURES_DIR = DATA_DIR / "features"          # Person C writes here
SPLITS_DIR = DATA_DIR / "splits"              # Person C writes here
MODELS_DIR = REPO_ROOT / "models"             # Person D writes here
OUTPUT_DIR = REPO_ROOT / "output"             # Person D writes here
SAMPLES_DIR = REPO_ROOT / "samples"

# Contract file names
NORMALIZED_FILE = "source{n}_normalized{suffix}.parquet"   # suffix '' for train, '_test' for test
CANDIDATES_TRAIN = CANDIDATES_DIR / "candidate_pairs_train.parquet"
CANDIDATES_TEST = CANDIDATES_DIR / "candidate_pairs_test.parquet"
FEATURES_TRAIN = FEATURES_DIR / "feature_matrix_train.parquet"
FEATURES_TEST = FEATURES_DIR / "feature_matrix_test.parquet"
SPLIT_TRAIN_IDS = SPLITS_DIR / "train_entity_ids.txt"
SPLIT_VAL_IDS = SPLITS_DIR / "val_entity_ids.txt"
SPLIT_VAL_US_IDS = SPLITS_DIR / "val_us_only_ids.txt"
SPLIT_VAL_INDIA_IDS = SPLITS_DIR / "val_india_only_ids.txt"
MODEL_PHASE1 = MODELS_DIR / "lgbm_phase1.txt"
MODEL_PHASE2 = MODELS_DIR / "lgbm_phase2.txt"
CALIBRATOR = MODELS_DIR / "isotonic_calibrator.pkl"
MATCHING_RESULTS = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_PAIRS_TSV = OUTPUT_DIR / "candidate_pairs.tsv"

# --------------------------------------------------------------------------- #
# Raw source columns
# --------------------------------------------------------------------------- #
RAW_COLS = ["entity_id", "business_name", "business_address", "country"]
GT_COLS = ["source1_entity_id", "matched_entity_ids"]

# --------------------------------------------------------------------------- #
# CONTRACT 1 - normalized records (Person A -> B, C)
# --------------------------------------------------------------------------- #
NORMALIZED_COLS = [
    "entity_id", "country", "name_raw", "name_norm", "name_core", "legal_suffix",
    "name_tokens", "name_transliterated", "is_non_latin", "address_raw", "address_norm",
    "address_numbers", "address_pin",
]
TOKEN_SEP = "|"

# --------------------------------------------------------------------------- #
# CONTRACT 2 - candidate pairs (Person B -> C, D)
# --------------------------------------------------------------------------- #
PAIR_KEY_COLS = ["source1_entity_id", "candidate_entity_id", "candidate_source", "country"]
BLOCK_FLAG_COLS = [
    "name_tfidf_block_hit", "char_ngram_block_hit", "address_block_hit", "exact_key_hit",
    "pin_block_hit", "phonetic_block_hit", "embedding_block_hit", "bidirectional_block_hit",
]
CANDIDATE_COLS = PAIR_KEY_COLS + BLOCK_FLAG_COLS + ["num_legs_retrieved", "blocking_score"]
CANDIDATE_TRAIN_COLS = CANDIDATE_COLS + ["is_true_match"]   # train file only
MAX_CANDIDATES_PER_S1 = 30

# --------------------------------------------------------------------------- #
# CONTRACT 3 - feature matrix (Person C -> D)
# --------------------------------------------------------------------------- #
FEATURE_PREFIXES = ("name__", "addr__", "cross__", "sem__", "comp__", "block__")
LABEL_COL = "label"
# Columns Person D relies on BY EXACT NAME (hard vetoes / diagnostics). Person C must ship these.
FEAT_HOUSE_CONFLICT = "addr__house_number_conflict"
FEAT_PIN_EXACT = "addr__pin_exact"
FEAT_ADDR_SIM = "addr__lev_ratio"
FEAT_NAME_SIM = "name__lev_ratio"
FEAT_EMB_COS = "sem__embedding_cosine"
FORBIDDEN_COMP_SUBSTRINGS = ("n_cand", "g_size", "cvc_", "count", "pool_size")


def feature_columns(columns) -> list[str]:
    """All model input columns: anything with a contract prefix."""
    return [c for c in columns if c.startswith(FEATURE_PREFIXES)]


def assert_no_raw_counts(columns) -> None:
    bad = [c for c in columns if c.startswith("comp__") and any(s in c for s in FORBIDDEN_COMP_SUBSTRINGS)]
    if bad:
        raise ValueError(f"density-sensitive comp__ features are forbidden: {bad}")


# --------------------------------------------------------------------------- #
# CONTRACT 4 - outputs (Person D)
# --------------------------------------------------------------------------- #
MATCHING_HEADER = ["source1_entity_id", "matched_entity_ids"]
CANDIDATE_HEADER = ["source1_entity_id", "candidate_entity_ids"]
