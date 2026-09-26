"""Stage 3-5 (Person B): Contract 1 normalized records -> Contract 2 candidate pairs.

Contract: see src/common/schema.py (CANDIDATE_COLS / CANDIDATE_TRAIN_COLS) and
team_task_split.md. Hard requirements:
  * country partitioning on every leg (open-set string label, never hard-coded)
  * <= S.MAX_CANDIDATES_PER_S1 candidates per Source 1 entity after adaptive-K pruning
  * blocking recall diagnostic against train_ground_truth.tsv printed before hand-off

CLI:
  python src/blocking.py --split train   # writes data/candidates/candidate_pairs_train.parquet
  python src/blocking.py --split test    # writes data/candidates/candidate_pairs_test.parquet

A complete working reference (country-partitioned word TF-IDF + char n-gram TF-IDF via
sparse_dot_topn, bidirectional pass, 5 exact-key legs, heuristic top-K cut, recall report)
lives in src/reference/blocking_ref.py. It uses its own column names; map them to the
contract when producing the output.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import schema as S  # noqa: E402


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


def generate_candidates(s1: pd.DataFrame, gallery: pd.DataFrame, ground_truth: dict | None = None) -> pd.DataFrame:
    """Contract 1 frames in -> Contract 2 frame out (is_true_match only when ground_truth given).

    TODO (Person B): implement the 7 legs + union + adaptive-K prune + recall diagnostic.
    """
    raise NotImplementedError("Person B: implement generate_candidates (see docstring / reference)")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Stage 3-5: blocking / candidate generation")
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--normalized-dir", default=str(S.NORMALIZED_DIR))
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    suffix = "" if args.split == "train" else "_test"
    nd = Path(args.normalized_dir)
    s1 = pd.read_parquet(nd / S.NORMALIZED_FILE.format(n=1, suffix=suffix))
    gal = pd.concat([pd.read_parquet(nd / S.NORMALIZED_FILE.format(n=n, suffix=suffix)) for n in (2, 3)], ignore_index=True)
    gt = None
    if args.split == "train":
        from decision import load_full_truth  # local import to keep the stub importable
        gt = load_full_truth(S.dataset_dir() / "train" / "train_ground_truth.tsv")
    out = generate_candidates(s1, gal, gt)
    expected = S.CANDIDATE_TRAIN_COLS if args.split == "train" else S.CANDIDATE_COLS
    assert list(out.columns) == expected, "Contract 2 column mismatch"
    assert out.groupby("source1_entity_id").size().max() <= S.MAX_CANDIDATES_PER_S1
    assert not out.duplicated(["source1_entity_id", "candidate_entity_id"]).any()
    path = Path(args.out) if args.out else (S.CANDIDATES_TRAIN if args.split == "train" else S.CANDIDATES_TEST)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    print(f"wrote {path}: {len(out):,} pairs")


if __name__ == "__main__":
    main()
