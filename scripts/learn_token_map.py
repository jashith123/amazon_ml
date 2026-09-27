"""Learned token normalisation from training pairs (plan document, section 1).

For every matched (S1, gallery) pair in the training ground truth, align each gallery name
token that does not occur in the S1 name to the most similar S1 token (Jaro-Winkler >= thr).
Mappings seen often enough and dominant for their source token become a dictionary
(e.g. praivet->private, phuds->foods, tredimg->trading) that is applied to gallery names
before blocking and features. Only the provided training data is used.

  python scripts/learn_token_map.py --pairs 2000000 --out models/token_map.json
  python scripts/learn_token_map.py --eval            # recall effect on the Delhi slice

Output: JSON {gallery_token: s1_token}. blocking.py / features.py apply it when
BLOCK_TOKEN_MAP points at the file (or models/token_map.json exists).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz.distance import JaroWinkler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import schema as S  # noqa: E402


def learn(pairs_limit: int, thr: float, min_count: int, min_dominance: float, seed: int = S.RANDOM_SEED) -> dict[str, str]:
    t0 = time.time()
    nd = S.NORMALIZED_DIR
    s1 = pd.read_parquet(nd / "source1_normalized.parquet", columns=["entity_id", "name_norm"]).set_index("entity_id")["name_norm"]
    gal = pd.concat([pd.read_parquet(nd / f"source{n}_normalized.parquet", columns=["entity_id", "name_norm"]) for n in (2, 3)]).set_index("entity_id")["name_norm"]
    gt = pd.read_csv(S.dataset_dir() / "train" / "train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
    gt = gt[gt.matched_entity_ids != ""]
    ex = gt.assign(m=gt.matched_entity_ids.str.split(",")).explode("m")
    if len(ex) > pairs_limit:
        ex = ex.sample(pairs_limit, random_state=seed)
    a = s1.reindex(ex.source1_entity_id).fillna("").to_numpy(dtype=object)
    b = gal.reindex(ex.m).fillna("").to_numpy(dtype=object)
    print(f"aligning {len(ex):,} pairs ({time.time()-t0:.0f}s)", flush=True)
    counts: dict[str, Counter] = defaultdict(Counter)
    src_total: Counter = Counter()
    for x, y in zip(a, b):
        xs = x.split()
        if not xs:
            continue
        xset = set(xs)
        for t in y.split():
            if t in xset or len(t) < 3 or t.isdigit():
                continue
            best, best_s = None, 0.0
            for u in xs:
                if u.isdigit() or abs(len(u) - len(t)) > 4:
                    continue
                s = JaroWinkler.normalized_similarity(t, u)
                if s > best_s:
                    best, best_s = u, s
            if best is not None and best_s >= thr:
                counts[t][best] += 1
            src_total[t] += 1
    mapping = {}
    for t, c in counts.items():
        u, n = c.most_common(1)[0]
        if n >= min_count and n / src_total[t] >= min_dominance and u != t:
            mapping[t] = u
    print(f"learned {len(mapping):,} mappings from {len(counts):,} candidate tokens ({time.time()-t0:.0f}s)")
    return mapping


def apply_map(names: pd.Series, mapping: dict[str, str]) -> pd.Series:
    if not mapping:
        return names
    return names.fillna("").astype(str).map(lambda s: " ".join(mapping.get(t, t) for t in s.split()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=2_000_000)
    ap.add_argument("--thr", type=float, default=0.80)
    ap.add_argument("--min-count", type=int, default=5)
    ap.add_argument("--min-dominance", type=float, default=0.6)
    ap.add_argument("--out", default=str(S.MODELS_DIR / "token_map.json"))
    args = ap.parse_args()
    mapping = learn(args.pairs, args.thr, args.min_count, args.min_dominance)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(mapping, ensure_ascii=False, indent=0), encoding="utf-8")
    sample = sorted(mapping.items(), key=lambda kv: kv[0])[:0]
    common = Counter()
    print("examples:", ", ".join(f"{k}->{v}" for k, v in list(mapping.items())[:25]))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
