"""Dense-slice blocking diagnosis: reproduce full-data density on one city.

Random samples of S1 hide the real problem (a sampled entity does not face its real
neighbours). Taking EVERY record whose address mentions one city keeps the neighbourhood
intact, so recall@K here tracks the full run. Prints recall@K for the union, the share of true
pairs never retrieved by any leg, which legs found the low-ranked true pairs, and examples.

  python scripts/diagnose_blocking.py --country India --city "delhi|dl" --queries 60000
  python scripts/diagnose_blocking.py --country US --city "tx" --queries 60000

Requires data/normalized/*.parquet (train) and the ground truth. Uses src/blocking.py as-is,
so any change to a leg or the score shows up here in ~1-2 minutes.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import blocking  # noqa: E402
from common import schema as S  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--country", default="India")
    ap.add_argument("--city", default="delhi|dl", help="regex of address tokens that define the slice")
    ap.add_argument("--queries", type=int, default=60000)
    ap.add_argument("--union-k", type=int, default=300)
    ap.add_argument("--normalized-dir", default=str(S.NORMALIZED_DIR))
    ap.add_argument("--save", default=None, help="parquet path to save the union with ranks/labels")
    args = ap.parse_args()

    S.MAX_CANDIDATES_PER_S1 = args.union_k
    blocking.S.MAX_CANDIDATES_PER_S1 = args.union_k
    cols = blocking.BLOCKING_COLS
    pat = r"\b(?:" + args.city + r")\b"
    t0 = time.time()
    nd = Path(args.normalized_dir)
    s1 = pd.read_parquet(nd / "source1_normalized.parquet", columns=cols)
    s1 = s1[(s1.country == args.country) & s1.address_norm.str.contains(pat, regex=True)].reset_index(drop=True)
    gal = pd.concat([pd.read_parquet(nd / f"source{n}_normalized.parquet", columns=cols) for n in (2, 3)], ignore_index=True)
    gal = gal[(gal.country == args.country) & gal.address_norm.str.contains(pat, regex=True)].reset_index(drop=True)
    gt = blocking.load_ground_truth(S.dataset_dir() / "train" / "train_ground_truth.tsv")
    gal_ids = set(gal.entity_id)
    print(f"slice {args.country}/{args.city}: S1={len(s1):,} gallery={len(gal):,} ({time.time()-t0:.0f}s)", flush=True)

    q = s1.sample(min(args.queries, len(s1)), random_state=0).reset_index(drop=True)
    out = blocking.generate_candidates_for_partition(q, gal, args.country, threads=8, chunk_size=30000)
    q_truth = {(s, m) for s in q.entity_id for m in gt.get(s, ()) if m in gal_ids}
    out["true"] = [(s, c) in q_truth for s, c in zip(out.source1_entity_id, out.candidate_entity_id)]
    out.sort_values(["source1_entity_id", "blocking_score"], ascending=[True, False], inplace=True)
    out["rank"] = out.groupby("source1_entity_id").cumcount() + 1
    got = set(zip(out.source1_entity_id[out["true"]], out.candidate_entity_id[out["true"]]))
    never = q_truth - got
    print(f"union: {len(out):,} pairs ({len(out)/len(q):.1f}/S1) in {time.time()-t0:.0f}s")
    print(f"true pairs: {len(q_truth):,} | never retrieved: {len(never):,} ({len(never)/len(q_truth):.1%})")
    for k in (30, 40, 50, 75, 100, 150, args.union_k):
        r = out[(out["rank"] <= k) & out["true"]].shape[0] / max(1, len(q_truth))
        print(f"  recall@{k}: {r:.2%}")
    tp = out[out["true"]]
    flags = S.BLOCK_FLAG_COLS
    print("leg hit rate among true pairs, rank<=30 vs rank>30:")
    print(pd.DataFrame({"rank<=30": tp[tp["rank"] <= 30][flags].mean().round(3),
                        "rank>30": tp[tp["rank"] > 30][flags].mean().round(3)}).to_string())
    s1i = s1.set_index("entity_id"); gali = gal.set_index("entity_id")
    print("--- never-retrieved examples:")
    for s, m in list(never)[:8]:
        print(" S1:", s1i.loc[s, "name_norm"], "|", s1i.loc[s, "address_norm"][:60])
        print("  G:", gali.loc[m, "name_norm"], "|", gali.loc[m, "address_norm"][:60])
    print("--- true pairs ranked >30 examples:")
    for _, r in tp[tp["rank"] > 30].sample(min(6, (tp["rank"] > 30).sum()), random_state=1).iterrows():
        print(f" rank {r['rank']} score {r.blocking_score:.2f} S1:", s1i.loc[r.source1_entity_id, "name_norm"], "|", s1i.loc[r.source1_entity_id, "address_norm"][:50])
        print("   G:", gali.loc[r.candidate_entity_id, "name_norm"], "|", gali.loc[r.candidate_entity_id, "address_norm"][:50])
    if args.save:
        out.to_parquet(args.save, index=False)
        print("saved", args.save)


if __name__ == "__main__":
    main()
