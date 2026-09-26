"""Train the blocking pruner (stage 5): LightGBM ranker over cheap pair features of the union.

Collects labelled union pairs (top BLOCK_UNION_K by heuristic, pruner disabled) from several
dense city slices of the TRAIN data, trains LightGBM, saves models/pruner.txt, and reports
recall@30 before (heuristic) and after (pruner) on a held-out slice.

  python scripts/train_pruner.py --slices "India:delhi|dl" "India:mumbai|mh" "US:tx" "US:ca" \
      --holdout "India:bangalore|bengaluru|ka" --queries 40000
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ["BLOCK_RETURN_PRUNER_FEATURES"] = "1"
os.environ["BLOCK_NO_PRUNER"] = "1"

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import blocking  # noqa: E402
from common import schema as S  # noqa: E402


def load_slice(country: str, city_re: str, nd: Path):
    cols = blocking.BLOCKING_COLS
    pat = r"\b(?:" + city_re + r")\b"
    s1 = pd.read_parquet(nd / "source1_normalized.parquet", columns=cols)
    s1 = s1[(s1.country == country) & s1.address_norm.str.contains(pat, regex=True)].reset_index(drop=True)
    gal = pd.concat([pd.read_parquet(nd / f"source{n}_normalized.parquet", columns=cols) for n in (2, 3)], ignore_index=True)
    gal = gal[(gal.country == country) & gal.address_norm.str.contains(pat, regex=True)].reset_index(drop=True)
    return s1, gal


def union_with_labels(country, city_re, nd, gt, queries, union_k):
    s1, gal = load_slice(country, city_re, nd)
    gal_ids = set(gal.entity_id)
    q = s1.sample(min(queries, len(s1)), random_state=0).reset_index(drop=True)
    S.MAX_CANDIDATES_PER_S1 = union_k
    blocking.S.MAX_CANDIDATES_PER_S1 = union_k
    out = blocking.generate_candidates_for_partition(q, gal, country, threads=8, chunk_size=20000)
    truth = {(s, m) for s in q.entity_id for m in gt.get(s, ()) if m in gal_ids}
    out["label"] = [(s, c) in truth for s, c in zip(out.source1_entity_id, out.candidate_entity_id)]
    print(f"  {country}/{city_re}: S1={len(s1):,} gal={len(gal):,} queries={len(q):,} union={len(out):,} truth={len(truth):,}", flush=True)
    return out, len(truth)


def recall_at(out, score_col, k, n_truth):
    o = out.sort_values(["source1_entity_id", score_col], ascending=[True, False])
    r = o.groupby("source1_entity_id").cumcount() + 1
    return float(o[(r <= k) & o["label"]].shape[0] / max(1, n_truth))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices", nargs="+", default=["India:delhi|dl", "India:mumbai|mh", "US:tx", "US:ca"])
    ap.add_argument("--holdout", default="India:bangalore|bengaluru|ka")
    ap.add_argument("--queries", type=int, default=40000)
    ap.add_argument("--union-k", type=int, default=int(os.environ.get("BLOCK_UNION_K", "80")))
    ap.add_argument("--out", default=str(S.MODELS_DIR / "pruner.txt"))
    args = ap.parse_args()
    nd = S.NORMALIZED_DIR
    gt = blocking.load_ground_truth(S.dataset_dir() / "train" / "train_ground_truth.tsv")
    t0 = time.time()
    feats = [f"pf_{c}" for c in blocking.PRUNER_FEATURES]
    train_parts = []
    for spec in args.slices:
        country, city = spec.split(":", 1)
        out, _ = union_with_labels(country, city, nd, gt, args.queries, args.union_k)
        train_parts.append(out)
    tr = pd.concat(train_parts, ignore_index=True)
    hc, hcity = args.holdout.split(":", 1)
    ho, ho_truth = union_with_labels(hc, hcity, nd, gt, args.queries, args.union_k)
    print(f"collected {len(tr):,} train pairs (pos rate {tr.label.mean():.3f}), holdout {len(ho):,} in {time.time()-t0:.0f}s", flush=True)

    params = dict(objective="binary", learning_rate=0.08, num_leaves=63, min_child_samples=100, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.9, verbose=-1, seed=S.RANDOM_SEED, n_jobs=-1)
    dtr = lgb.Dataset(tr[feats].to_numpy(np.float32), label=tr.label.astype(int).to_numpy(), feature_name=blocking.PRUNER_FEATURES)
    dho = lgb.Dataset(ho[feats].to_numpy(np.float32), label=ho.label.astype(int).to_numpy(), reference=dtr)
    model = lgb.train(params, dtr, num_boost_round=600, valid_sets=[dho], callbacks=[lgb.early_stopping(40, verbose=False)])
    ho["pruner"] = model.predict(ho[feats].to_numpy(np.float32))
    for k in (30, 40, 50):
        print(f"holdout recall@{k}: heuristic {recall_at(ho, 'blocking_score', k, ho_truth):.2%} -> pruner {recall_at(ho, 'pruner', k, ho_truth):.2%}")
    print(f"union recall@{args.union_k}: {recall_at(ho, 'blocking_score', args.union_k, ho_truth):.2%}")
    imp = pd.Series(model.feature_importance("gain"), index=blocking.PRUNER_FEATURES).sort_values(ascending=False)
    print("top pruner features:", ", ".join(f"{k}={v:.0f}" for k, v in imp.head(8).items()))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    model.save_model(args.out)
    print(f"saved {args.out} (best_iter={model.best_iteration}) in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
