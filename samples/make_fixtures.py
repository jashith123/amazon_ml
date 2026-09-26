"""Hand-made fixtures for every contract so each person can develop without waiting.

    python samples/make_fixtures.py

Writes into samples/:
  source1_normalized.parquet, source2_normalized.parquet      (Contract 1, 6 rows each)
  candidate_pairs_train.parquet, candidate_pairs_test.parquet  (Contract 2)
  feature_matrix_train.parquet, feature_matrix_test.parquet    (Contract 3, synthetic ~300 S1)
  splits/*.txt, test_source1.tsv, train_ground_truth.tsv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import schema as S  # noqa: E402

OUT = Path(__file__).resolve().parent
rng = np.random.RandomState(S.RANDOM_SEED)


def contract1():
    rows = [
        # plain US row
        ("S1-1", "US", "Orelee's Barbershop", "orelees barbershop", "orelees barbershop", "", "barbershop|orelees",
         "orelees barbershop", False, "1795 Westchester Drive, High Point, NC", "1795 westchester drive high point nc", "1795", ""),
        # India row, Devanagari name
        ("S2-1", "India", "राम मार्केटिंग प्राइवेट लिमिटेड", "राम मार्केटिंग प्राइवेट लिमिटेड", "राम मार्केटिंग", "प्राइवेट लिमिटेड",
         "मार्केटिंग|राम", "ram marketing private limited", True, "KH NO. -570/13, NEW DELHI, Delhi", "kh no 570 13 new delhi delhi", "570|13", ""),
        # legal suffix
        ("S1-2", "India", "ABC Pvt. Ltd.", "abc pvt ltd", "abc", "pvt ltd", "abc", "abc pvt ltd", False,
         "21 MG Road, Bengaluru, Karnataka 560001", "21 mg road bengaluru karnataka 560001", "21|560001", "560001"),
        # no address
        ("S3-1", "India", "International South Consultants Private Ltd", "international south consultants private ltd",
         "international south consultants", "private ltd", "consultants|international|south", "international south consultants private ltd",
         False, "", "", "", ""),
        # PIN code only
        ("S2-2", "India", "Svl Tubes Pvt", "svl tubes pvt", "svl tubes", "pvt", "svl|tubes", "svl tubes pvt", False,
         "Chennai 600029", "chennai 600029", "600029", "600029"),
        # house number
        ("S3-2", "US", "Moyna's Coffee", "moynas coffee", "moynas coffee", "", "coffee|moynas", "moynas coffee", False,
         "1 Ivanhoe Ave, Cincinnati, Ohio", "1 ivanhoe avenue cincinnati ohio", "1", ""),
    ]
    df = pd.DataFrame(rows, columns=S.NORMALIZED_COLS)
    df[df.entity_id.str.startswith("S1")].to_parquet(OUT / "source1_normalized.parquet", index=False)
    df[~df.entity_id.str.startswith("S1")].to_parquet(OUT / "source2_normalized.parquet", index=False)
    return df


def contract2():
    flags = S.BLOCK_FLAG_COLS
    rows = [
        # obvious true match: many legs, high score
        ("S1-2", "S2-9", "S2", "India", 1, 1, 1, 1, 1, 0, 0, 1, 6, 0.95, True),
        # hard negative: name similar, address different
        ("S1-2", "S3-9", "S3", "India", 1, 1, 0, 0, 0, 1, 0, 0, 3, 0.55, False),
        # random negative
        ("S1-2", "S2-8", "S2", "India", 1, 0, 0, 0, 0, 0, 0, 0, 1, 0.20, False),
        # single-leg candidate
        ("S1-2", "S3-7", "S3", "India", 0, 0, 0, 0, 0, 1, 0, 0, 1, 0.50, False),
    ]
    df = pd.DataFrame(rows, columns=S.PAIR_KEY_COLS + flags + ["num_legs_retrieved", "blocking_score", "is_true_match"])
    for c in flags:
        df[c] = df[c].astype(bool)
    df.to_parquet(OUT / "candidate_pairs_train.parquet", index=False)
    df.drop(columns=["is_true_match"]).to_parquet(OUT / "candidate_pairs_test.parquet", index=False)
    return df


def synthetic_features(n_entities: int, prefix: str, with_label: bool):
    """Synthetic Contract-3 rows: ~4 candidates per S1, features correlated with the label."""
    recs = []
    truth = {}
    for i in range(n_entities):
        s1 = f"{prefix}{i:05d}"
        country = "US" if i % 2 == 0 else "India"
        n_true = rng.choice([0, 1, 2, 3], p=[0.08, 0.30, 0.40, 0.22])
        n_cand = n_true + rng.randint(1, 4)
        labels = np.array([1] * n_true + [0] * (n_cand - n_true))
        rng.shuffle(labels)
        truth[s1] = set()
        for j, lab in enumerate(labels):
            cid = f"S{2 + (j % 2)}-{i:05d}{j}"
            if lab:
                truth[s1].add(cid)
            base = 0.85 if lab else 0.35
            name_sim = np.clip(rng.normal(base, 0.12), 0, 1)
            addr_sim = np.clip(rng.normal(base, 0.15), 0, 1)
            recs.append({
                "source1_entity_id": s1, "candidate_entity_id": cid, "candidate_source": cid[:2], "country": country,
                "name__lev_ratio": name_sim, "name__jaro_winkler": np.clip(name_sim + rng.normal(0, .05), 0, 1),
                "name__token_jaccard": np.clip(name_sim - 0.1 + rng.normal(0, .1), 0, 1),
                "name__exact_core_match": float(name_sim > 0.95),
                "addr__lev_ratio": addr_sim, "addr__token_jaccard": np.clip(addr_sim - 0.1 + rng.normal(0, .1), 0, 1),
                "addr__house_number_conflict": float((not lab) and rng.rand() < 0.3),
                "addr__pin_exact": float(lab and rng.rand() < 0.5), "addr__both_have_address": 1.0,
                "cross__name_addr_product": name_sim * addr_sim, "cross__min": min(name_sim, addr_sim),
                "cross__max": max(name_sim, addr_sim), "cross__country_exact_match": 1.0,
                "sem__embedding_cosine": 0.0, "sem__embedding_rank": 0.0,
                "block__num_legs_retrieved": float(rng.randint(1, 4) + 2 * lab),
                "block__exact_key_hit": float(lab and rng.rand() < 0.6),
                "label": int(lab),
            })
    df = pd.DataFrame(recs)
    grp = df.groupby("source1_entity_id")["cross__name_addr_product"]
    df["comp__rank"] = grp.rank(ascending=False, method="first")
    df["comp__score_percentile"] = grp.rank(pct=True)
    df["comp__score_gap_top1"] = grp.transform("max") - df["cross__name_addr_product"]
    if not with_label:
        df = df.drop(columns=["label"])
    return df, truth


def contract3():
    tr, truth = synthetic_features(400, "S1-T", True)
    tr.to_parquet(OUT / "feature_matrix_train.parquet", index=False)
    te, _ = synthetic_features(60, "S1-X", False)
    te.to_parquet(OUT / "feature_matrix_test.parquet", index=False)
    ents = np.array(tr["source1_entity_id"].unique(), dtype=object)
    rng.shuffle(ents)
    n_val = int(0.2 * len(ents))
    val, train = ents[:n_val], ents[n_val:]
    sd = OUT / "splits"
    sd.mkdir(exist_ok=True)
    (sd / S.SPLIT_TRAIN_IDS.name).write_text("\n".join(train))
    (sd / S.SPLIT_VAL_IDS.name).write_text("\n".join(val))
    country = tr.drop_duplicates("source1_entity_id").set_index("source1_entity_id")["country"]
    (sd / S.SPLIT_VAL_US_IDS.name).write_text("\n".join(e for e in val if country[e] == "US"))
    (sd / S.SPLIT_VAL_INDIA_IDS.name).write_text("\n".join(e for e in val if country[e] == "India"))
    pd.DataFrame({"entity_id": te["source1_entity_id"].unique()}).assign(
        business_name="x", business_address="y", country="US")[S.RAW_COLS].to_csv(OUT / "test_source1.tsv", sep="\t", index=False)
    te[["source1_entity_id", "candidate_entity_id"]].assign(candidate_source="S2", country="US").to_parquet(
        OUT / "candidate_pairs_test_synth.parquet", index=False)
    pd.DataFrame({"source1_entity_id": list(truth), "matched_entity_ids": [",".join(sorted(v)) for v in truth.values()]}).to_csv(
        OUT / "train_ground_truth.tsv", sep="\t", index=False)


if __name__ == "__main__":
    contract1()
    contract2()
    contract3()
    print("fixtures written to", OUT)
