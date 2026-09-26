"""Unit tests for Stage 3-5 Candidate Generation (Person B).

Tests Contract 2 compliance, country partitioning, multi-leg retrieval,
adaptive-K pruning, and blocking recall diagnostic.
"""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from common import schema as S
import blocking


def make_test_data():
    s1_rows = [
        # US entity
        ("S1-100", "US", "Alpha Technologies Inc", "alpha technologies inc", "alpha technologies", "inc",
         "alpha|technologies", "alpha technologies inc", False, "100 Market St, San Francisco, CA 94105",
         "100 market street san francisco ca 94105", "100|94105", "94105"),
        # India entity with PIN
        ("S1-101", "India", "Tata Consultancy Services", "tata consultancy services", "tata consultancy services", "",
         "consultancy|services|tata", "tata consultancy services", False, "Plot 42 Cyber City, Gurgaon 122002",
         "plot 42 cyber city gurgaon 122002", "42|122002", "122002"),
        # India entity for phonetic / transliterated match
        ("S1-102", "India", "Sharma Enterprises", "sharma enterprises", "sharma", "enterprises",
         "sharma", "sharma enterprises", False, "12 MG Road Bangalore 560001",
         "12 mg road bangalore 560001", "12|560001", "560001"),
        # France entity (unseen country test)
        ("S1-103", "France", "Boulangerie Patisserie Paris", "boulangerie patisserie paris", "boulangerie patisserie paris", "",
         "boulangerie|paris|patisserie", "boulangerie patisserie paris", False, "15 Rue de Rivoli 75001 Paris",
         "15 rue de rivoli 75001 paris", "15|75001", "75001"),
    ]
    s1 = pd.DataFrame(s1_rows, columns=S.NORMALIZED_COLS)

    gal_rows = [
        # True match for S1-100 (US)
        ("S2-100", "US", "Alpha Technologies Corp", "alpha technologies corp", "alpha technologies", "corp",
         "alpha|technologies", "alpha technologies corp", False, "100 Market Street, San Francisco, CA 94105",
         "100 market street san francisco ca 94105", "100|94105", "94105"),
        # Hard negative for S1-100 (US)
        ("S3-100", "US", "Alpha Logistics Inc", "alpha logistics inc", "alpha logistics", "inc",
         "alpha|logistics", "alpha logistics inc", False, "500 Howard Street, San Francisco, CA 94105",
         "500 howard street san francisco ca 94105", "500|94105", "94105"),
        # True match for S1-101 (India)
        ("S2-101", "India", "TCS Ltd", "tcs ltd", "tcs", "ltd",
         "tcs", "tcs ltd", False, "Cyber City, Gurgaon 122002",
         "cyber city gurgaon 122002", "122002", "122002"),
        # Cross-country distractor with identical name in India (should NEVER match S1-100)
        ("S2-999", "India", "Alpha Technologies Inc", "alpha technologies inc", "alpha technologies", "inc",
         "alpha|technologies", "alpha technologies inc", False, "100 Market St, Mumbai",
         "100 market st mumbai", "100", ""),
        # True match for S1-102 (India) - phonetic variant
        ("S3-102", "India", "Scharma Enterprise", "scharma enterprise", "scharma", "enterprise",
         "scharma", "scharma enterprise", False, "12 MG Road Bangalore 560001",
         "12 mg road bangalore 560001", "12|560001", "560001"),
        # Match for France S1-103
        ("S2-103", "France", "Boulangerie Parisienne", "boulangerie parisienne", "boulangerie parisienne", "",
         "boulangerie|parisienne", "boulangerie parisienne", False, "15 Rue de Rivoli 75001 Paris",
         "15 rue de rivoli 75001 paris", "15|75001", "75001"),
    ]
    gal = pd.DataFrame(gal_rows, columns=S.NORMALIZED_COLS)

    gt = {
        "S1-100": {"S2-100"},
        "S1-101": {"S2-101"},
        "S1-102": {"S3-102"},
        "S1-103": {"S2-103"},
    }
    return s1, gal, gt


import unittest

class TestBlocking(unittest.TestCase):
    def test_contract2_schema_train(self):
        s1, gal, gt = make_test_data()
        out = blocking.generate_candidates(s1, gal, ground_truth=gt)
        self.assertEqual(list(out.columns), S.CANDIDATE_TRAIN_COLS)
        self.assertIn("is_true_match", out.columns)
        self.assertEqual(out["embedding_block_hit"].sum(), 0, "Leg G skipped: embedding_block_hit must be False everywhere")
        self.assertLessEqual(out.groupby("source1_entity_id").size().max(), S.MAX_CANDIDATES_PER_S1)
        self.assertFalse(out.duplicated(["source1_entity_id", "candidate_entity_id"]).any())
        self.assertTrue((out["num_legs_retrieved"] == out[S.BLOCK_FLAG_COLS].sum(axis=1)).all())

    def test_contract2_schema_test(self):
        s1, gal, _ = make_test_data()
        out = blocking.generate_candidates(s1, gal, ground_truth=None)
        self.assertEqual(list(out.columns), S.CANDIDATE_COLS)
        self.assertNotIn("is_true_match", out.columns)
        self.assertEqual(out["embedding_block_hit"].sum(), 0)

    def test_country_partitioning_no_leakage(self):
        s1, gal, gt = make_test_data()
        out = blocking.generate_candidates(s1, gal, ground_truth=gt)
        us_cands = out[out["source1_entity_id"] == "S1-100"]["candidate_entity_id"].tolist()
        self.assertNotIn("S2-999", us_cands, "Country partition violated: cross-country match found!")

        s1_country = s1.set_index("entity_id")["country"]
        gal_country = gal.set_index("entity_id")["country"]
        for _, row in out.iterrows():
            self.assertEqual(s1_country[row["source1_entity_id"]], row["country"])
            self.assertEqual(gal_country[row["candidate_entity_id"]], row["country"])

    def test_adaptive_k_pruning_cap(self):
        candidates = [f"cand_{i}" for i in range(50)]
        scores = [1.0 - 0.01 * i for i in range(50)]
        pruned = blocking.adaptive_k_prune(candidates, scores, k_min=5, gap=0.1, k_max=30)
        self.assertLessEqual(len(pruned), 30)
        self.assertGreaterEqual(len(pruned), 5)
        self.assertEqual(pruned, candidates[:11])

    def test_blocking_recall_diagnostic(self):
        s1, gal, gt = make_test_data()
        out = blocking.generate_candidates(s1, gal, ground_truth=gt)
        true_pairs = out[out["is_true_match"]]
        for _, row in true_pairs.iterrows():
            self.assertIn(row["candidate_entity_id"], gt[row["source1_entity_id"]])


if __name__ == "__main__":
    unittest.main()

