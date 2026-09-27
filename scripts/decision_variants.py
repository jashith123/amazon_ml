"""Produce several matching_results.tsv variants from saved pair probabilities, for the
leaderboard to arbitrate (multiple uploads allowed). Each variant is validated and zipped
into submissions/<name>_matching_results.zip for the team leader.

  python scripts/decision_variants.py --probs output/work_tokenmap/pair_probs.parquet \
      --variants "alpha1.0" "alpha0.7" "alpha0.53" "alpha0.4" "alpha0.53_floor0.3"

Variant syntax: alpha<a>[_floor<f>][_pdf]  (a = prior-shift odds multiplier, f = min prob)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import schema as S  # noqa: E402
from decision import decide, write_matching_results  # noqa: E402


def parse_variant(name: str):
    alpha, floor, formula = 1.0, 0.0, "full"
    for part in name.split("_"):
        if part.startswith("alpha"):
            alpha = float(part[5:])
        elif part.startswith("floor"):
            floor = float(part[5:])
        elif part == "pdf":
            formula = "pdf"
    return alpha, floor, formula


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", required=True)
    ap.add_argument("--variants", nargs="+", default=["alpha1.0", "alpha0.7", "alpha0.53", "alpha0.4"])
    ap.add_argument("--source1", default=str(S.dataset_dir() / "test" / "test_source1.tsv"))
    ap.add_argument("--out-root", default=str(S.OUTPUT_DIR / "variants"))
    ap.add_argument("--zip-dir", default=str(ROOT / "submissions"))
    args = ap.parse_args()
    base = pd.read_parquet(args.probs)
    s1_ids = pd.read_csv(args.source1, sep="\t", dtype=str, keep_default_na=False, usecols=["entity_id"])["entity_id"].tolist()
    print(f"{len(base):,} scored pairs, {len(s1_ids):,} entities")
    p0 = base["prob"].to_numpy(np.float64)
    validator = ROOT / "student_resource" / "student_resource" / "utils" / "validate_submission.py"
    test_dir = Path(args.source1).parent
    rows = []
    for name in args.variants:
        alpha, floor, formula = parse_variant(name)
        pairs = base[["source1_entity_id", "candidate_entity_id"]].copy()
        pairs["prob"] = (alpha * p0 / (alpha * p0 + (1.0 - p0))).astype(np.float32) if alpha != 1.0 else p0.astype(np.float32)
        pred = decide(pairs, formula=formula, exclusivity=True, min_prob=floor)
        out_dir = Path(args.out_root) / name
        out_dir.mkdir(parents=True, exist_ok=True)
        write_matching_results(pred, s1_ids, out_dir / "matching_results.tsv")
        n_match = sum(len(v) for v in pred.values())
        n_single = sum(1 for s in s1_ids if not pred.get(s))
        ok = "?"
        if validator.exists():
            r = subprocess.run([sys.executable, str(validator), "--matching", str(out_dir / "matching_results.tsv"),
                                "--test-dir", str(test_dir)], capture_output=True, text=True)
            ok = "PASS" if r.returncode == 0 else "FAIL"
        zip_path = Path(args.zip_dir) / f"{name}_matching_results.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
            z.write(out_dir / "matching_results.tsv", "matching_results.tsv")
        rows.append((name, alpha, floor, formula, n_match, n_match / len(s1_ids), n_single / len(s1_ids), ok, zip_path.name))
        print(f"{name:22s} alpha={alpha:<5} floor={floor:<5} matches={n_match:,} ({n_match/len(s1_ids):.3f}/entity) "
              f"singletons={n_single/len(s1_ids):.1%} validator={ok} -> {zip_path.name}", flush=True)
    pd.DataFrame(rows, columns=["variant", "alpha", "floor", "formula", "matches", "matches_per_entity", "singleton_frac", "validator", "zip"]).to_csv(
        Path(args.out_root) / "variants_summary.csv", index=False)


if __name__ == "__main__":
    main()
