"""Integration script: runs every stage in contract order.

  python src/run_pipeline.py --stages all
  python src/run_pipeline.py --stages normalize,block
  python src/run_pipeline.py --stages train,decide

Each stage reads the previous stage's real output path from src/common/schema.py.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import schema as S  # noqa: E402

PY = sys.executable
SRC = Path(__file__).resolve().parent


def sh(*args):
    print("$", " ".join(str(a) for a in args), flush=True)
    subprocess.run([PY, *map(str, args)], check=True)


def stage_normalize():
    ds = S.dataset_dir()
    for split, suffix in (("train", ""), ("test", "_test")):
        for n in (1, 2, 3):
            sh(SRC / "normalization.py", "--input", ds / split / f"{split}_source{n}.tsv",
               "--output", S.NORMALIZED_DIR / S.NORMALIZED_FILE.format(n=n, suffix=suffix))


def stage_block():
    sh(SRC / "blocking.py", "--split", "train")
    sh(SRC / "blocking.py", "--split", "test")


def stage_features():
    sh(SRC / "features.py", "--split", "train")
    sh(SRC / "features.py", "--split", "test")


def stage_train():
    sh(SRC / "training.py")


def stage_decide():
    sh(SRC / "decision.py")
    validator = S.REPO_ROOT / "student_resource" / "student_resource" / "utils" / "validate_submission.py"
    if not validator.exists():
        validator = S.REPO_ROOT / "utils" / "validate_submission.py"
    if validator.exists():
        sh(validator, "--matching", S.MATCHING_RESULTS, "--candidate", S.CANDIDATE_PAIRS_TSV,
           "--test-dir", S.dataset_dir() / "test")


STAGES = {"normalize": stage_normalize, "block": stage_block, "features": stage_features,
          "train": stage_train, "decide": stage_decide}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="all", help="comma list of: " + ",".join(STAGES))
    args = ap.parse_args(argv)
    names = list(STAGES) if args.stages == "all" else args.stages.split(",")
    for n in names:
        print(f"\n===== stage: {n} =====", flush=True)
        STAGES[n]()


if __name__ == "__main__":
    main()
