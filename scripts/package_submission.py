"""Assemble <team>_submission.zip in the layout the challenge requires.

  python scripts/package_submission.py --team TEAMNAME [--outputs output/work_improved]

Layout:
  <team>_submission.zip
  ├── output/matching_results.tsv, output/candidate_pairs.tsv
  ├── code/business_entity_resolution/{src/, scripts/, models/pruner.txt, README.md, requirements.txt}
  └── Documentation_template.md

Runs the official validator on the chosen outputs first and refuses to package on failure.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True)
    ap.add_argument("--outputs", default="output", help="folder holding matching_results.tsv + candidate_pairs.tsv")
    ap.add_argument("--skip-validate", action="store_true")
    args = ap.parse_args()
    out_dir = ROOT / args.outputs
    matching = out_dir / "matching_results.tsv"
    cands = out_dir / "candidate_pairs.tsv"
    for f in (matching, cands):
        if not f.exists():
            sys.exit(f"missing {f}")
    validator = ROOT / "student_resource" / "student_resource" / "utils" / "validate_submission.py"
    test_dir = ROOT / "student_resource" / "student_resource" / "dataset" / "test"
    if not test_dir.is_dir():
        test_dir = ROOT / "dataset" / "test"
    if not args.skip_validate and validator.exists():
        r = subprocess.run([sys.executable, str(validator), "--matching", str(matching), "--candidate", str(cands),
                            "--test-dir", str(test_dir)], capture_output=True, text=True)
        print(r.stdout[-600:])
        if r.returncode != 0:
            sys.exit("validator FAILED; not packaging")

    stage = ROOT / "build" / f"{args.team}_submission"
    if stage.exists():
        shutil.rmtree(stage)
    (stage / "output").mkdir(parents=True)
    shutil.copy(matching, stage / "output" / "matching_results.tsv")
    shutil.copy(cands, stage / "output" / "candidate_pairs.tsv")
    code = stage / "code" / "business_entity_resolution"
    shutil.copytree(ROOT / "src", code / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(ROOT / "scripts", code / "scripts", ignore=shutil.ignore_patterns("__pycache__"))
    (code / "models").mkdir()
    for m in ("pruner.txt", "lgbm_phase2.txt", "lgbm_phase1.txt", "isotonic_calibrator.pkl", "training_summary.json"):
        p = ROOT / "models" / m
        if p.exists():
            shutil.copy(p, code / "models" / m)
    shutil.copy(ROOT / "requirements.txt", code / "requirements.txt")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    (code / "README.md").write_text(
        "# Business Entity Resolution - runnable pipeline\n\n"
        "Reproduce end to end (data -> blocking -> matching -> output):\n\n"
        "```bash\npip install -r requirements.txt\n"
        "# unzip the challenge data so that dataset/train and dataset/test exist next to src/\n"
        "python src/run_pipeline.py --stages normalize\n"
        "python src/blocking.py --split train        # uses models/pruner.txt if present\n"
        "python src/features.py --split train --max-s1 300000\n"
        "python src/training.py\n"
        "python src/blocking.py --split test\n"
        "python src/features.py --split test\n"
        "python src/run_pipeline.py --stages decide  # writes output/*.tsv and runs the validator\n```\n\n"
        "Set BLOCK_MAX_DF_ABS=10000 (default) for the TF-IDF term cap. The pruner can be retrained with\n"
        "`python scripts/train_pruner.py`. Models shipped in models/ are the ones used for the submitted outputs.\n\n"
        "---\n\nOriginal team README follows.\n\n" + readme, encoding="utf-8")
    shutil.copy(ROOT / "Documentation_template.md", stage / "Documentation_template.md")

    zip_path = ROOT / "build" / f"{args.team}_submission.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(stage.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(stage.parent))
    print(f"wrote {zip_path} ({zip_path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
