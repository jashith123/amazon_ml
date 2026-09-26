# Amazon ML Challenge 2026 — Business Entity Resolution

Team repo. The pipeline design is `amazon_ml_final_pipeline.pdf` (11 stages); the
4-person split and the **schema contracts everyone codes against** are in
`team_task_split.md`. Column names live in one place: `src/common/schema.py` — import
from it, never retype a column name.

## Layout

```
dataset/                       <- NOT in git. Unzip the challenge data here (or leave it under
                                  student_resource/student_resource/dataset — both paths work)
samples/                       <- hand-made fixtures for every contract (python samples/make_fixtures.py)
data/normalized/               <- Person A writes   (Contract 1)
data/candidates/               <- Person B writes   (Contract 2)
data/features/, data/splits/   <- Person C writes   (Contract 3)
models/, output/               <- Person D writes   (Contract 4)
src/common/schema.py           <- shared constants
src/normalization.py           <- Person A   (stub, contract-checked CLI)
src/blocking.py                <- Person B   (stub + adaptive_k_prune, contract-checked CLI)
src/features.py                <- Person C   (stub + write_splits + contract validator)
src/training.py                <- Person D   two-phase LightGBM + hard negatives + isotonic calibration  [DONE]
src/decision.py                <- Person D   expected-F0.5 selection + exclusivity + vetoes + writers  [DONE]
src/run_pipeline.py            <- integration wiring (normalize -> block -> features -> train -> decide)
src/reference/                 <- optional working reference code for A and B (see below)
code/business_entity_resolution/  <- final submission package is assembled here at the end
```

## Setup (everyone)

```bash
python -m venv .venv                     # Python 3.10+ (we run 3.14)
.venv/Scripts/activate                   # Windows;  source .venv/bin/activate on mac/linux
pip install -r requirements.txt
python samples/make_fixtures.py          # builds the fixtures for all contracts
```

The venv is not committed (600 MB of platform binaries); `requirements.txt` pins every
version we use. `sentence-transformers` / `faiss-cpu` (Leg G, stretch goal) are not pinned
because there is no PyTorch wheel for Python 3.14 yet — install them in a 3.12 venv if you
attempt Leg G.

## Smoke test of the finished stages (Person D)

```bash
python src/training.py --fixture
python src/decision.py --features samples/feature_matrix_test.parquet \
    --candidates samples/candidate_pairs_test_synth.parquet --source1 samples/test_source1.tsv \
    --model samples/models/lgbm_phase2.txt --calibrator samples/models/isotonic_calibrator.pkl \
    --out-dir samples/output
python student_resource/student_resource/utils/validate_submission.py \
    --matching samples/output/matching_results.tsv --candidate samples/output/candidate_pairs.tsv --test-dir samples
```

## Full run (once A, B, C have delivered)

```bash
python src/run_pipeline.py --stages all          # or: normalize,block,features,train,decide
```

`training.py` prints validation macro-F0.5 for the expected-F0.5 layer vs. the best plain
threshold, overall and leave-one-country-out (US-only / India-only), plus the reliability
diagram and top features. `decision.py` writes `output/matching_results.tsv` and
`output/candidate_pairs.tsv` and `run_pipeline.py` runs the official validator on them.

## Reference implementations (optional for A and B)

`src/reference/normalization_ref.py` — multi-representation normalisation: ASCII/Indic
transliteration via `anyascii`, legal-suffix extraction, alias (`aka`/`f/k/a`) and
domain-name handling, US/India state-name folding, address abbreviation expansion, house
number / postcode / numeric-token extraction. Multiprocess, ~10 M rows in minutes.

`src/reference/blocking_ref.py` — country-partitioned candidate generation: word TF-IDF
(name+address) and char 3-4-gram TF-IDF (name) top-N via `sparse_dot_topn`, bidirectional
pass, 5 exact-key legs with bucket caps, union with leg bit-flags, heuristic top-K cut,
recall@K report against ground truth.

Both use their own column names; map to the contract columns when writing outputs.

## Data facts (from EDA on the real files)

| fact | value |
|---|---|
| train S1 / S2 / S3 rows | 2.21 M / 5.03 M / 5.29 M |
| test S1 / S2 / S3 rows | 1.73 M / 4.89 M / 5.08 M (France = 15% of test S1, absent from train) |
| matches per S1 (train) | mean 3.46, singletons 5.6% |
| cross-country matches in GT | 0 (country partitioning is free) |
| S2/S3 id matched to >1 S1 | 0 (exclusivity constraint is safe) |
| gallery names in Indic script | ~9% of S2, ~7% of S3; S1 is 100% ASCII |
| domain-style names (`abc.com`) | ~4% of gallery |

Rules: no external lookups of any kind, `sep="\t"` always, country is an open string set.
