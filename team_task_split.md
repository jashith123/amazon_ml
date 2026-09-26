# Amazon ML Challenge 2026 — 4-Person Task Split
Business Entity Resolution — tonight's execution plan

## Why this split
The 11-stage pipeline is sequential (normalize → block → featurize → train/decide), which
normally forces people to wait on each other. To make 4 people work **in parallel starting
right now**, every person builds against a fixed schema contract (exact file paths + exact
column names, given below) and tests against a small hand-made fixture instead of waiting
for a teammate's real output. Real outputs get swapped in at integration time — since the
contract doesn't change, nothing needs to be rewritten.

**Assignment:**
- Person A — Normalization (Stage 1–2)
- Person B — Blocking & Candidate Generation (Stage 3–5)
- Person C — Feature Engineering & Train/Val Split (Stage 6–7)
- Person D — Training, Calibration & F0.5 Decision Layer (Stage 8–11)

---

## SHARED — identical for all 4 people, read this first

**Environment**
- Python 3.10+, same venv for everyone: `pandas`, `numpy`, `pyarrow`, `scikit-learn`,
  `lightgbm`, `rapidfuzz`, `jellyfish` (soundex/metaphone), `indic-transliteration`
  (pip package `indic_transliteration`, use its `sanscript` module), `sentence-transformers`,
  `faiss-cpu`. Pin versions once, share a `requirements.txt` in repo root.
- All TSVs read with `sep='\t'` — never without it.
- Fix `RANDOM_SEED = 42` everywhere (numpy, lightgbm, sklearn splits).

**Repo layout (create this now, before writing any code)**
```
dataset/train/train_source{1,2,3}.tsv, train_ground_truth.tsv
dataset/test/test_source{1,2,3}.tsv
samples/                          <- tiny hand-made fixtures, see below
data/normalized/                  <- Person A writes here
data/candidates/                  <- Person B writes here
data/features/                    <- Person C writes here
data/splits/                      <- Person C writes here
models/                           <- Person D writes here
output/                           <- Person D writes here
src/normalization.py              <- Person A
src/blocking.py                   <- Person B
src/features.py                   <- Person C
src/training.py                   <- Person D
src/decision.py                   <- Person D
src/common/schema.py              <- shared constants, see below (anyone can start this)
src/run_pipeline.py               <- integration script, written last
```

**`src/common/schema.py`** — create this together in the first 5 minutes (or one person
does it and pastes it to the other three immediately). It just holds the column name lists
below as Python constants so no one typo's a column name. Everyone imports from it.

**Do not wait for a teammate's real file.** Each prompt below tells you how to hand-build
a 5–10 row fixture that matches the contract exactly. Use that to develop and unit-test.
Swap in the real upstream file only at integration time — the code doesn't change.

**ID conventions:** `S1-...`, `S2-...`, `S3-...` prefixes are part of the ID strings already
(don't strip them). `country` is an open string ('US', 'India', possibly others — never
hard-code a `{US, India}` set anywhere).

---

## CONTRACT 1 — Normalized records schema (Person A produces, B/C consume)

One file per source, both train and test:
`data/normalized/source1_normalized.parquet`, `source2_normalized.parquet`,
`source3_normalized.parquet`, and `source1_normalized_test.parquet` etc. for test.

Exact columns, exact names, in this order:

| column | dtype | notes |
|---|---|---|
| entity_id | str | unchanged, e.g. `S1-925783039` |
| country | str | unchanged |
| name_raw | str | original `business_name` |
| name_norm | str | NFKD → lowercase → strip punctuation → collapse whitespace |
| name_core | str | `name_norm` minus legal suffix |
| legal_suffix | str | `''` if none, e.g. `pvt ltd` |
| name_tokens | str | sorted, de-duplicated tokens joined by `\|` e.g. `abc\|technologies` |
| name_transliterated | str | Indic→Latin + French ligature fold applied to name_norm |
| is_non_latin | bool | True if name_raw contained non-Latin script |
| address_raw | str | original `business_address` (empty string if null) |
| address_norm | str | lowercase, abbreviations expanded (Rd→Road etc.) |
| address_numbers | str | all numeric tokens, joined by `\|`, `''` if none |
| address_pin | str | 5–6 digit postal code if found, else `''` |

**Fixture for B/C to hand-build now:** 6 rows covering: one plain US row, one India row with
a Devanagari/Gurmukhi name, one row with a legal suffix, one row with no address, one row
with a PIN code, one row with a house number. Fill all 12 columns by hand.

---

## CONTRACT 2 — Candidate pairs schema (Person B produces, C/D consume)

`data/candidates/candidate_pairs_train.parquet` and `candidate_pairs_test.parquet`.
One row per (S1 entity, candidate) pair, after adaptive-K pruning (≤30 per S1).

| column | dtype | notes |
|---|---|---|
| source1_entity_id | str | |
| candidate_entity_id | str | S2- or S3- id |
| candidate_source | str | `'S2'` or `'S3'` |
| country | str | the shared country of the pair |
| name_tfidf_block_hit | bool | |
| char_ngram_block_hit | bool | |
| address_block_hit | bool | |
| exact_key_hit | bool | |
| pin_block_hit | bool | |
| phonetic_block_hit | bool | |
| embedding_block_hit | bool | `False` for all rows if Leg G is skipped for time — see priority note below |
| bidirectional_block_hit | bool | |
| num_legs_retrieved | int | 0–7 |
| blocking_score | float | best/combined score across legs that hit, used for adaptive-K ranking |
| is_true_match | bool | **train file only** — joined from `train_ground_truth.tsv`; omit this column entirely in the test file |

**Fixture for C/D to hand-build now:** for one fake S1 id, 4 candidate rows: one obvious
true match (high blocking_score, several legs hit), one hard negative (name similar,
address different), one random negative, one candidate with `num_legs_retrieved=1`.

**Priority note for Person B (from the source doc's own ranking):** Legs A, B, C, D, E, F
(TF-IDF, char n-gram, bidirectional, exact keys, PIN, phonetic) are lexical, fast, and rank
highest-impact — build all of these first and get the recall diagnostic passing. Leg G
(BGE-M3/dense embedding ANN) and any fine-tuning are explicitly lower priority in the
source ranking (7th of 10) — treat as a stretch goal. If you run low on time, ship with
`embedding_block_hit = False` everywhere and tell C/D so semantic features degrade gracefully
instead of crashing.

---

## CONTRACT 3 — Feature matrix schema (Person C produces, D consumes)

`data/features/feature_matrix_train.parquet` (has `label`) and `feature_matrix_test.parquet`
(no `label`). One row per candidate pair (same keys as Contract 2), plus feature columns.
**Naming convention — every feature is prefixed by its group, no exceptions**, so D can
select columns by prefix without a lookup table:

- `source1_entity_id`, `candidate_entity_id`, `candidate_source`, `country` (join keys, carried over from Contract 2)
- `name__*` (18 features) — e.g. `name__lev_ratio`, `name__jaro_winkler`, `name__token_jaccard`, `name__exact_core_match`, `name__phonetic_match`, `name__trigram_jaccard`, ...
- `addr__*` (16 features) — e.g. `addr__lev_ratio`, `addr__house_number_conflict`, `addr__pin_exact`, `addr__both_have_address`, ...
- `cross__*` (10 features) — e.g. `cross__name_addr_product` (most important single feature per the source doc), `cross__numeric_conflict`, `cross__country_exact_match`, ...
- `sem__*` (4 features) — e.g. `sem__embedding_cosine`, `sem__embedding_rank`. Fill with `0.0` if Person B skipped Leg G — do not leave NaN.
- `comp__*` — **rank/percentile only, never raw counts.** No column named anything like `n_cand`, `g_size`, or any absolute pool size may exist in this table. e.g. `comp__rank`, `comp__score_percentile`, `comp__score_gap_top1`.
- `block__*` — mirrors Contract 2's boolean flags plus `block__num_legs_retrieved`.
- `label` — **train only**, copied straight from Contract 2's `is_true_match`.

`data/splits/train_entity_ids.txt`, `val_entity_ids.txt`, `val_us_only_ids.txt`,
`val_india_only_ids.txt` — one `source1_entity_id` per line, split at the **entity level**
(never split individual candidate rows across files).

**Fixture for D to hand-build now:** take the 4-row fixture from Contract 2, add ~10 fake
feature columns (a few per prefix) and a `label` column, so you can write and test the
training loop without waiting on real features.

---

## CONTRACT 4 — Final outputs (Person D produces, this is the deliverable)

- `output/matching_results.tsv` — columns `source1_entity_id`, `matched_entity_ids`
  (comma-separated S2-/S3- ids, empty string for singletons). One row per S1 test entity.
- `output/candidate_pairs.tsv` — columns `source1_entity_id`, `candidate_entity_ids`
  (all candidates, comma-separated — this is just Contract 2's test file collapsed to one
  row per S1; D can write a 5-line helper for this, doesn't need new logic).
- `models/lgbm_phase1.txt`, `models/lgbm_phase2.txt` (LightGBM native format),
  `models/isotonic_calibrator.pkl` (joblib).

---

# PROMPT FOR PERSON A — Normalization (Stage 1–2)

You own `src/normalization.py`. Your job: turn each raw source file into the normalized
schema in **CONTRACT 1** above, for all three sources, train and test (6 files total).

**Must-have tonight (in priority order):**
1. `name_norm` (NFKD, lowercase, strip punctuation, collapse whitespace) and `address_norm`
   (lowercase, expand common abbreviations: Rd→Road, St→Street, Blvd→Boulevard, Ave→Avenue).
2. Legal suffix extraction → `legal_suffix` + `name_core`. Keep a small dictionary
   (pvt, ltd, corp, inc, llc, gmbh, sarl, ...) — extend as you notice more in the data.
3. `address_numbers` (regex-extract digit runs) and `address_pin` (5–6 digit token, prefer
   the one at the end of the address string).
4. `name_transliterated`: run `indic_transliteration.sanscript` (or similar) to convert
   Devanagari/Gurmukhi/Bengali/etc. to Latin, then apply a French ligature fold
   (`œ→oe, æ→ae, ß→ss, ô→o, ñ→n, é/è/ê→e, ç→c, à/â→a`) on top. `is_non_latin` = True if
   `name_raw` contains any character outside basic Latin.
5. `name_tokens` (sorted, de-duplicated, pipe-joined).

**Stretch if time allows:** script-specific fixes (schwa deletion, nukta stripping per
script — see the source doc's script table); the "learned normalization from training
pairs" step (aligning matched pairs to learn extra token mappings like pvt→private). Skip
these if pressed for time — items 1–5 above are what B and C actually need to function.

**Function signature:** `normalize_source(df: pd.DataFrame) -> pd.DataFrame` taking a raw
dataframe with columns `entity_id, business_name, business_address, country` and returning
the 12-column Contract 1 dataframe. CLI: `python src/normalization.py --input <tsv> --output <parquet>`.

**Test before integrating:**
- Row count in == row count out, no dropped entity_ids.
- Spot-check 5 India rows with non-Latin names: `is_non_latin=True` and `name_transliterated`
  is readable Latin text, not empty/garbage.
- Assert `address_pin` is either `''` or all-digit length 5–6, on a sample of 1000 rows.
- Hand-verify one row through the whole doc's example: `ABC Pvt. Ltd.` → `name_norm='abc pvt ltd'`,
  `name_core='abc'`, `legal_suffix='pvt ltd'`, `name_tokens='abc'`.

**Handoff:** the moment your 6 normalized parquet files exist, tell B and C — they were
already coding against the fixture, so this is a drop-in swap, not a rewrite.

---

# PROMPT FOR PERSON B — Blocking & Candidate Generation (Stage 3–5)

You own `src/blocking.py`. Your job: given normalized records (real files once A delivers,
your own fixture until then), produce **CONTRACT 2**'s candidate_pairs parquet, train and
test, with blocking recall ≥99% on the training fold.

**Must-have tonight, in this order (matches the source doc's own priority ranking):**
1. **Country partitioning** — every leg only compares records within the same `country`.
   Non-negotiable, biggest free win (eliminates 50–70% of the search space, zero recall cost).
2. Leg A: word TF-IDF on `name_norm + ' ' + address_norm`, chunked/sparse, top-40 per S1.
3. Leg B: character 4-gram TF-IDF on `name_norm` (spaces removed), top-40.
4. Leg C: **bidirectional** — reuse Leg A's index, run gallery (S2/S3) as queries too, union
   with the forward pass. Cheap, do not skip (it's flagged CRITICAL and often missing).
5. Leg D: exact/near-exact key blocking on `name_core`, `name_core+first address number`,
   `name_tokens` (sorted-signature), acronym of `name_tokens`. Bucket cap 200, top-30/bucket.
6. Leg E: PIN/ZIP bucket join on `address_pin`.
7. Leg F: Soundex/Metaphone (`jellyfish.soundex` / `.metaphone`) on `name_core` tokens, plus
   first-letter acronym keys. Cap 200/bucket, skip oversized buckets.
8. Union all legs → set `num_legs_retrieved`, the boolean hit flags, and `blocking_score`
   (just use the max TF-IDF cosine among legs that produced a real similarity score; exact/
   phonetic-only hits can get a fixed placeholder score like 0.5).
9. Adaptive-K prune to ≤30/S1 (closed-form pruner is already written in the source doc's
   §10 — reuse it verbatim, function name `adaptive_k_prune`).
10. **Blocking recall diagnostic** — join against `train_ground_truth.tsv`, compute recall
    per country, average candidates/S1. Do not hand off until this is ≥99% (or you've told
    the team it's lower and why).

**Stretch if time allows:** Leg G (BGE-M3/dense embedding ANN via `sentence-transformers` +
`faiss-cpu`), fine-tuning the encoder. If skipped, set `embedding_block_hit=False` for all
rows and say so explicitly when you hand off — C's semantic features and D's decision layer
both degrade gracefully (not crash) when this is False everywhere.

**Test before integrating:**
- Run the recall diagnostic on your fixture and on a random 2,000-S1 sample of the real
  train data as soon as A's normalized files land — don't wait for the full 2.2M rows to
  validate correctness.
- Assert every `is_true_match=True` pair in your candidate set actually appears (sanity:
  recall check IS this, but also assert no duplicate (source1_entity_id, candidate_entity_id)
  rows).
- Assert max 30 candidates per S1 after pruning.

**Handoff:** once recall is verified, hand the parquet files to C and D. Tell them your
actual `embedding_block_hit` status (on/off) so they don't waste time debugging a feature
that's intentionally all-zero.

---

# PROMPT FOR PERSON C — Feature Engineering & Split (Stage 6–7)

You own `src/features.py`. Your job: given normalized records + candidate pairs (real once
A and B deliver, your own fixtures until then), produce **CONTRACT 3**'s feature matrix
(train + test) and the entity-level train/val split files.

**Must-have tonight, in this order:**
1. `name__*` group (18 features): Levenshtein ratio and Jaro-Winkler via `rapidfuzz`,
   char TF-IDF cosine (fit on training corpus), token Jaccard/overlap, token-sort ratio,
   exact match flags on `name_norm`/`name_core`, phonetic match (compare soundex codes
   computed on the fly from `name_core`), prefix overlap, length diff, suffix agreement
   flags, trigram Jaccard.
2. `addr__*` group (16 features): same similarity metrics applied to `address_norm`, plus
   `addr__house_number_exact`/`addr__house_number_conflict` (compare `address_numbers`),
   `addr__pin_exact` (compare `address_pin`), `addr__both_have_address` flag.
   **Flag `addr__house_number_conflict` and `addr__pin_exact` clearly** — D needs these by
   exact name for hard-veto logic.
3. `cross__*` group (10 features): products/sums/min/max of the name and address similarity
   scores, the two threshold-flag features (`name_high_addr_low`, `name_low_addr_high`),
   `cross__numeric_conflict`, `cross__country_exact_match`. Build `cross__name_addr_product`
   correctly — it's called out as the single most important feature in the source doc.
4. `block__*` group: straight passthrough of Contract 2's boolean flags and
   `num_legs_retrieved`, just renamed with the `block__` prefix.
5. `comp__*` group: **rank and percentile only.** Within each S1's candidate set, compute
   `comp__rank` (1 = best blocking_score), `comp__score_percentile`, `comp__score_gap_top1`.
   Never write a column that is an absolute count (no `n_candidates`, no pool size) —
   this is explicitly called out as a known leaderboard-vs-validation trap in the source doc.
6. `sem__*` group: if B shipped embeddings, compute cosine similarity + rank; if not,
   fill all four columns with `0.0` (not NaN — D's model can't handle NaN by default
   without extra config, keep it simple).
7. `label` column (train only): copy `is_true_match` straight from Contract 2.
8. Entity-aware split: split unique `source1_entity_id` values 80/20 into
   `train_entity_ids.txt` / `val_entity_ids.txt` (never split by row). Also write
   `val_us_only_ids.txt` and `val_india_only_ids.txt` (entities whose country is US-only /
   India-only, for the leave-one-country-out check D needs).

**Test before integrating:**
- No NaNs anywhere in the feature matrix — `assert feature_df.isna().sum().sum() == 0`.
- No column name matches a "raw count" pattern — grep for anything that isn't rank/
  percentile/boolean/similarity in the `comp__` group.
- On your fixture, hand-verify `cross__name_addr_product` = `name__lev_ratio *`-equivalent
  product actually computes correctly for the known true-match row.
- Every `source1_entity_id` appears in exactly one of train/val split files, no overlap.

**Handoff:** feature matrix + split files to D. Tell D explicitly whether `sem__*` is real
or all-zero (mirrors B's note to you).

---

# PROMPT FOR PERSON D — Training, Calibration & Decision Layer (Stage 8–11)

You own `src/training.py` and `src/decision.py`. Your job: given the feature matrix (real
once C delivers, your own fixture until then), train the model and produce **CONTRACT 4**'s
final submission files.

**Must-have tonight, in this order:**
1. **Two-phase LightGBM training.**
   - Phase 1: train on true positives (`label=1`) + random negatives (`label=0`) from
     `train_entity_ids.txt` only. Hyperparameters from the source doc:
     `n_estimators=1000+, learning_rate=0.05, num_leaves=63+, subsample=0.8,
     colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1, min_child_samples=20,
     early_stopping_rounds=50`, metric=AUC. Save as `models/lgbm_phase1.txt`.
   - Phase 2: run Phase 1 model on all training candidates, extract hard negatives
     (`label=0` rows with `P(match)≥0.3`), retrain with true positives : hard negatives :
     random negatives ≈ 2:1:1. Save as `models/lgbm_phase2.txt`. **This step is called out
     as the single highest-impact training improvement — don't skip it for time.**
2. **Isotonic calibration** — fit `sklearn.isotonic.IsotonicRegression(out_of_bounds='clip')`
   on out-of-fold predictions from `val_entity_ids.txt`. Save as
   `models/isotonic_calibrator.pkl`. Sanity-plot the reliability diagram before moving on.
3. **Expected-F0.5 decision layer** (`src/decision.py`) — the closed-form selection is
   already written in the source doc's §10 as `expected_f05_select(probs)`; reuse it
   verbatim, per S1 entity, on calibrated probabilities.
4. **Exclusivity constraint** — the source doc's §10 also gives `enforce_exclusivity`
   verbatim; run it after the per-entity selection. First verify the assumption it depends
   on: `assert no S2-/S3- id repeats across S1 rows in train_ground_truth.tsv`.
5. **Hard vetoes** (apply regardless of P): reject if `addr__house_number_conflict` is True
   AND address similarity is low; reject if both `name__lev_ratio` and `sem__embedding_cosine`
   (or 0.0 if B/C didn't ship embeddings — then skip this specific veto) are very low.
6. Write `output/matching_results.tsv` and `output/candidate_pairs.tsv` per Contract 4.

**Stretch if time allows:** the cross-encoder reranker for uncertain pairs
(0.3 ≤ P ≤ 0.6) — this is explicitly the lowest-priority, optional-second-stage item in
the source doc's own ranking. Skip it first if the night is running short.

**Test before integrating:**
- Confirm on your fixture that a clear true-match row scores high P and survives; a clear
  random-negative row scores low P and is dropped.
- Confirm singleton logic: an S1 entity with only weak candidates should get `k=0` selected
  → empty `matched_entity_ids`, not a forced match.
- Once real data is in: compute macro-F0.5 on `val_entity_ids.txt`, then again on
  `val_us_only_ids.txt` / `val_india_only_ids.txt` (leave-one-country-out) — big gaps between
  these flag a generalization problem before you burn a leaderboard submission on it.
- Run the official validator: `python3 utils/validate_submission.py --matching
  output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test`
  and confirm PASS before calling it done.

---

## Integration (whoever's free first, once all 4 pieces land)

Write `src/run_pipeline.py` that just calls, in order:
`normalization.py` (×6 files) → `blocking.py` → `features.py` → `training.py` →
`decision.py`, each reading the previous stage's real output path from Contract 1–4 (no
new logic needed — if everyone stuck to the schemas above, this is wiring, not coding).
Run it once end-to-end on a 5,000-S1 sample to catch integration bugs fast, then kick off
the full run.
