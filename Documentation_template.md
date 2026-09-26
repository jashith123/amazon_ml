# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [TEAM NAME]
**Team Members:** Kanwal Raj Singh (normalisation), ksv (candidate generation), Hiresh Goyal (feature engineering), Jashith Narang (training, calibration, decision layer, integration)
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

We resolve Source 2 / Source 3 records against the Source 1 reference with a blocking-plus-classifier pipeline whose decision layer is built directly around the macro-F0.5 metric. Country-partitioned lexical blocking (word and character TF-IDF, a bidirectional pass, exact keys) feeds a learned LightGBM pruner that keeps the 30 best candidates per Source 1 entity; a second LightGBM classifier on 60 pairwise features, calibrated with isotonic regression, produces match probabilities; and a closed-form expected-F0.5 subset selection per entity, followed by an exclusivity constraint, turns probabilities into matches. No external data, models or services are used at any stage; the only model is LightGBM (MIT licence).

---

## 2. Methodology

### 2.1 Problem Analysis

Exploratory analysis of the training files established the facts the design rests on:

| Fact | Value | Consequence |
|---|---|---|
| Records (train S1 / S2 / S3) | 2.21M / 5.03M / 5.29M | everything must scale linearly |
| Records (test S1 / S2 / S3) | 1.73M / 4.89M / 5.08M; France = 15% of test S1, absent from training | country is an open string label; nothing may be tuned per country |
| Matches per S1 entity | mean 3.46; only 5.6% singletons | the model must handle one-to-many, and empty predictions matter but are rare |
| Cross-country matches in ground truth | 0 | blocking can partition by country with zero recall cost |
| S2/S3 records matched to more than one S1 | 0 | an exclusivity constraint is safe and is a pure precision gain |
| Gallery names in Indic scripts | ~9% of S2, ~7% of S3; S1 is 100% ASCII | transliteration must happen before blocking |
| Domain-style names (`abc.com`) | ~4% of gallery | a space-free name key is needed |

Noise patterns observed in matched pairs: legal-suffix variants (Pvt/Private, Ltd/Limited), word reordering ("of Group Bnp Companies-Delhi"), typos and transliteration spellings ("Rseouerees" for "Resources", "praivet limited"), leading junk ("##1900", ">>"), alias prefixes ("Fluxiri aka ..."), address abbreviation and component reordering, extra or missing components (PIN, plot number, floor), and Indic-script state names inside otherwise Latin addresses.

### 2.2 Solution Strategy

**Approach Type:** Blocking + learned pruner + pairwise classifier + metric-optimised decision layer.

**Core Innovation:** (1) a learned pruner on cheap union features that lifts blocking recall at a fixed 30 candidates per entity by about 3 points on dense Indian cities; (2) a per-entity expected-F0.5 subset selection with the exact denominator (expected number of true matches among all candidates) and a singleton option scored as Π(1-p), which outperformed the best global threshold by a wide margin (0.92 vs 0.67 macro-F0.5 on 60k held-out training entities).

Stages, in execution order:

1. Load and validate (tab-separated reads, duplicate-ID checks, the assertions above).
2. Multi-representation normalisation: ASCII folding and Indic transliteration (`anyascii`), legal-suffix extraction, address abbreviation expansion, house-number / postcode / numeric-token extraction, sorted token signature.
3. Country-partitioned candidate generation (Section 3).
4. Blocking recall diagnostic against ground truth (gate).
5. Learned pruner → top-30 per entity → `candidate_pairs.tsv`.
6. Pairwise features (Section 4).
7. Entity-level train/validation split, plus US-only and India-only validation lists.
8. Two-phase LightGBM training with hard-negative mining.
9. Isotonic calibration on a held-out calibration fold.
10. Expected-F0.5 decision layer, exclusivity, hard vetoes.
11. Output writing and the official validator.

---

## 3. Candidate Generation (Blocking)

All legs run inside one country partition; the gallery index (Source 2 ∪ Source 3 of that country) is fitted once and Source 1 records are queried in chunks of 100,000.

- **Blocking keys / legs used:**
  - Word TF-IDF (sublinear tf, l2) over normalised name + address, top-40 by cosine via `sparse_dot_topn`.
  - Character 3-4-gram TF-IDF over the space-free name, top-40.
  - Bidirectional pass: every gallery record as a query against the Source 1 index, top-3.
  - Exact keys with bucket caps: name core, name core + first house number, sorted token signature, acronym, full normalised address (≥12 chars), house number + first two address words, PIN/ZIP, Soundex and Metaphone of the first name token.
  - An absolute document-frequency cap (10,000 records) on TF-IDF terms: a fractional cap makes the cost quadratic in gallery size; with the absolute cap the full training set blocks in about 75 minutes on a laptop.
- **Union scoring and pruning:** every union pair gets a heuristic score (word cosine + char cosine + a bonus per key leg + bidirectional bonus). The top 80 by heuristic are re-scored by a LightGBM pruner on 19 cheap features (the two cosines, leg flags, name and address token-set ratios, name ratio, house-number agreement/conflict, token counts, and a chunk-local gallery-side competition ratio: this pair's score relative to the best Source 1 for the same candidate). The top 30 by pruner probability form the candidate set.
- **Candidate pairs generated:** 30 per Source 1 entity, 66.2M pairs for the training set, [TEST PAIRS] for the test set.
- **How we ensured true matches were not lost:** the recall diagnostic on the full training set, and a dense-slice harness (`scripts/diagnose_blocking.py`) that keeps every record of one city so an entity faces its real neighbours. Findings: on Delhi, 3.4% of true pairs were never retrieved by any leg (heavily transliterated names with truncated addresses), 5.4% were retrieved but ranked 31-300 by the heuristic, union recall@100 was 96.5%. The learned pruner recovers most of the second group: on a held-out Bangalore slice recall@30 rose from 93.2% to 96.3% against a union ceiling of 96.4%.

Full training-set blocking recall: heuristic top-30 88.9% (India 83.0%, US 92.9%); with the pruner [PRUNER RECALL]. Note that the Windows laptop used for the full run (16 cores, 23.6 GB RAM) forced two engineering choices that cost recall: the absolute df cap (about 0.6 points on a 50k-entity sample) and the 30-candidate cut.

---

## 4. Matching Model

**Features used (60, six groups, every column prefixed by its group):**
- Name features (18): Levenshtein ratio, Jaro-Winkler, token-sort and token-set ratios, partial ratio, char 2-3-gram TF-IDF cosine, token Jaccard and overlap, n-gram Jaccard, prefix overlap, length difference and ratio, exact-normalised / exact-core / exact-raw flags, Soundex match, legal-suffix match and conflict.
- Address features (16): the same twelve similarity metrics on the normalised address, plus house-number exact and conflict (first number only), PIN exact, both-have-address.
- Cross-field (10): product, sum, min, max, difference and harmonic mean of name and address similarity, name-high/address-low and name-low/address-high flags, country match, numeric-token conflict.
- Blocking evidence (9): the eight leg flags and the number of legs that retrieved the pair.
- Competition (3, density-safe by construction): rank, percentile and gap-to-top of the blocking score within the entity's candidate set. No absolute counts are used anywhere.
- Semantic (4): reserved for a multilingual sentence-embedding cosine (`intfloat/multilingual-e5-small`, MIT); zero in this submission.

Records are vectorised once per country (char TF-IDF, binary token sets, numeric-token sets); pair metrics are sparse row-wise products, and the edit-distance metrics use `rapidfuzz.process.cpdist`. The feature stage runs at about 22,000 pairs per second.

**Model type:** LightGBM binary classifier (MIT licence, far below the 8B-parameter limit), 63 leaves, learning rate 0.05, early stopping on validation AUC. Phase 1 trains on all positives plus an equal number of random negatives; phase 2 adds the negatives phase 1 scored ≥ 0.3 (hard negatives, weight 2) at a 2:1:1 ratio and retrains. Trained on 300,000 stratified Source 1 entities (9.0M candidate pairs); 15% of training entities are held out for isotonic calibration.

**Threshold selection method:** none. For each entity with calibrated probabilities p₁ ≥ p₂ ≥ … the expected F0.5 of accepting the top k is 1.25·Σᵢ≤ₖpᵢ / (k + 0.25·Σ_all pᵢ), and predicting no match has expected score Π(1-pᵢ); we take the best k. An exclusivity pass then keeps each Source 2/3 id only for the Source 1 entity that claims it with the highest probability. Hard veto: house-number conflict with address similarity below 0.3.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro), validation on 60,000 held-out training entities:**

| Run | All | US only | India only |
|---|---|---|---|
| Baseline (heuristic top-30) | 0.920 | 0.944 | 0.885 |
| With learned pruner | [IMPROVED] | [IMPROVED US] | [IMPROVED IN] |
| Best global threshold, for comparison | 0.672 | 0.682 | 0.656 |

Phase-2 validation AUC 0.9995. Public leaderboard: [LB SCORE].

- **Common false positives (wrong merges):** neighbouring businesses at the same address (same building, different tenants) with generic names; chains and franchises sharing a name with different branch addresses; the model relies on the house-number conflict and the competition features to separate these.
- **Common false negatives (missed matches):** dominated by blocking recall (the ceiling): transliteration spellings that no character n-gram survives ("sauth phuds praivet limited"), combined with truncated addresses; names that are aliases or domain forms with a different address representation. India recall is 10 points below US for this reason.

---

## 6. Conclusion

A carefully engineered lexical blocker with a learned pruner, a compact LightGBM matcher on well-designed features, and a decision layer that optimises the metric directly give a strong, fully reproducible solution on a laptop. The remaining headroom is almost entirely in blocking recall for India (transliterated names), where a multilingual embedding leg is the natural next step.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/src/` (mirror of the repository's `src/`):

| file | stage |
|---|---|
| `common/schema.py` | column contracts and paths shared by every stage |
| `normalization.py` | stages 1-2 |
| `blocking.py` | stages 3-5 (legs, union, learned pruner, recall diagnostic) |
| `features.py` | stages 6-7 |
| `training.py` | stages 8-9 |
| `decision.py` | stages 10-11 |
| `run_pipeline.py` | end-to-end driver |
| `../scripts/train_pruner.py`, `../scripts/diagnose_blocking.py` | pruner training and the dense-slice diagnosis harness |

Reproduce: `pip install -r requirements.txt`, unzip the data into `dataset/`, then `python src/run_pipeline.py --stages all` (or the per-stage commands in the README). `output/matching_results.tsv` and `output/candidate_pairs.tsv` are written by the decision stage and pass `utils/validate_submission.py`.

### B. Additional Results

Blocking recall at fixed K on a 50,000-entity sample at the full data's distractor density, before the pruner: K=30 97.9%, K=40 98.3%, K=50 98.6% (uncapped TF-IDF terms: 98.5% at K=30). Dense Delhi slice (122k S1, 545k gallery): recall@30 91.2%, @50 94.2%, @100 96.5%; never retrieved 3.4%.
