# sample50k — real train data for fast experiments

50,000 Source-1 entities (stratified US/India), all their true matches, plus random
distractors at the full data's density (237,561 gallery records). Everything an experiment
needs, in the team's contract formats:

| file | contract | rows |
|---|---|---|
| `source1_normalized.parquet` | Contract 1 | 50,000 |
| `source2_normalized.parquet`, `source3_normalized.parquet` | Contract 1 | 237,561 together |
| `candidate_pairs_train.parquet` | Contract 2 (with `is_true_match`) | 1,499,385 (top-30 per S1) |
| `train_ground_truth.tsv` | ground truth for these 50k entities | 172,561 true pairs |

Blocking recall on this sample with the current blocker: 97.9% overall (India 96.4%, US 98.9%).
The whole chain (features → training → decision) runs on it in about two minutes:

```bash
python src/features.py --split train \
    --normalized-dir samples/sample50k \
    --candidates samples/sample50k/candidate_pairs_train.parquet \
    --out data/features/feature_matrix_sample50k.parquet
python src/training.py --features data/features/feature_matrix_sample50k.parquet \
    --ground-truth samples/sample50k/train_ground_truth.tsv --models-dir models/sample50k
```

## Experiment contract: semantic similarity (Leg G / `sem__*` features)

Today the four `sem__*` feature columns are all zero. To test an embedding model:

1. Python 3.12 venv, `pip install sentence-transformers` (PyTorch has no 3.14 wheel).
2. Model must be MIT/Apache and ≤ 8B params, e.g. `intfloat/multilingual-e5-small` (MIT).
3. Embed `name_norm` (or `name_transliterated`) of every record; for each row of
   `candidate_pairs_train.parquet` compute the cosine of the two embeddings.
4. Write `sem_scores.parquet` with columns `source1_entity_id`, `candidate_entity_id`,
   `sem__embedding_cosine` (float32, 0.0 when either name is empty). Optional: fine-tune the
   encoder on the true pairs in `train_ground_truth.tsv` (MultipleNegativesRankingLoss) and
   report recall@k of nearest-neighbour retrieval as a possible 8th blocking leg.

We join that file into the feature matrix, retrain, and compare validation F0.5 against
the lexical-only baseline. No external data or APIs of any kind (challenge rule).
