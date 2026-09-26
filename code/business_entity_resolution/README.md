# Final submission package (assembled at the end)

This folder becomes `code/business_entity_resolution/` inside `<team_name>_submission.zip`.
Development happens at the repo root (`src/`, see the root README.md). Before packaging:

1. copy the finished `src/` (all stages) into `src/` here,
2. keep `requirements.txt` in sync with the root one,
3. describe the exact end-to-end run here (data -> blocking -> matching -> output):

```bash
pip install -r requirements.txt
python src/run_pipeline.py --stages all     # writes output/matching_results.tsv and output/candidate_pairs.tsv
```

`src/` currently holds the reference normalisation (`data_utils.py`) and blocking
(`blocking.py`) modules; they are superseded by whatever the team ships at the root.
