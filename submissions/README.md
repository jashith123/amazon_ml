# Leaderboard files for the team leader to upload

| file | run | validation macro-F0.5 | validator |
|---|---|---|---|
| `improved_matching_results.zip` | learned pruner end to end (27 Sep 04:06) | 0.940 (US 0.951, India 0.924) | PASS |

Unzip, then upload `matching_results.tsv` on the portal. Each zip holds exactly one
`matching_results.tsv` (1,732,544 rows, tab-separated). The full submission package
(code + candidate_pairs.tsv + write-up) is built with `scripts/package_submission.py`
on the machine that ran the pipeline and shared separately (347 MB, too large for git).
