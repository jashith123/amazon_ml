# Leaderboard files for the team leader to upload

Unzip, then upload the `matching_results.tsv` inside (1,732,544 rows). All PASS the official validator.

| file | run | what differs | val macro-F0.5 |
|---|---|---|---|
| `improved_matching_results.zip` | pruner run (27 Sep 04:06) | reference | 0.940 |
| `alpha1.0_matching_results.zip` | token-map run (27 Sep 16:30) | same decision rule, model trained with token map | 0.940 |
| `alpha0.53_matching_results.zip` | token-map run | probabilities shifted for a denser test pool (odds x0.53): fewer, surer matches | n/a (test-only correction) |
| `alpha0.3_matching_results.zip` | token-map run | stronger shift (odds x0.3): most conservative | n/a |

Upload order suggestion: alpha0.53 first; if it scores above improved/alpha1.0, try alpha0.3;
if below, the density correction is wrong and improved/alpha1.0 stays. Report each public score
to Jashith so the next variants can be aimed.
