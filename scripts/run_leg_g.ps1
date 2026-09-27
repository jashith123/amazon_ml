$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = "python"
if (Test-Path ".venv\Scripts\python.exe") { $py = Join-Path $root ".venv\Scripts\python.exe" }

"=== Running Leg G on samples/sample50k/ ==="
& $py src/semantic_similarity.py `
    --normalized samples/sample50k/source1_normalized.parquet samples/sample50k/source2_normalized.parquet samples/sample50k/source3_normalized.parquet `
    --candidates samples/sample50k/candidate_pairs_train.parquet `
    --output samples/sample50k/sem_scores.parquet

if ($LASTEXITCODE -ne 0) { "Failed on sample50k!"; exit 1 }
"=== sample50k Leg G DONE ==="

if (Test-Path "data\candidates\candidate_pairs_train.parquet") {
    "=== Running Leg G on full TRAIN data ==="
    & $py src/semantic_similarity.py `
        --normalized data/normalized/source1_normalized.parquet data/normalized/source2_normalized.parquet data/normalized/source3_normalized.parquet `
        --candidates data/candidates/candidate_pairs_train.parquet `
        --output data/candidates/sem_scores_train.parquet
        
    if ($LASTEXITCODE -ne 0) { "Failed on full train data!"; exit 1 }
} else {
    "Full TRAIN candidates not found in data/candidates/, skipping."
}

if (Test-Path "data\candidates\candidate_pairs_test.parquet") {
    "=== Running Leg G on full TEST data ==="
    & $py src/semantic_similarity.py `
        --normalized data/normalized/source1_normalized_test.parquet data/normalized/source2_normalized_test.parquet data/normalized/source3_normalized_test.parquet `
        --candidates data/candidates/candidate_pairs_test.parquet `
        --output data/candidates/sem_scores_test.parquet
        
    if ($LASTEXITCODE -ne 0) { "Failed on full test data!"; exit 1 }
} else {
    "Full TEST candidates not found in data/candidates/, skipping."
}

"ALL DONE."
