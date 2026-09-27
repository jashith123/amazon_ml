# Waits for the final run's test feature file to be written, then runs the decision stage
# immediately (skipping the 700 MB candidate TSV), validates, zips the leaderboard file into
# submissions/ and logs everything to work_decision_final.log.
$ErrorActionPreference = "Continue"
$root = "C:\Hackathon\Amazon"
Set-Location $root
$env:PYTHONIOENCODING = "utf-8"
$py = Join-Path $root ".venv\Scripts\python.exe"
$log = Join-Path $root "work_decision_final.log"
"waiter start $(Get-Date -Format s)" | Out-File $log -Append -Encoding utf8
while (-not ((Test-Path (Join-Path $root "work_features_final.log")) -and (Select-String -Path (Join-Path $root "work_features_final.log") -Pattern "wrote .*feature_matrix_test.parquet" -Quiet))) { Start-Sleep -Seconds 15 }
"features ready $(Get-Date -Format s)" | Out-File $log -Append -Encoding utf8
New-Item -ItemType Directory -Force -Path (Join-Path $root "output\work_final") | Out-Null
& $py "src\decision.py" --out-dir "output\work_final" --skip-candidates 2>&1 | Out-File $log -Append -Encoding utf8
"decision exit $LASTEXITCODE $(Get-Date -Format s)" | Out-File $log -Append -Encoding utf8
& $py "student_resource\student_resource\utils\validate_submission.py" --matching "output\work_final\matching_results.tsv" --test-dir "student_resource\student_resource\dataset\test" 2>&1 | Out-File $log -Append -Encoding utf8
"validator exit $LASTEXITCODE $(Get-Date -Format s)" | Out-File $log -Append -Encoding utf8
& $py -c "import zipfile; z=zipfile.ZipFile('submissions/final_matching_results.zip','w',zipfile.ZIP_DEFLATED,compresslevel=6); z.write('output/work_final/matching_results.tsv','matching_results.tsv'); z.close(); print('zipped')" 2>&1 | Out-File $log -Append -Encoding utf8
"ZIP READY $(Get-Date -Format s)" | Out-File $log -Append -Encoding utf8
