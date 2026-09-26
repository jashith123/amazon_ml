# Runs the stages after train blocking sequentially, stopping at the first failure.
#   -Start      first step to run: features_train | training | blocking_test | features_test | decision
#   -NoPruner   "1" disables the learned blocking pruner (baseline consistency)
#   -Log        log file name (in repo root)
#   -MaxS1      train entities sampled for the feature/training stages
param([int]$MaxS1 = 300000, [string]$Start = "features_train", [string]$NoPruner = "0", [string]$Log = "work_rest.log",
      [string]$WaitFor = "")
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONIOENCODING = "utf-8"
$env:BLOCK_MAX_DF_ABS = "10000"
if ($NoPruner -eq "1") { $env:BLOCK_NO_PRUNER = "1" }
$py = Join-Path $root ".venv\Scripts\python.exe"
$log = Join-Path $root $Log
"runner start=$Start nopruner=$NoPruner $(Get-Date -Format s)" | Out-File $log -Append -Encoding utf8
if ($WaitFor -ne "") {
    $wl = Join-Path $root $WaitFor
    "waiting for ALL DONE in $WaitFor" | Out-File $log -Append -Encoding utf8
    while (-not ((Test-Path $wl) -and (Select-String -Path $wl -Pattern "ALL DONE|ALL STAGES DONE" -Quiet))) { Start-Sleep -Seconds 30 }
    if (Select-String -Path $wl -Pattern "exit -|Traceback|MemoryError|STOPPED" -Quiet) {
        "$WaitFor reports failure; not continuing" | Out-File $log -Append -Encoding utf8; exit 1
    }
}

function Run-Step($name, $argv) {
    # NB: never name this parameter $args - PowerShell reserves it and it comes through empty.
    "=== $name $(Get-Date -Format s)" | Out-File $log -Append -Encoding utf8
    & $py @argv 2>&1 | Out-File $log -Append -Encoding utf8
    "=== $name exit $LASTEXITCODE $(Get-Date -Format s)" | Out-File $log -Append -Encoding utf8
    if ($LASTEXITCODE -ne 0) { "STOPPED at $name" | Out-File $log -Append -Encoding utf8; exit 1 }
}

$steps = @(
    @("blocking_train",  @("src\blocking.py", "--split", "train")),
    @("features_train",  @("src\features.py", "--split", "train", "--max-s1", "$MaxS1")),
    @("training",        @("src\training.py")),
    @("blocking_test",   @("src\blocking.py", "--split", "test")),
    @("features_test",   @("src\features.py", "--split", "test")),
    @("decision",        @("src\run_pipeline.py", "--stages", "decide"))
)
$go = $false
foreach ($s in $steps) {
    if ($s[0] -eq $Start) { $go = $true }
    if ($go) { Run-Step $s[0] $s[1] }
}
# keep this run's outputs under output\<log name>\ so later runs do not overwrite them
$tag = [System.IO.Path]::GetFileNameWithoutExtension($Log)
$keep = Join-Path $root "output\$tag"
New-Item -ItemType Directory -Force -Path $keep | Out-Null
Copy-Item (Join-Path $root "output\matching_results.tsv") $keep -Force -ErrorAction SilentlyContinue
Copy-Item (Join-Path $root "output\candidate_pairs.tsv") $keep -Force -ErrorAction SilentlyContinue
Copy-Item (Join-Path $root "models\training_summary.json") $keep -Force -ErrorAction SilentlyContinue
"outputs copied to $keep" | Out-File $log -Append -Encoding utf8
"ALL STAGES DONE $(Get-Date -Format s)" | Out-File $log -Append -Encoding utf8
