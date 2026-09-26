# Waits for the train blocking run (work_block.log "ALL DONE"), then runs the remaining stages
# sequentially, stopping at the first failure. Log: work_rest.log
param([int]$MaxS1 = 300000)
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONIOENCODING = "utf-8"
$env:BLOCK_MAX_DF_ABS = "10000"
$py = Join-Path $root ".venv\Scripts\python.exe"
$log = Join-Path $root "work_rest.log"
$blockLog = Join-Path $root "work_block.log"
"waiting for train blocking $(Get-Date -Format s)" | Out-File $log -Append -Encoding utf8
while (-not ((Test-Path $blockLog) -and (Select-String -Path $blockLog -Pattern "ALL DONE" -Quiet))) { Start-Sleep -Seconds 30 }
if (Select-String -Path $blockLog -Pattern "exit -|Traceback|MemoryError" -Quiet) {
    "train blocking failed; not continuing $(Get-Date -Format s)" | Out-File $log -Append -Encoding utf8
    exit 1
}

function Run-Step($name, $argv) {
    # NB: never name this parameter $args - PowerShell reserves it and it comes through empty,
    # which launched a bare python REPL that error-looped in the hidden window.
    "=== $name $(Get-Date -Format s)" | Out-File $log -Append -Encoding utf8
    & $py @argv 2>&1 | Out-File $log -Append -Encoding utf8
    "=== $name exit $LASTEXITCODE $(Get-Date -Format s)" | Out-File $log -Append -Encoding utf8
    if ($LASTEXITCODE -ne 0) { "STOPPED at $name" | Out-File $log -Append -Encoding utf8; exit 1 }
}

Run-Step "features_train" @("src\features.py", "--split", "train", "--max-s1", "$MaxS1")
Run-Step "training"       @("src\training.py")
Run-Step "blocking_test"  @("src\blocking.py", "--split", "test")
Run-Step "features_test"  @("src\features.py", "--split", "test")
Run-Step "decision"       @("src\run_pipeline.py", "--stages", "decide")
"ALL STAGES DONE $(Get-Date -Format s)" | Out-File $log -Append -Encoding utf8
