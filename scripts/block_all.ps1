# Runs the blocking stage (train then test) as a detached process. Log: work_block.log
# Usage: powershell -File scripts\block_all.ps1 [train|test|both]
param([string]$which = "both")
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONIOENCODING = "utf-8"
$py = Join-Path $root ".venv\Scripts\python.exe"
New-Item -ItemType Directory -Force -Path (Join-Path $root "data\candidates") | Out-Null
$log = Join-Path $root "work_block.log"
"start $which $(Get-Date -Format s)" | Out-File $log -Append -Encoding utf8
$splits = if ($which -eq "both") { @("train", "test") } else { @($which) }
foreach ($split in $splits) {
    "=== blocking $split $(Get-Date -Format s)" | Out-File $log -Append -Encoding utf8
    & $py "src\blocking.py" --split $split 2>&1 | Out-File $log -Append -Encoding utf8
    "=== blocking $split exit $LASTEXITCODE $(Get-Date -Format s)" | Out-File $log -Append -Encoding utf8
}
"ALL DONE $(Get-Date -Format s)" | Out-File $log -Append -Encoding utf8
