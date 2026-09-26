# Runs Person A's normalisation on all six source files, three at a time (memory-safe),
# each as its own process so the stage uses several cores. Log: work_normalize.log
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONIOENCODING = "utf-8"
$py = Join-Path $root ".venv\Scripts\python.exe"
$ds = Join-Path $root "student_resource\student_resource\dataset"
if (Test-Path (Join-Path $root "dataset\train")) { $ds = Join-Path $root "dataset" }
New-Item -ItemType Directory -Force -Path (Join-Path $root "data\normalized") | Out-Null
$log = Join-Path $root "work_normalize.log"
"start $(Get-Date -Format s)" | Out-File $log -Encoding utf8

foreach ($wave in @(@("train", ""), @("test", "_test"))) {
    $split = $wave[0]; $suffix = $wave[1]
    $procs = @()
    foreach ($n in 1, 2, 3) {
        $in = Join-Path $ds "$split\${split}_source$n.tsv"
        $out = Join-Path $root "data\normalized\source${n}_normalized$suffix.parquet"
        $procs += Start-Process -FilePath $py -ArgumentList @("src\normalization.py", "--input", $in, "--output", $out) `
            -NoNewWindow -PassThru -RedirectStandardError (Join-Path $root "work_normalize_${split}_$n.err")
        "launched $split source$n pid $($procs[-1].Id) $(Get-Date -Format s)" | Out-File $log -Append -Encoding utf8
    }
    $procs | Wait-Process
    foreach ($p in $procs) { "finished pid $($p.Id) exit $($p.ExitCode) $(Get-Date -Format s)" | Out-File $log -Append -Encoding utf8 }
}
"ALL DONE $(Get-Date -Format s)" | Out-File $log -Append -Encoding utf8
