# Reproduces the whole pipeline end to end (Windows PowerShell). Uses the .venv at the repository root.
# Duration: several hours depending on thermal pauses (RTX 4050 Laptop).
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$py = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
& $py -m pytest -q                                   ; if (-not $?) { exit 1 }
& $py -m ttamon.thermal_guard --probe                # supported sensors
& $py scripts\train_source.py --seed 45              # does not overwrite an existing checkpoint
& $py scripts\train_source.py --seed 46
& $py scripts\verify_audit.py                        # audit checks (V1-V7)
& $py scripts\collect_predictions.py --plan val main ablation stream
& $py scripts\analyze.py
& $py scripts\make_figures.py
