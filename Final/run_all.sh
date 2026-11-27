#!/usr/bin/env bash
# Linux/macOS equivalent (python = project environment)
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
$PY -m pytest -q
$PY -m ttamon.thermal_guard --probe
$PY scripts/train_source.py --seed 45
$PY scripts/train_source.py --seed 46
$PY scripts/verify_audit.py
$PY scripts/collect_predictions.py --plan val main ablation stream
$PY scripts/analyze.py
$PY scripts/make_figures.py
