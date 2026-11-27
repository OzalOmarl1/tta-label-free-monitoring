# Label-Free Reliability Monitoring after TTA (`ttamon`)

**Can We Trust the Adapted Model Without Labels? Validity of Label-Free Accuracy Estimators after Test-Time Adaptation and an Adapt-or-Not Gate**

Ozal Omarli (260801900) — Advisor: Asst. Prof. Ramin Abbaszadi — OSTİM Technical University, 2026–2027 Fall

## What does it do?
Source, BN-adapt and TENT are run on 5 independent ResNet-18 source models, 17 distribution domains (clean CIFAR-10, 15× CIFAR-10-C severity 5, CIFAR-10.1) and 2 stream conditions (i.i.d. / label-sorted, N=64). The post-TTA error structure of the AC, AC-TS, DoC, ATC-MC, ATC-NE and GDE (agreement between independent models) estimators is then measured with preregistered hypothesis tests, and a label-free "adapt / do not adapt" gate is evaluated.

## Folder structure
```
Final/
├── ttamon/                 # package: data, models, methods (Source/BN-adapt/TENT), estimators, metrics, stats, thermal_guard, config
├── scripts/
│   ├── train_source.py         # source model training (--seed)
│   ├── verify_audit.py         # reproduces the claims of the original repository (AUDIT V1–V7)
│   ├── collect_predictions.py  # all TTA runs -> results/raw/*.npz (resumes where it stopped)
│   ├── analyze.py              # metrics, estimates, hypothesis tests -> results/summary.json, tables/
│   └── make_figures.py         # figures/*.png
├── results/                # all_runs.csv, summary.json, tables/, logs/ (incl. thermal logs), raw/ (logits, see Releases)
├── figures/
├── thermal/THERMAL_SUPPORT.md
├── tests/                  # pytest (method integrity, streams, estimators, statistics, thermal guard)
├── PREREGISTRATION.md      # preregistered hypotheses + deviations
├── requirements.txt
└── run_all.ps1 / run_all.sh
```
Data (`../data/`) and checkpoints (`../src/models/checkpoints/`) are shared at the repository root (`ttamon/paths.py`; can be changed with the `TTAMON_DATA` and `TTAMON_CKPT` environment variables).

## Installation and running
```powershell
# at the repository root
uv venv .venv --python 3.12
uv pip install --python .venv\Scripts\python.exe -r Final\requirements.txt --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple
cd Final
..\.venv\Scripts\python.exe -m pytest -q
.\run_all.ps1
```

## Reproducibility
* **Seeds:** models 42, 43, 44 (original, SHA256 verified), 45, 46 (this study); main stream seed 2026; stream ablation 11/12/13 (model 42).
* **Determinism:** `torch.backends.cudnn.deterministic=True`, `benchmark=False`; Python/NumPy/Torch seeds fixed.
* **Hash record:** `results/logs/checkpoint_hashes_final.json`.
* **Source of the numbers:** every reported number is generated automatically into `results/summary.json` and `results/tables/`; no result is entered by hand.
* **Hardware:** NVIDIA RTX 4050 Laptop GPU (6 GB), i5-12500H; Python 3.12.15, PyTorch 2.11.0+cu128.

## Main results (automatic: results/summary.json)

| Condition | Method | Shifted accuracy (%) | ECE (%) | GDE MAE (pp) | ATC-MC src MAE (pp) |
|---|---|---|---|---|---|
| iid64 | Source | 56.33 ± 0.52 | 30.03 ± 0.54 | 15.77 ± 0.37 | 15.03 ± 1.04 |
| iid64 | BN-adapt | 76.62 ± 0.86 | 13.66 ± 0.46 | 1.64 ± 0.50 | 3.57 ± 0.88 |
| iid64 | TENT | 79.82 ± 0.59 | 13.81 ± 0.43 | 1.38 ± 0.42 | 7.05 ± 0.87 |
| sorted64 | Source | 56.33 ± 0.52 | 30.03 ± 0.54 | 15.77 ± 0.37 | 15.03 ± 1.04 |
| sorted64 | BN-adapt | 20.99 ± 0.59 | 57.10 ± 0.43 | 27.07 ± 0.52 | 33.50 ± 2.19 |
| sorted64 | TENT | 19.74 ± 0.38 | 63.84 ± 0.30 | 25.99 ± 0.47 | 45.82 ± 1.69 |

Adapt-or-not gate regret (pp, 2 conditions): GDE gate 0.06 ± 0.05; always Source 11.85 ± 0.18; always TENT 18.40 ± 0.16.

Hypothesis decisions: **H1** — not supported (insufficient evidence); **H2** — AC: not supported, AC-TS: supported, DoC: not supported, ATC-MC: supported, ATC-NE: supported; **H3** — BN-adapt: not supported, TENT: supported; **H4** — all 4 tests supported; **H5** — supported.

## Thermal safety
`ttamon/thermal_guard.py` logs CPU/GPU temperatures in the background; it pauses the workload at CPU 85 °C / GPU 80 °C and resumes after cooling down. BIOS/firmware/voltage/overclock/fan settings are never touched. Details: `thermal/THERMAL_SUPPORT.md`.

## Known limitations
Single architecture (ResNet-18), CIFAR scale, 5 model seeds, unofficial CIFAR-10-C mirror (alignment verified, official checksum could not be verified), N=16 condition dropped for thermal reasons. Details: `PREREGISTRATION.md`.
