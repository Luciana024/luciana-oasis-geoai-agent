# Reproducing the project

## 1. Environment

Create an isolated environment and install the package from the repository
root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,web]'
pytest -q
```

For the exact package versions used in the final verification, install
`requirements-tested.txt`. GPU-enabled PyTorch builds are platform-specific;
use the PyTorch build appropriate for the evaluator's CUDA or CPU platform.

## 2. Stage external artefacts

Create `data/raw`, `data/results`, `outputs`, `logs` and `checkpoints` only as
needed. Copy the official data/checkpoint archive into the repository without
changing relative paths. Required path groups are documented in
`ARTIFACTS.md`; canonical source definitions are in `configs/data.yaml` and
`configs/data_sources.yaml`.

Do not regenerate or overwrite the archived 70/15/15 experiment. The main
experiment uses the frozen S1 windows recut chronologically by target date to
65/10/25.

## 3. Verify source contracts

```bash
pytest -q
PYTHONPATH=src python -m agent --help
PYTHONPATH=src python -m graph --help
```

These checks require no private or large research data.

The expected source-only result is `181 passed, 9 deselected`. After staging
the external archive, run `pytest -q -m external_data` to execute the remaining
nine data/checkpoint integration checks.

## 4. Rebuild the analytical pipeline

After staging source data, run the stages in order:

```bash
# COVID preparation: never omit the explicit source and year
PYTHONPATH=src python -m agent --years 2020 2021 2022 2023 --source local

# Fixed L7/H7/S1 temporal data
PYTHONPATH=src python -m agent --task forecast_prepare

# Graphs are built through the graph CLI; inspect city-specific arguments
PYTHONPATH=src python -m graph --help

# Main model and leakage-safe rolling test
python scripts/run_training.py
python scripts/run_rolling.py

# Operational forecast, explanation and exports
python scripts/run_operational_forecast.py
python scripts/run_geoshapley.py
python scripts/run_allocation.py --help
python scripts/export_website_data.py
```

The selected rolling window is 730 days and the final frozen update is U10.
The validation partition is divided chronologically into checkpoint-selection
and calibration subsets. Test observations must not be used for training,
checkpoint selection, calibration or uncertainty thresholds.

## 5. Expected retrospective results

The official result archive should reproduce these rounded overall metrics:

| City | Model MAE | Persistence MAE | Model RMSE | Model R2 |
|---|---:|---:|---:|---:|
| Edinburgh | 46.53 | 50.66 | 69.06 | 0.67 |
| Glasgow | 44.73 | 47.97 | 63.32 | 0.60 |

Expected empirical calibrated coverage is 0.817/0.949 for Edinburgh and
0.822/0.952 for Glasgow at nominal 80%/95%. The operational 4 March 2023 layer
has no target-day ground truth and must not be included in these metrics.

## 6. Dashboard

The dashboard reads frozen website and planning artefacts. It does not train a
model automatically:

```bash
python scripts/start_dashboard.py
```

If an artefact is absent, restore it at the documented relative path rather
than modifying code to invent a substitute.
