# Luciana / OASIS GeoAI Agent

Luciana is a rule-based, tool-using GeoAI agent for seven-day COVID-19 rate
forecasting and deterministic six-site vaccination planning in Edinburgh and
Glasgow. It is not an LLM agent: forecasts are produced by a probabilistic
multi-graph DCRNN and sites are selected by a deterministic greedy solver.

## Scope

- Seven historical rolling seven-day rates predict the rate reported seven
  days later for each 2011 Intermediate Zone (IZ).
- Geographic, transport and commuting graphs share one learned softmax fusion
  vector.
- The model returns a mean, predictive standard deviation and empirically
  calibrated intervals.
- Target-local GeoShapley explains forecasts but is not used for site selection.
- Coverage, equity, preventive and balanced policies select exactly six
  recorded GP, pharmacy or mobile-stop candidates.

The validated study areas are City of Edinburgh (`S12000036`, 111 IZs) and
Glasgow City (`S12000049`, 136 IZs). The 4 March 2023 planning layer is an
unverified operational extrapolation and is excluded from retrospective
accuracy metrics.

## Repository layout

```text
configs/       experiment and workflow configuration
docs/          model, data and tool specifications
scripts/       supported command-line entry points
src/           production Python packages
tests/         unit and workflow contract tests
web/           Streamlit dashboard and static assets
```

This official submission repository includes the trained checkpoints and
curated frozen results needed to run the dashboard and reproduce the submitted
Edinburgh and Glasgow outputs. Restricted source data used only for training
from the beginning are distributed separately. See
[REPRODUCING.md](REPRODUCING.md) and [ARTIFACTS.md](ARTIFACTS.md).

## Quick verification

Python 3.10 or newer is required. Python 3.13.9 was used for the final source
verification.

Reviewers should follow the complete, platform-specific instructions in
[REVIEWER_GUIDE.md](REVIEWER_GUIDE.md). The guide explains how to clone the
exact tagged version, install dependencies, verify the tests, start an
independent local dashboard and reproduce each of the three user tasks.

For a section-by-section explanation of the interface, Agent interactions,
planning policies, model explanations and result history, see
[AGENT_INTERFACE_GUIDE.md](AGENT_INTERFACE_GUIDE.md).

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,web]'
pytest -q
PYTHONPATH=src python -m agent --help
```

The expected source-only result is `177 passed, 9 deselected`. After staging
the external artefact archive, run `pytest -q -m external_data` for the nine
data/checkpoint integration checks; the combined suite contains 186 tests.

The curated inference bundle in this repository supplies checkpoints and
published outputs, but intentionally omits the multi-gigabyte raw OSM and OD
sources required by some `external_data` graph-rebuild tests.

### Restricted raw data

Some source public-health and origin-destination (OD) mobility data cannot be
published directly in this repository because of access permissions,
copyright, licensing and privacy constraints. A permitted subset is available
from the [controlled competition reproduction data folder](https://drive.google.com/drive/folders/1_b6JuLO_Rd1fRhxewm0QJjb-Rbn23EW7?usp=drive_link).
It is provided solely for reproducing the competition results and remains
subject to the original data owners' access conditions and usage restrictions.

The external raw-data bundle is needed only to rebuild and train the models
from the beginning. It is not required to run the dashboard or reproduce the
displayed results, because the trained checkpoints, frozen predictions,
explanations and site-allocation outputs are included in this repository.

## Main commands

```bash
# Data agent; years and source are mandatory
PYTHONPATH=src python -m agent --years 2022 --source local

# Prepare fixed seven-day lookback/seven-day target windows
PYTHONPATH=src python -m agent --task forecast_prepare

# Train and evaluate after staging the required artefacts
python scripts/run_training.py
python scripts/run_rolling.py
python scripts/run_operational_forecast.py

# Start the planning dashboard after staging website artefacts
python scripts/start_dashboard.py
```

Configuration `configs/model.yaml` is the main 65/10/25 rolling experiment.
`docs/model.md` is the authoritative model specification.
Training requests CUDA as recorded in the experiment configuration. Frozen
operational inference defaults to CPU so reviewers can reload U10 without a
GPU; this does not change checkpoint parameters or retrospective metrics.

## Reproducibility boundaries

Network retrieval depends on upstream PHS services. Road graph construction
depends on externally distributed Geofabrik/OSM data. The repository never
silently substitutes a missing year, data source, checkpoint, candidate site
or IZ. Missing required artefacts therefore produce an explicit error.

## Licence

Copyright (c) 2026 The Luciana project authors. All rights reserved. This
private repository is licensed only for OASIS 2026 evaluation and
reproducibility assessment; redistribution, publication, commercial use and
distribution of derivative works require prior written permission. See
`LICENSE`. Third-party components remain subject to the terms recorded in
`THIRD_PARTY_NOTICES`.
