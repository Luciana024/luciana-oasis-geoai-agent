# Reviewer Reproduction Guide

This repository contains the complete runnable Luciana GeoAI Health Agent submission. It includes the application source code and the frozen inference artefacts needed to inspect the Edinburgh and Glasgow results. A GPU, model retraining, and access to the authors' computer are not required for the standard review workflow.

## 1. System requirements

- Git
- Python 3.10 or newer
- Approximately 4 GB of free disk space
- A modern web browser

Python 3.13.9 was used for the final verification. The commands below use an isolated virtual environment and do not modify the system Python installation.

## 2. Download an exact copy

```bash
git clone https://github.com/Luciana024/luciana-oasis-geoai-agent.git
cd luciana-oasis-geoai-agent
git checkout oasis-2026-submission-v10
```

The tag identifies the exact reviewed version. Reviewers who want the newest development version may remain on the `main` branch instead.

## 3. Create the environment

### Linux or macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,web]'
```

### Windows PowerShell

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,web]"
```

Dependency installation requires an internet connection. Subsequent use of the frozen dashboard results does not require remote data retrieval.

## 4. Verify the repository

Run the automated test suite from the repository root:

```bash
pytest -q
```

The expected result is:

```text
185 passed, 9 deselected
```

The nine deselected tests rebuild or directly inspect separately distributed raw OpenStreetMap, mobility, lookup, and model-training inputs. Those large or access-controlled inputs are not required to run the dashboard or inspect its frozen results.

## 5. Start the dashboard

```bash
python scripts/start_dashboard.py
```

Streamlit prints a local address in the terminal, normally:

```text
http://localhost:8501
```

Open the printed address in a browser. Every reviewer runs an independent local copy; the public demonstration URL and the authors' terminal are not required. Stop the server with `Ctrl+C` in the terminal.

## 6. Reproduce the interface workflow

1. Read **About values, policies and responsible use** on the home page.
2. Choose Edinburgh or Glasgow and select an available forecast date.
3. Select **Task 1: Show the forecast** and choose either **Whole region** or an Intermediate Zone.
4. For a Whole region result, answer the Agent's follow-up question about the alpha graph-mix explanation. The explanation is displayed only after the user chooses **Yes, show alpha results**.
5. Return to the forecast or choose an Intermediate Zone using the options presented after the alpha result.
6. Select **Task 2: Plan 6 vaccination sites** to choose a policy and travel constraint. Each policy button includes a plain-language explanation.
7. Select **Task 3: Compare four policies** to compare coverage, equity, preventive, and balanced planning perspectives.
8. Use the floating **Terminology Guide** on the right side of the page for plain-language definitions. It follows the page while scrolling and expands on hover or keyboard focus.
9. Reopen an earlier result by selecting its associated message in the conversation history.

The available date selector is authoritative: it lists only dates for which a stored forecast exists. Invalid free-text dates are rejected before the analytical workflow starts.

## 7. What is included

- Production source code, configuration, tests, and technical documentation
- Final Edinburgh and Glasgow U10 checkpoints and calibration metadata
- Temporal tensors, graph supports, node-order records, and rolling predictions
- Operational forecast and GeoShapley exports
- Candidate-site, travel-time, and four-policy allocation outputs
- Streamlit dashboard assets and frozen publication tables

The agent is rule-based and tool-using; it is not a large language model. The DCRNN produces forecasts, GeoShapley explains forecasts, and the deterministic greedy solver selects exactly six recorded candidates. The 4 March 2023 planning layer is an unverified extrapolation and is excluded from retrospective accuracy metrics.

## 8. Troubleshooting

- **The browser does not open automatically:** copy the `Local URL` printed by Streamlit into the browser.
- **Port 8501 is already in use:** Streamlit may select another port; always use the address printed in the terminal.
- **`python` is not found:** use `python3` on Linux or macOS, or `py` on Windows.
- **A module is missing:** confirm that the virtual environment is active and rerun `python -m pip install -e '.[dev,web]'` (use double quotes in PowerShell).
- **The page shows stale content:** stop the server with `Ctrl+C`, restart it, and refresh the browser without cache.

Full raw-data retraining artefacts are maintained in a separate controlled archive because redistribution rights and file sizes differ from the codebase. See [REPRODUCING.md](REPRODUCING.md) and [ARTIFACTS.md](ARTIFACTS.md) for the technical reproduction boundaries.

## 9. Restricted raw data for training from scratch

Some source public-health and origin-destination (OD) mobility data cannot be published directly in this GitHub repository because of access permissions, copyright, licensing, and privacy constraints. The permitted subset of raw reproduction data is available through the following controlled link:

[Download the competition reproduction data from Google Drive](https://drive.google.com/drive/folders/1_b6JuLO_Rd1fRhxewm0QJjb-Rbn23EW7?usp=drive_link)

These raw files are provided solely to reproduce the competition results and remain subject to the original data owners' access conditions, licences, and usage restrictions. They must not be redistributed or used for unrelated purposes without the appropriate permission.

The external raw-data bundle is needed only when rebuilding and training the models from the beginning. It is not required to install the application, run the dashboard, inspect the uploaded frozen predictions and explanations, or reproduce the displayed site-allocation results. Those trained checkpoints and processed outputs are already included in this repository.
