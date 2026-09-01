# Reviewer guide

This private repository is the complete runnable Luciana agent submission. It
contains source code plus frozen Edinburgh and Glasgow inference artefacts.

## Fast verification

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,web]'
pytest -q
python scripts/start_dashboard.py
```

Expected test result: `177 passed, 9 deselected`. The nine deselected tests
rebuild or directly inspect separately distributed raw OSM, mobility, lookup
and checkpoint inputs; they are not needed to inspect frozen results or run the
dashboard.

## What is included

- complete production source, configuration, tests and technical documentation;
- final Edinburgh and Glasgow U10 checkpoints and calibration metadata;
- temporal tensors, graph supports, node-order records and rolling predictions;
- operational forecast and GeoShapley exports;
- candidate-site, travel-time and four-policy allocation outputs;
- Streamlit dashboard assets and frozen publication tables.

## Important interpretation

The agent is rule-based and tool-using; it is not an LLM agent. The DCRNN
produces forecasts, GeoShapley explains forecasts, and the deterministic greedy
solver selects exactly six recorded candidates. The 4 March 2023 planning
layer is an unverified extrapolation and is excluded from retrospective
accuracy metrics.

Full raw-data retraining artefacts are maintained as a separate controlled
archive because redistribution rights and file size differ from the codebase.
