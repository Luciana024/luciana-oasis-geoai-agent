# Frozen inference/result reproduction artefacts

Use this archive together with `oasis_geoai_agent_source_release.tar.gz` to
verify the published Edinburgh and Glasgow forecasts, explanations and
six-site allocations without retraining U01--U10.

The archive contains the final U10 checkpoints and their manifests,
calibration/scaler files, temporal inputs, graph supports, canonical node
orders, operational exports, GeoShapley output, candidate sites, travel-time
matrices, allocation results and frozen publication/dashboard exports.

Extract this archive over the source-release root so that paths begin with
`data/`. Run the source-only tests first, then:

```bash
pytest -q -m external_data
python scripts/run_operational_forecast.py
python scripts/run_allocation.py --help
python scripts/start_dashboard.py
```

The 4 March 2023 operational layer is an unverified extrapolation and must not
be included in retrospective accuracy metrics. Exact GPU floating-point bytes
may vary by platform; compare reported metrics at their published precision.
