"""Names aligned with docs/model.md and configs/model.yaml.

Do not hard-code N=111 inside layers. 111 is an Edinburgh data check only.
"""

from __future__ import annotations

# SIMD 2020 one-indicator-per-domain names. Must match data/dataset.py arrays.
STATIC_FEATURE_COLUMNS = (
    "income_rate",
    "employment_rate",
    "university_rate",
    "overcrowded_rate",
    "crime_rate",
    "pt_gp_min",
)

MODEL_NAME = (
    "Probabilistic Adaptive Multi-Graph DCRNN Encoder with Contextual Node Embedding"
)
HEAD_NAME = "UQGNN-inspired univariate probabilistic prediction head"
CALIBRATION_METHOD = "finite-sample corrected empirical calibration"
EXPLANATION_SCOPE = "target_iz_local"

GRAPH_GEO = "geo"
GRAPH_TRANSPORT = "transport"
GRAPH_MOBILITY = "mobility"
THREE_GRAPH_SET = (GRAPH_GEO, GRAPH_TRANSPORT, GRAPH_MOBILITY)
TWO_GRAPH_SET = (GRAPH_GEO, GRAPH_TRANSPORT)

# GeoShapley player names; order must match STATIC_FEATURE_COLUMNS.
FEATURE_PLAYER_NAMES = (
    "income_deprivation",
    "employment_deprivation",
    "higher_education",
    "overcrowding",
    "crime",
    "public_transport_time_to_gp",
)
LOCATION_PLAYER = "location"
FEATURE_TO_PLAYER = dict(zip(STATIC_FEATURE_COLUMNS, FEATURE_PLAYER_NAMES, strict=True))
INTERACTION_PLAYER_NAMES = tuple(f"location_x_{name}" for name in FEATURE_PLAYER_NAMES)

DISPLAY_CALIBRATED = "calibrated_95"
DISPLAY_RAW = "raw_gaussian_95"

CANONICAL_HASH_JOIN = "\n"
LEGACY_HASH_JOIN = "|"

EXCHANGEABILITY_LIMITATION = (
    "Rolling seven-day COVID-19 outcomes are temporally overlapping and "
    "spatially dependent, so calibrated intervals are empirical uncertainty "
    "intervals rather than intervals with a formal exchangeability-based "
    "coverage guarantee."
)
LOCATION_LABEL = (
    "Residual spatial/location contribution conditional on the fixed graph structure."
)
INTERACTION_LABEL = (
    "Location–feature interaction for the named original variable, "
    "conditional on the fixed graph structure."
)
