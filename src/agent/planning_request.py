"""Validate a planning request. Number of sites is fixed at six and not asked."""

from __future__ import annotations

from typing import Any

from allocation.contracts import (
    DEMO_FORECAST_DATE,
    EDINBURGH_CA,
    N_SITES,
    SCENARIO_ALIASES,
    SCENARIOS,
    SITE_TYPES,
    TRAVEL_MODES,
)
from common.errors import ModelError
from common.utils import LOCAL_AUTHORITY_CODE, LOCAL_AUTHORITY_NAME

DEFAULT_THRESHOLD_MIN = {"drive": 20.0, "walk": 30.0}


def parse_planning_request(payload: dict[str, Any]) -> dict[str, Any]:
    area_code = str(payload.get("area_code") or LOCAL_AUTHORITY_CODE).strip()
    forecast_date = str(payload.get("forecast_date") or payload.get("decision_date") or DEMO_FORECAST_DATE).strip()
    scenario = SCENARIO_ALIASES.get(str(payload.get("scenario") or "balanced").strip().lower())
    if scenario not in SCENARIOS:
        raise ModelError(
            f"Planning scenario must be one of {list(SCENARIOS)}.",
            code="invalid_config",
        )
    travel_mode = str(payload.get("travel_mode") or "drive").strip().lower()
    if travel_mode not in TRAVEL_MODES:
        raise ModelError(
            f"Travel mode must be one of {list(TRAVEL_MODES)}.",
            code="invalid_config",
        )
    raw_types = payload.get("eligible_site_types") or list(SITE_TYPES)
    if isinstance(raw_types, str):
        raw_types = [part.strip() for part in raw_types.split(",") if part.strip()]
    site_types = [str(item).strip().lower() for item in raw_types]
    unknown = [item for item in site_types if item not in SITE_TYPES]
    if unknown:
        raise ModelError(
            f"Unknown candidate-site types {unknown}. Allowed: {list(SITE_TYPES)}.",
            code="invalid_config",
        )
    threshold = payload.get("travel_time_threshold_min")
    if threshold is None:
        threshold = DEFAULT_THRESHOLD_MIN[travel_mode]
    threshold = float(threshold)
    if threshold <= 0:
        raise ModelError("travel_time_threshold_min must be positive.", code="invalid_config")
    return {
        "area_code": area_code,
        "area_name": str(payload.get("area_name") or (LOCAL_AUTHORITY_NAME if area_code == EDINBURGH_CA else area_code)),
        "forecast_date": forecast_date,
        "scenario": scenario,
        "travel_mode": travel_mode,
        "travel_time_threshold_min": threshold,
        "eligible_site_types": site_types,
        "priority_population": str(payload.get("priority_population") or "all").strip(),
        "n_sites": N_SITES,
        "n_sites_is_fixed": True,
        "confirm_rolling_update": bool(payload.get("confirm_rolling_update")),
        "confirm_new_region_training": bool(payload.get("confirm_new_region_training")),
        "prefer_live_forecast": bool(payload.get("prefer_live_forecast")),
    }
