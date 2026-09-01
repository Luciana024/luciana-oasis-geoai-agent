"""Replaceable allocation contract. The language model does not choose sites."""

from __future__ import annotations

from typing import Any, Callable, Protocol

N_SITES = 6
SCENARIOS = ("coverage", "equity", "preventive", "balanced")
SCENARIO_LABELS = {
    "coverage": "Coverage-priority",
    "equity": "Equity-priority",
    "preventive": "Preventive-priority",
    "balanced": "Balanced",
}
DEFAULT_TRAVEL_MODE = "drive"
DEFAULT_THRESHOLD_MIN = 20.0
TRAVEL_MODES = ("drive", "walk")
SITE_TYPES = ("gp", "pharmacy", "mobile_stop")
EDINBURGH_CA = "S12000036"
DEMO_FORECAST_DATE = "2023-03-04"
DEMO_ISSUE_DATE = "2023-02-25"
U10_CHECKPOINT = (
    "data/results/model/rolling_v1_split65_10_25/final_test/W730/U10/checkpoint.pt"
)
WEBSITE_DIR = "data/results/exports/website_article_v1/website"
FORECAST_CACHE = f"{WEBSITE_DIR}/future_forecast_20230304.csv"
RETROSPECTIVE_CACHE = f"{WEBSITE_DIR}/retrospective_predictions.csv"
DATE_SELECTOR = f"{WEBSITE_DIR}/date_selector.csv"
GEOSHAPLEY_CACHE = f"{WEBSITE_DIR}/geoshapley.csv"
ROLLING_ALPHA = f"{WEBSITE_DIR}/rolling_alpha.csv"
BOUNDARIES_GEOJSON = f"{WEBSITE_DIR}/edinburgh_iz_boundaries.geojson"
MODEL_METADATA = f"{WEBSITE_DIR}/model_metadata.json"

SCENARIO_ALIASES = {
    "coverage": "coverage",
    "coverage priority": "coverage",
    "coverage-priority": "coverage",
    "equity": "equity",
    "equity priority": "equity",
    "equity-priority": "equity",
    "preventive": "preventive",
    "preventive priority": "preventive",
    "preventive-priority": "preventive",
    "balanced": "balanced",
}


class AllocationSolver(Protocol):
    def __call__(self, payload: dict[str, Any]) -> dict[str, Any]:
        ...


def empty_allocation_result(
    *,
    status: str,
    message: str,
    scenario: str,
) -> dict[str, Any]:
    """Same shape the real solver must return. Sites are never invented here."""
    return {
        "status": status,
        "n_sites_required": N_SITES,
        "n_sites_selected": 0,
        "scenario": scenario,
        "selected_sites": [],
        "assignments": [],
        "metrics": {
            "population_covered": None,
            "iz_covered": None,
            "mean_travel_time_min": None,
            "max_travel_time_min": None,
            "unserved_population": None,
            "unserved_iz": None,
        },
        "diagnostics": {"message": message},
        "selection_reasons": {},
        "invented": False,
    }
