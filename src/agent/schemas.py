"""Agent task names and tool return envelope."""

from __future__ import annotations

from typing import Any

ALLOWED_TASKS = (
    "covid_prepare",
    "inventory",
    "forecast_prepare",
    "travel_time_prepare",
    "candidate_sites_prepare",
    "planning",
)
TASKS_NEEDING_YEARS = {"covid_prepare"}
TASKS_NEEDING_SOURCE = {"covid_prepare", "travel_time_prepare", "candidate_sites_prepare"}
TRAVEL_TIME_SOURCES = ("local", "osm")
CANDIDATE_SITE_SOURCES = ("local", "api", "osm")

# Tools the agent may call. Implementations are registered in agent.tools.
CALLABLE_TOOL_NAMES = (
    "acquire_data",
    "preprocess_covid",
    "inventory_raw_datasets",
    "prepare_forecast_dataset",
    "validate_inputs",
    "load_temporal_dataset",
    "build_graph_supports",
    "train_model",
    "load_checkpoint",
    "forecast_single_target",
    "evaluate_test_period",
    "explain_target_iz_with_geoshapley",
    "export_map_ready_results",
    "run_rolling_evaluation",
    "export_operational_forecast",
    "prepare_travel_time",
    "load_travel_time",
    "prepare_candidate_sites",
    "load_candidate_sites",
    "load_healthcare_layers",
    "acquire_healthcare_table",
    "export_iz_origins",
    "check_model_compatibility",
    "prepare_or_validate_region",
    "select_checkpoint",
    "forecast_inference",
    "trigger_rolling_update",
    "trigger_new_region_training",
    "run_location_allocation",
    "validate_allocation_result",
    "get_site_iz_info",
    "compare_allocation_scenarios",
    "generate_web_layers",
)


def tool_envelope(
    status: str,
    outputs: dict[str, Any],
    warnings: list[dict[str, Any]],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    return {
        "status": status,
        "outputs": outputs,
        "warnings": warnings,
        "provenance": provenance,
    }
