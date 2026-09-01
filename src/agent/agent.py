"""COVID prepare agent: parse user choices, call registered tools, stop when input is missing.

    PYTHONPATH=src python -m agent --years 2022 --source local
    PYTHONPATH=src python -m agent --task forecast_prepare
    PYTHONPATH=src python -m agent --task forecast_prepare --lookback 14 --horizon 7
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from agent.guardrails import request_approval
from agent.instructions import WINDOW_PROMPT
from agent.schemas import ALLOWED_TASKS, TASKS_NEEDING_YEARS, TRAVEL_TIME_SOURCES, CANDIDATE_SITE_SOURCES
from agent.state import AgentState
from agent.validators import (
    coerce_source,
    coerce_task,
    coerce_years,
    extract_source,
    extract_task,
    extract_window,
    extract_years,
    window_from_payload,
)
from common.utils import (
    ALLOWED_DATA_SOURCES,
    LOCAL_AUTHORITY_CODE,
    LOCAL_AUTHORITY_NAME,
    get_logger,
    write_run_log,
)

LOGGER = get_logger("agent")


def interpret_request(request: str | dict[str, Any]) -> AgentState:
    """Extract task, years, source, and optional forecast window.

    covid_prepare still requires years and data_source. forecast_prepare
    defaults lookback and horizon to 7 if the user omitted them.
    """
    if isinstance(request, dict):
        text = str(request.get("request") or request.get("text") or "")
        years = coerce_years(request.get("years"))
        source = coerce_source(request.get("source") or request.get("data_source"))
        if source is None:
            source = extract_source(text)
        area_code = str(request.get("area_code") or LOCAL_AUTHORITY_CODE)
        osm_place = request.get("osm_place")
        sites_path = request.get("sites_path")
        task = coerce_task(request.get("task")) or extract_task(text)
        lookback, horizon = window_from_payload(request)
        extracted = extract_window(text)
        if extracted is not None:
            if lookback is None:
                lookback = extracted[0]
            if horizon is None:
                horizon = extracted[1]
    else:
        text = str(request or "")
        years = extract_years(text)
        source = extract_source(text)
        area_code = LOCAL_AUTHORITY_CODE
        osm_place = None
        sites_path = None
        task = extract_task(text)
        extracted = extract_window(text)
        lookback, horizon = extracted if extracted is not None else (None, None)

    state = AgentState(request=text)
    state.task = task or "covid_prepare"
    state.years = years
    state.data_source = source
    state.lookback_days = lookback
    state.forecast_horizon_days = horizon
    state.area_code = area_code
    if area_code == LOCAL_AUTHORITY_CODE:
        state.area_name = LOCAL_AUTHORITY_NAME
    else:
        state.area_name = str(request.get("area_name") or area_code) if isinstance(request, dict) else area_code
    state.osm_place = str(osm_place).strip() if osm_place else None
    state.sites_path = str(sites_path).strip() if sites_path else None

    if state.task in TASKS_NEEDING_YEARS and not state.years:
        state.missing_parameters.append("years")
    if state.task == "covid_prepare" and state.data_source not in ALLOWED_DATA_SOURCES:
        state.missing_parameters.append("data_source")
    if state.task == "travel_time_prepare" and state.data_source not in TRAVEL_TIME_SOURCES:
        state.missing_parameters.append("data_source")
    if state.task == "candidate_sites_prepare" and state.data_source not in CANDIDATE_SITE_SOURCES:
        state.missing_parameters.append("data_source")
    if state.task == "forecast_prepare":
        from data.dataset import FORECAST_HORIZON_DAYS, LOOKBACK_DAYS

        defaulted = False
        if state.lookback_days is None:
            state.lookback_days = LOOKBACK_DAYS
            state.warnings.append(f"lookback_days defaulted to {LOOKBACK_DAYS}")
            defaulted = True
        if state.forecast_horizon_days is None:
            state.forecast_horizon_days = FORECAST_HORIZON_DAYS
            state.warnings.append(f"forecast_horizon_days defaulted to {FORECAST_HORIZON_DAYS}")
            defaulted = True
        if defaulted:
            state.warnings.append(WINDOW_PROMPT)
    state.status = "awaiting_user" if state.missing_parameters else "planned"
    return state


def run_plan(request: str | dict[str, Any]) -> dict[str, Any]:
    """Run one registered data task. Required user choices are never invented."""
    from agent.tools import get_registry, register_default_tools

    register_default_tools()
    state = interpret_request(request)
    if state.missing_parameters:
        result = {
            "status": "awaiting_user",
            "task": state.task,
            "missing_parameters": state.missing_parameters,
            "prompts": request_approval(state),
            "state": state.as_dict(),
        }
        write_run_log({"event": "blocked_missing_parameters", **result})
        return result

    LOGGER.info("Task: %s", state.task)
    registry = get_registry()
    if state.task == "covid_prepare":
        return _run_covid_prepare(state, registry)
    if state.task == "forecast_prepare":
        return _run_forecast_prepare(state, registry)
    if state.task == "travel_time_prepare":
        return _run_travel_time_prepare(state, registry)
    if state.task == "candidate_sites_prepare":
        return _run_candidate_sites_prepare(state, registry)
    if state.task == "inventory":
        return {"status": "ok", "task": state.task, **registry["inventory_raw_datasets"]()}
    if state.task == "planning":
        from agent.planning import run_planning

        payload = request if isinstance(request, dict) else {"request": request, "task": "planning"}
        return run_planning(payload)
    raise ValueError(f"Unknown task: {state.task}")


def _run_covid_prepare(state: AgentState, registry: dict) -> dict[str, Any]:
    plan = ["acquire_data", "preprocess_covid"]
    LOGGER.info("Plan: %s (source=%s)", plan, state.data_source)
    acquired = registry["acquire_data"](
        years=state.years,
        area_code=state.area_code,
        source=state.data_source,
    )
    from data.covid import load_iz_master

    iz_master = load_iz_master(area_code=state.area_code)
    prepared = registry["preprocess_covid"](
        frames=acquired["frames"],
        years=state.years,
        iz_master=iz_master,
        area_code=state.area_code,
        area_name=state.area_name,
    )
    result = {
        "status": "ok",
        "task": "covid_prepare",
        "years": state.years,
        "data_source": state.data_source,
        "area_code": state.area_code,
        "plan": plan,
        "selected_paths": [item["output_path"] for item in acquired["provenance"]],
        "primary_scenario": prepared["primary_scenario"],
        "sensitivity_scenarios": prepared["sensitivity_scenarios"],
        "output_paths": prepared["output_paths"],
        "report_path": prepared["report_path"],
        "provenance_path": acquired["provenance_path"],
        "preprocessing_report": prepared["report"],
        "warnings": [
            *acquired.get("warnings", []),
            *prepared.get("warnings", []),
        ],
        "n_rows": prepared["report"]["n_rows_after_area_filter"],
        "n_suppressed": prepared["report"]["response_status_counts"].get("disclosure_controlled_0_2", 0),
    }
    write_run_log(
        {
            "event": "covid_prepare_complete",
            **{k: v for k, v in result.items() if k != "preprocessing_report"},
        }
    )
    return result


def _run_forecast_prepare(state: AgentState, registry: dict) -> dict[str, Any]:
    LOGGER.info(
        "Plan: prepare_forecast_dataset (lookback=%s, horizon=%s)",
        state.lookback_days,
        state.forecast_horizon_days,
    )
    prepared = registry["prepare_forecast_dataset"](
        lookback_days=state.lookback_days,
        forecast_horizon_days=state.forecast_horizon_days,
    )
    result = {
        "status": "ok",
        "task": "forecast_prepare",
        "lookback_days": state.lookback_days,
        "forecast_horizon_days": state.forecast_horizon_days,
        "plan": ["prepare_forecast_dataset"],
        **prepared,
    }
    write_run_log(
        {
            "event": "forecast_prepare_complete",
            "task": result["task"],
            "lookback_days": state.lookback_days,
            "forecast_horizon_days": state.forecast_horizon_days,
            "config_id": prepared.get("config_id"),
            "n_valid_samples": prepared.get("n_valid_samples"),
            "n_excluded_samples": prepared.get("n_excluded_samples"),
        }
    )
    return result


def _run_travel_time_prepare(state: AgentState, registry: dict) -> dict[str, Any]:
    LOGGER.info(
        "Plan: prepare_travel_time (source=%s, area_code=%s)",
        state.data_source,
        state.area_code,
    )
    prepared = registry["prepare_travel_time"](
        area_code=state.area_code,
        source=state.data_source,
        osm_place=state.osm_place,
        sites_path=state.sites_path,
    )
    result = {
        "status": prepared.get("status", "ok"),
        "task": "travel_time_prepare",
        "area_code": state.area_code,
        "data_source": state.data_source,
        "plan": ["prepare_travel_time"],
        **prepared,
    }
    write_run_log(
        {
            "event": "travel_time_prepare_complete",
            "task": result["task"],
            "area_code": state.area_code,
            "data_source": state.data_source,
            "output_path": prepared.get("output_path"),
            "n_rows": prepared.get("n_rows"),
        }
    )
    return result


def _run_candidate_sites_prepare(state: AgentState, registry: dict) -> dict[str, Any]:
    LOGGER.info(
        "Plan: prepare_candidate_sites (source=%s, area_code=%s)",
        state.data_source,
        state.area_code,
    )
    prepared = registry["prepare_candidate_sites"](
        area_code=state.area_code,
        source=state.data_source,
        osm_place=state.osm_place,
    )
    result = {
        "status": prepared.get("status", "ok"),
        "task": "candidate_sites_prepare",
        "area_code": state.area_code,
        "data_source": state.data_source,
        "plan": ["prepare_candidate_sites"],
        **prepared,
    }
    write_run_log(
        {
            "event": "candidate_sites_prepare_complete",
            "task": result["task"],
            "area_code": state.area_code,
            "data_source": state.data_source,
            "output_path": prepared.get("output_path"),
            "n_sites": prepared.get("n_sites"),
        }
    )
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Retrieve, merge and complete Edinburgh COVID neighbourhood extracts."
    )
    parser.add_argument("request", nargs="?", default="", help="Natural-language request")
    parser.add_argument("--years", nargs="+", type=int, help="Report years, e.g. 2021 2022")
    parser.add_argument(
        "--source",
        choices=["api", "local", "osm"],
        help="covid: api/local. travel_time: local (CSV) or osm (compute once).",
    )
    parser.add_argument("--task", choices=list(ALLOWED_TASKS), help="Data tool to run")
    parser.add_argument("--forecast-date", help="Planning: forecast/decision date, e.g. 2023-03-04")
    parser.add_argument(
        "--scenario",
        choices=["coverage", "equity", "preventive", "balanced"],
        help="Planning scenario",
    )
    parser.add_argument("--travel-mode", choices=["drive", "walk"], help="Planning travel mode")
    parser.add_argument("--threshold", type=float, help="Planning travel-time threshold in minutes")
    parser.add_argument("--site-types", help="Planning: comma-separated gp,pharmacy,mobile_stop")
    parser.add_argument("--priority-population", help="Planning priority population label")
    parser.add_argument(
        "--confirm-rolling-update",
        action="store_true",
        help="Required before a rolling model update. Never silent.",
    )
    parser.add_argument(
        "--confirm-new-region-training",
        action="store_true",
        help="Required before new-region training. Never silent.",
    )
    parser.add_argument("--area-code", help="Study area CA code. Default S12000036 (City of Edinburgh).")
    parser.add_argument("--osm-place", help="OSMnx place string when computing travel time for a city.")
    parser.add_argument("--sites-path", help="Candidate-site shapefile or CSV for travel-time destinations.")
    parser.add_argument(
        "--window",
        type=int,
        help="Forecast task: set lookback and horizon to the same number. Use --lookback and --horizon if they differ.",
    )
    parser.add_argument("--lookback", type=int, help="Forecast task: input lookback days (consecutive daily reports)")
    parser.add_argument(
        "--horizon",
        type=int,
        help="Forecast task: lead time in days to one rolling-seven-day target. Not an H-day cumulative rate.",
    )
    args = parser.parse_args(argv)
    payload: dict[str, Any] = {"request": args.request}
    if args.years:
        payload["years"] = args.years
    if args.source:
        payload["source"] = args.source
    if args.task:
        payload["task"] = args.task
    if args.window is not None:
        payload["window"] = args.window
    if args.lookback is not None:
        payload["lookback"] = args.lookback
    if args.horizon is not None:
        payload["horizon"] = args.horizon
    if args.area_code:
        payload["area_code"] = args.area_code
    if args.osm_place:
        payload["osm_place"] = args.osm_place
    if args.sites_path:
        payload["sites_path"] = args.sites_path
    if args.forecast_date:
        payload["forecast_date"] = args.forecast_date
    if args.scenario:
        payload["scenario"] = args.scenario
    if args.travel_mode:
        payload["travel_mode"] = args.travel_mode
    if args.threshold is not None:
        payload["travel_time_threshold_min"] = args.threshold
    if args.site_types:
        payload["eligible_site_types"] = args.site_types
    if args.priority_population:
        payload["priority_population"] = args.priority_population
    payload["confirm_rolling_update"] = bool(args.confirm_rolling_update)
    payload["confirm_new_region_training"] = bool(args.confirm_new_region_training)
    result = run_plan(payload)
    print(json.dumps({k: v for k, v in result.items() if k != "state"}, indent=2, default=str))


if __name__ == "__main__":
    main()
