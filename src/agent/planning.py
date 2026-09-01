"""End-to-end planning workflow. Tools do all numbers; the agent does not siting."""

from __future__ import annotations

from typing import Any

from agent.planning_request import parse_planning_request
from allocation.contracts import EDINBURGH_CA
from agent.planning_tools import (
    check_model_compatibility,
    compare_allocation_scenarios,
    forecast_inference,
    generate_web_layers,
    prepare_or_validate_region,
    run_location_allocation,
    select_checkpoint,
    trigger_new_region_training,
    trigger_rolling_update,
    validate_allocation_result,
)
from common.utils import get_logger, write_run_log

LOGGER = get_logger("agent.planning")


def run_planning(payload: dict[str, Any]) -> dict[str, Any]:
    request = parse_planning_request(payload)
    compatibility = check_model_compatibility(request)
    mode = compatibility["mode"]
    model_action = "reused"
    blockers: list[dict[str, Any]] = []

    checkpoint = None
    forecast = None
    if mode == "rolling_update":
        update = trigger_rolling_update(request)
        blockers.append(update)
        if update.get("status") != "ok":
            result = _bundle(request, compatibility, model_action="blocked", blockers=blockers)
            write_run_log({"event": "planning_blocked_rolling", **_summary(result)})
            return result
        model_action = "updated"
    elif mode == "new_region_training":
        train = trigger_new_region_training(request)
        blockers.append(train)
        if train.get("status") != "ok":
            result = _bundle(request, compatibility, model_action="blocked", blockers=blockers)
            write_run_log({"event": "planning_blocked_training", **_summary(result)})
            return result
        model_action = "retrained" if train.get("retrained") else "reused"
        checkpoint = {
            "status": "ok",
            "checkpoint_id": train.get("checkpoint_id"),
            "checkpoint_path": train.get("checkpoint_path"),
            "update_id": train.get("checkpoint_id"),
            "target_date": request["forecast_date"],
        }
        forecast = {
            "status": "ok",
            "source": "new_region_training",
            "forecast_path": train.get("forecast_path"),
            "geoshapley_path": train.get("geoshapley_path"),
            "forecast_status": "unverified_extrapolation",
            "retrained": bool(train.get("retrained")),
            "checkpoint_id": train.get("checkpoint_id"),
            "n_iz": train.get("n_iz"),
            "target_date": request["forecast_date"],
            "label": train.get("message"),
        }

    if checkpoint is None:
        checkpoint = select_checkpoint(compatibility)
    if forecast is None:
        forecast = forecast_inference(request, compatibility)
    region = prepare_or_validate_region(request)
    allocation = run_location_allocation(request, forecast, region)
    validation = validate_allocation_result(allocation, request)
    comparison = compare_allocation_scenarios(request, forecast, region)
    bundle = _bundle(
        request,
        compatibility,
        model_action=model_action,
        checkpoint=checkpoint,
        region=region,
        forecast=forecast,
        allocation=allocation,
        validation=validation,
        comparison=comparison,
        blockers=blockers,
    )
    web_dir = None
    if request["area_code"] != EDINBURGH_CA:
        from agent.region_training import region_output_dir

        web_dir = region_output_dir(request["area_code"]) / "planning"
    bundle["website"] = generate_web_layers(bundle, output_dir=web_dir)
    write_run_log({"event": "planning_complete", **_summary(bundle)})
    return bundle


def _bundle(request: dict[str, Any], compatibility: dict[str, Any], **parts: Any) -> dict[str, Any]:
    allocation = parts.get("allocation") or {}
    forecast = parts.get("forecast") or {}
    status = "ok"
    if compatibility.get("mode") != "inference" and parts.get("model_action") == "blocked":
        status = "needs_confirmation"
    elif allocation.get("status") == "not_wired":
        status = "awaiting_allocator"
    elif forecast.get("status") not in (None, "ok"):
        status = "failed"
    return {
        "status": status,
        "task": "planning",
        "request": request,
        "compatibility": compatibility,
        "model_action": parts.get("model_action"),
        "checkpoint": parts.get("checkpoint"),
        "region": parts.get("region"),
        "forecast": forecast,
        "allocation": allocation,
        "validation": parts.get("validation"),
        "comparison": parts.get("comparison"),
        "blockers": parts.get("blockers") or [],
        "n_sites_fixed": request["n_sites"],
        "forecast_label": forecast.get("label")
        or "4 March 2023 forecast is an unverified extrapolation.",
    }


def _summary(bundle: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": bundle.get("status"),
        "task": "planning",
        "mode": (bundle.get("compatibility") or {}).get("mode"),
        "model_action": bundle.get("model_action"),
        "n_sites": (bundle.get("allocation") or {}).get("n_sites_selected"),
        "forecast_source": (bundle.get("forecast") or {}).get("source"),
    }
