"""Controlled planning tools. Real forecast cache now; allocator is swappable."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from allocation.contracts import (
    DATE_SELECTOR,
    DEMO_FORECAST_DATE,
    DEMO_ISSUE_DATE,
    EDINBURGH_CA,
    FORECAST_CACHE,
    N_SITES,
    RETROSPECTIVE_CACHE,
    SCENARIOS,
    SCENARIO_LABELS,
    U10_CHECKPOINT,
)
from allocation.engine import run_allocation as _run_allocation
from allocation.validate import validate_allocation_result as _validate_allocation
from common.errors import ModelError
from common.utils import get_logger, project_root, write_json
from data.node_order import sha256_file

LOGGER = get_logger("agent.planning_tools")
CANONICAL_HASH = "8f625000ca42af45709b4e887a429c93971443f30f2fbddbe07863342ca16d34"


def _path(rel: str) -> Path:
    path = Path(rel)
    return path if path.is_absolute() else project_root() / path


def _retrospective_lookup(forecast_date: str) -> dict[str, str] | None:
    selector = _path(DATE_SELECTOR)
    if not selector.exists():
        return None
    frame = pd.read_csv(selector)
    if "target_report_date" not in frame.columns:
        return None
    hit = frame.loc[frame["target_report_date"].astype(str) == str(forecast_date)]
    if hit.empty:
        return None
    row = hit.iloc[0]
    return {
        "target_report_date": str(row["target_report_date"]),
        "issue_date": str(row["issue_date"]) if "issue_date" in hit.columns else "",
        "update_id": str(row["update_id"]) if "update_id" in hit.columns else "",
        "checkpoint_id": str(row.get("checkpoint_id") or row.get("update_id") or ""),
    }


def check_model_compatibility(request: dict[str, Any]) -> dict[str, Any]:
    """Decide inference vs rolling update vs new-region training. Do not train here."""
    area_code = str(request["area_code"])
    forecast_date = str(request["forecast_date"])
    checkpoint = _path(U10_CHECKPOINT)
    cache = _path(FORECAST_CACHE)
    retro = _path(RETROSPECTIVE_CACHE)
    same_area = area_code == EDINBURGH_CA
    demo_date = forecast_date == DEMO_FORECAST_DATE
    checkpoint_ok = checkpoint.exists()
    cache_ok = cache.exists()
    retro_row = _retrospective_lookup(forecast_date)
    node_hash = None
    if cache_ok:
        sample = pd.read_csv(cache, nrows=1)
        if "node_order_hash" in sample.columns:
            node_hash = str(sample.iloc[0]["node_order_hash"])
    hash_ok = node_hash == CANONICAL_HASH if node_hash else checkpoint_ok or cache_ok
    replay = False
    checkpoint_id = "U10" if demo_date else (retro_row or {}).get("checkpoint_id") or None
    if same_area and demo_date and (checkpoint_ok or cache_ok) and hash_ok:
        mode = "inference"
        reason = "Edinburgh, 2011 Intermediate Zones, 4 March 2023 forecast from the saved model."
    elif same_area and retro_row is not None and retro.exists():
        mode = "inference"
        replay = True
        reason = (
            f"Replaying stored predictions for {forecast_date}."
        )
    elif same_area and not demo_date:
        mode = "rolling_update"
        reason = "Same area, but the requested forecast date is outside the stored prediction tables."
    else:
        mode = "new_region_training"
        reason = "This city is not Edinburgh, so a city-specific model is required."
    return {
        "status": "ok",
        "mode": mode,
        "reason": reason,
        "area_code": area_code,
        "forecast_date": forecast_date,
        "checkpoint_id": checkpoint_id,
        "checkpoint_path": str(checkpoint) if checkpoint_ok else None,
        "checkpoint_exists": checkpoint_ok or replay or (demo_date and cache_ok),
        "forecast_cache_path": str(cache) if cache_ok else None,
        "node_order_hash": node_hash or CANONICAL_HASH,
        "hash_ok": hash_ok,
        "replay_from_table": replay,
        "needs_confirmation": mode != "inference",
        "will_not_train_silently": True,
    }


def select_checkpoint(compatibility: dict[str, Any]) -> dict[str, Any]:
    if not compatibility.get("checkpoint_exists") and not compatibility.get("replay_from_table"):
        raise ModelError("The Edinburgh model file is missing. Refusing to invent a model.", code="missing_checkpoint")
    checkpoint_id = str(compatibility.get("checkpoint_id") or "U10")
    return {
        "status": "ok",
        "checkpoint_id": checkpoint_id,
        "checkpoint_path": compatibility.get("checkpoint_path"),
        "update_id": checkpoint_id,
        "issue_date": DEMO_ISSUE_DATE if checkpoint_id == "U10" else None,
        "target_date": compatibility.get("forecast_date") or DEMO_FORECAST_DATE,
        "replay_from_table": bool(compatibility.get("replay_from_table")),
    }


def prepare_or_validate_region(request: dict[str, Any]) -> dict[str, Any]:
    """Check candidate sites and travel time exist. Do not invent either."""
    from data.candidate_sites import load_candidate_sites
    from data.travel_time import load_travel_time

    area = request["area_code"]
    warnings: list[str] = []
    sites = None
    matrix = None
    try:
        sites = load_candidate_sites(area_code=area)
    except ModelError as error:
        warnings.append(str(error))
    try:
        matrix = load_travel_time(area_code=area)
    except ModelError as error:
        warnings.append(str(error))
    return {
        "status": "ok" if sites is not None and matrix is not None else "incomplete",
        "n_sites": int(len(sites)) if sites is not None else 0,
        "n_travel_rows": int(len(matrix)) if matrix is not None else 0,
        "site_types": (
            sites["site_type"].astype(str).value_counts().to_dict()
            if sites is not None and "site_type" in sites.columns
            else {}
        ),
        "warnings": warnings,
    }


def _write_forecast_slice(frame: pd.DataFrame, forecast_date: str) -> Path:
    folder = project_root() / "data" / "results" / "planning"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"forecast_{forecast_date}.csv"
    frame.to_csv(path, index=False)
    return path


def forecast_inference(request: dict[str, Any], compatibility: dict[str, Any]) -> dict[str, Any]:
    """U10 cache, stored retrospective table, or optional live U10. Does not retrain."""
    cache = _path(FORECAST_CACHE)
    forecast_date = str(request.get("forecast_date") or DEMO_FORECAST_DATE)
    source = "cache"
    if request.get("prefer_live_forecast") and forecast_date == DEMO_FORECAST_DATE:
        try:
            from model.operational import run_operational_forecast

            live = run_operational_forecast()
            source = "live_u10"
            LOGGER.info("Live operational forecast completed.")
            return {
                "status": "ok",
                "source": source,
                "n_iz": int(live.get("n_nodes") or 111),
                "forecast_path": live.get("output_csv") or str(cache),
                "forecast_status": "unverified_extrapolation",
                "retrained": False,
                "live": live,
            }
        except Exception as exc:
            LOGGER.warning("Live forecast failed (%s); using cached U10 table.", exc)
            source = "cache_fallback"
    if forecast_date == DEMO_FORECAST_DATE:
        if not cache.exists():
            raise ModelError("Forecast cache missing and live inference failed.", code="missing_forecast")
        frame = pd.read_csv(cache)
        return {
            "status": "ok",
            "source": source,
            "n_iz": int(frame["iz_code"].nunique()) if "iz_code" in frame.columns else int(len(frame)),
            "forecast_path": str(cache),
            "forecast_status": "unverified_extrapolation",
            "issue_date": DEMO_ISSUE_DATE,
            "target_date": DEMO_FORECAST_DATE,
            "retrained": False,
            "checkpoint_id": "U10",
            "label": "4 March 2023 forecast is an unverified extrapolation.",
        }
    retro = _path(RETROSPECTIVE_CACHE)
    if not retro.exists():
        raise ModelError("Retrospective prediction table missing.", code="missing_forecast")
    frame = pd.read_csv(retro)
    if "target_report_date" not in frame.columns:
        raise ModelError("Retrospective table has no target_report_date.", code="invalid_config")
    slice_df = frame.loc[frame["target_report_date"].astype(str) == forecast_date].copy()
    if slice_df.empty:
        raise ModelError(f"No stored predictions for {forecast_date}.", code="missing_forecast")
    path = _write_forecast_slice(slice_df, forecast_date)
    checkpoint_id = str(compatibility.get("checkpoint_id") or slice_df.iloc[0].get("checkpoint_id") or "")
    return {
        "status": "ok",
        "source": "retrospective_table",
        "n_iz": int(slice_df["iz_code"].nunique()) if "iz_code" in slice_df.columns else int(len(slice_df)),
        "forecast_path": str(path),
        "forecast_status": "retrospective_evaluation",
        "issue_date": str(slice_df.iloc[0].get("issue_date") or ""),
        "target_date": forecast_date,
        "retrained": False,
        "checkpoint_id": checkpoint_id,
        "label": f"{forecast_date} uses stored predictions.",
    }


def trigger_rolling_update(request: dict[str, Any]) -> dict[str, Any]:
    if not request.get("confirm_rolling_update"):
        return {
            "status": "needs_confirmation",
            "mode": "rolling_update",
            "executed": False,
            "message": "Rolling update requires explicit confirmation. It will not run silently.",
        }
    return {
        "status": "not_wired",
        "mode": "rolling_update",
        "executed": False,
        "message": "Rolling-update interface is registered. The pipeline is not invoked in this demonstration.",
    }


def trigger_new_region_training(request: dict[str, Any]) -> dict[str, Any]:
    from agent.region_training import region_artefacts_ready, run_new_region_training

    area_code = str(request.get("area_code") or "").strip()
    if region_artefacts_ready(area_code):
        return run_new_region_training(request)
    if not request.get("confirm_new_region_training"):
        return {
            "status": "needs_confirmation",
            "mode": "new_region_training",
            "executed": False,
            "message": (
                "I stopped because this city does not yet have a saved model. "
                "Tick the box on the left if you want me to build one. "
                "I will not do that unless you ask. "
                "Edinburgh's files will not be overwritten."
            ),
        }
    return run_new_region_training(request)


def run_location_allocation(
    request: dict[str, Any],
    forecast: dict[str, Any],
    region: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "area_code": request["area_code"],
        "scenario": request["scenario"],
        "travel_mode": request["travel_mode"],
        "travel_time_threshold_min": request["travel_time_threshold_min"],
        "eligible_site_types": request["eligible_site_types"],
        "priority_population": request["priority_population"],
        "n_sites": N_SITES,
        "forecast_path": forecast.get("forecast_path"),
        "forecast_date": request.get("forecast_date"),
        "n_candidate_sites": region.get("n_sites"),
    }
    result = _run_allocation(payload)
    result["forecast_path"] = payload.get("forecast_path")
    if result.get("status") == "ok":
        from allocation.export import export_allocation

        export_dir = None
        area = str(request.get("area_code") or "")
        if area and area != EDINBURGH_CA:
            from agent.region_training import region_output_dir

            export_dir = region_output_dir(area) / "allocation" / str(request.get("scenario") or "balanced")
        result["export_paths"] = export_allocation(result, output_dir=export_dir)
    return result


def validate_allocation_result(
    allocation: dict[str, Any],
    request: dict[str, Any],
) -> dict[str, Any]:
    site_ids = None
    iz_codes = None
    try:
        from data.candidate_sites import load_candidate_sites

        sites = load_candidate_sites(area_code=request["area_code"])
        if "site_type" in sites.columns:
            allowed = set(request["eligible_site_types"])
            sites = sites.loc[sites["site_type"].astype(str).isin(allowed)]
        site_ids = sites["site_id"].astype(str).tolist()
    except Exception:
        site_ids = None
    forecast_file = allocation.get("forecast_path") or (
        str(_path(FORECAST_CACHE)) if request.get("area_code") == EDINBURGH_CA else None
    )
    if forecast_file and Path(forecast_file).exists():
        iz_codes = pd.read_csv(forecast_file, usecols=["iz_code"])["iz_code"].astype(str).tolist()
    return _validate_allocation(allocation, candidate_site_ids=site_ids, iz_codes=iz_codes)


def get_site_iz_info(site_id: str | None = None, iz_code: str | None = None) -> dict[str, Any]:
    """Look up recorded tables only. Do not invent attributes."""
    out: dict[str, Any] = {"status": "ok"}
    if site_id:
        try:
            from data.candidate_sites import load_candidate_sites

            sites = load_candidate_sites()
            hit = sites.loc[sites["site_id"].astype(str) == str(site_id)]
            if hit.empty:
                raise ModelError(f"Unknown site_id {site_id}.", code="unknown_site")
            row = hit.iloc[0]
            out["site"] = {
                "site_id": str(row["site_id"]),
                "site_name": str(row["site_name"]) if "site_name" in hit.columns else None,
                "site_type": str(row["site_type"]) if "site_type" in hit.columns else None,
            }
        except ModelError:
            raise
    if iz_code:
        cache = _path(FORECAST_CACHE)
        if not cache.exists():
            raise ModelError("Forecast table missing for IZ lookup.", code="missing_forecast")
        frame = pd.read_csv(cache)
        hit = frame.loc[frame["iz_code"].astype(str) == str(iz_code)]
        if hit.empty:
            raise ModelError(f"Unknown iz_code {iz_code}.", code="unknown_iz")
        row = hit.iloc[0]
        out["iz"] = {
            "iz_code": str(row["iz_code"]),
            "predicted_rate": float(row["predicted_rate"]) if pd.notna(row.get("predicted_rate")) else None,
            "predicted_sigma": float(row["predicted_sigma"]) if pd.notna(row.get("predicted_sigma")) else None,
            "forecast_status": str(row.get("forecast_status") or ""),
        }
    return out


def compare_allocation_scenarios(request: dict[str, Any], forecast: dict[str, Any], region: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for scenario in SCENARIOS:
        variant = dict(request)
        variant["scenario"] = scenario
        result = run_location_allocation(variant, forecast, region)
        rows.append(
            {
                "scenario": scenario,
                "label": SCENARIO_LABELS[scenario],
                "status": result.get("status"),
                "n_sites_selected": result.get("n_sites_selected", 0),
                "metrics": result.get("metrics") or {},
                "message": (result.get("diagnostics") or {}).get("message"),
            }
        )
    return {"status": "ok", "n_sites": N_SITES, "scenarios": rows}


def generate_web_layers(bundle: dict[str, Any], output_dir: str | Path | None = None) -> dict[str, Any]:
    from presentation.planning_page import write_planning_page

    return write_planning_page(bundle, output_dir=output_dir)
