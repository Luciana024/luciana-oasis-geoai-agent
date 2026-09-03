"""Join forecast, population, SIMD, candidate sites and travel time. Do not invent rows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from allocation.contracts import FORECAST_CACHE, SITE_TYPES
from allocation.objectives import attach_equity_labels
from common.errors import ModelError
from common.utils import PANEL_CSV, project_root, resolve_project_path
from data.candidate_sites import load_candidate_sites
from data.travel_time import load_travel_time

SIMD_PATH = Path("data") / "results" / "simd_iz.csv"
PANEL_PATH = Path("data") / "results" / PANEL_CSV


def prepare_allocation_inputs(payload: dict[str, Any]) -> dict[str, Any]:
    """Build IZ demand, eligible sites and the IZ×site travel matrix for one mode."""
    if payload.get("iz") is not None:
        iz = pd.DataFrame(payload["iz"]).copy()
        sites = pd.DataFrame(payload["sites"]).copy()
        travel = pd.DataFrame(payload["travel"]).copy()
    else:
        iz = _load_iz_table(payload)
        sites = _load_sites(payload)
        travel = _load_travel(payload)

    required_iz = {"iz_code", "population", "income_rate", "pt_gp_min", "predicted_rate"}
    missing = required_iz - set(iz.columns)
    if missing:
        raise ModelError(f"Allocation IZ table missing {sorted(missing)}.", code="missing_dataset")
    if "predicted_sigma" not in iz.columns:
        iz["predicted_sigma"] = 0.0
    iz["iz_code"] = iz["iz_code"].astype(str)
    iz["population"] = pd.to_numeric(iz["population"], errors="coerce")
    if iz["population"].isna().any() or (iz["population"] <= 0).any():
        raise ModelError("IZ population is missing or non-positive. Refusing to invent it.", code="missing_dataset")
    iz = attach_equity_labels(iz)

    sites["site_id"] = sites["site_id"].astype(str)
    allowed_types = [str(item).lower() for item in payload.get("eligible_site_types") or list(SITE_TYPES)]
    if "site_type" in sites.columns:
        sites = sites.loc[sites["site_type"].astype(str).str.lower().isin(allowed_types)].copy()
    if sites.empty:
        raise ModelError("No eligible candidate sites remain after type filter.", code="missing_dataset")

    mode = str(payload.get("travel_mode") or "drive")
    travel["iz_code"] = travel["iz_code"].astype(str)
    travel["site_id"] = travel["site_id"].astype(str)
    if "mode" in travel.columns:
        travel = travel.loc[travel["mode"].astype(str) == mode].copy()
    travel["travel_time_min"] = pd.to_numeric(travel["travel_time_min"], errors="coerce")

    site_ids = sites["site_id"].astype(str).tolist()
    iz_codes = iz["iz_code"].astype(str).tolist()
    travel = travel.loc[travel["site_id"].isin(site_ids) & travel["iz_code"].isin(iz_codes)].copy()
    if travel.empty:
        raise ModelError("Travel-time matrix has no overlapping IZ–site pairs.", code="missing_dataset")

    threshold = float(payload.get("travel_time_threshold_min") or 20.0)
    wide = travel.pivot_table(index="iz_code", columns="site_id", values="travel_time_min", aggfunc="min")
    wide = wide.reindex(index=iz_codes, columns=site_ids)
    coverable = wide.le(threshold) & wide.notna()
    return {
        "iz": iz.set_index("iz_code"),
        "sites": sites.set_index("site_id"),
        "travel_wide": wide,
        "coverable": coverable,
        "threshold": threshold,
        "mode": mode,
        "scenario": str(payload.get("scenario") or "balanced"),
        "n_sites": int(payload.get("n_sites") or 6),
        "priority_population": str(payload.get("priority_population") or "all"),
    }


def _load_iz_table(payload: dict[str, Any]) -> pd.DataFrame:
    forecast_path = (
        resolve_project_path(payload["forecast_path"])
        if payload.get("forecast_path")
        else project_root() / FORECAST_CACHE
    )
    if not forecast_path.exists():
        raise ModelError(f"Forecast table missing: {forecast_path}", code="missing_forecast")
    forecast = pd.read_csv(forecast_path)
    if "target_report_date" in forecast.columns and forecast.duplicated("iz_code").any():
        target = str(payload.get("forecast_date") or "")
        dates = forecast["target_report_date"].astype(str)
        if target and target in set(dates):
            forecast = forecast.loc[dates == target].copy()
        else:
            forecast = forecast.loc[dates == dates.max()].copy()
        forecast = forecast.drop_duplicates("iz_code", keep="last")
    if "predicted_rate" not in forecast.columns:
        if "predicted_mu_original" in forecast.columns:
            forecast["predicted_rate"] = forecast["predicted_mu_original"]
        elif "predicted_mean" in forecast.columns:
            forecast["predicted_rate"] = forecast["predicted_mean"]
        elif "display_mean" in forecast.columns:
            forecast["predicted_rate"] = forecast["display_mean"]
    if "predicted_sigma" not in forecast.columns and "predicted_sigma_original" in forecast.columns:
        forecast["predicted_sigma"] = forecast["predicted_sigma_original"]
    if "iz_code" not in forecast.columns or "predicted_rate" not in forecast.columns:
        raise ModelError("Forecast table must contain iz_code and predicted_rate.", code="invalid_config")
    area = str(payload.get("area_code") or "S12000036")
    if area not in {"S12000036", ""}:
        region = project_root() / "data" / "results" / "regions" / area
        simd_path = region / "simd_iz.csv"
        panel_path = region / "covid" / "fill1.csv"
        planning_population_path = region / "planning_population.csv"
    else:
        simd_path = project_root() / SIMD_PATH
        panel_path = project_root() / PANEL_PATH
        planning_population_path = None
    if not simd_path.exists():
        raise ModelError(f"SIMD IZ table missing: {simd_path}", code="missing_dataset")
    simd = pd.read_csv(simd_path)
    simd = simd.rename(columns={"IntZone": "iz_code"})
    if planning_population_path is not None and planning_population_path.exists():
        pop = pd.read_csv(planning_population_path)
        required = {"iz_code", "population"}
        if not required.issubset(pop.columns):
            raise ModelError(
                f"Planning population table must contain {sorted(required)}: {planning_population_path}",
                code="missing_dataset",
            )
        pop = pop[["iz_code", "population"]].drop_duplicates("iz_code")
    elif panel_path.exists():
        panel = pd.read_csv(panel_path)
        last_date = panel["Date"].max()
        pop_cols = [col for col in ("IntZone", "iz_code") if col in panel.columns]
        if not pop_cols or "Population" not in panel.columns:
            raise ModelError("Population table must contain IntZone and Population.", code="missing_dataset")
        zone_col = pop_cols[0]
        pop = panel.loc[panel["Date"] == last_date, [zone_col, "Population"]].drop_duplicates(zone_col)
        pop = pop.rename(columns={zone_col: "iz_code", "Population": "population"})
    elif "total_population" in simd.columns:
        # Packaged regional results intentionally omit the raw COVID panel.
        # The frozen SIMD aggregation contains the same planning population.
        pop = simd[["iz_code", "total_population"]].drop_duplicates("iz_code")
        pop = pop.rename(columns={"total_population": "population"})
    else:
        raise ModelError(
            f"Population missing: neither {panel_path} nor total_population in {simd_path} is available.",
            code="missing_dataset",
        )
    out = forecast.merge(pop, on="iz_code", how="left").merge(
        simd[["iz_code", "income_rate", "pt_gp_min"]],
        on="iz_code",
        how="left",
    )
    missing = out.loc[out["population"].isna() | out["income_rate"].isna(), "iz_code"].astype(str).tolist()
    if missing:
        raise ModelError(
            f"Population or SIMD missing for IZs {missing[:10]}. Do not invent them.",
            code="missing_dataset",
        )
    return out


def _load_sites(payload: dict[str, Any]):
    return pd.DataFrame(load_candidate_sites(area_code=str(payload.get("area_code") or "S12000036")))


def _load_travel(payload: dict[str, Any]) -> pd.DataFrame:
    return load_travel_time(area_code=str(payload.get("area_code") or "S12000036"))
