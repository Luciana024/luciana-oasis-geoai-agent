"""Deterministic six-site allocation for coverage, equity, preventive and balanced."""

from __future__ import annotations

from typing import Any

import pandas as pd

from allocation.assignment import assign_iz_to_sites, greedy_cover, travel_cover_scores
from allocation.contracts import AllocationSolver, N_SITES, SCENARIOS
from allocation.diagnostics import selection_diagnostics
from allocation.metrics import allocation_metrics
from allocation.objectives import SCENARIO_WEIGHTS, iz_demand_weights
from allocation.prepare_inputs import prepare_allocation_inputs
from common.errors import ModelError

_SOLVER: AllocationSolver | None = None


def set_solver(fn: AllocationSolver | None) -> None:
    """Optional override. Default is the four-scenario greedy covering solver."""
    global _SOLVER
    _SOLVER = fn


def get_solver() -> AllocationSolver | None:
    return _SOLVER


def greedy_scenario_solver(payload: dict[str, Any]) -> dict[str, Any]:
    scenario = str(payload.get("scenario") or "balanced")
    if scenario not in SCENARIOS:
        raise ModelError(f"Scenario must be one of {list(SCENARIOS)}.", code="invalid_config")
    prepared = prepare_allocation_inputs(payload)
    n_sites = int(prepared.get("n_sites") or N_SITES)
    if n_sites != N_SITES:
        raise ModelError(f"Prototype fixes n_sites at {N_SITES}.", code="invalid_config")
    if len(prepared["sites"]) < n_sites:
        raise ModelError(
            f"Only {len(prepared['sites'])} eligible sites; cannot select {n_sites} without inventing them.",
            code="missing_dataset",
        )
    warnings: list[str] = []
    iz_tbl = prepared["iz"]
    if "high_predicted_risk" in iz_tbl.columns:
        n_hot = int((iz_tbl["high_predicted_risk"].astype(bool) | iz_tbl["high_uncertainty"].astype(bool)).sum())
        if scenario == "preventive":
            warnings.append(
                f"Preventive demand is restricted to {n_hot} IZs that are high predicted risk "
                "and/or high uncertainty; remaining IZs have weight 0."
            )
    if str(prepared.get("priority_population") or "all") not in {"", "all"}:
        warnings.append(
            "Priority population label is recorded, but the panel only has total IZ population. "
            "Weights were not split by age or other subgroups."
        )
    weights = iz_demand_weights(prepared["iz"], scenario)
    cover_kind = "binary"
    cover_scores = travel_cover_scores(prepared["travel_wide"], prepared["threshold"], kind=cover_kind)
    primary = "cover" if scenario in {"coverage", "preventive"} else "access"
    selected_ids, reasons, gains = greedy_cover(
        cover_scores,
        weights,
        n_sites,
        travel_wide=prepared["travel_wide"],
        primary=primary,
        threshold=prepared["threshold"],
    )
    assignments = assign_iz_to_sites(prepared["travel_wide"], selected_ids, prepared["threshold"])
    metrics = allocation_metrics(prepared["iz"], assignments)
    sites = prepared["sites"]
    selected_sites = []
    for site_id in selected_ids:
        row = sites.loc[site_id]
        selected_sites.append(
            {
                "site_id": str(site_id),
                "site_name": str(row["site_name"]) if "site_name" in sites.columns else str(site_id),
                "site_type": str(row["site_type"]) if "site_type" in sites.columns else None,
            }
        )
    assignment_rows = []
    for record in assignments.itertuples(index=False):
        assignment_rows.append(
            {
                "iz_code": str(record.iz_code),
                "site_id": None if pd.isna(record.site_id) else str(record.site_id),
                "travel_time_min": None if pd.isna(record.travel_time_min) else float(record.travel_time_min),
                "served": bool(record.served),
            }
        )
    return {
        "status": "ok",
        "n_sites_required": N_SITES,
        "n_sites_selected": len(selected_sites),
        "scenario": scenario,
        "travel_mode": prepared["mode"],
        "travel_time_threshold_min": prepared["threshold"],
        "selected_sites": selected_sites,
        "assignments": assignment_rows,
        "metrics": metrics,
        "diagnostics": selection_diagnostics(
            scenario,
            selected_ids,
            gains,
            metrics,
            SCENARIO_WEIGHTS[scenario],
            warnings,
            method="greedy_cover_then_pmedian" if primary == "cover" else "greedy_pmedian",
        ),
        "selection_reasons": reasons,
        "invented": False,
    }


def run_allocation(payload: dict[str, Any]) -> dict[str, Any]:
    """Select six sites and assign IZs. Never a language-model choice."""
    solver = _SOLVER or greedy_scenario_solver
    result = solver(payload)
    if not isinstance(result, dict):
        raise TypeError("Allocation solver must return a dict.")
    result.setdefault("invented", False)
    return result


def run(*args, **kwargs):
    """CLI/script entry. Prefer run_allocation(payload)."""
    if args and isinstance(args[0], dict) and not kwargs:
        return run_allocation(args[0])
    if kwargs:
        return run_allocation(kwargs)
    raise TypeError("run() expects an allocation payload dict.")
