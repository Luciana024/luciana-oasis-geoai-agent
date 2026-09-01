"""Greedy covering selects six recorded candidate sites."""

import pandas as pd
import pytest

from allocation.engine import run, run_allocation, set_solver
from allocation.objectives import iz_demand_weights
from common.errors import ModelError


def _toy_payload():
    iz = pd.DataFrame(
        {
            "iz_code": ["A", "B", "C", "D"],
            "population": [1000, 2000, 500, 4000],
            "income_rate": [0.05, 0.20, 0.10, 0.02],
            "pt_gp_min": [10.0, 30.0, 20.0, 8.0],
            "predicted_rate": [20.0, 80.0, 40.0, 10.0],
            "predicted_sigma": [5.0, 15.0, 8.0, 4.0],
        }
    )
    sites = pd.DataFrame(
        {
            "site_id": ["S1", "S2", "S3", "S4", "S5", "S6", "S7"],
            "site_name": list("ABCDEFG"),
            "site_type": ["gp"] * 7,
        }
    )
    rows = []
    times = {
        "S1": [5, 40, 40, 40],
        "S2": [40, 5, 40, 40],
        "S3": [40, 40, 5, 40],
        "S4": [40, 40, 40, 5],
        "S5": [8, 8, 40, 40],
        "S6": [40, 40, 8, 8],
        "S7": [30, 30, 30, 30],
    }
    for site_id, vals in times.items():
        for iz_code, minutes in zip(["A", "B", "C", "D"], vals):
            rows.append({"iz_code": iz_code, "site_id": site_id, "mode": "drive", "travel_time_min": minutes})
    return {
        "scenario": "coverage",
        "travel_mode": "drive",
        "travel_time_threshold_min": 10.0,
        "n_sites": 6,
        "eligible_site_types": ["gp"],
        "iz": iz,
        "sites": sites,
        "travel": pd.DataFrame(rows),
    }


def test_greedy_coverage_selects_six_sites():
    result = run_allocation(_toy_payload())
    assert result["status"] == "ok"
    assert result["n_sites_selected"] == 6
    assert result["invented"] is False
    assert len(result["selected_sites"]) == 6
    ids = [row["site_id"] for row in result["selected_sites"]]
    assert "S4" in ids
    served = [row for row in result["assignments"] if row["served"]]
    assert len(served) == 4


def test_scenarios_change_demand_weights():
    iz = pd.DataFrame(
        {
            "iz_code": ["A", "B"],
            "population": [100.0, 100.0],
            "income_rate": [0.01, 0.50],
            "pt_gp_min": [5.0, 40.0],
            "predicted_rate": [10.0, 90.0],
            "predicted_sigma": [1.0, 20.0],
        }
    ).set_index("iz_code")
    cov = iz_demand_weights(iz, "coverage")
    eq = iz_demand_weights(iz, "equity")
    prev = iz_demand_weights(iz, "preventive")
    assert cov["A"] == cov["B"]
    assert eq["B"] > eq["A"]
    assert prev["B"] > prev["A"]


def test_preventive_high_risk_or_high_uncertainty():
    iz = pd.DataFrame(
        {
            "iz_code": ["low", "high_risk", "high_sigma"],
            "population": [100.0, 100.0, 100.0],
            "income_rate": [0.10, 0.10, 0.10],
            "pt_gp_min": [10.0, 10.0, 10.0],
            "predicted_rate": [10.0, 90.0, 10.0],
            "predicted_sigma": [1.0, 1.0, 20.0],
            "uncertainty_flag": ["normal", "normal", "high"],
        }
    ).set_index("iz_code")
    prev = iz_demand_weights(iz, "preventive")
    assert prev["low"] == 0
    assert prev["high_risk"] > 0
    assert prev["high_sigma"] > 0


def test_equity_and_coverage_can_differ_and_mobile_can_win():
    payload = _toy_payload()
    payload["eligible_site_types"] = ["gp", "mobile_stop"]
    extra = pd.DataFrame(
        {"site_id": ["MS_1"], "site_name": ["Edge Park"], "site_type": ["mobile_stop"]}
    )
    payload["sites"] = pd.concat([payload["sites"], extra], ignore_index=True)
    extra_rows = [
        {"iz_code": iz, "site_id": "MS_1", "mode": "drive", "travel_time_min": t}
        for iz, t in zip(["A", "B", "C", "D"], [40.0, 1.0, 40.0, 40.0])
    ]
    payload["travel"] = pd.concat([payload["travel"], pd.DataFrame(extra_rows)], ignore_index=True)
    coverage_ids = [row["site_id"] for row in run_allocation({**payload, "scenario": "coverage"})["selected_sites"]]
    equity_ids = [row["site_id"] for row in run_allocation({**payload, "scenario": "equity"})["selected_sites"]]
    assert "S6" in coverage_ids
    assert "MS_1" in equity_ids
    assert coverage_ids != equity_ids


def test_balanced_threshold_changes_selected_sites():
    tight = run_allocation({**_toy_payload(), "scenario": "balanced", "travel_time_threshold_min": 10.0})
    loose = run_allocation({**_toy_payload(), "scenario": "balanced", "travel_time_threshold_min": 35.0})
    tight_ids = [row["site_id"] for row in tight["selected_sites"]]
    loose_ids = [row["site_id"] for row in loose["selected_sites"]]
    assert tight_ids != loose_ids
    assert "S7" not in tight_ids
    assert "S7" in loose_ids


def test_run_payload_alias():
    result = run(_toy_payload())
    assert result["n_sites_selected"] == 6


def test_unknown_scenario_fails():
    payload = _toy_payload()
    payload["scenario"] = "random"
    with pytest.raises(ModelError):
        run_allocation(payload)


def test_set_solver_override_and_restore():
    set_solver(lambda payload: {"status": "ok", "n_sites_selected": 6, "selected_sites": [{"site_id": "X"}] * 6, "invented": False})
    try:
        result = run_allocation({"scenario": "balanced"})
        assert result["selected_sites"][0]["site_id"] == "X"
    finally:
        set_solver(None)
    result = run_allocation(_toy_payload())
    assert result["selected_sites"][0]["site_id"] != "X"
