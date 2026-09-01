"""Planning workflow: compatibility, U10 cache inference, no invented sites."""

from pathlib import Path

import pandas as pd
import pytest

from agent.planning import run_planning
from agent.planning_request import parse_planning_request
from agent.planning_tools import trigger_new_region_training
from allocation.contracts import N_SITES
from allocation.engine import run_allocation, set_solver
from allocation.validate import validate_allocation_result
from common.errors import ModelError


def test_parse_request_fixes_six_sites():
    parsed = parse_planning_request({"scenario": "coverage", "travel_mode": "walk"})
    assert parsed["n_sites"] == N_SITES
    assert parsed["n_sites_is_fixed"] is True
    assert parsed["scenario"] == "coverage"
    assert parsed["travel_mode"] == "walk"


@pytest.mark.external_data
def test_edinburgh_demo_uses_inference_not_training():
    result = run_planning({"area_code": "S12000036", "forecast_date": "2023-03-04", "scenario": "balanced"})
    assert result["compatibility"]["mode"] == "inference"
    assert result["model_action"] == "reused"
    assert result["forecast"]["retrained"] is False
    assert result["forecast"]["checkpoint_id"] == "U10"
    assert "unverified extrapolation" in result["forecast_label"].lower()
    assert result["allocation"]["invented"] is False
    assert result["allocation"]["n_sites_selected"] == 6
    assert result["status"] == "ok"
    assert Path(result["website"]["page"]).exists()


def test_new_area_does_not_train_silently():
    result = run_planning({"area_code": "S12000033", "forecast_date": "2023-03-04"})
    assert result["compatibility"]["mode"] == "new_region_training"
    assert result["status"] == "needs_confirmation"
    assert result["model_action"] == "blocked"
    assert result["blockers"][0]["executed"] is False


def test_existing_region_checkpoint_reuses_without_confirm(monkeypatch):
    monkeypatch.setattr("agent.region_training.region_artefacts_ready", lambda code: True)
    monkeypatch.setattr(
        "agent.region_training.run_new_region_training",
        lambda request: {
            "status": "ok",
            "mode": "new_region_training",
            "executed": True,
            "retrained": False,
            "message": "reused",
            "area_code": request["area_code"],
        },
    )
    result = trigger_new_region_training(
        {"area_code": "S12000049", "confirm_new_region_training": False}
    )
    assert result["status"] == "ok"
    assert result["retrained"] is False


def test_confirmed_new_region_calls_training_pipeline(monkeypatch):
    monkeypatch.setattr(
        "agent.region_training.run_new_region_training",
        lambda request: {
            "status": "ok",
            "mode": "new_region_training",
            "executed": True,
            "message": "trained",
            "area_code": request["area_code"],
        },
    )
    result = trigger_new_region_training(
        {"area_code": "S12000049", "confirm_new_region_training": True}
    )
    assert result["status"] == "ok"
    assert result["executed"] is True


def test_allocation_forecast_maps_mu_to_predicted_rate(tmp_path):
    from agent.region_training import _write_allocation_forecast

    src = tmp_path / "forecast_map.csv"
    src.write_text(
        "iz_code,target_report_date,predicted_mu_original,predicted_sigma_original\n"
        "S02003001,2023-02-25,10.0,1.0\n"
        "S02003001,2023-03-04,12.5,1.5\n"
        "S02003002,2023-03-04,8.0,0.5\n"
    )
    dest = tmp_path / "forecast_for_allocation.csv"
    out = _write_allocation_forecast(src, dest, forecast_date="2023-03-04")
    frame = pd.read_csv(out)
    assert list(frame["iz_code"]) == ["S02003001", "S02003002"]
    assert list(frame["predicted_rate"]) == [12.5, 8.0]
    assert list(frame["predicted_sigma"]) == [1.5, 0.5]


def test_swappable_allocator_is_validated():
    def fake_solver(payload):
        return {
            "status": "ok",
            "n_sites_required": 6,
            "n_sites_selected": 6,
            "scenario": payload["scenario"],
            "selected_sites": [{"site_id": f"GP_{i}", "site_name": f"Site {i}", "site_type": "gp"} for i in range(6)],
            "assignments": [{"iz_code": "S02001576", "site_id": "GP_0"}],
            "metrics": {"iz_covered": 1},
            "diagnostics": {},
            "selection_reasons": {"GP_0": "highest coverage in the solver output"},
            "invented": False,
        }

    set_solver(fake_solver)
    try:
        result = run_allocation({"scenario": "balanced"})
        assert result["n_sites_selected"] == 6
        assert result["selected_sites"][0]["site_id"] == "GP_0"
    finally:
        set_solver(None)


def test_allocator_cannot_mark_invented_true():
    with pytest.raises(ModelError) as error:
        validate_allocation_result(
            {
                "status": "ok",
                "invented": True,
                "selected_sites": [{"site_id": "FAKE"}],
            }
        )
    assert error.value.code == "invented_sites"


def test_train_model_can_load_config():
    import agent.tools as tools

    assert callable(tools.load_model_config)
    assert callable(tools.fit_cross_section_scaler)
    cfg = tools.load_model_config()
    assert "paths" in cfg
    assert "expected_edinburgh_iz_count" in cfg
