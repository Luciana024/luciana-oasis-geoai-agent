"""Dashboard adapter: GeoMind UI labels map onto the local allocator."""

from pathlib import Path

import pandas as pd
import pytest

from agent.dashboard_bridge import (
    bundle_to_ui,
    extract_forecast_date,
    map_scenario,
    map_study_area,
    match_zone_names,
    narrate_planning,
    parse_agent_intent,
    text_wants_iz_view,
    text_wants_region_view,
    GLASGOW_CA,
    GLASGOW_NAME,
)
from agent.planning_request import parse_planning_request
from allocation.contracts import EDINBURGH_CA, N_SITES
from presentation.display_labels import (
    FEATURE_LABELS,
    SITE_TYPE_LABELS,
    iz_code_from_map_event,
    label_of,
    simple_geoshapley_contributions,
    site_id_from_map_event,
)


def test_ui_scenario_aliases():
    assert map_scenario("coverage priority") == "coverage"
    parsed = parse_planning_request({"scenario": "preventive priority"})
    assert parsed["scenario"] == "preventive"
    assert parsed["n_sites"] == N_SITES


def test_edinburgh_area_maps_to_ca():
    code, name = map_study_area("City of Edinburgh")
    assert code == EDINBURGH_CA
    assert "Edinburgh" in name


def test_glasgow_area_maps_to_ca():
    code, name = map_study_area("Glasgow City")
    assert code == GLASGOW_CA
    assert name == GLASGOW_NAME


@pytest.mark.external_data
def test_bundle_to_ui_uses_real_population_fields():
    bundle = {
        "status": "ok",
        "request": {
            "scenario": "balanced",
            "travel_mode": "drive",
            "travel_time_threshold_min": 20.0,
            "forecast_date": "2023-03-04",
        },
        "compatibility": {"mode": "inference"},
        "forecast": {
            "status": "ok",
            "forecast_path": str(
                Path("data/results/exports/website_article_v1/website/future_forecast_20230304.csv")
            ),
            "forecast_status": "unverified_extrapolation",
            "checkpoint_id": "U10",
            "target_date": "2023-03-04",
        },
        "allocation": {
            "status": "ok",
            "invented": False,
            "scenario": "balanced",
            "selected_sites": [
                {"site_id": "GP_71011", "site_name": "Polwarth Medical Practice", "site_type": "gp"}
            ],
            "assignments": [
                {
                    "iz_code": "S02001576",
                    "site_id": "GP_71011",
                    "travel_time_min": 5.2,
                    "served": True,
                }
            ],
            "metrics": {
                "population_covered": 5276.0,
                "iz_covered": 1,
                "unserved_iz": 0,
                "n_iz": 1,
                "mean_travel_time_min": 5.2,
                "max_travel_time_min": 5.2,
                "unserved_population": 0.0,
            },
            "selection_reasons": {"GP_71011": "Step 1 (access): covering gain 0.1, travel-time reduction 1.0"},
            "diagnostics": {"message": "ok", "warnings": []},
        },
        "comparison": {"scenarios": []},
        "checkpoint": {"checkpoint_id": "U10"},
    }
    # forecast_path in bundle may be relative; bundle_to_ui reads it as given
    root_cache = Path(__file__).resolve().parents[2] / bundle["forecast"]["forecast_path"]
    bundle["forecast"]["forecast_path"] = str(root_cache)
    ui = bundle_to_ui(bundle)
    assert ui["success"] is True
    assert ui["forecast"]["is_unverified"] is True
    assert ui["allocation"]["diagnostics"]["covered_population"] == 5276
    assert ui["allocation"]["selected_sites"][0]["site_id"] == "GP_71011"
    assert isinstance(ui["comparison_df"], pd.DataFrame)


def test_region_population_falls_back_to_frozen_simd(tmp_path, monkeypatch):
    from agent.dashboard_bridge import _population_by_iz

    region = tmp_path / "data" / "results" / "regions" / "S12000049"
    region.mkdir(parents=True)
    (region / "simd_iz.csv").write_text(
        "IntZone,total_population\nIZ1,1234\nIZ2,5678\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("agent.dashboard_bridge.project_root", lambda: tmp_path)

    assert _population_by_iz("S12000049") == {"IZ1": 1234.0, "IZ2": 5678.0}


def test_blocked_bundle_does_not_invent_sites():
    ui = bundle_to_ui(
        {
            "status": "needs_confirmation",
            "compatibility": {"mode": "new_region_training", "reason": "other city"},
            "forecast": {},
            "blockers": [{"message": "New-region training requires explicit confirmation."}],
        }
    )
    assert ui["success"] is False
    assert "confirmation" in ui["message"].lower()


def test_display_labels_drop_underscores():
    assert label_of(SITE_TYPE_LABELS, "mobile_stop") == "Mobile stop (car park)"
    assert label_of(FEATURE_LABELS, "location_x_public_transport_time_to_gp") == (
        "Location × public transport time to GP"
    )
    assert "_" not in label_of(FEATURE_LABELS, "income_deprivation")


def test_map_click_reads_red_site_id():
    plottable = [{"site_id": "GP_71011"}, {"site_id": "MS_14"}]
    event = {"selection": {"points": [{"curve_number": 1, "point_index": 1, "customdata": ["MS_14"]}]}}
    assert site_id_from_map_event(event, plottable) == "MS_14"
    nested = {"selection": {"points": [{"curve_number": 2, "customdata": [["MS_14"]]}]}}
    assert site_id_from_map_event(nested, plottable) == "MS_14"
    zone_click = {"selection": {"points": [{"curve_number": 0, "hovertext": "Assigned site=MS_14"}]}}
    assert site_id_from_map_event(zone_click, plottable) is None
    assert site_id_from_map_event({"selection": {"points": []}}, plottable) is None


def test_map_click_reads_iz_code():
    codes = ["S02001576", "S02001577"]
    event = {"selection": {"points": [{"curve_number": 0, "location": 1}]}}
    assert iz_code_from_map_event(event, codes) == "S02001577"
    assert iz_code_from_map_event({"selection": {"points": [{"hovertext": "S02001576"}]}}, codes) == "S02001576"


def test_parse_agent_intent_routes_tasks():
    assert parse_agent_intent("Plan 6 sites for Edinburgh")["intent"] == "plan"
    equity = parse_agent_intent("use equity priority")
    assert equity["intent"] == "plan"
    assert equity["scenario"] == "equity priority"
    assert parse_agent_intent("compare the four policies")["intent"] == "compare"
    assert parse_agent_intent("show the forecast and uncertainty")["intent"] == "forecast"
    assert parse_agent_intent("why GP_71011?", site_ids=["GP_71011"]) == {
        "intent": "explain_site",
        "site_id": "GP_71011",
    }
    assert parse_agent_intent("explain S02001576")["iz_code"] == "S02001576"
    assert parse_agent_intent("按左边的 brief 规划 6 个点")["intent"] == "plan"
    assert parse_agent_intent("比较四种情景")["intent"] == "compare"
    coverage_zh = parse_agent_intent("覆盖")
    assert coverage_zh["intent"] == "plan"
    assert coverage_zh["scenario"] == "coverage priority"
    assert parse_agent_intent("比较覆盖")["intent"] == "compare"
    assert parse_agent_intent("看候选点")["intent"] == "allocation"
    assert parse_agent_intent("我想要格拉斯哥的结果")["intent"] == "forecast"
    assert parse_agent_intent("I want Glasgow results")["intent"] == "forecast"
    assert parse_agent_intent("show Glasgow results")["intent"] == "forecast"
    assert parse_agent_intent("格拉斯哥")["intent"] == "help"
    assert parse_agent_intent(
        "I would like to know the distribution of vaccination sites in the Edinburgh area."
    )["intent"] == "allocation"
    from agent.dashboard_bridge import looks_like_allocation

    assert looks_like_allocation(
        "I would like to know the distribution of vaccination sites in the Edinburgh area."
    )
    assert parse_agent_intent("我想看格拉斯哥2022年6月15日的预报")["intent"] == "forecast"
    assert parse_agent_intent("show me the infection rate map for Edinburgh")["intent"] == "forecast"
    assert parse_agent_intent("爱丁堡感染率怎么样")["intent"] == "forecast"
    assert parse_agent_intent("where can I get vaccinated in Glasgow")["intent"] == "allocation"
    assert parse_agent_intent("疫苗接种点在哪")["intent"] == "allocation"
    assert parse_agent_intent("contrast the four policies")["intent"] == "compare"
    assert parse_agent_intent("哪个政策更好")["intent"] == "compare"
    assert parse_agent_intent("please place six vaccination clinics")["intent"] == "plan"


def test_extract_forecast_date_reads_city_and_day_in_one_sentence():
    options = ["2023-03-04 (extrapolation)", "2022-06-15", "2022-03-04"]
    latest = options[0]
    assert extract_forecast_date("我想看格拉斯哥 2022-06-15 的预测", options=options, latest=latest) == "2022-06-15"
    assert extract_forecast_date("2022年6月15日爱丁堡的预报", options=options, latest=latest) == "2022-06-15"
    assert extract_forecast_date("I want the Glasgow forecast for 15 June 2022", options=options, latest=latest) == "2022-06-15"
    assert extract_forecast_date("Glasgow 2022/06/15", options=options, latest=latest) == "2022-06-15"
    assert extract_forecast_date("最新一天", options=options, latest=latest) == latest
    assert extract_forecast_date("2022年3月4日", options=options, latest=latest) == "2022-03-04"
    assert extract_forecast_date("3月4日", options=options, latest=latest) == latest


def test_typed_button_equivalents_for_forecast_views():
    names = {"S02000001": "Anderston", "S02000002": "City Centre East"}
    assert match_zone_names("我想看 Anderston 的预测", names) == ["S02000001"]
    assert match_zone_names("please show City Centre East", names) == ["S02000002"]
    assert match_zone_names("Anderston", names) == ["S02000001"]
    assert text_wants_region_view("全区域展示（整座城市的全部中间区）")
    assert text_wants_region_view("我想看格拉斯哥的全区域预测")
    assert not text_wants_iz_view("全区域展示（整座城市的全部中间区）")
    assert text_wants_iz_view("按所选中间区（Intermediate Zone）展示")
    assert text_wants_iz_view("换一个街区")
    assert not text_wants_region_view("按所选中间区展示 Anderston")


def test_chat_replies_to_smalltalk():
    from agent.dashboard_bridge import classify_chat, narrate_chat

    assert classify_chat("你好") == "greet"
    assert classify_chat("谢谢") == "thanks"
    assert classify_chat("你是谁") == "identity"
    assert classify_chat("hi there") == "greet"
    assert parse_agent_intent("你好")["chat"] == "greet"
    hello = narrate_chat("你好", lang="zh", chat="greet", step="city")
    assert "你好" in hello
    assert "格拉斯哥" in hello or "爱丁堡" in hello
    weather = narrate_chat("今天天气真好", lang="zh", chat="other", step="city")
    assert "天气" in weather
    assert "格拉斯哥" in weather or "爱丁堡" in weather
    thanks = narrate_chat("thanks", lang="en", chat="thanks", step="task")
    assert "welcome" in thanks.lower()
    assert "forecast" in thanks.lower() or "vaccination" in thanks.lower()


def test_detect_ui_lang_follows_typed_language():
    from presentation.ui_i18n import detect_ui_lang

    assert detect_ui_lang("我想要格拉斯哥的结果") == "zh"
    assert detect_ui_lang("hello") == "en"
    assert detect_ui_lang("I want Glasgow results") == "en"
    assert detect_ui_lang("Show the forecast") is None
    assert detect_ui_lang("Glasgow City") is None
    assert detect_ui_lang("2023-03-04") is None


def test_plan_from_brief_binds_left_panel_to_agent():
    from agent.dashboard_bridge import brief_fingerprint, plan_from_brief_text

    text = plan_from_brief_text(
        study_area="Glasgow City",
        forecast_date="2023-03-04",
        scenario_label="Equity priority",
        travel_threshold=20.0,
        travel_label="Drive",
        eligible_labels=["GP", "Pharmacy"],
    )
    assert "task brief" in text
    assert "Glasgow City" in text
    assert parse_agent_intent(text)["intent"] == "plan"
    edinburgh = brief_fingerprint(
        study_area="City of Edinburgh",
        forecast_date="2023-03-04",
        scenario="balanced",
        travel_mode="drive",
        travel_threshold=20.0,
        eligible_types=["gp", "pharmacy"],
        confirm_new_region=False,
    )
    glasgow = brief_fingerprint(
        study_area="Glasgow City",
        forecast_date="2023-03-04",
        scenario="balanced",
        travel_mode="drive",
        travel_threshold=20.0,
        eligible_types=["gp", "pharmacy"],
        confirm_new_region=True,
    )
    assert edinburgh != glasgow


def test_narrate_planning_does_not_claim_agent_picked_sites():
    text = narrate_planning(
        {
            "success": True,
            "forecast": {"forecast_date": "2023-03-04", "checkpoint_used": "U10", "total_zones": 111},
            "allocation": {
                "diagnostics": {
                    "scenario": "balanced",
                    "fixed_site_count": 6,
                    "travel_time_threshold_min": 20,
                    "travel_mode": "drive",
                    "covered_zones": 111,
                    "total_zones": 111,
                    "coverage_percentage": 100.0,
                    "covered_population": 527620,
                    "mean_travel_time_min": 5.2,
                    "unserved_zones": 0,
                },
                "selected_sites": [{"site_id": "GP_71011", "site_name": "Polwarth Medical Practice"}],
            },
        }
    )
    assert "did not choose" not in text
    assert "不是我选" not in narrate_planning(
        {
            "success": True,
            "forecast": {"forecast_date": "2023-03-04", "total_zones": 111},
            "allocation": {
                "diagnostics": {
                    "scenario": "balanced",
                    "fixed_site_count": 6,
                    "travel_time_threshold_min": 20,
                    "travel_mode": "drive",
                    "covered_zones": 111,
                    "total_zones": 111,
                    "coverage_percentage": 100.0,
                    "covered_population": 527620,
                    "mean_travel_time_min": 5.2,
                    "unserved_zones": 0,
                },
                "selected_sites": [{"site_id": "GP_71011", "site_name": "Polwarth Medical Practice"}],
            },
        },
        lang="zh",
    )
    assert "GP_71011" in text
    assert "U10" not in text
    assert "rolling_v1" not in text
    assert "checkpoint" not in text.lower()
    assert "trained" not in text.lower()
    assert "retrain" not in text.lower()
    from agent.dashboard_bridge import AGENT_GREETING, plain_agent_text

    for sample in (AGENT_GREETING, text, plain_agent_text("Compatible U10 checkpoint. I did not retrain.")):
        lower = sample.lower()
        assert "u10" not in lower
        assert "checkpoint" not in lower
        assert "trained" not in lower
        assert "retrain" not in lower


def test_simple_geoshapley_adds_main_and_interaction():
    frame = pd.DataFrame(
        [
            {"feature_name": "baseline", "shapley_value": 30.0, "predicted_rate": 40.0},
            {"feature_name": "public_transport_time_to_gp", "shapley_value": 4.0, "predicted_rate": 40.0},
            {"feature_name": "location_x_public_transport_time_to_gp", "shapley_value": 6.0, "predicted_rate": 40.0},
            {"feature_name": "location", "shapley_value": -2.0, "predicted_rate": 40.0},
            {"feature_name": "income_deprivation", "shapley_value": 0.0, "predicted_rate": 40.0},
            {"feature_name": "employment_deprivation", "shapley_value": 0.0, "predicted_rate": 40.0},
            {"feature_name": "higher_education", "shapley_value": 0.0, "predicted_rate": 40.0},
            {"feature_name": "overcrowding", "shapley_value": 0.0, "predicted_rate": 40.0},
            {"feature_name": "crime", "shapley_value": 0.0, "predicted_rate": 40.0},
        ]
    )
    out = simple_geoshapley_contributions(frame)
    pt = out["table"].set_index("indicator").loc["public_transport_time_to_gp"]
    assert pt["contribution"] == 10.0
    assert out["baseline"] == 30.0
    assert out["predicted_rate"] == 40.0
    train_export = pd.DataFrame(
        [
            {
                "player_name": "public_transport_time_to_gp",
                "phi": 4.0,
                "phi_0": 32.0,
                "reconstructed_prediction": 41.0,
            },
            {
                "player_name": "location_x_public_transport_time_to_gp",
                "phi": 5.0,
                "phi_0": 32.0,
                "reconstructed_prediction": 41.0,
            },
        ]
    )
    train_out = simple_geoshapley_contributions(train_export)
    assert train_out["baseline"] == 32.0
    assert train_out["predicted_rate"] == 41.0
    assert (
        train_out["table"].set_index("indicator").loc["public_transport_time_to_gp"]["contribution"] == 9.0
    )
