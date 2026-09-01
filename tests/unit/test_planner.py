from agent import extract_source, extract_window, extract_years, interpret_request


def test_extracts_single_year():
    assert extract_years("请用2022年数据") == [2022]


def test_extracts_year_range():
    assert extract_years("合并2021到2022年") == [2021, 2022]


def test_ignores_years_without_approved_extracts():
    assert extract_years("用2019年数据") == []


def test_extracts_api_source():
    assert extract_source("请用API拉取2022年数据") == "api"


def test_extracts_local_source():
    assert extract_source("请用本地2022年数据") == "local"


def test_blocks_when_year_and_source_missing():
    state = interpret_request("请合并爱丁堡COVID数据")
    assert state.status == "awaiting_user"
    assert "years" in state.missing_parameters
    assert "data_source" in state.missing_parameters


def test_blocks_when_source_missing():
    state = interpret_request({"years": [2022], "request": "merge"})
    assert "data_source" in state.missing_parameters


def test_dict_request_uses_explicit_year_and_source():
    state = interpret_request({"years": [2022], "source": "local", "request": "merge"})
    assert state.years == [2022]
    assert state.data_source == "local"
    assert state.missing_parameters == []


def test_extracts_equal_forecast_window():
    assert extract_window("请用7天预测7天") == (7, 7)
    assert extract_window("14天预测14天") == (14, 14)


def test_extracts_unequal_forecast_window():
    assert extract_window("7天预测14天") == (7, 14)
    assert extract_window("14天预测7天") == (14, 7)
    assert extract_window("14 predicting 7") == (14, 7)


def test_forecast_task_defaults_to_seven_seven():
    state = interpret_request({"task": "forecast_prepare", "request": "做预测样本"})
    assert state.status == "planned"
    assert state.lookback_days == 7
    assert state.forecast_horizon_days == 7
    assert "forecast_window" not in state.missing_parameters
    assert any("lookback_days defaulted to 7" in item for item in state.warnings)
    assert any("forecast_horizon_days defaulted to 7" in item for item in state.warnings)


def test_forecast_task_accepts_window_flag():
    state = interpret_request({"task": "forecast_prepare", "window": 14, "request": "forecast"})
    assert state.lookback_days == 14
    assert state.forecast_horizon_days == 14
    assert state.missing_parameters == []


def test_forecast_task_accepts_unequal_lookback_and_horizon():
    state = interpret_request(
        {"task": "forecast_prepare", "lookback": 14, "horizon": 7, "request": "forecast"}
    )
    assert state.status == "planned"
    assert state.lookback_days == 14
    assert state.forecast_horizon_days == 7
    assert state.missing_parameters == []


def test_travel_time_task_asks_for_source():
    state = interpret_request({"task": "travel_time_prepare", "request": "算出行时间"})
    assert state.task == "travel_time_prepare"
    assert state.status == "awaiting_user"
    assert "data_source" in state.missing_parameters
    assert "years" not in state.missing_parameters


def test_travel_time_task_accepts_osm_and_other_city():
    state = interpret_request(
        {
            "task": "travel_time_prepare",
            "source": "osm",
            "area_code": "S12000049",
            "osm_place": "Glasgow, UK",
            "request": "travel time",
        }
    )
    assert state.status == "planned"
    assert state.data_source == "osm"
    assert state.area_code == "S12000049"
    assert state.osm_place == "Glasgow, UK"
    assert state.area_name != "City of Edinburgh"


def test_extracts_travel_time_task_from_text():
    from agent.validators import extract_task

    assert extract_task("请算出行时间矩阵") == "travel_time_prepare"
    assert extract_source("用 osm 计算出行时间") == "osm"


def test_candidate_sites_task_accepts_api():
    state = interpret_request(
        {
            "task": "candidate_sites_prepare",
            "source": "api",
            "area_code": "S12000036",
            "request": "用API做候选点",
        }
    )
    assert state.status == "planned"
    assert state.data_source == "api"
    state = interpret_request({"task": "candidate_sites_prepare", "request": "做候选点"})
    assert state.task == "candidate_sites_prepare"
    assert "data_source" in state.missing_parameters
    assert "years" not in state.missing_parameters
