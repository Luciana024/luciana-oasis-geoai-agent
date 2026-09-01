from pathlib import Path

import pandas as pd
import pytest

from data.covid import CovidPreprocessError, preprocess_covid


FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "covid_sample.csv"
REQUIRED = [
    "Date",
    "IntZone",
    "IntZoneName",
    "CA",
    "CAName",
    "Positive7Day",
    "Positive7DayQF",
    "Population",
    "CrudeRate7DayPositive",
    "CrudeRate7DayPositiveQF",
]


def _master(codes: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"IntZone": codes, "node_index": list(range(len(codes)))})


def _sample() -> pd.DataFrame:
    return pd.read_csv(FIXTURE, dtype=str, keep_default_na=False)


def _run(frames, years, iz_master, tmp_path, monkeypatch, **kwargs):
    import data.covid as covid_mod

    monkeypatch.setattr(covid_mod, "project_root", lambda: tmp_path)
    return preprocess_covid(frames, years=years, iz_master=iz_master, **kwargs)


def test_ckan_sql_ident_rejects_non_uuid():
    from data.covid import AcquisitionError, _sql_ident

    assert _sql_ident("8906de12-f413-4b3f-95a0-11ed15e61773") == '"8906de12-f413-4b3f-95a0-11ed15e61773"'
    with pytest.raises(AcquisitionError):
        _sql_ident("covid; drop table")


def test_empty_frames_fail(tmp_path, monkeypatch):
    with pytest.raises(CovidPreprocessError, match="frames is empty"):
        _run([], [2022], _master(["S02001576"]), tmp_path, monkeypatch)


def test_empty_years_fail(tmp_path, monkeypatch):
    with pytest.raises(CovidPreprocessError, match="years is empty"):
        _run([_sample()], [], _master(["S02001576"]), tmp_path, monkeypatch)


def test_missing_required_columns_fail(tmp_path, monkeypatch):
    frame = _sample().drop(columns=["Population"])
    with pytest.raises(CovidPreprocessError, match="Population"):
        _run([frame], [2022], _master(["S02001576", "S02001577"]), tmp_path, monkeypatch)


def test_invalid_suppression_configuration_fails(tmp_path, monkeypatch):
    frame = _sample()
    master = _master(["S02001576", "S02001577"])
    with pytest.raises(CovidPreprocessError, match="unique"):
        _run([frame], [2022], master, tmp_path, monkeypatch, suppression_fill_values=(1, 1))
    with pytest.raises(CovidPreprocessError, match="primary_scenario_fill"):
        _run([frame], [2022], master, tmp_path, monkeypatch, primary_scenario_fill=1, suppression_fill_values=(0, 2))


def test_yyyymmdd_and_iso_dates_parse(tmp_path, monkeypatch):
    frame = _sample()
    frame.loc[frame["Date"] == "20220102", "Date"] = "2022-01-02"
    master = _master(["S02001576", "S02001577"])
    result = _run([frame], [2022], master, tmp_path, monkeypatch)
    dates = set(result["scenario_frames"]["fill1"]["Date"].dt.strftime("%Y-%m-%d"))
    assert dates == {"2022-01-01", "2022-01-02"}


def test_invalid_non_empty_date_fails(tmp_path, monkeypatch):
    frame = _sample()
    frame.loc[0, "Date"] = "16/01/2022"
    with pytest.raises(CovidPreprocessError, match="not YYYYMMDD"):
        _run([frame], [2022], _master(["S02001576", "S02001577"]), tmp_path, monkeypatch)


def test_unexpected_year_fails(tmp_path, monkeypatch):
    frame = _sample()
    with pytest.raises(CovidPreprocessError, match="not requested"):
        _run([frame], [2023], _master(["S02001576", "S02001577"]), tmp_path, monkeypatch)


def test_missing_requested_year_warns(tmp_path, monkeypatch):
    result = _run(
        [_sample()],
        [2022, 2023],
        _master(["S02001576", "S02001577"]),
        tmp_path,
        monkeypatch,
    )
    assert any("2023" in warning for warning in result["warnings"])


def test_keeps_edinburgh_and_rejects_wrong_caname(tmp_path, monkeypatch):
    master = _master(["S02001576", "S02001577"])
    result = _run([_sample()], [2022], master, tmp_path, monkeypatch)
    fill1 = result["scenario_frames"]["fill1"]
    assert set(fill1["CA"].unique()) == {"S12000036"}
    assert result["primary_scenario"] == "fill1"
    assert result["sensitivity_scenarios"] == ["fill0", "fill2"]

    bad = _sample()
    bad.loc[bad["CA"] == "S12000036", "CAName"] = "Glasgow City"
    with pytest.raises(CovidPreprocessError, match="unexpected CAName"):
        _run([bad], [2022], master, tmp_path, monkeypatch)


def test_unexpected_iz_fails_and_missing_expected_iz_is_reported(tmp_path, monkeypatch):
    master = _master(["S02001576", "S02001577", "S02001578"])
    result = _run([_sample()], [2022], master, tmp_path, monkeypatch)
    assert "S02001578" in result["report"]["iz"]["missing_expected_iz_codes"]
    extra = _sample()
    extra.loc[extra["CA"] == "S12000033", "CA"] = "S12000036"
    extra.loc[extra["CA"] == "S12000036", "CAName"] = "City of Edinburgh"
    with pytest.raises(CovidPreprocessError, match="absent from iz_master"):
        _run([extra], [2022], _master(["S02001576", "S02001577"]), tmp_path, monkeypatch)


def test_final_order_follows_node_index(tmp_path, monkeypatch):
    master = pd.DataFrame({"IntZone": ["S02001577", "S02001576"], "node_index": [0, 1]})
    result = _run([_sample()], [2022], master, tmp_path, monkeypatch)
    ordered = result["scenario_frames"]["fill1"]
    first_day = ordered.loc[ordered["Date"] == ordered["Date"].min()]
    assert list(first_day["IntZone"]) == ["S02001577", "S02001576"]
    assert list(first_day["node_index"]) == [0, 1]


def test_exact_duplicates_reduced_and_conflicts_fail(tmp_path, monkeypatch):
    frame = _sample()
    exact = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    master = _master(["S02001576", "S02001577"])
    result = _run([exact], [2022], master, tmp_path, monkeypatch)
    assert result["report"]["duplicates"]["n_exact_duplicate_rows_removed"] == 1

    conflict = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    conflict.loc[len(conflict) - 1, "Positive7Day"] = "99"
    with pytest.raises(CovidPreprocessError, match="conflicting"):
        _run([conflict], [2022], master, tmp_path, monkeypatch)


def test_response_classification_and_scenario_fills(tmp_path, monkeypatch):
    master = _master(["S02001576", "S02001577"])
    result = _run([_sample()], [2022], master, tmp_path, monkeypatch)
    fill0 = result["scenario_frames"]["fill0"]
    fill1 = result["scenario_frames"]["fill1"]
    fill2 = result["scenario_frames"]["fill2"]
    suppressed = fill1["response_status"].eq("disclosure_controlled_0_2")
    missing = fill1["response_status"].eq("true_missing_not_suppressed")
    observed = fill1["response_status"].eq("observed")
    assert list(fill0.loc[suppressed, "case_count_used"]) == [0]
    assert list(fill1.loc[suppressed, "case_count_used"]) == [1]
    assert list(fill2.loc[suppressed, "case_count_used"]) == [2]
    assert fill0.loc[missing, "case_count_used"].isna().all()
    assert fill1.loc[missing, "case_count_used"].isna().all()
    assert fill2.loc[missing, "case_count_used"].isna().all()
    assert fill0.loc[observed, "case_count_used"].tolist() == fill1.loc[observed, "case_count_used"].tolist()
    assert fill1.loc[observed, "case_count_used"].tolist() == fill2.loc[observed, "case_count_used"].tolist()
    observed_rate = fill1.loc[fill1["Positive7Day_original"] == "10", "infection_rate"].iloc[0]
    assert abs(float(observed_rate) - 200.0) < 1e-9
    assert fill1["is_primary_scenario"].all()
    assert not fill0["is_primary_scenario"].any()
    for path in result["output_paths"].values():
        written = pd.read_csv(path, dtype=str, keep_default_na=False)
        assert Path(path).exists()
        assert Path(path).name in {"fill0.csv", "fill1.csv", "fill2.csv"}
        assert written["Date"].tolist() == written["Date_original"].tolist()
        assert "infection_rate" in written.columns
        assert not written["Date"].str.contains("-").any()


def test_invalid_non_numeric_is_separate_from_missing(tmp_path, monkeypatch):
    frame = _sample()
    frame.loc[3, "Positive7Day"] = "abc"
    result = _run([frame], [2022], _master(["S02001576", "S02001577"]), tmp_path, monkeypatch)
    fill1 = result["scenario_frames"]["fill1"]
    invalid = fill1["response_status"].eq("invalid_non_numeric_value")
    assert int(invalid.sum()) == 1
    assert fill1.loc[invalid, "invalid_value_flag"].eq(1).all()
    assert fill1.loc[invalid, "case_count_used"].isna().all()
    for other in result["scenario_frames"].values():
        assert other.loc[invalid, "case_count_used"].isna().all()


def test_negative_and_non_integer_observed_counts_fail(tmp_path, monkeypatch):
    master = _master(["S02001576", "S02001577"])
    negative = _sample()
    negative.loc[0, "Positive7Day"] = "-1"
    with pytest.raises(CovidPreprocessError, match="Negative"):
        _run([negative], [2022], master, tmp_path, monkeypatch)
    fractional = _sample()
    fractional.loc[0, "Positive7Day"] = "10.5"
    with pytest.raises(CovidPreprocessError, match="not integers"):
        _run([fractional], [2022], master, tmp_path, monkeypatch)


def test_invalid_population_fails_for_observed_rows(tmp_path, monkeypatch):
    master = _master(["S02001576", "S02001577"])
    for value in ["", "0", "-10"]:
        frame = _sample()
        frame.loc[0, "Population"] = value
        with pytest.raises(CovidPreprocessError, match="Population"):
            _run([frame], [2022], master, tmp_path, monkeypatch)


def test_calendar_gaps_distinguish_partial_and_full_missing_dates(tmp_path, monkeypatch):
    result = _run(
        [_sample()],
        [2022],
        _master(["S02001576", "S02001577"]),
        tmp_path,
        monkeypatch,
    )
    gaps = result["report"]["calendar_gaps"]
    assert gaps["n_missing_iz_date_cells"] == 0
    assert gaps["n_dates_with_partial_iz_coverage"] == 0
    assert gaps["n_fully_missing_dates"] == 0

    frame = _sample()
    frame = frame.loc[~((frame["Date"] == "20220102") & (frame["IntZone"] == "S02001577"))]
    result = _run([frame], [2022], _master(["S02001576", "S02001577"]), tmp_path, monkeypatch)
    gaps = result["report"]["calendar_gaps"]
    assert gaps["n_missing_iz_date_cells"] == 1
    assert gaps["n_dates_with_partial_iz_coverage"] == 1
    assert gaps["n_fully_missing_dates"] == 0


def test_cross_scenario_only_suppressed_counts_differ(tmp_path, monkeypatch):
    result = _run(
        [_sample()],
        [2022],
        _master(["S02001576", "S02001577"]),
        tmp_path,
        monkeypatch,
    )
    fill0 = result["scenario_frames"]["fill0"]
    fill1 = result["scenario_frames"]["fill1"]
    fill2 = result["scenario_frames"]["fill2"]
    assert len(fill0) == len(fill1) == len(fill2)
    assert list(zip(fill0["Date"], fill0["IntZone"])) == list(zip(fill1["Date"], fill1["IntZone"]))
    assert list(fill0["node_index"]) == list(fill1["node_index"]) == list(fill2["node_index"])
    suppressed = fill1["suppression_flag"].eq(1)
    left = fill0.loc[~suppressed, "case_count_used"]
    right = fill1.loc[~suppressed, "case_count_used"]
    assert left.isna().equals(right.isna())
    comparable = left.notna()
    assert (left.loc[comparable].astype("int64") == right.loc[comparable].astype("int64")).all()
    assert result["report"]["primary_scenario"] == "fill1"
    assert result["report"]["published_rate_discrepancy"]["published_rate_kind"] == "disclosure_band"
