from pathlib import Path

from agent.region_training import region_output_dir, region_rolling_alpha_path
from allocation.contracts import EDINBURGH_CA
from data.dataset import chronological_fraction_cuts


def test_glasgow_boundaries_stay_in_region_dir():
    from agent.region_training import region_boundaries_path

    path = region_boundaries_path("S12000049")
    assert "S12000049" in str(path)
    assert "website_article_v1" not in str(path)
    assert path.name == "iz_boundaries.geojson"


def test_glasgow_rolling_alpha_stays_in_region_dir():
    path = region_rolling_alpha_path("S12000049")
    text = str(path)
    assert "S12000049" in text
    assert "rolling_v1" not in text
    assert path == region_output_dir("S12000049") / "rolling" / "final_test" / "W730" / "rolling_alpha.csv"


def test_edinburgh_region_dir_is_not_the_frozen_export():
    out = region_output_dir(EDINBURGH_CA)
    assert "website_article_v1" not in str(out)


def test_glasgow_valid_windows_support_split65():
    cuts = chronological_fraction_cuts(453, 0.65, 0.10)
    assert cuts["train"] == (0, 294)
    assert cuts["validation"] == (294, 339)
    assert cuts["test"] == (339, 453)
    assert cuts["test"][1] - cuts["test"][0] == 114


def test_update_dates_skip_empty_calendar_slots():
    import pandas as pd

    from model.rolling import update_dates

    issues = pd.DatetimeIndex(["2022-07-31", "2022-08-01", "2022-10-28", "2022-11-20", "2023-02-18"])
    dates = update_dates(issues, 28)
    assert [str(d.date()) for d in dates] == ["2022-07-31", "2022-10-28", "2023-02-18"]
