"""SIMD Data Zone aggregation. Zero-population DZs are omitted, not filled."""

import pandas as pd
import pytest

from data.deprivation import DeprivationError, _aggregate_to_iz, _to_numeric_indicators


def _dz_rows():
    return pd.DataFrame(
        {
            "DataZone": ["S01000001", "S01000002", "S01000003"],
            "IntZone": ["S02000001", "S02000001", "S02000002"],
            "Total_population": ["100", "0", "80"],
            "Working_age_population": ["60", "0", "50"],
            "Income_count": ["10", "0", "8"],
            "Employment_count": ["5", "2", "4"],
            "overcrowded_count": ["1", "243", "2"],
            "crime_count": ["3", "*", "1"],
            "University": ["0.1", "*", "0.2"],
            "PT_GP": ["10", "7.3", "12"],
        }
    )


def test_zero_population_data_zones_are_omitted_from_iz_rates():
    dz = _to_numeric_indicators(_dz_rows())
    master = pd.DataFrame({"IntZone": ["S02000001", "S02000002"], "node_index": [0, 1]})
    out = _aggregate_to_iz(dz, master)
    one = out.loc[out["IntZone"] == "S02000001"].iloc[0]
    assert one["n_data_zones"] == 1
    assert one["total_population"] == 100
    assert one["income_rate"] == pytest.approx(0.1)
    assert one["employment_rate"] == pytest.approx(5 / 60)


def test_missing_total_population_is_still_an_error():
    rows = _dz_rows()
    rows.loc[0, "Total_population"] = ""
    with pytest.raises(DeprivationError, match="missing Total_population"):
        _to_numeric_indicators(rows)
