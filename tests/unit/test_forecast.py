from datetime import date, timedelta
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

from data.dataset import (
    FORECAST_HORIZON_DAYS,
    LOOKBACK_DAYS,
    STATIC_FEATURE_COLUMNS,
    ForecastError,
    prepare_forecast_dataset,
)


def _master(codes: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"IntZone": codes, "node_index": list(range(len(codes)))})


def _panel(
    start: date,
    n_days: int,
    codes: list[str],
    rate_fn,
    skip_dates: set[date] | None = None,
    drop_cells: set[tuple[date, str]] | None = None,
    extra_rows: list[dict] | None = None,
    node_index_map: dict[str, int] | None = None,
) -> pd.DataFrame:
    skip_dates = skip_dates or set()
    drop_cells = drop_cells or set()
    node_index_map = node_index_map or {zone: i for i, zone in enumerate(codes)}
    rows = []
    for offset in range(n_days):
        day = start + timedelta(days=offset)
        if day in skip_dates:
            continue
        for zone in codes:
            if (day, zone) in drop_cells:
                continue
            value = rate_fn(day, zone)
            rows.append(
                {
                    "Date": day.strftime("%Y%m%d"),
                    "IntZone": zone,
                    "infection_rate": "" if value is None else str(value),
                    "case_count_used": "1",
                    "Population": "1000",
                    "suppression_flag": "0",
                    "missing_reason": "",
                    "node_index": str(node_index_map[zone]),
                }
            )
    if extra_rows:
        rows.extend(extra_rows)
    return pd.DataFrame(rows)


def _simd_table(codes: list[str]) -> pd.DataFrame:
    rows = []
    for i, zone in enumerate(codes):
        rows.append(
            {
                "IntZone": zone,
                "node_index": i,
                "income_rate": 0.01 * (i + 1),
                "employment_rate": 0.02 * (i + 1),
                "university_rate": 0.10 * (i + 1),
                "overcrowded_rate": 0.03 * (i + 1),
                "crime_rate": 0.04 * (i + 1),
                "pt_gp_min": 10.0 + i,
            }
        )
    return pd.DataFrame(rows)


def _run(tmp_path: Path, panel: pd.DataFrame, master: pd.DataFrame, **kwargs):
    covid_path = tmp_path / "fill1.csv"
    covid_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(covid_path, index=False)
    params = {
        "covid_path": covid_path,
        "iz_master": master,
        "output_dir": tmp_path / "forecast",
    }
    if "simd_path" not in kwargs:
        simd_path = tmp_path / "simd_iz.csv"
        _simd_table(master["IntZone"].tolist()).to_csv(simd_path, index=False)
        params["simd_path"] = simd_path
    params.update(kwargs)
    return prepare_forecast_dataset(**params)


def _complete_panel(start: date = date(2021, 1, 1), n_days: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    codes = ["S020ZZ001", "S020AA002"]
    master = _master(codes)

    def rate_fn(day: date, zone: str) -> float:
        base = (day - start).days + 1
        return float(base if zone == codes[0] else 10 + base)

    return _panel(start, n_days, codes, rate_fn), master


def test_window_geometry_and_shapes(tmp_path):
    panel, master = _complete_panel()
    result = _run(tmp_path, panel, master)
    valid = pd.read_csv(result["output_paths"]["valid_samples"])
    train = np.load(result["output_paths"]["train_arrays"])

    assert (valid["lookback_days"] == LOOKBACK_DAYS).all()
    assert (valid["forecast_horizon_days"] == FORECAST_HORIZON_DAYS).all()
    starts = pd.to_datetime(valid["input_start_date"])
    ends = pd.to_datetime(valid["input_end_date"])
    origins = pd.to_datetime(valid["forecast_origin_date"])
    targets = pd.to_datetime(valid["target_date"])
    assert ((ends - starts).dt.days == 6).all()
    assert ((ends - starts).dt.days + 1 == 7).all()
    assert (origins == ends).all()
    assert ((targets - origins).dt.days == 7).all()

    ordered = valid.sort_values("forecast_origin_date")
    origin_gap = pd.to_datetime(ordered["forecast_origin_date"]).diff().dt.days.dropna()
    target_gap = pd.to_datetime(ordered["target_date"]).diff().dt.days.dropna()
    assert (origin_gap == 1).all()
    assert (target_gap == 1).all()
    shared = (
        pd.to_datetime(ordered["input_end_date"]).shift(1) - pd.to_datetime(ordered["input_start_date"])
    ).dt.days.dropna() + 1
    assert (shared == 6).all()

    n_train = result["split_counts"]["train"]
    assert train["X_dynamic_raw"].shape == (n_train, 7, 2, 1)
    assert train["y_target_raw"].shape == (n_train, 2, 1)
    assert train["y_target_raw"].shape[-1] == 1
    assert train["y_target_raw"].ndim == 3


def test_node_order_follows_master_not_file_order(tmp_path):
    codes = ["S020ZZ001", "S020AA002"]
    master = _master(codes)
    start = date(2021, 1, 1)
    file_order = ["S020AA002", "S020ZZ001"]
    node_index_map = {zone: i for i, zone in enumerate(codes)}

    def rate_fn(day: date, zone: str) -> float:
        return 100.0 if zone == "S020ZZ001" else 1.0

    panel = _panel(start, 20, file_order, rate_fn, node_index_map=node_index_map)
    result = _run(tmp_path, panel, master)
    train = np.load(result["output_paths"]["train_arrays"])
    assert train["X_dynamic_raw"][0, 0, 0, 0] == 100.0
    assert train["X_dynamic_raw"][0, 0, 1, 0] == 1.0
    nodes = pd.read_csv(result["output_paths"]["node_order"])
    assert nodes["IntZone"].tolist() == codes


def test_duplicate_date_iz_raises(tmp_path):
    panel, master = _complete_panel()
    panel = pd.concat([panel, panel.iloc[[0]]], ignore_index=True)
    with pytest.raises(ForecastError, match="Duplicate Date-IntZone"):
        _run(tmp_path, panel, master)


def test_unknown_iz_raises(tmp_path):
    panel, master = _complete_panel()
    panel.loc[0, "IntZone"] = "S020UNKNOWN"
    with pytest.raises(ForecastError, match="unknown IntZone"):
        _run(tmp_path, panel, master)


def test_missing_rate_is_not_replaced_with_zero(tmp_path):
    codes = ["S020ZZ001", "S020AA002"]
    master = _master(codes)
    start = date(2021, 1, 1)
    hole = start + timedelta(days=10)

    def rate_fn(day: date, zone: str) -> float | None:
        if day == hole and zone == codes[0]:
            return None
        return 5.0

    panel = _panel(start, 40, codes, rate_fn)
    result = _run(tmp_path, panel, master)
    excluded = pd.read_csv(result["output_paths"]["excluded_samples"])
    assert not excluded.empty
    train = np.load(result["output_paths"]["train_arrays"])
    assert not np.any(train["X_dynamic_raw"] == 0)
    quality = json.loads(Path(result["output_paths"]["panel_quality"]).read_text(encoding="utf-8"))
    assert int(quality["n_missing_infection_rate"]) == 1
    assert int(quality["n_zero_infection_rate"]) == 0


def test_incomplete_sample_excluded_and_reported(tmp_path):
    codes = ["S020ZZ001", "S020AA002"]
    master = _master(codes)
    start = date(2021, 1, 1)
    skip = date(2021, 1, 10)

    def rate_fn(day: date, zone: str) -> float:
        return 3.0

    panel = _panel(start, 40, codes, rate_fn, skip_dates={skip})
    result = _run(tmp_path, panel, master)
    excluded = pd.read_csv(result["output_paths"]["excluded_samples"])
    assert result["n_excluded_samples"] > 0
    blob = excluded.astype(str).to_csv(index=False)
    assert skip.isoformat() in blob


def test_chronological_split_keeps_all_iz_together(tmp_path):
    panel, master = _complete_panel()
    result = _run(tmp_path, panel, master)
    manifest = pd.read_csv(result["output_paths"]["split_manifest"])
    assert manifest.groupby("target_date")["split"].nunique().max() == 1
    by_split = {
        label: pd.to_datetime(part["target_date"])
        for label, part in manifest.groupby("split")
    }
    assert by_split["train"].max() < by_split["validation"].min()
    assert by_split["validation"].max() < by_split["test"].min()


def test_scaler_uses_unique_training_inputs_only(tmp_path):
    codes = ["S020ZZ001", "S020AA002"]
    master = _master(codes)
    start = date(2021, 1, 1)

    def rate_fn(day: date, zone: str) -> float:
        offset = (day - start).days
        if zone == codes[0]:
            return 100.0 if offset == 0 else 0.0
        return float(offset + 1)

    panel = _panel(start, 20, codes, rate_fn)
    result = _run(tmp_path, panel, master)
    scaler = pd.read_csv(result["output_paths"]["scaler"])
    row = scaler.loc[scaler["IntZone"] == codes[0]].iloc[0]
    n_unique = int(row["n_unique_training_input_observations"])
    expected_mean = 100.0 / n_unique
    expected_std = float(np.std(np.array([100.0] + [0.0] * (n_unique - 1)), ddof=0))
    assert row["train_mean"] == pytest.approx(expected_mean)
    assert row["train_std"] == pytest.approx(expected_std)

    train = np.load(result["output_paths"]["train_arrays"])
    val = np.load(result["output_paths"]["validation_arrays"])
    test = np.load(result["output_paths"]["test_arrays"])
    mean = row["train_mean"]
    scale = row["effective_scale"]
    assert val["X_dynamic_scaled"][:, :, 0, 0] == pytest.approx(
        (val["X_dynamic_raw"][:, :, 0, 0] - mean) / scale
    )
    assert test["y_target_scaled"][:, 0, 0] == pytest.approx(
        (test["y_target_raw"][:, 0, 0] - mean) / scale
    )
    assert train["y_target_raw"].min() >= 0


def test_validation_and_test_do_not_affect_scaler(tmp_path):
    codes = ["S020ZZ001", "S020AA002"]
    master = _master(codes)
    start = date(2021, 1, 1)

    def rate_fn(day: date, zone: str) -> float:
        offset = (day - start).days
        if offset >= 14:
            return 9999.0
        return 2.0

    panel = _panel(start, 20, codes, rate_fn)
    result = _run(tmp_path, panel, master)
    scaler = pd.read_csv(result["output_paths"]["scaler"])
    assert scaler["train_mean"].to_numpy() == pytest.approx([2.0, 2.0])
    last_train_input = pd.to_datetime(scaler["last_training_input_date"].iloc[0])
    assert last_train_input < pd.Timestamp("2021-01-15")


def test_zero_variance_scale_is_one(tmp_path):
    codes = ["S020ZZ001", "S020AA002"]
    master = _master(codes)
    start = date(2021, 1, 1)

    def rate_fn(day: date, zone: str) -> float:
        if zone == codes[0]:
            return 4.0
        return float((day - start).days + 1)

    panel = _panel(start, 20, codes, rate_fn)
    result = _run(tmp_path, panel, master)
    scaler = pd.read_csv(result["output_paths"]["scaler"])
    constant = scaler.loc[scaler["IntZone"] == codes[0]].iloc[0]
    assert constant["train_std"] == pytest.approx(0.0)
    assert constant["effective_scale"] == pytest.approx(1.0)
    train = np.load(result["output_paths"]["train_arrays"])
    z = train["X_dynamic_scaled"][:, :, 0, 0]
    assert z == pytest.approx(4.0 - constant["train_mean"])


def test_raw_targets_keep_original_units(tmp_path):
    panel, master = _complete_panel()
    result = _run(tmp_path, panel, master)
    train = np.load(result["output_paths"]["train_arrays"])
    first_origin = date(2021, 1, 7)
    first_target = first_origin + timedelta(days=7)
    expected = float((first_target - date(2021, 1, 1)).days + 1)
    assert train["y_target_raw"][0, 0, 0] == expected


def test_rerun_is_deterministic(tmp_path):
    panel, master = _complete_panel()
    first = _run(tmp_path / "a", panel, master)
    second = _run(tmp_path / "b", panel, master)
    for key in ("train_arrays", "validation_arrays", "test_arrays"):
        left = np.load(first["output_paths"][key])
        right = np.load(second["output_paths"][key])
        for name in left.files:
            np.testing.assert_array_equal(left[name], right[name])
    left_manifest = pd.read_csv(first["output_paths"]["split_manifest"])
    right_manifest = pd.read_csv(second["output_paths"]["split_manifest"])
    pd.testing.assert_frame_equal(left_manifest, right_manifest)


def test_default_config_is_l7_h7_s1(tmp_path):
    panel, master = _complete_panel()
    result = _run(tmp_path, panel, master)
    assert result["lookback_days"] == 7
    assert result["forecast_horizon_days"] == 7
    assert result["window_stride_days"] == 1
    assert result["config_id"] == "L7_H7_S1"
    assert result["target_period_start_offset"] == 1
    assert result["target_period_end_offset"] == 7
    assert result["target_overlaps_latest_input_period"] is False
    for payload in result["array_shapes"].values():
        assert payload["y_target_raw"][-1] == 1
        assert len(payload["y_target_raw"]) == 3


def test_lookback_14_horizon_7_keeps_next_week_target(tmp_path):
    panel, master = _complete_panel(n_days=40)
    result = _run(tmp_path, panel, master, lookback_days=14, forecast_horizon_days=7)
    assert result["config_id"] == "L14_H7_S1"
    assert result["lookback_days"] == 14
    assert result["forecast_horizon_days"] == 7
    assert result["target_period_start_offset"] == 1
    assert result["target_period_end_offset"] == 7
    valid = pd.read_csv(result["output_paths"]["valid_samples"])
    starts = pd.to_datetime(valid["input_start_date"])
    ends = pd.to_datetime(valid["input_end_date"])
    origins = pd.to_datetime(valid["forecast_origin_date"])
    targets = pd.to_datetime(valid["target_date"])
    assert ((ends - starts).dt.days + 1 == 14).all()
    assert (origins == ends).all()
    assert ((targets - origins).dt.days == 7).all()
    train = np.load(result["output_paths"]["train_arrays"])
    n_train = result["split_counts"]["train"]
    assert train["X_dynamic_raw"].shape == (n_train, 14, 2, 1)
    assert train["y_target_raw"].shape == (n_train, 2, 1)


def test_lookback_7_horizon_14_is_not_a_14_day_cumulative(tmp_path):
    panel, master = _complete_panel(n_days=40)
    result = _run(tmp_path, panel, master, lookback_days=7, forecast_horizon_days=14)
    assert result["config_id"] == "L7_H14_S1"
    assert result["lookback_days"] == 7
    assert result["forecast_horizon_days"] == 14
    assert result["target_period_start_offset"] == 8
    assert result["target_period_end_offset"] == 14
    assert result["target_overlaps_latest_input_period"] is False
    assert "14-day cumulative" in result["target_period_interpretation"]
    valid = pd.read_csv(result["output_paths"]["valid_samples"])
    origins = pd.to_datetime(valid["forecast_origin_date"])
    targets = pd.to_datetime(valid["target_date"])
    assert ((targets - origins).dt.days == 14).all()
    train = np.load(result["output_paths"]["train_arrays"])
    n_train = result["split_counts"]["train"]
    assert train["X_dynamic_raw"].shape == (n_train, 7, 2, 1)
    assert train["y_target_raw"].shape == (n_train, 2, 1)
    assert train["y_target_raw"].ndim == 3


def test_unequal_configs_are_distinct(tmp_path):
    panel, master = _complete_panel(n_days=40)
    left = _run(tmp_path / "l14h7", panel, master, lookback_days=14, forecast_horizon_days=7)
    right = _run(tmp_path / "l7h14", panel, master, lookback_days=7, forecast_horizon_days=14)
    assert left["config_id"] != right["config_id"]
    assert left["config_id"] == "L14_H7_S1"
    assert right["config_id"] == "L7_H14_S1"


def test_static_simd_is_in_all_splits_and_not_covid_scaled(tmp_path):
    panel, master = _complete_panel()
    result = _run(tmp_path, panel, master)
    expected = _simd_table(master["IntZone"].tolist())[list(STATIC_FEATURE_COLUMNS)].to_numpy(dtype="float64")
    names = np.array(STATIC_FEATURE_COLUMNS, dtype="U32")
    for key in ("train_arrays", "validation_arrays", "test_arrays"):
        payload = np.load(result["output_paths"][key])
        assert payload["X_static_raw"].shape == (2, 6)
        assert payload["X_static_raw"] == pytest.approx(expected)
        assert payload["X_static_feature_names"].tolist() == names.tolist()
    static = pd.read_csv(result["output_paths"]["static_features"])
    assert static["IntZone"].tolist() == master["IntZone"].tolist()
    assert result["static_scaling_mode"] == "none"
    train = np.load(result["output_paths"]["train_arrays"])
    covid_mean = pd.read_csv(result["output_paths"]["scaler"])["train_mean"].to_numpy()
    assert not np.allclose(train["X_static_raw"][:, 0], expected[:, 0] - covid_mean)


def test_missing_simd_iz_raises(tmp_path):
    panel, master = _complete_panel()
    simd_path = tmp_path / "simd_iz.csv"
    _simd_table([master["IntZone"].iloc[0]]).to_csv(simd_path, index=False)
    with pytest.raises(ForecastError, match="no SIMD row"):
        _run(tmp_path, panel, master, simd_path=simd_path)
