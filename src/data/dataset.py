"""Chronological COVID sliding windows, dataset split and rate standardisation.

The input COVID panel is already filtered to the City of Edinburgh by the
upstream harmonisation module. This module does not repeat the area filter.
It validates the study geography by requiring an exact match between panel IZ
codes and the authoritative Edinburgh IZ master.

Each Y_t is a daily-reported rolling seven-day infection rate (PHS window
ending on date t), not a daily-new-case rate and not an H-day cumulative rate.
Lookback L is how many consecutive daily reports the model sees.
Horizon H is how many calendar days after origin t the single target is dated:

    [Y_(t-L+1), ..., Y_t] -> Y_(t+H)

Y_(t+H) is still one rolling-seven-day rate. It approximately covers t+H-6
through t+H, because the response window is always seven days (6 = 7-1).
The output is one value per IZ, never H daily forecasts.

Lookback and horizon need not be equal. Defaults are L=7, H=7, stride=1:
input seven recent reports and predict the rate reported seven days later
(target period approximately t+1 through t+7). Examples:
- L=14, H=7: more history, same next-week target.
- L=7, H=14: predict the 7-day rate dated 14 days later (about t+8..t+14),
  not a 14-day cumulative rate.

Stride is one calendar day so the origin can roll forward daily. Adjacent
lookbacks overlap. The split is chronological by target date. The scaler is
fit on unique training-input Date-IZ rates only. Node order comes from the
IZ master. SIMD 2020 indicators are attached as static node features
X_static_raw [IZ, F], identical in train/validation/test and aligned to
node_order. They are not tiled across the lookback and are not standardised
by the COVID rate scaler.

A model is identified by config_id L{L}_H{H}_S{S}. A checkpoint trained for
one configuration must not be used for another without retraining.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from math import floor
from pathlib import Path
from typing import Any
import math
import shutil

import numpy as np
import pandas as pd

from data.covid import MODELLING_RATE_COLUMN, load_iz_master
from data.node_order import NodeOrder, assert_edinburgh_count, load_node_order
from common.errors import ModelError
from common.utils import (
    EXPECTED_IZ_COUNT,
    GEOGRAPHY_VINTAGE,
    LOCAL_AUTHORITY_CODE,
    LOCAL_AUTHORITY_NAME,
    NODE_KEY,
    get_logger,
    project_root,
    read_table,
    results_dir,
    write_json,
    write_run_log,
    write_table,
)
from model.constants import STATIC_FEATURE_COLUMNS

LOGGER = get_logger("forecast")

LOOKBACK_DAYS = 7
FORECAST_HORIZON_DAYS = 7
WINDOW_STRIDE_DAYS = 1
# PHS response length. Distinct from lookback (input reports) and horizon (lead time).
RESPONSE_WINDOW_DAYS = 7
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
SCALE_EPSILON = 1e-8
SCALE_DDOF = 0
SCALING_MODE = "per_iz"
STATIC_SCALING_MODE = "none"
PIPELINE_VERSION = "0.1.0"
# One SIMD 2020 indicator per domain: see model.constants.STATIC_FEATURE_COLUMNS.
SPLIT_LABELS = ("train", "validation", "test")
FITTING_POLICY = "unique_training_input_date_iz_observations"
MISSING_POLICY = (
    "Do not invent Date-IZ rows, interpolate, or replace ordinary missing "
    "infection rates with zero. Exclude the entire origin if any required "
    "input or target cell is absent or non-finite."
)
OUTPUT_FILENAMES = (
    "node_order.csv",
    "panel_quality.json",
    "valid_samples.csv",
    "excluded_samples.csv",
    "split_manifest.csv",
    "split_summary.json",
    "scaler.csv",
    "static_features.csv",
    "array_integrity.json",
    "train.npz",
    "validation.npz",
    "test.npz",
    "run_metadata.json",
)
IZ_MASTER_RELATIVE_PATH = Path("data") / "raw" / "boundaries" / "Code lookup.csv"
EXCLUDED_COLUMNS = [
    "forecast_origin_date",
    "target_date",
    "exclusion_reason",
    "fully_missing_input_dates",
    "partially_missing_input_dates",
    "n_absent_input_rows",
    "n_missing_input_rate_values",
    "target_date_fully_missing",
    "target_date_partial_coverage",
    "n_absent_target_rows",
    "n_missing_target_rate_values",
    "n_affected_iz",
    "affected_iz_sample",
]


class ForecastError(ValueError):
    """Sliding-window construction cannot continue without guessing."""


@dataclass(frozen=True)
class WindowConfig:
    """User-configurable sample geometry. Lookback and horizon need not match.

    lookback_days: consecutive daily rolling-7-day reports used as input.
    forecast_horizon_days: calendar lead time from origin t to the single target
        report date t+H. Not an H-day cumulative infection total.
    window_stride_days: calendar step between candidate origins.
    """

    lookback_days: int
    forecast_horizon_days: int
    window_stride_days: int = WINDOW_STRIDE_DAYS

    @property
    def config_id(self) -> str:
        return (
            f"L{self.lookback_days}_H{self.forecast_horizon_days}_S{self.window_stride_days}"
        )


@dataclass(frozen=True)
class WindowRecord:
    sample_id: int
    input_dates: tuple[date, ...]
    origin: date
    target: date


def prepare_forecast_dataset(
    lookback_days: int = LOOKBACK_DAYS,
    forecast_horizon_days: int = FORECAST_HORIZON_DAYS,
    covid_path: Path | None = None,
    iz_master: pd.DataFrame | None = None,
    area_code: str = LOCAL_AUTHORITY_CODE,
    start_date: str | date | None = None,
    end_date: str | date | None = None,
    split_boundaries: dict[str, tuple[str | date, str | date]] | None = None,
    epsilon: float = SCALE_EPSILON,
    output_dir: Path | None = None,
    overwrite: bool = False,
    window_stride_days: int = WINDOW_STRIDE_DAYS,
    simd_path: Path | None = None,
) -> dict[str, Any]:
    """Build chronological COVID windows, split them, and standardise rates.

    Defaults are lookback=7, horizon=7, stride=1. The Agent may override them.
    Horizon H is the lead time to one rolling-seven-day report Y_(t+H), not
    an H-day cumulative rate and not H daily outputs.

    SIMD is loaded from simd_iz.csv (one 2020 row per IZ) and stored as
    X_static_raw. fill1.csv stays COVID-only.

    Returns output paths and summary metadata. Does not train a model.
    The input panel is already Edinburgh-filtered upstream; this function does
    not drop rows by local-authority code.
    """
    window = WindowConfig(
        lookback_days=int(lookback_days),
        forecast_horizon_days=int(forecast_horizon_days),
        window_stride_days=int(window_stride_days),
    )
    _validate_config(window=window, epsilon=epsilon)
    horizon_notes = _horizon_interpretation(window)
    LOGGER.info("%s: %s", window.config_id, horizon_notes["target_period_interpretation"])
    covid_path = Path(covid_path) if covid_path is not None else results_dir() / "fill1.csv"
    if iz_master is None:
        iz_master_source = str(project_root() / IZ_MASTER_RELATIVE_PATH)
        iz_master = load_iz_master(area_code=area_code)
    else:
        iz_master_source = "supplied DataFrame"

    nodes = _validate_iz_master(iz_master)
    node_order = _node_order_info(nodes, iz_master_source=iz_master_source)
    panel = _load_covid_panel(covid_path, nodes, area_code=area_code)
    start = _coerce_date(start_date) if start_date is not None else panel["Date"].min().date()
    end = _coerce_date(end_date) if end_date is not None else panel["Date"].max().date()
    if start > end:
        raise ForecastError(f"analysis start {start.isoformat()} is after end {end.isoformat()}.")
    _validate_analysis_span(start, end, window=window)

    if output_dir is None:
        output_dir = _default_output_dir(start, end, window=window)
    else:
        output_dir = Path(output_dir)
    _assert_overwrite_allowed(output_dir, overwrite=overwrite)
    output_dir.mkdir(parents=True, exist_ok=True)

    in_period = (panel["Date"].dt.date >= start) & (panel["Date"].dt.date <= end)
    panel = panel.loc[in_period].copy()
    quality = _panel_quality(panel, nodes, start=start, end=end)
    calendar = pd.date_range(start, end, freq="D")
    rate_mat, present_mat = _date_node_matrices(panel, nodes, calendar)

    valid, excluded = _build_windows(calendar, rate_mat, present_mat, nodes, window=window)
    if not valid:
        raise ForecastError("No valid sliding-window samples were produced.")
    assigned = _assign_splits(valid, split_boundaries)
    _require_nonempty_splits(assigned)

    simd_path = Path(simd_path) if simd_path is not None else results_dir() / "simd_iz.csv"
    static_table, static_mat = _load_static_simd(simd_path, nodes)
    scaler = _fit_scaler(assigned, calendar, rate_mat, nodes, epsilon=epsilon)
    arrays = _build_split_arrays(
        assigned, calendar, rate_mat, scaler, window=window, static_mat=static_mat
    )
    paths = _write_outputs(
        output_dir=output_dir,
        covid_path=covid_path,
        simd_path=simd_path,
        nodes=nodes,
        node_order=node_order,
        quality=quality,
        valid=valid,
        excluded=excluded,
        assigned=assigned,
        scaler=scaler,
        static_table=static_table,
        arrays=arrays,
        start=start,
        end=end,
        split_boundaries=split_boundaries,
        epsilon=epsilon,
        iz_master_source=iz_master_source,
        overwrite=overwrite,
        window=window,
        horizon_notes=horizon_notes,
    )
    summary = _run_summary(
        paths, quality, valid, excluded, assigned, scaler, arrays, window, horizon_notes
    )
    write_run_log({"event": "forecast_prepare_complete", **summary}, filename="forecast_prepare.jsonl")
    LOGGER.info(
        "Wrote %s valid samples (%s excluded) to %s",
        len(valid),
        len(excluded),
        output_dir,
    )
    return summary


def inverse_transform_targets(
    y_scaled: np.ndarray,
    scaler: pd.DataFrame,
    nodes: pd.DataFrame,
) -> np.ndarray:
    """Convert standardised targets back to infection-rate per 100,000.

    Do not use standardised values as public-health risk. Mapped forecasts and
    allocation inputs must use this inverse transform.
    """
    y = np.asarray(y_scaled, dtype="float64")
    if y.ndim != 3 or y.shape[-1] != 1:
        raise ForecastError("Predictions must have shape [samples, IZs, 1].")
    nodes_ordered = _validate_iz_master(nodes)
    n_iz = int(len(nodes_ordered))
    if y.shape[1] != n_iz:
        raise ForecastError(
            f"Prediction IZ dimension is {y.shape[1]}, expected {n_iz} from the IZ master."
        )
    required = {NODE_KEY, "node_index", "train_mean", "effective_scale"}
    missing = required - set(scaler.columns)
    if missing:
        raise ForecastError(f"Scaler missing columns {sorted(missing)}.")
    aligned = nodes_ordered.merge(
        scaler[[NODE_KEY, "node_index", "train_mean", "effective_scale"]],
        on=[NODE_KEY, "node_index"],
        how="left",
        validate="one_to_one",
    )
    if aligned["train_mean"].isna().any() or aligned["effective_scale"].isna().any():
        raise ForecastError("Scaler node order does not match the authoritative IZ master.")
    mean = aligned["train_mean"].to_numpy(dtype="float64").reshape(1, n_iz, 1)
    scale = aligned["effective_scale"].to_numpy(dtype="float64").reshape(1, n_iz, 1)
    return y * scale + mean


def _require_positive_int(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ForecastError(f"{name} must be a positive integer.")


def _horizon_interpretation(window: WindowConfig) -> dict[str, Any]:
    """Describe the 7-day epidemiological period implied by horizon H.

    The PHS response is always a rolling 7-day rate, so Y_(t+H) covers about
    t+H-6 through t+H. H is lead time, not the length of a cumulative rate.
    """
    horizon = window.forecast_horizon_days
    start_offset = horizon - (RESPONSE_WINDOW_DAYS - 1)
    end_offset = horizon
    overlaps = horizon < RESPONSE_WINDOW_DAYS
    if horizon < RESPONSE_WINDOW_DAYS:
        text = (
            "Horizon is shorter than the 7-day response window, so the target "
            "rolling-seven-day period overlaps the epidemiological period of "
            "the last input report."
        )
    elif horizon == RESPONSE_WINDOW_DAYS:
        text = (
            "Horizon equals 7, so the target is the immediately subsequent "
            "complete seven-day period after the last input report "
            "(approximately t+1 through t+7)."
        )
    else:
        text = (
            "Horizon is longer than 7, so there is a gap between the last input "
            "report's underlying epidemiological period and the target period. "
            f"Y_(t+{horizon}) is still one rolling-seven-day rate "
            f"(approximately t+{start_offset} through t+{end_offset}), not a "
            f"{horizon}-day cumulative infection rate."
        )
    return {
        "config_id": window.config_id,
        "lookback_days": window.lookback_days,
        "forecast_horizon_days": horizon,
        "window_stride_days": window.window_stride_days,
        "response_window_days": RESPONSE_WINDOW_DAYS,
        "target_period_start_offset": start_offset,
        "target_period_end_offset": end_offset,
        "target_period_interpretation": text,
        "target_overlaps_latest_input_period": overlaps,
    }


def _validate_config(window: WindowConfig, epsilon: float) -> None:
    _require_positive_int("lookback_days", window.lookback_days)
    _require_positive_int("forecast_horizon_days", window.forecast_horizon_days)
    _require_positive_int("window_stride_days", window.window_stride_days)
    if epsilon <= 0:
        raise ForecastError("epsilon must be > 0.")
    if not 0 < TRAIN_FRACTION < 1:
        raise ForecastError("TRAIN_FRACTION must be in (0, 1).")
    if not 0 < VALIDATION_FRACTION < 1:
        raise ForecastError("VALIDATION_FRACTION must be in (0, 1).")
    if TRAIN_FRACTION + VALIDATION_FRACTION >= 1:
        raise ForecastError("TRAIN_FRACTION + VALIDATION_FRACTION must be < 1.")


def _validate_analysis_span(start: date, end: date, window: WindowConfig) -> None:
    n_days = (end - start).days + 1
    needed = window.lookback_days + window.forecast_horizon_days
    if n_days < needed:
        raise ForecastError(
            f"Analysis period has {n_days} calendar days; "
            f"need at least lookback_days + forecast_horizon_days = {needed}."
        )


def _default_output_dir(start: date, end: date, window: WindowConfig) -> Path:
    name = f"{window.config_id}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}"
    return results_dir() / "forecast" / name


def _planned_output_paths(output_dir: Path) -> list[Path]:
    return [output_dir / name for name in OUTPUT_FILENAMES]


def _assert_overwrite_allowed(output_dir: Path, overwrite: bool) -> None:
    existing = [path for path in _planned_output_paths(output_dir) if path.exists()]
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise ForecastError(
            f"Output already exists ({names}). Pass overwrite=True or --overwrite to replace "
            "only these forecast files."
        )


def _validate_iz_master(iz_master: pd.DataFrame) -> pd.DataFrame:
    """Authoritative node order: unique IntZone, contiguous node_index 0..N-1."""
    if NODE_KEY not in iz_master.columns or "node_index" not in iz_master.columns:
        raise ForecastError("IZ master must contain IntZone and node_index.")
    nodes = iz_master[[NODE_KEY, "node_index"]].copy()
    nodes[NODE_KEY] = nodes[NODE_KEY].astype("string").str.strip()
    nodes["node_index"] = pd.to_numeric(nodes["node_index"], errors="coerce")
    if nodes[NODE_KEY].eq("").any() or nodes[NODE_KEY].isna().any():
        raise ForecastError("IZ master contains empty IntZone codes.")
    if nodes["node_index"].isna().any():
        raise ForecastError("IZ master contains non-integer node_index values.")
    nodes["node_index"] = nodes["node_index"].astype("int64")
    if nodes[NODE_KEY].duplicated().any():
        raise ForecastError("IZ master has duplicate IntZone codes.")
    if nodes["node_index"].duplicated().any():
        raise ForecastError("IZ master has duplicate node_index values.")
    nodes = nodes.sort_values("node_index").reset_index(drop=True)
    expected = list(range(len(nodes)))
    if nodes["node_index"].tolist() != expected:
        raise ForecastError("node_index must be contiguous from 0 to N-1.")
    return nodes


def _node_order_hash(nodes: pd.DataFrame) -> str:
    ordered_codes = nodes.sort_values("node_index")[NODE_KEY].astype(str).tolist()
    return hashlib.sha256("|".join(ordered_codes).encode("utf-8")).hexdigest()


def _node_order_info(nodes: pd.DataFrame, iz_master_source: str) -> dict[str, Any]:
    digest = _node_order_hash(nodes)
    return {
        "geography_type": "2011 Intermediate Zone",
        "geography_vintage": GEOGRAPHY_VINTAGE,
        "n_nodes": int(len(nodes)),
        "node_index_rule": "contiguous integer node_index from 0 to N-1, ordered by the IZ master",
        "node_order_sha256": digest,
        "iz_master_source": iz_master_source,
        "reference": (
            f"2011 Intermediate Zone; vintage={GEOGRAPHY_VINTAGE}; "
            f"n={len(nodes)}; node_index=0..N-1; sha256={digest}"
        ),
    }


def _load_covid_panel(
    covid_path: Path,
    nodes: pd.DataFrame,
    area_code: str = LOCAL_AUTHORITY_CODE,
) -> pd.DataFrame:
    """Load the already Edinburgh-filtered panel. Do not filter by CA again."""
    if not covid_path.exists():
        raise ForecastError(f"COVID table missing: {covid_path}")
    frame = read_table(covid_path)
    required = ["Date", NODE_KEY, MODELLING_RATE_COLUMN]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ForecastError(f"COVID table missing columns {missing}.")

    out = frame.copy()
    out[NODE_KEY] = out[NODE_KEY].astype("string").str.strip()
    out["Date"] = _parse_panel_dates(out["Date"])
    if out["Date"].isna().any():
        raise ForecastError("COVID table contains empty or unparseable Date values.")

    _check_optional_area_code(out, area_code=area_code)

    covid_zones = set(out[NODE_KEY].tolist())
    master_zones = set(nodes[NODE_KEY].tolist())
    unknown = sorted(covid_zones - master_zones)
    omitted = sorted(master_zones - covid_zones)
    if unknown:
        raise ForecastError(f"{len(unknown)} unknown IntZone codes: {unknown[:10]}.")
    if omitted:
        raise ForecastError(f"{len(omitted)} expected IntZone codes are omitted: {omitted[:10]}.")
    if covid_zones != master_zones:
        raise ForecastError("Panel IZ codes do not exactly match the IZ master.")

    dup = out.duplicated(["Date", NODE_KEY], keep=False)
    if bool(dup.any()):
        raise ForecastError("Duplicate Date-IntZone records are present.")

    had_node_index = "node_index" in out.columns
    out = out.merge(nodes, on=NODE_KEY, how="left", suffixes=("_covid", ""))
    if had_node_index:
        out = _validate_covid_node_index(out)
    if out["node_index"].isna().any():
        raise ForecastError("Inconsistent node mapping after IZ-master join.")
    out["node_index"] = out["node_index"].astype("int64")

    out[MODELLING_RATE_COLUMN] = _parse_infection_rate(out[MODELLING_RATE_COLUMN])
    finite = out[MODELLING_RATE_COLUMN].notna()
    if bool((out.loc[finite, MODELLING_RATE_COLUMN] < 0).any()):
        raise ForecastError("Finite infection rates must be non-negative.")

    out = out.sort_values(["Date", "node_index"]).reset_index(drop=True)
    return out


def _check_optional_area_code(frame: pd.DataFrame, area_code: str) -> None:
    """Optional consistency check only. Never used to drop rows."""
    for column in ("CA", "local_authority_code"):
        if column not in frame.columns:
            continue
        values = frame[column].astype("string").str.strip()
        nonempty = values.notna() & values.ne("") & values.str.lower().ne("nan")
        disagree = nonempty & values.ne(area_code)
        if bool(disagree.any()):
            samples = values.loc[disagree].head(5).tolist()
            raise ForecastError(
                f"{column} values disagree with expected area code {area_code}: {samples}."
            )


def _validate_covid_node_index(out: pd.DataFrame) -> pd.DataFrame:
    if "node_index_covid" not in out.columns:
        raise ForecastError("COVID table contains missing or invalid node_index values.")
    covid_index = pd.to_numeric(out["node_index_covid"], errors="coerce")
    if covid_index.isna().any():
        raise ForecastError("COVID table contains missing or invalid node_index values.")
    if not np.equal(covid_index, np.floor(covid_index)).all():
        raise ForecastError("COVID table contains non-integer node_index values.")
    covid_index = covid_index.astype("int64")
    master_index = out["node_index"].astype("int64")
    if not covid_index.equals(master_index):
        mismatch_rows = out.loc[
            covid_index.ne(master_index),
            [NODE_KEY, "node_index_covid", "node_index"],
        ].head(10)
        raise ForecastError(
            "COVID node_index disagrees with the authoritative IZ master. "
            f"Examples: {mismatch_rows.to_dict(orient='records')}"
        )
    return out.drop(columns=["node_index_covid"])


def _parse_panel_dates(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    parsed = pd.to_datetime(text, format="%Y%m%d", errors="coerce")
    remaining = parsed.isna() & text.ne("") & text.notna()
    parsed.loc[remaining] = pd.to_datetime(text.loc[remaining], format="%Y-%m-%d", errors="coerce")
    bad = text.ne("") & text.notna() & parsed.isna()
    if bool(bad.any()):
        samples = text.loc[bad].head(5).tolist()
        raise ForecastError(f"Date values are not YYYYMMDD or YYYY-MM-DD: {samples}.")
    return parsed


def _parse_infection_rate(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    empty = text.isna() | text.eq("") | text.str.lower().isin(["nan", "none", "<na>", "null"])
    parsed = pd.to_numeric(text.where(~empty, pd.NA), errors="coerce")
    invalid = ~empty & parsed.isna()
    if bool(invalid.any()):
        samples = text.loc[invalid].head(5).tolist()
        raise ForecastError(f"infection_rate contains non-numeric values: {samples}.")
    if bool(np.isinf(parsed.fillna(0.0)).any()):
        raise ForecastError("infection_rate contains infinite values.")
    return parsed.astype("Float64")


def _to_float64_with_nan(series: pd.Series) -> np.ndarray:
    """Convert a nullable numeric Series to float64, mapping pd.NA to np.nan, never to 0."""
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64", na_value=np.nan)


def _panel_quality(
    panel: pd.DataFrame,
    nodes: pd.DataFrame,
    start: date,
    end: date,
) -> dict[str, Any]:
    calendar = pd.date_range(start, end, freq="D")
    n_dates = int(len(calendar))
    n_iz = int(len(nodes))
    expected = n_dates * n_iz
    observed = int(len(panel))
    present = panel.groupby(panel["Date"].dt.normalize()).size()
    present = present.reindex(calendar, fill_value=0)
    missing_all = [ts.date().isoformat() for ts in calendar[present.eq(0)]]
    partial = [ts.date().isoformat() for ts in calendar[(present > 0) & (present < n_iz)]]
    n_missing_rate = int(panel[MODELLING_RATE_COLUMN].isna().sum())
    n_zero_rate = int((panel[MODELLING_RATE_COLUMN] == 0).sum())
    coverage = (
        panel.groupby(NODE_KEY, observed=True)
        .size()
        .reindex(nodes[NODE_KEY], fill_value=0)
    )
    affected = coverage.loc[coverage.lt(n_dates)].index.astype(str).tolist()
    warnings: list[str] = []
    if missing_all:
        warnings.append(f"{len(missing_all)} calendar dates are absent for every IZ and were not invented.")
    if partial:
        warnings.append(f"{len(partial)} calendar dates have partial IZ coverage.")
    if n_missing_rate:
        warnings.append(f"{n_missing_rate} observed rows have missing infection_rate.")
    if n_iz != EXPECTED_IZ_COUNT:
        warnings.append(f"IZ count is {n_iz}, not the Edinburgh expected {EXPECTED_IZ_COUNT}.")
    status = "ok_with_warnings" if warnings else "ok"
    return {
        "analysis_start_date": start.isoformat(),
        "analysis_end_date": end.isoformat(),
        "n_calendar_dates": n_dates,
        "n_iz": n_iz,
        "expected_date_iz_observations": expected,
        "observed_date_iz_observations": observed,
        "n_absent_date_iz_rows": expected - observed,
        "n_missing_infection_rate": n_missing_rate,
        "n_zero_infection_rate": n_zero_rate,
        "zero_is_not_used_as_missing_fill": True,
        "dates_missing_for_every_iz": missing_all,
        "dates_with_partial_iz_coverage": partial,
        "affected_iz_codes": affected,
        "study_area": LOCAL_AUTHORITY_NAME,
        "local_authority_code": LOCAL_AUTHORITY_CODE,
        "geography_type": "2011 Intermediate Zone",
        "geography_vintage": GEOGRAPHY_VINTAGE,
        "area_filter_repeated": False,
        "warnings": warnings,
        "validation_status": status,
    }


def _date_node_matrices(
    panel: pd.DataFrame,
    nodes: pd.DataFrame,
    calendar: pd.DatetimeIndex,
) -> tuple[np.ndarray, np.ndarray]:
    n_dates = len(calendar)
    n_iz = len(nodes)
    rate = np.full((n_dates, n_iz), np.nan, dtype="float64")
    present = np.zeros((n_dates, n_iz), dtype=bool)
    indexer = calendar.get_indexer(panel["Date"].dt.normalize())
    ok = indexer >= 0
    i = indexer[ok]
    j = panel.loc[ok, "node_index"].to_numpy(dtype="int64")
    values = _to_float64_with_nan(panel.loc[ok, MODELLING_RATE_COLUMN])
    present[i, j] = True
    rate[i, j] = values
    return rate, present


def _build_windows(
    calendar: pd.DatetimeIndex,
    rate_mat: np.ndarray,
    present_mat: np.ndarray,
    nodes: pd.DataFrame,
    window: WindowConfig,
) -> tuple[list[WindowRecord], list[dict[str, Any]]]:
    """Slide origins daily: L consecutive reports at t map to one target at t+H.

    Lookback and horizon are independent. Incomplete Date-IZ samples are
    excluded entirely; missing rates are not filled with zero.
    """
    n_dates, n_iz = rate_mat.shape
    lookback = window.lookback_days
    horizon = window.forecast_horizon_days
    stride = window.window_stride_days
    first_origin = lookback - 1
    last_origin = n_dates - 1 - horizon
    valid: list[WindowRecord] = []
    excluded: list[dict[str, Any]] = []
    iz_codes = nodes[NODE_KEY].tolist()

    origin_idx = first_origin
    while origin_idx <= last_origin:
        input_idx = list(range(origin_idx - lookback + 1, origin_idx + 1))
        target_idx = origin_idx + horizon
        input_dates = tuple(calendar[i].date() for i in input_idx)
        origin = calendar[origin_idx].date()
        target = calendar[target_idx].date()
        x = rate_mat[input_idx]
        y = rate_mat[target_idx]
        x_present = present_mat[input_idx]
        y_present = present_mat[target_idx]

        fully_missing_input = [
            input_dates[row] for row in range(lookback) if not bool(x_present[row].any())
        ]
        partially_missing_input = [
            input_dates[row]
            for row in range(lookback)
            if bool(x_present[row].any()) and not bool(x_present[row].all())
        ]
        n_absent_input = int((~x_present).sum())
        n_missing_input_rate = int((x_present & ~np.isfinite(x)).sum())
        target_fully_missing = not bool(y_present.any())
        target_partial = bool(y_present.any()) and not bool(y_present.all())
        n_absent_target = int((~y_present).sum())
        n_missing_target_rate = int((y_present & ~np.isfinite(y)).sum())
        incomplete_nodes = (
            (~x_present).any(axis=0)
            | (~y_present)
            | (~np.isfinite(x)).any(axis=0)
            | (~np.isfinite(y))
        )
        affected_idx = np.flatnonzero(incomplete_nodes)
        reasons: list[str] = []
        if fully_missing_input:
            reasons.append("fully_missing_input_dates")
        if partially_missing_input:
            reasons.append("partially_missing_input_dates")
        if n_absent_input:
            reasons.append("absent_input_rows")
        if n_missing_input_rate:
            reasons.append("missing_input_rates")
        if target_fully_missing:
            reasons.append("fully_missing_target_date")
        if target_partial:
            reasons.append("partial_target_coverage")
        if n_absent_target:
            reasons.append("absent_target_rows")
        if n_missing_target_rate:
            reasons.append("missing_target_rates")

        complete = (
            len(input_dates) == lookback
            and _consecutive_days(input_dates)
            and origin == input_dates[-1]
            and target == origin + timedelta(days=horizon)
            and not reasons
            and bool(x_present.all())
            and bool(y_present.all())
            and bool(np.isfinite(x).all())
            and bool(np.isfinite(y).all())
        )
        if complete:
            valid.append(
                WindowRecord(
                    sample_id=len(valid),
                    input_dates=input_dates,
                    origin=origin,
                    target=target,
                )
            )
        else:
            excluded.append(
                {
                    "forecast_origin_date": origin.isoformat(),
                    "target_date": target.isoformat(),
                    "exclusion_reason": "|".join(reasons) if reasons else "invalid_window",
                    "fully_missing_input_dates": "|".join(d.isoformat() for d in fully_missing_input),
                    "partially_missing_input_dates": "|".join(
                        d.isoformat() for d in partially_missing_input
                    ),
                    "n_absent_input_rows": n_absent_input,
                    "n_missing_input_rate_values": n_missing_input_rate,
                    "target_date_fully_missing": target_fully_missing,
                    "target_date_partial_coverage": target_partial,
                    "n_absent_target_rows": n_absent_target,
                    "n_missing_target_rate_values": n_missing_target_rate,
                    "n_affected_iz": int(len(affected_idx)),
                    "affected_iz_sample": "|".join(iz_codes[int(i)] for i in affected_idx[:10]),
                }
            )
        origin_idx += stride
    return valid, excluded


def _consecutive_days(days: tuple[date, ...]) -> bool:
    return all(days[i] - days[i - 1] == timedelta(days=1) for i in range(1, len(days)))


def _assign_splits(
    valid: list[WindowRecord],
    split_boundaries: dict[str, tuple[str | date, str | date]] | None,
) -> dict[str, list[WindowRecord]]:
    ordered = sorted(valid, key=lambda sample: (sample.target, sample.origin, sample.sample_id))
    if split_boundaries is None:
        return _fraction_splits(ordered)
    return _boundary_splits(ordered, split_boundaries)


def _fraction_splits(ordered: list[WindowRecord]) -> dict[str, list[WindowRecord]]:
    n = len(ordered)
    n_train = floor(TRAIN_FRACTION * n)
    n_validation = floor(VALIDATION_FRACTION * n)
    n_test = n - n_train - n_validation
    return {
        "train": ordered[:n_train],
        "validation": ordered[n_train : n_train + n_validation],
        "test": ordered[n_train + n_validation : n_train + n_validation + n_test],
    }


def _boundary_splits(
    ordered: list[WindowRecord],
    split_boundaries: dict[str, tuple[str | date, str | date]],
) -> dict[str, list[WindowRecord]]:
    missing = [label for label in SPLIT_LABELS if label not in split_boundaries]
    if missing:
        raise ForecastError(f"split_boundaries missing {missing}.")
    ranges: dict[str, tuple[date, date]] = {}
    for label in SPLIT_LABELS:
        low, high = split_boundaries[label]
        start, end = _coerce_date(low), _coerce_date(high)
        if start > end:
            raise ForecastError(f"{label} split boundary is reversed.")
        ranges[label] = (start, end)
    if not (ranges["train"][1] < ranges["validation"][0] <= ranges["validation"][1] < ranges["test"][0]):
        raise ForecastError("Explicit split boundaries must be chronological and non-overlapping.")

    assigned = {label: [] for label in SPLIT_LABELS}
    for sample in ordered:
        matched = [
            label
            for label, (start, end) in ranges.items()
            if start <= sample.target <= end
        ]
        if len(matched) != 1:
            raise ForecastError(
                f"target_date {sample.target.isoformat()} is not in exactly one split interval."
            )
        assigned[matched[0]].append(sample)
    return assigned


def _require_nonempty_splits(assigned: dict[str, list[WindowRecord]]) -> None:
    empty = [label for label, rows in assigned.items() if not rows]
    if empty:
        raise ForecastError(f"Split(s) {empty} contain no samples.")
    train_last = assigned["train"][-1].target
    val_first = assigned["validation"][0].target
    val_last = assigned["validation"][-1].target
    test_first = assigned["test"][0].target
    if train_last >= val_first:
        raise ForecastError("Validation target dates must follow training target dates.")
    if val_last >= test_first:
        raise ForecastError("Test target dates must follow validation target dates.")


def _fit_scaler(
    assigned: dict[str, list[WindowRecord]],
    calendar: pd.DatetimeIndex,
    rate_mat: np.ndarray,
    nodes: pd.DataFrame,
    epsilon: float,
) -> pd.DataFrame:
    """Fit one mean/std per IZ from unique training-input dates (ddof=0).

    Per-IZ scaling learns deviations from each IZ's own training-period level.
    Mapped forecasts must be inverse-transformed to original rate units.
    A global scaler would be a separate modelling decision.
    """
    date_pos = {ts.date(): i for i, ts in enumerate(calendar)}
    train_dates = sorted({day for sample in assigned["train"] for day in sample.input_dates})
    if not train_dates:
        raise ForecastError("Training split has no input dates for scaler fitting.")
    rows = [date_pos[day] for day in train_dates]
    block = rate_mat[rows]
    if not np.isfinite(block).all():
        raise ForecastError("Training-input scaler block contains non-finite rates.")
    mean = block.mean(axis=0)
    std = block.std(axis=0, ddof=SCALE_DDOF)
    scale = np.where(std < epsilon, 1.0, std)
    table = nodes.copy()
    table["train_mean"] = mean
    table["train_std"] = std
    table["effective_scale"] = scale
    table["n_unique_training_input_observations"] = len(train_dates)
    table["first_training_input_date"] = train_dates[0].isoformat()
    table["last_training_input_date"] = train_dates[-1].isoformat()
    table["epsilon"] = epsilon
    table["ddof"] = SCALE_DDOF
    table["fitting_policy"] = FITTING_POLICY
    table["scaling_mode"] = SCALING_MODE
    table["zero_or_near_zero_variance"] = std < epsilon
    return table


def _load_static_simd(simd_path: Path, nodes: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Load SIMD 2020 IZ indicators aligned to node_order. Do not invent values.

    SIMD is a static snapshot, not a daily series. Missing IZs or non-finite
    modelling columns fail; they are not filled with zero.
    """
    if not simd_path.exists():
        raise ForecastError(f"SIMD table missing: {simd_path}")
    frame = read_table(simd_path)
    if NODE_KEY not in frame.columns:
        raise ForecastError("SIMD table must contain IntZone.")
    missing_cols = [column for column in STATIC_FEATURE_COLUMNS if column not in frame.columns]
    if missing_cols:
        raise ForecastError(f"SIMD table missing columns {missing_cols}.")

    out = frame.copy()
    out[NODE_KEY] = out[NODE_KEY].astype("string").str.strip()
    if out[NODE_KEY].eq("").any() or out[NODE_KEY].isna().any():
        raise ForecastError("SIMD table contains empty IntZone codes.")
    if out[NODE_KEY].duplicated().any():
        raise ForecastError("SIMD table has duplicate IntZone codes.")

    simd_zones = set(out[NODE_KEY].tolist())
    master_zones = set(nodes[NODE_KEY].tolist())
    unknown = sorted(simd_zones - master_zones)
    omitted = sorted(master_zones - simd_zones)
    if unknown:
        raise ForecastError(f"{len(unknown)} unknown SIMD IntZone codes: {unknown[:10]}.")
    if omitted:
        raise ForecastError(f"{len(omitted)} expected IntZone codes have no SIMD row: {omitted[:10]}.")

    aligned = nodes.merge(out[[NODE_KEY, *STATIC_FEATURE_COLUMNS]], on=NODE_KEY, how="left", validate="one_to_one")
    for column in STATIC_FEATURE_COLUMNS:
        aligned[column] = _parse_static_feature(aligned[column], column)
    matrix = aligned[list(STATIC_FEATURE_COLUMNS)].to_numpy(dtype="float64")
    if not np.isfinite(matrix).all():
        raise ForecastError("SIMD modelling columns contain non-finite values.")
    table = aligned[[NODE_KEY, "node_index", *STATIC_FEATURE_COLUMNS]].copy()
    table["static_scaling_mode"] = STATIC_SCALING_MODE
    return table, matrix


def _parse_static_feature(series: pd.Series, name: str) -> pd.Series:
    text = series.astype("string").str.strip()
    empty = text.isna() | text.eq("") | text.str.lower().isin(["nan", "none", "<na>", "null"])
    parsed = pd.to_numeric(text.where(~empty, pd.NA), errors="coerce")
    invalid = ~empty & parsed.isna()
    if bool(invalid.any()):
        samples = text.loc[invalid].head(5).tolist()
        raise ForecastError(f"{name} contains non-numeric values: {samples}.")
    if bool(empty.any()):
        raise ForecastError(f"{name} contains missing values; they are not invented.")
    if bool(np.isinf(parsed.fillna(0.0)).any()):
        raise ForecastError(f"{name} contains infinite values.")
    return parsed.astype("float64")


def _build_split_arrays(
    assigned: dict[str, list[WindowRecord]],
    calendar: pd.DatetimeIndex,
    rate_mat: np.ndarray,
    scaler: pd.DataFrame,
    window: WindowConfig,
    static_mat: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    date_pos = {ts.date(): i for i, ts in enumerate(calendar)}
    mean = scaler["train_mean"].to_numpy(dtype="float64")
    scale = scaler["effective_scale"].to_numpy(dtype="float64")
    n_iz = rate_mat.shape[1]
    out: dict[str, dict[str, np.ndarray]] = {}
    for label, samples in assigned.items():
        n = len(samples)
        x_raw = np.empty((n, window.lookback_days, n_iz, 1), dtype="float64")
        y_raw = np.empty((n, n_iz, 1), dtype="float64")
        sample_id = np.empty(n, dtype="int64")
        origin_dates = np.empty(n, dtype="U10")
        target_dates = np.empty(n, dtype="U10")
        for i, sample in enumerate(samples):
            input_idx = [date_pos[day] for day in sample.input_dates]
            x_raw[i, :, :, 0] = rate_mat[input_idx]
            y_raw[i, :, 0] = rate_mat[date_pos[sample.target]]
            sample_id[i] = sample.sample_id
            origin_dates[i] = sample.origin.isoformat()
            target_dates[i] = sample.target.isoformat()
        x_scaled = (x_raw - mean.reshape(1, 1, n_iz, 1)) / scale.reshape(1, 1, n_iz, 1)
        y_scaled = (y_raw - mean.reshape(1, n_iz, 1)) / scale.reshape(1, n_iz, 1)
        if static_mat.shape != (n_iz, len(STATIC_FEATURE_COLUMNS)):
            raise ForecastError(
                f"X_static_raw shape is {static_mat.shape}, expected {(n_iz, len(STATIC_FEATURE_COLUMNS))}."
            )
        out[label] = {
            "X_dynamic_raw": x_raw,
            "X_dynamic_scaled": x_scaled,
            "y_target_raw": y_raw,
            "y_target_scaled": y_scaled,
            "X_static_raw": np.array(static_mat, copy=True),
            "X_static_feature_names": np.array(STATIC_FEATURE_COLUMNS, dtype="U32"),
            "sample_id": sample_id,
            "forecast_origin_date": origin_dates,
            "target_date": target_dates,
        }
    return out


def _write_outputs(
    output_dir: Path,
    covid_path: Path,
    simd_path: Path,
    nodes: pd.DataFrame,
    node_order: dict[str, Any],
    quality: dict[str, Any],
    valid: list[WindowRecord],
    excluded: list[dict[str, Any]],
    assigned: dict[str, list[WindowRecord]],
    scaler: pd.DataFrame,
    static_table: pd.DataFrame,
    arrays: dict[str, dict[str, np.ndarray]],
    start: date,
    end: date,
    split_boundaries: dict[str, tuple[str | date, str | date]] | None,
    epsilon: float,
    iz_master_source: str,
    overwrite: bool,
    window: WindowConfig,
    horizon_notes: dict[str, Any],
) -> dict[str, str]:
    reference = node_order["reference"]
    split_of = {sample.sample_id: label for label, rows in assigned.items() for sample in rows}
    n_nodes = int(len(nodes))
    valid_rows = [
        _sample_row(sample, split_of[sample.sample_id], n_nodes, reference, window)
        for sample in valid
    ]
    split_rows = [
        {
            "sample_id": sample.sample_id,
            "input_start_date": sample.input_dates[0].isoformat(),
            "input_end_date": sample.input_dates[-1].isoformat(),
            "forecast_origin_date": sample.origin.isoformat(),
            "target_date": sample.target.isoformat(),
            "lookback_days": window.lookback_days,
            "forecast_horizon_days": window.forecast_horizon_days,
            "window_stride_days": window.window_stride_days,
            "config_id": window.config_id,
            "split": label,
        }
        for label, rows in assigned.items()
        for sample in rows
    ]
    node_path = write_table(nodes, output_dir / "node_order.csv")
    valid_path = write_table(pd.DataFrame(valid_rows), output_dir / "valid_samples.csv")
    excluded_path = write_table(
        pd.DataFrame(excluded, columns=EXCLUDED_COLUMNS),
        output_dir / "excluded_samples.csv",
    )
    split_path = write_table(pd.DataFrame(split_rows), output_dir / "split_manifest.csv")
    scaler_path = write_table(scaler, output_dir / "scaler.csv")
    static_path = write_table(static_table, output_dir / "static_features.csv")
    quality_path = write_json(quality, output_dir / "panel_quality.json")
    split_summary = _split_summary(assigned, excluded, nodes, node_order, window)
    split_summary_path = write_json(split_summary, output_dir / "split_summary.json")

    array_paths: dict[str, str] = {}
    integrity: dict[str, Any] = {
        "config_id": window.config_id,
        "lookback_days": window.lookback_days,
        "forecast_horizon_days": window.forecast_horizon_days,
        "window_stride_days": window.window_stride_days,
        "scaling_mode": SCALING_MODE,
        "static_scaling_mode": STATIC_SCALING_MODE,
        "static_feature_names": list(STATIC_FEATURE_COLUMNS),
        "node_order_reference": reference,
        "node_order_sha256": node_order["node_order_sha256"],
        "node_index_sequence": nodes["node_index"].tolist(),
        "iz_codes": nodes[NODE_KEY].tolist(),
    }
    for label, payload in arrays.items():
        path = output_dir / f"{label}.npz"
        np.savez_compressed(path, **payload)
        array_paths[label] = str(path)
        rate_names = ("X_dynamic_raw", "X_dynamic_scaled", "y_target_raw", "y_target_scaled", "X_static_raw")
        integrity[label] = {name: list(payload[name].shape) for name in payload}
        integrity[f"{label}_contains_nan"] = {
            name: bool(np.isnan(payload[name]).any()) for name in rate_names
        }
        integrity[f"{label}_sample_id_order"] = payload["sample_id"].tolist()
    integrity_path = write_json(integrity, output_dir / "array_integrity.json")

    near_zero = scaler.loc[scaler["zero_or_near_zero_variance"], NODE_KEY].astype(str).tolist()
    metadata = {
        "input_data_path": str(covid_path),
        "input_data_mtime": datetime.fromtimestamp(covid_path.stat().st_mtime, tz=timezone.utc).isoformat(),
        "simd_path": str(simd_path),
        "processing_timestamp": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": PIPELINE_VERSION,
        "study_area": LOCAL_AUTHORITY_NAME,
        "local_authority_code": LOCAL_AUTHORITY_CODE,
        "geography_type": "2011 Intermediate Zone",
        "geography_vintage": GEOGRAPHY_VINTAGE,
        "upstream_edinburgh_filter": True,
        "area_filter_repeated": False,
        "iz_master_source": iz_master_source,
        "analysis_period": {"start": start.isoformat(), "end": end.isoformat()},
        "response_variable_definition": (
            "Daily-reported rolling seven-day infection rate per 100,000 "
            f"({MODELLING_RATE_COLUMN} = case_count_used / Population * 100000). "
            "Not a daily-new-case rate."
        ),
        "lookback_days": window.lookback_days,
        "forecast_horizon_days": window.forecast_horizon_days,
        "window_stride_days": window.window_stride_days,
        "config_id": window.config_id,
        "target_definition": (
            f"single rolling-seven-day infection_rate at forecast_origin + "
            f"{window.forecast_horizon_days} calendar days; not an "
            f"{window.forecast_horizon_days}-day cumulative rate and not "
            f"{window.forecast_horizon_days} daily forecasts"
        ),
        **{k: v for k, v in horizon_notes.items() if k not in {"lookback_days", "forecast_horizon_days", "window_stride_days", "config_id"}},
        "split_rule": (
            "explicit target-date boundaries"
            if split_boundaries is not None
            else f"chronological by target_date; floor({TRAIN_FRACTION}) / floor({VALIDATION_FRACTION}) / remainder"
        ),
        "scaler_fitting_policy": FITTING_POLICY,
        "scaling_mode": SCALING_MODE,
        "static_feature_names": list(STATIC_FEATURE_COLUMNS),
        "static_feature_definition": (
            "SIMD 2020v2 IZ indicators; one static row per IZ. "
            "crime_rate is the fill1 primary scenario. Not a daily series "
            "and not standardised by the COVID rate scaler."
        ),
        "static_scaling_mode": STATIC_SCALING_MODE,
        "scaler_ddof": SCALE_DDOF,
        "scaler_epsilon": epsilon,
        "missing_data_policy": MISSING_POLICY,
        "node_order": node_order,
        "node_order_reference": reference,
        "overwrite": overwrite,
        "zero_or_near_zero_variance_iz": near_zero,
        "warnings": [
            *list(quality.get("warnings", [])),
            horizon_notes["target_period_interpretation"],
        ],
        "output_paths": {},
    }
    paths = {
        "node_order": str(node_path),
        "panel_quality": str(quality_path),
        "valid_samples": str(valid_path),
        "excluded_samples": str(excluded_path),
        "split_manifest": str(split_path),
        "split_summary": str(split_summary_path),
        "scaler": str(scaler_path),
        "static_features": str(static_path),
        "array_integrity": str(integrity_path),
        **{f"{label}_arrays": path for label, path in array_paths.items()},
    }
    metadata["output_paths"] = paths
    metadata_path = write_json(metadata, output_dir / "run_metadata.json")
    paths["run_metadata"] = str(metadata_path)
    metadata["output_paths"] = paths
    write_json(metadata, metadata_path)
    return paths


def _sample_row(
    sample: WindowRecord,
    split: str,
    n_nodes: int,
    node_order_reference: str,
    window: WindowConfig,
) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "input_start_date": sample.input_dates[0].isoformat(),
        "input_end_date": sample.input_dates[-1].isoformat(),
        "input_dates": "|".join(day.isoformat() for day in sample.input_dates),
        "forecast_origin_date": sample.origin.isoformat(),
        "target_date": sample.target.isoformat(),
        "lookback_days": window.lookback_days,
        "forecast_horizon_days": window.forecast_horizon_days,
        "window_stride_days": window.window_stride_days,
        "number_of_nodes": n_nodes,
        "node_order_reference": node_order_reference,
        "config_id": window.config_id,
        "split": split,
    }


def _split_summary(
    assigned: dict[str, list[WindowRecord]],
    excluded: list[dict[str, Any]],
    nodes: pd.DataFrame,
    node_order: dict[str, Any],
    window: WindowConfig,
) -> dict[str, Any]:
    n_valid = sum(len(rows) for rows in assigned.values())
    payload: dict[str, Any] = {
        "n_valid_samples": n_valid,
        "n_excluded_samples": len(excluded),
        "lookback_days": window.lookback_days,
        "forecast_horizon_days": window.forecast_horizon_days,
        "window_stride_days": window.window_stride_days,
        "config_id": window.config_id,
        "n_iz": int(len(nodes)),
        "node_order": node_order,
        "node_order_reference": node_order["reference"],
        "splits": {},
        "actual_proportions": {},
    }
    for label, rows in assigned.items():
        payload["splits"][label] = {
            "n_samples": len(rows),
            "first_forecast_origin_date": rows[0].origin.isoformat() if rows else None,
            "last_forecast_origin_date": rows[-1].origin.isoformat() if rows else None,
            "first_target_date": rows[0].target.isoformat() if rows else None,
            "last_target_date": rows[-1].target.isoformat() if rows else None,
        }
        payload["actual_proportions"][label] = (len(rows) / n_valid) if n_valid else 0.0
    return payload


def _run_summary(
    paths: dict[str, str],
    quality: dict[str, Any],
    valid: list[WindowRecord],
    excluded: list[dict[str, Any]],
    assigned: dict[str, list[WindowRecord]],
    scaler: pd.DataFrame,
    arrays: dict[str, dict[str, np.ndarray]],
    window: WindowConfig,
    horizon_notes: dict[str, Any],
) -> dict[str, Any]:
    reasons: dict[str, int] = {}
    for row in excluded:
        reasons[row["exclusion_reason"]] = reasons.get(row["exclusion_reason"], 0) + 1
    near_zero = scaler.loc[scaler["zero_or_near_zero_variance"], NODE_KEY].astype(str).tolist()
    return {
        "status": "ok",
        "n_valid_samples": len(valid),
        "n_excluded_samples": len(excluded),
        "exclusion_reason_counts": reasons,
        "split_counts": {label: len(rows) for label, rows in assigned.items()},
        "array_shapes": {
            label: {
                name: list(arr.shape)
                for name, arr in payload.items()
                if name in {
                    "X_dynamic_raw",
                    "X_dynamic_scaled",
                    "y_target_raw",
                    "y_target_scaled",
                    "X_static_raw",
                }
            }
            for label, payload in arrays.items()
        },
        "split_target_ranges": {
            label: {
                "first": rows[0].target.isoformat(),
                "last": rows[-1].target.isoformat(),
            }
            for label, rows in assigned.items()
        },
        "scaler_training_input_range": {
            "first": str(scaler["first_training_input_date"].iloc[0]),
            "last": str(scaler["last_training_input_date"].iloc[0]),
        },
        "lookback_days": window.lookback_days,
        "forecast_horizon_days": window.forecast_horizon_days,
        "window_stride_days": window.window_stride_days,
        "config_id": window.config_id,
        "target_period_start_offset": horizon_notes["target_period_start_offset"],
        "target_period_end_offset": horizon_notes["target_period_end_offset"],
        "target_period_interpretation": horizon_notes["target_period_interpretation"],
        "target_overlaps_latest_input_period": horizon_notes["target_overlaps_latest_input_period"],
        "scaling_mode": SCALING_MODE,
        "static_feature_names": list(STATIC_FEATURE_COLUMNS),
        "static_scaling_mode": STATIC_SCALING_MODE,
        "zero_or_near_zero_variance_iz": near_zero,
        "output_paths": paths,
        "panel_quality_status": quality["validation_status"],
    }


def _coerce_date(value: str | date | datetime | pd.Timestamp) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(text[:10] if fmt == "%Y-%m-%d" and "-" in text else text, fmt).date()
        except ValueError:
            continue
    raise ForecastError(f"Cannot parse date: {value!r}.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build chronological COVID sliding-window arrays from fill1.csv."
    )
    parser.add_argument("--covid", type=Path, help="COVID panel CSV (default: data/results/fill1.csv)")
    parser.add_argument(
        "--simd",
        type=Path,
        help="IZ-level SIMD CSV (default: data/results/simd_iz.csv). Attached as X_static_raw.",
    )
    parser.add_argument("--start", help="Analysis start date YYYY-MM-DD")
    parser.add_argument("--end", help="Analysis end date YYYY-MM-DD")
    parser.add_argument(
        "--window",
        type=int,
        help="Set lookback and horizon to the same number. Use --lookback and --horizon if they differ.",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        help="Input lookback days (default 7). Overrides --window for lookback.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        help="Lead time in calendar days to the single target report (default 7). "
        "Not an H-day cumulative rate. Overrides --window for horizon.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output folder (default: data/results/forecast/L{L}_H{H}_S{S}_START_END)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing forecast output files in the chosen directory.",
    )
    args = parser.parse_args(argv)
    lookback = LOOKBACK_DAYS
    horizon = FORECAST_HORIZON_DAYS
    if args.window is not None:
        lookback = args.window
        horizon = args.window
    if args.lookback is not None:
        lookback = args.lookback
    if args.horizon is not None:
        horizon = args.horizon
    result = prepare_forecast_dataset(
        lookback_days=lookback,
        forecast_horizon_days=horizon,
        covid_path=args.covid,
        simd_path=args.simd,
        start_date=args.start,
        end_date=args.end,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    print(json.dumps({k: v for k, v in result.items() if k != "state"}, indent=2, default=str))


# ---------------------------------------------------------------------------
# Frozen S1 loader (does not rebuild windows).
# ---------------------------------------------------------------------------

SPLIT_FILES = {
    "train": "train.npz",
    "validation": "validation.npz",
    "test": "test.npz",
}
REQUIRED_ARRAYS = (
    "X_dynamic_raw",
    "X_dynamic_scaled",
    "y_target_raw",
    "y_target_scaled",
    "X_static_raw",
    "sample_id",
    "forecast_origin_date",
    "target_date",
)


@dataclass
class CovidScaler:
    """Frozen per-IZ COVID scaler from scaler.csv. Never refit at inference."""

    codes: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray

    def as_dict(self) -> dict[str, Any]:
        return {
            "iz_codes": list(self.codes),
            "train_mean": self.mean.tolist(),
            "effective_scale": self.scale.tolist(),
        }


@dataclass
class SplitArrays:
    name: str
    x_dynamic_scaled: np.ndarray
    y_target_scaled: np.ndarray
    x_dynamic_raw: np.ndarray
    y_target_raw: np.ndarray
    sample_id: np.ndarray
    forecast_origin_date: np.ndarray
    target_date: np.ndarray
    x_dynamic_model: np.ndarray | None = None
    y_anchor_raw: np.ndarray | None = None
    y_delta_raw: np.ndarray | None = None
    y_delta_scaled: np.ndarray | None = None


@dataclass
class TemporalDataset:
    """Loaded S1 tensors. N comes from node_order, not from a layer constant."""

    directory: Path
    node_order: NodeOrder
    covid_scaler: CovidScaler
    x_static_raw: np.ndarray
    static_feature_names: tuple[str, ...]
    lookback_days: int
    forecast_horizon_days: int
    config_id: str
    splits: dict[str, SplitArrays]
    metadata: dict[str, Any]
    validation_selection_index: np.ndarray
    validation_calibration_index: np.ndarray
    internal_split_provenance: dict[str, Any] = field(default_factory=dict)
    residual_scalers: Any = None

    @property
    def n_nodes(self) -> int:
        return self.node_order.n_nodes


def _as_datetime64(values: np.ndarray) -> np.ndarray:
    return pd.to_datetime(values).to_numpy()


def _load_split(path: Path, name: str, n_nodes: int, lookback: int) -> tuple[SplitArrays, np.ndarray]:
    if not path.is_file():
        raise ModelError(f"Missing split file: {path}", code="missing_dataset")
    with np.load(path, allow_pickle=True) as payload:
        missing = [key for key in REQUIRED_ARRAYS if key not in payload.files]
        if missing:
            raise ModelError(
                f"{path} is missing arrays: {missing}",
                code="missing_dataset",
            )
        x_scaled = np.asarray(payload["X_dynamic_scaled"], dtype=np.float64)
        y_scaled = np.asarray(payload["y_target_scaled"], dtype=np.float64)
        x_raw = np.asarray(payload["X_dynamic_raw"], dtype=np.float64)
        y_raw = np.asarray(payload["y_target_raw"], dtype=np.float64)
        sample_id = np.asarray(payload["sample_id"])
        origin = _as_datetime64(payload["forecast_origin_date"])
        target = _as_datetime64(payload["target_date"])
        x_static = np.asarray(payload["X_static_raw"], dtype=np.float64)

    batch = x_scaled.shape[0]
    if x_scaled.ndim != 4 or x_scaled.shape[1:] != (lookback, n_nodes, 1):
        raise ModelError(
            f"{name} X_dynamic_scaled shape is {x_scaled.shape}, "
            f"expected [B, {lookback}, {n_nodes}, 1].",
            code="invalid_tensor_shape",
        )
    if y_scaled.shape != (batch, n_nodes, 1):
        raise ModelError(
            f"{name} y_target_scaled shape is {y_scaled.shape}, expected [{batch}, {n_nodes}, 1]. "
            "This model forecasts one target date, not an H-step sequence.",
            code="invalid_tensor_shape",
        )
    if x_raw.shape != x_scaled.shape or y_raw.shape != y_scaled.shape:
        raise ModelError(f"{name} raw/scaled tensor shapes do not match.", code="invalid_tensor_shape")
    if x_static.shape != (n_nodes, len(STATIC_FEATURE_COLUMNS)):
        raise ModelError(
            f"{name} X_static_raw shape is {x_static.shape}, "
            f"expected [{n_nodes}, {len(STATIC_FEATURE_COLUMNS)}].",
            code="invalid_tensor_shape",
        )
    if origin.shape[0] != batch or target.shape[0] != batch:
        raise ModelError(f"{name} date vectors do not match batch size.", code="invalid_tensor_shape")
    return SplitArrays(
        name=name,
        x_dynamic_scaled=x_scaled,
        y_target_scaled=y_scaled,
        x_dynamic_raw=x_raw,
        y_target_raw=y_raw,
        sample_id=sample_id,
        forecast_origin_date=origin,
        target_date=target,
    ), x_static


def _load_covid_scaler(path: Path, node_order: NodeOrder) -> CovidScaler:
    table = pd.read_csv(path)
    table = table.sort_values("node_index").reset_index(drop=True)
    codes = tuple(table[NODE_KEY].astype(str).tolist())
    if codes != node_order.codes:
        raise ModelError(
            "scaler.csv IZ sequence does not match node_order.csv.",
            code="node_order_mismatch",
        )
    mean = table["train_mean"].to_numpy(dtype=np.float64)
    scale = table["effective_scale"].to_numpy(dtype=np.float64)
    if np.any(~np.isfinite(mean)) or np.any(~np.isfinite(scale)) or np.any(scale <= 0):
        raise ModelError("COVID scaler mean/scale must be finite and scale must be positive.", code="invalid_scaler")
    return CovidScaler(codes=codes, mean=mean, scale=scale)


def split_validation_internal(
    target_dates: np.ndarray,
    *,
    selection_fraction: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Chronological 50/50 split of the outer validation set by target_date.

    selection: early stopping / checkpoint choice.
    calibration: frozen q95 and P90 sigma only; never used to update parameters.
    """
    if not 0.0 < selection_fraction < 1.0:
        raise ModelError("selection_fraction must be in (0, 1).", code="invalid_config")
    order = np.argsort(target_dates, kind="mergesort")
    n_samples = int(len(order))
    n_selection = int(np.floor(n_samples * selection_fraction))
    selection = order[:n_selection]
    calibration = order[n_selection:]
    provenance = {
        "order_by": "target_date",
        "selection_fraction": float(selection_fraction),
        "n_validation": n_samples,
        "n_validation_selection": int(len(selection)),
        "n_validation_calibration": int(len(calibration)),
        "selection_target_date_start": None
        if selection.size == 0
        else str(np.datetime_as_string(target_dates[selection[0]], unit="D")),
        "selection_target_date_end": None
        if selection.size == 0
        else str(np.datetime_as_string(target_dates[selection[-1]], unit="D")),
        "calibration_target_date_start": None
        if calibration.size == 0
        else str(np.datetime_as_string(target_dates[calibration[0]], unit="D")),
        "calibration_target_date_end": None
        if calibration.size == 0
        else str(np.datetime_as_string(target_dates[calibration[-1]], unit="D")),
    }
    return selection, calibration, provenance


def load_temporal_dataset(
    directory: str | Path,
    *,
    lookback_days: int = 7,
    forecast_horizon_days: int = 7,
    expected_iz_count: int | None = 111,
    selection_fraction: float = 0.5,
) -> TemporalDataset:
    """Load L7_H7_S1. Do not call forecast.prepare_forecast_dataset."""
    directory = Path(directory)
    if not directory.is_dir():
        raise ModelError(f"Dataset directory does not exist: {directory}", code="missing_dataset")

    node_order = load_node_order(directory / "node_order.csv")
    if expected_iz_count is not None:
        assert_edinburgh_count(node_order.n_nodes, expected=expected_iz_count)

    metadata_path = directory / "run_metadata.json"
    metadata: dict[str, Any] = {}
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        stored_l = int(metadata.get("lookback_days", lookback_days))
        stored_h = int(metadata.get("forecast_horizon_days", forecast_horizon_days))
        if stored_l != lookback_days or stored_h != forecast_horizon_days:
            raise ModelError(
                f"Dataset L/H is L{stored_l}_H{stored_h}, expected L{lookback_days}_H{forecast_horizon_days}.",
                code="config_mismatch",
            )

    integrity_path = directory / "array_integrity.json"
    if integrity_path.is_file():
        integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
        integrity_codes = tuple(str(code) for code in integrity.get("iz_codes", []))
        if integrity_codes and integrity_codes != node_order.codes:
            raise ModelError(
                "array_integrity.json IZ sequence does not match node_order.csv.",
                code="node_order_mismatch",
            )
        # forecast.py stored a pipe-join hash; keep it as legacy only.
        stored_legacy = str(integrity.get("node_order_sha256") or "")
        if stored_legacy and stored_legacy != node_order.legacy_hash and stored_legacy != node_order.canonical_hash:
            # Compare sequences first; hash algorithm may differ.
            if integrity_codes and integrity_codes == node_order.codes:
                pass
            elif integrity_codes:
                raise ModelError(
                    "Stored node-order hash does not match the IZ sequence.",
                    code="node_order_mismatch",
                )

    covid_scaler = _load_covid_scaler(directory / "scaler.csv", node_order)
    splits: dict[str, SplitArrays] = {}
    static_from_split: np.ndarray | None = None
    for name, filename in SPLIT_FILES.items():
        split, x_static = _load_split(
            directory / filename,
            name,
            n_nodes=node_order.n_nodes,
            lookback=lookback_days,
        )
        splits[name] = split
        if static_from_split is None:
            static_from_split = x_static
        elif not np.array_equal(static_from_split, x_static):
            raise ModelError("X_static_raw differs across splits.", code="invalid_tensor_shape")

    static_path = directory / "static_features.csv"
    if static_path.is_file():
        static_table = pd.read_csv(static_path)
        static_codes = tuple(static_table.sort_values("node_index")[NODE_KEY].astype(str).tolist())
        if static_codes != node_order.codes:
            raise ModelError("static_features.csv IZ sequence does not match node_order.", code="node_order_mismatch")

    validation = splits["validation"]
    selection_index, calibration_index, internal_prov = split_validation_internal(
        validation.target_date,
        selection_fraction=selection_fraction,
    )
    return TemporalDataset(
        directory=directory,
        node_order=node_order,
        covid_scaler=covid_scaler,
        x_static_raw=static_from_split,
        static_feature_names=STATIC_FEATURE_COLUMNS,
        lookback_days=lookback_days,
        forecast_horizon_days=forecast_horizon_days,
        config_id=str(metadata.get("config_id", "L7_H7_S1")),
        splits=splits,
        metadata=metadata,
        validation_selection_index=selection_index,
        validation_calibration_index=calibration_index,
        internal_split_provenance=internal_prov,
    )


def subset_split(split: SplitArrays, index: np.ndarray) -> SplitArrays:
    """Select rows by integer index. Used for validation_selection / calibration."""
    index = np.asarray(index)
    out = SplitArrays(
        name=split.name,
        x_dynamic_scaled=split.x_dynamic_scaled[index],
        y_target_scaled=split.y_target_scaled[index],
        x_dynamic_raw=split.x_dynamic_raw[index],
        y_target_raw=split.y_target_raw[index],
        sample_id=split.sample_id[index],
        forecast_origin_date=split.forecast_origin_date[index],
        target_date=split.target_date[index],
    )
    if split.x_dynamic_model is not None:
        out.x_dynamic_model = split.x_dynamic_model[index]
        out.y_anchor_raw = split.y_anchor_raw[index]
        out.y_delta_raw = split.y_delta_raw[index]
        out.y_delta_scaled = split.y_delta_scaled[index]
    return out


def concatenate_splits(splits: list[SplitArrays], name: str) -> SplitArrays:
    """Stack existing S1 windows. Does not rebuild forecast.py arrays."""
    if not splits:
        raise ModelError("Cannot concatenate an empty split list.", code="invalid_tensor_shape")
    out = SplitArrays(
        name=name,
        x_dynamic_scaled=np.concatenate([item.x_dynamic_scaled for item in splits], axis=0),
        y_target_scaled=np.concatenate([item.y_target_scaled for item in splits], axis=0),
        x_dynamic_raw=np.concatenate([item.x_dynamic_raw for item in splits], axis=0),
        y_target_raw=np.concatenate([item.y_target_raw for item in splits], axis=0),
        sample_id=np.concatenate([item.sample_id for item in splits], axis=0),
        forecast_origin_date=np.concatenate([item.forecast_origin_date for item in splits], axis=0),
        target_date=np.concatenate([item.target_date for item in splits], axis=0),
    )
    if all(item.x_dynamic_model is not None for item in splits):
        out.x_dynamic_model = np.concatenate([item.x_dynamic_model for item in splits], axis=0)
        out.y_anchor_raw = np.concatenate([item.y_anchor_raw for item in splits], axis=0)
        out.y_delta_raw = np.concatenate([item.y_delta_raw for item in splits], axis=0)
        out.y_delta_scaled = np.concatenate([item.y_delta_scaled for item in splits], axis=0)
    return out


def sort_split_by_target_date(split: SplitArrays) -> SplitArrays:
    order = np.argsort(split.target_date, kind="mergesort")
    return subset_split(split, order)


def chronological_fraction_cuts(n_samples: int, train_frac: float, val_frac: float) -> dict[str, tuple[int, int]]:
    """floor(train) / floor(val) / remainder, same rule as forecast.py."""
    if n_samples < 3:
        raise ModelError("Need at least 3 windows to recut the outer split.", code="invalid_config")
    n_train = int(math.floor(train_frac * n_samples))
    n_val = int(math.floor(val_frac * n_samples))
    n_test = n_samples - n_train - n_val
    if min(n_train, n_val, n_test) < 1:
        raise ModelError(
            "Outer split produced an empty partition.",
            code="invalid_config",
            details={"n_train": n_train, "n_validation": n_val, "n_test": n_test},
        )
    return {
        "train": (0, n_train),
        "validation": (n_train, n_train + n_val),
        "test": (n_train + n_val, n_samples),
    }


def _date_str(values: np.ndarray, index: int) -> str:
    return str(pd.Timestamp(values[index]).normalize().date())


def write_chronological_resplit(
    dataset: TemporalDataset,
    output_dir: str | Path,
    *,
    train_frac: float = 0.60,
    val_frac: float = 0.15,
    lookback_days: int = 7,
) -> dict[str, Any]:
    """Recut existing S1 windows. Does not rebuild forecast.py samples or touch data/raw."""
    output_dir = Path(output_dir)
    source = dataset.directory.resolve()
    if output_dir.resolve() == source:
        raise ModelError("Refusing to overwrite the frozen 70/15/15 S1 directory.", code="rolling_overwrite_forbidden")
    pool = sort_split_by_target_date(
        concatenate_splits(
            [dataset.splits["train"], dataset.splits["validation"], dataset.splits["test"]],
            name="pool",
        )
    )
    cuts = chronological_fraction_cuts(pool.target_date.shape[0], train_frac, val_frac)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("node_order.csv", "static_features.csv", "panel_quality.json", "valid_samples.csv", "excluded_samples.csv"):
        src = source / name
        if src.is_file():
            shutil.copy2(src, output_dir / name)

    train_sl = pool.x_dynamic_raw[cuts["train"][0] : cuts["train"][1]]
    node_mean = np.nanmean(train_sl, axis=(0, 1))[:, 0]
    node_std = np.nanstd(train_sl, axis=(0, 1), ddof=0)[:, 0]
    epsilon = 1e-8
    node_scale = np.where(node_std >= epsilon, node_std, 1.0)
    mean = node_mean.reshape(1, 1, -1, 1)
    scale = node_scale.reshape(1, 1, -1, 1)

    split_rows = []
    split_summary_splits: dict[str, Any] = {}
    for name, (lo, hi) in cuts.items():
        part = subset_split(pool, np.arange(lo, hi))
        x_raw = part.x_dynamic_raw
        y_raw = part.y_target_raw
        payload = {
            "X_dynamic_raw": x_raw,
            "X_dynamic_scaled": (x_raw - mean) / scale,
            "y_target_raw": y_raw,
            "y_target_scaled": (y_raw - node_mean.reshape(1, -1, 1)) / node_scale.reshape(1, -1, 1),
            "X_static_raw": dataset.x_static_raw,
            "sample_id": part.sample_id,
            "forecast_origin_date": part.forecast_origin_date.astype("datetime64[D]").astype(str),
            "target_date": part.target_date.astype("datetime64[D]").astype(str),
        }
        np.savez_compressed(output_dir / f"{name}.npz", **payload)
        origins = part.forecast_origin_date
        targets = part.target_date
        split_summary_splits[name] = {
            "n_samples": int(hi - lo),
            "first_forecast_origin_date": _date_str(origins, 0),
            "last_forecast_origin_date": _date_str(origins, -1),
            "first_target_date": _date_str(targets, 0),
            "last_target_date": _date_str(targets, -1),
        }
        for i in range(part.target_date.shape[0]):
            origin = pd.Timestamp(origins[i]).normalize()
            split_rows.append(
                {
                    "sample_id": int(part.sample_id[i]) if np.isscalar(part.sample_id[i]) else int(part.sample_id[i]),
                    "input_start_date": str((origin - pd.Timedelta(days=lookback_days - 1)).date()),
                    "input_end_date": str(origin.date()),
                    "forecast_origin_date": str(origin.date()),
                    "target_date": _date_str(targets, i),
                    "lookback_days": lookback_days,
                    "forecast_horizon_days": 7,
                    "window_stride_days": 1,
                    "config_id": dataset.config_id,
                    "split": name,
                }
            )

    n = int(pool.target_date.shape[0])
    train_origins = pd.to_datetime(pool.forecast_origin_date[cuts["train"][0] : cuts["train"][1]])
    first_input = (train_origins.min() - pd.Timedelta(days=lookback_days - 1)).normalize()
    last_input = train_origins.max().normalize()
    unique_input_dates = pd.date_range(first_input, last_input, freq="D")
    scaler_src = pd.read_csv(source / "scaler.csv")
    scaler_src = scaler_src.sort_values("node_index").reset_index(drop=True)
    scaler_src["train_mean"] = node_mean
    scaler_src["train_std"] = node_std
    scaler_src["effective_scale"] = node_scale
    scaler_src["n_unique_training_input_observations"] = int(len(unique_input_dates))
    scaler_src["first_training_input_date"] = str(first_input.date())
    scaler_src["last_training_input_date"] = str(last_input.date())
    scaler_src["fitting_policy"] = "resplit_train_lookback_cells"
    scaler_src["zero_or_near_zero_variance"] = node_std < epsilon
    scaler_src.to_csv(output_dir / "scaler.csv", index=False)
    pd.DataFrame(split_rows).to_csv(output_dir / "split_manifest.csv", index=False)

    actual = {name: cuts[name][1] - cuts[name][0] for name in cuts}
    summary = {
        "n_valid_samples": n,
        "lookback_days": lookback_days,
        "forecast_horizon_days": 7,
        "config_id": dataset.config_id,
        "n_iz": dataset.n_nodes,
        "source_dataset": str(source),
        "split_rule": f"chronological by target_date; floor({train_frac}) / floor({val_frac}) / remainder",
        "requested_proportions": {"train": train_frac, "validation": val_frac, "test": 1.0 - train_frac - val_frac},
        "actual_proportions": {name: actual[name] / n for name in actual},
        "splits": split_summary_splits,
        "windows_not_rebuilt": True,
    }
    (output_dir / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    metadata = json.loads((source / "run_metadata.json").read_text(encoding="utf-8")) if (source / "run_metadata.json").is_file() else {}
    metadata["split_rule"] = summary["split_rule"]
    metadata["parent_dataset"] = str(source)
    metadata["windows_not_rebuilt"] = True
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    integrity = json.loads((source / "array_integrity.json").read_text(encoding="utf-8")) if (source / "array_integrity.json").is_file() else {}
    integrity["split_rule"] = summary["split_rule"]
    for name, (lo, hi) in cuts.items():
        integrity[name] = {
            "X_dynamic_raw": [int(hi - lo), lookback_days, dataset.n_nodes, 1],
            "y_target_raw": [int(hi - lo), dataset.n_nodes, 1],
        }
    (output_dir / "array_integrity.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    main()
