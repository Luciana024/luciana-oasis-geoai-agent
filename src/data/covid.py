"""Retrieve and prepare PHS neighbourhood COVID-19 extracts for Edinburgh 2011 IZs.

Retrieval: api (PHS CKAN) or local (data/raw/covid). The chosen source is never
switched silently, and raw files are never overwritten.

The response is a daily-reported rolling seven-day positive-case count/rate.
Date is the final specimen date of that window, not a daily-new-case date.

Three disclosure-control scenarios are written for counts known only to lie in
0-2: fill0 (lower bound), fill1 (midpoint, primary), and fill2 (upper bound).
Ordinary missing and invalid values stay missing. Absent Date x IntZone cells
are reported, never invented. Original source columns are kept.
"""

from __future__ import annotations

import json
import math
import re
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from common.utils import (
    ALLOWED_DATA_SOURCES,
    ALLOWED_YEARS,
    COVID_COLUMNS,
    KNOWN_QUALITY_FLAGS,
    LOCAL_AUTHORITY_CODE,
    LOCAL_AUTHORITY_NAME,
    PRIMARY_SCENARIO_FILL,
    RATE_DISCREPANCY_TOLERANCE,
    SUPPRESSION_FILL_VALUES,
    SUPPRESSION_FLAG,
    get_logger,
    load_yaml,
    project_root,
    read_table,
    write_json,
    write_table,
)

LOGGER = get_logger("covid")
PAGE_SIZE = 10000
SQL_DATE_CHUNK = 14
USER_AGENT = "oasis-geoai-agent/0.1 (OASIS 2026 Track B)"


def _results_dir():
    return project_root() / "data" / "results"

DUPLICATE_COMPARE_FIELDS = [
    "IntZoneName",
    "CA",
    "CAName",
    "Positive7Day",
    "Positive7DayQF",
    "Population",
    "CrudeRate7DayPositive",
    "CrudeRate7DayPositiveQF",
]
RESPONSE_STATUSES = (
    "observed",
    "disclosure_controlled_0_2",
    "true_missing_not_suppressed",
    "invalid_non_numeric_value",
)
SCENARIO_LABELS = {0: "fill0", 1: "fill1", 2: "fill2"}
MODELLING_RATE_COLUMN = "infection_rate"
QUANTILES = (0.25, 0.5, 0.75, 0.9, 0.95, 0.99)
CROSS_SCENARIO_IDENTITY_FIELDS = [
    "Date",
    "IntZone",
    "node_index",
    "IntZoneName",
    "CA",
    "CAName",
    "Positive7Day_original",
    "Positive7DayQF",
    "Population",
    "Population_original",
    "CrudeRate7DayPositive_original",
    "CrudeRate7DayPositiveQF",
    "response_status",
    "suppression_flag",
    "missing_flag",
    "invalid_value_flag",
    "missing_reason",
]


class AcquisitionError(RuntimeError):
    """Raised when a requested year cannot be retrieved from API or local cache."""


class CovidPreprocessError(ValueError):
    """COVID preprocessing cannot continue without guessing."""


def acquire_data(
    years: list[int],
    area_code: str = LOCAL_AUTHORITY_CODE,
    source: str = "api",
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Fetch one registered COVID extract per requested year from the chosen source."""
    years = _validate_requested_years(years)
    source = _validate_source(source)
    sources = load_yaml("configs/data.yaml")
    selected_dir = Path(output_dir) if output_dir is not None else _results_dir()
    selected_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    provenance: list[dict[str, Any]] = []
    warnings: list[str] = []

    for year in years:
        spec = sources["covid_iz_resources"][year]
        record: dict[str, Any] = {
            "year": year,
            "area_code": area_code,
            "chosen_source": source,
            "resource_id": spec["resource_id"],
            "resource_name": spec["resource_name"],
            "licence": sources["ckan"]["licence"],
            "source_url": sources["ckan"]["dataset_url"],
            "query_date": datetime.now(timezone.utc).date().isoformat(),
        }
        if source == "api":
            try:
                frame = _fetch_year_from_api(
                    resource_id=spec["resource_id"],
                    year=year,
                    area_code=area_code,
                    base_url=sources["ckan"]["base_url"],
                )
            except Exception as exc:  # noqa: BLE001 - fail closed, no silent local switch
                raise AcquisitionError(
                    f"API retrieval failed for {year}. Local files were not used because "
                    f"source='api'. Re-run with source='local' to read data/raw/covid. "
                    f"Original error: {exc}"
                ) from exc
            record["retrieval"] = "ckan_datastore_search"
        else:
            frame = _load_year_from_local(
                filename=spec["local_filename"],
                year=year,
                area_code=area_code,
            )
            record["retrieval"] = "local_raw"
            record["local_path"] = str(project_root() / "data" / "raw" / "covid" / spec["local_filename"])
        frame = _normalise_covid_frame(frame, year=year, area_code=area_code)
        source_name = spec["local_filename"] if source == "local" else spec["resource_id"]
        frame["source_file"] = source_name
        frame.attrs["source_file"] = source_name
        out_path = selected_dir / f"{year}.csv"
        write_table(frame, out_path)
        record["n_rows"] = int(len(frame))
        record["n_iz"] = int(frame["IntZone"].nunique()) if len(frame) else 0
        record["date_min"] = str(frame["Date"].min()) if len(frame) else None
        record["date_max"] = str(frame["Date"].max()) if len(frame) else None
        record["output_path"] = str(out_path)
        frames.append(frame)
        provenance.append(record)
        LOGGER.info("Acquired %s rows for year %s via %s", len(frame), year, record["retrieval"])

    provenance_path = record_provenance(provenance, output_dir=selected_dir)
    return {
        "frames": frames,
        "years": years,
        "area_code": area_code,
        "provenance": provenance,
        "provenance_path": provenance_path,
        "warnings": warnings,
    }


def record_provenance(records: list[dict[str, Any]], output_dir: Path | None = None) -> str:
    """Write source URL, licence, query date and retrieval mode next to the results."""
    payload = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "records": records,
    }
    folder = Path(output_dir) if output_dir is not None else _results_dir()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "sources.json"
    write_json(payload, path)
    return str(path)


def inventory_raw_datasets() -> dict[str, Any]:
    """List registered raw files without changing them or filling empty folders."""
    raw_root = project_root() / "data" / "raw"
    folders = [
        "covid",
        "population",
        "deprivation",
        "housing",
        "gp",
        "boundaries",
        "roads",
        "mobility",
        "candidate_sites",
        "travel_time",
    ]
    items: list[dict[str, Any]] = []
    missing_folders: list[str] = []
    for folder in folders:
        path = raw_root / folder
        if not path.exists():
            missing_folders.append(folder)
            continue
        files = [
            p
            for p in path.rglob("*")
            if p.is_file() and p.name not in {".gitkeep", ".DS_Store"}
        ]
        items.append(
            {
                "folder": folder,
                "path": str(path),
                "n_files": len(files),
                "files": [
                    {
                        "name": str(p.relative_to(raw_root)),
                        "bytes": p.stat().st_size,
                        "suffix": p.suffix.lower(),
                    }
                    for p in sorted(files)
                ],
                "status": "present" if files else "empty",
            }
        )
    payload = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "raw_root": str(raw_root),
        "datasets": items,
        "missing_folders": missing_folders,
        "note": "Empty folders are reported, not filled.",
    }
    out = _results_dir() / "inventory.json"
    write_json(payload, out)
    return {**payload, "output_path": str(out)}


def _validate_requested_years(years: list[int]) -> list[int]:
    if not years:
        raise AcquisitionError("No year provided. The agent must obtain the year from the user.")
    cleaned = sorted({int(year) for year in years})
    invalid = [year for year in cleaned if year not in ALLOWED_YEARS]
    if invalid:
        raise AcquisitionError(
            f"Unsupported year(s) {invalid}. Approved neighbourhood extracts exist for {list(ALLOWED_YEARS)}."
        )
    return cleaned


def _validate_source(source: str) -> str:
    chosen = str(source or "").strip().lower()
    if chosen not in ALLOWED_DATA_SOURCES:
        raise AcquisitionError(
            f"Unsupported data source {source!r}. Choose one of {list(ALLOWED_DATA_SOURCES)}."
        )
    return chosen


def _fetch_year_from_api(
    resource_id: str,
    year: int,
    area_code: str,
    base_url: str,
) -> pd.DataFrame:
    """Pull one council area from CKAN by date chunks.

    datastore_search with offset stops around 32k rows, so a filtered
    neighbourhood table is truncated. SQL date windows stay under that cap.
    """
    dates = _ckan_distinct_dates(base_url, resource_id, area_code)
    year_dates = [value for value in dates if str(value).startswith(str(year))]
    if not year_dates:
        raise AcquisitionError(f"CKAN returned no dates for {year} / {area_code}")
    rows: list[dict[str, Any]] = []
    for start in range(0, len(year_dates), SQL_DATE_CHUNK):
        chunk = year_dates[start : start + SQL_DATE_CHUNK]
        rows.extend(
            _ckan_sql_records(
                base_url,
                (
                    f'SELECT * FROM {_sql_ident(resource_id)} '
                    f'WHERE "CA" = {_sql_literal(area_code)} '
                    f'AND "Date" >= {_sql_literal(chunk[0])} '
                    f'AND "Date" <= {_sql_literal(chunk[-1])}'
                ),
            )
        )
    if not rows:
        raise AcquisitionError(f"CKAN returned no rows for {year} / {area_code}")
    frame = pd.DataFrame(rows)
    return _filter_year(frame, year)


def _sql_ident(resource_id: str) -> str:
    text = str(resource_id).strip()
    if not re.fullmatch(r"[0-9a-fA-F-]{36}", text):
        raise AcquisitionError(f"Unexpected CKAN resource id {resource_id!r}")
    return f'"{text}"'


def _sql_literal(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _ckan_sql_records(base_url: str, sql: str) -> list[dict[str, Any]]:
    payload = _ckan_get(f"{base_url}/datastore_search_sql", {"sql": sql})
    if not payload.get("success"):
        raise AcquisitionError("CKAN datastore_search_sql unsuccessful")
    return list((payload.get("result") or {}).get("records") or [])


def _ckan_distinct_dates(base_url: str, resource_id: str, area_code: str) -> list[str]:
    sql = (
        f'SELECT DISTINCT "Date" FROM {_sql_ident(resource_id)} '
        f'WHERE "CA" = {_sql_literal(area_code)} ORDER BY "Date"'
    )
    values = []
    for row in _ckan_sql_records(base_url, sql):
        raw = row.get("Date")
        if raw is None:
            continue
        values.append(str(raw).replace("-", "")[:8])
    return sorted({item for item in values if item.isdigit() and len(item) == 8})


def _ckan_get(url: str, params: dict[str, Any]) -> dict[str, Any]:
    query = urllib.parse.urlencode(params)
    request = urllib.request.Request(
        f"{url}?{query}",
        headers={"User-Agent": USER_AGENT},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def _load_year_from_local(filename: str, year: int, area_code: str) -> pd.DataFrame:
    path = project_root() / "data" / "raw" / "covid" / filename
    if not path.exists():
        raise AcquisitionError(f"Local extract missing: {path}")
    frame = read_table(path)
    return _filter_year(frame, year, area_code=area_code)


def _filter_year(frame: pd.DataFrame, year: int, area_code: str | None = None) -> pd.DataFrame:
    """Keep Edinburgh rows for one calendar year. Combined 2020-2021 files need this split."""
    out = frame.copy()
    out.columns = [str(col).lstrip("\ufeff") for col in out.columns]
    if area_code is not None and "CA" in out.columns:
        out = out.loc[out["CA"].astype(str) == area_code]
    dates = _parse_report_date(out["Date"])
    out = out.loc[dates.dt.year == year].copy()
    if out.empty:
        raise AcquisitionError(f"No rows remain after filtering year {year}")
    return out


def _parse_report_date(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    return pd.to_datetime(text, format="%Y%m%d", errors="coerce")


def _normalise_covid_frame(frame: pd.DataFrame, year: int, area_code: str) -> pd.DataFrame:
    """Harmonise API and local schemas; drop empty rows and non-IZ keys."""
    out = frame.copy()
    out.columns = [str(col).lstrip("\ufeff") for col in out.columns]
    for column in COVID_COLUMNS:
        if column not in out.columns:
            out[column] = ""
    parsed = _parse_report_date(out["Date"])
    mask = parsed.notna() & out["IntZone"].astype(str).str.startswith("S020")
    if "CA" in out.columns:
        mask &= out["CA"].astype(str) == area_code
    out = out.loc[mask, COVID_COLUMNS].copy()
    out["Date"] = parsed.loc[out.index].dt.strftime("%Y%m%d")
    out["year"] = year
    return out.reset_index(drop=True)


def load_iz_master(area_code: str = LOCAL_AUTHORITY_CODE) -> pd.DataFrame:
    """Build the authoritative Edinburgh 2011 IZ table from the official lookup."""
    path = project_root() / "data" / "raw" / "boundaries" / "Code lookup.csv"
    if not path.exists():
        raise CovidPreprocessError(f"IZ lookup missing: {path}")
    lookup = read_table(path)
    if "IntZone" not in lookup.columns or "CA" not in lookup.columns:
        raise CovidPreprocessError("IZ lookup must contain IntZone and CA.")
    edin = lookup.loc[lookup["CA"].astype("string").str.strip() == area_code].copy()
    edin["IntZone"] = edin["IntZone"].astype("string").str.strip()
    edin = edin.loc[edin["IntZone"].ne(""), ["IntZone"]].drop_duplicates()
    edin = edin.sort_values("IntZone").reset_index(drop=True)
    edin["node_index"] = range(len(edin))
    if edin.empty:
        raise CovidPreprocessError(f"No Intermediate Zones found for CA={area_code}.")
    return edin


CENTROID_RELATIVE_PATH = (
    Path("data")
    / "raw"
    / "boundaries"
    / "SG_IntermediateZoneCent_2011"
    / "SG_IntermediateZone_Cent_2011.dbf"
)


def attach_iz_centroids(
    covid_path: Path | None = None,
    area_code: str = LOCAL_AUTHORITY_CODE,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Join 2011 IZ population-weighted centroids onto fill1 by IntZone.

    Centroids are British National Grid metres (EPSG:27700), not lat/lon, and
    not geometric polygon centres. Missing IZs are reported, never invented.
    fill1.csv is not overwritten.
    """
    covid_path = Path(covid_path) if covid_path is not None else _results_dir() / "fill1.csv"
    if not covid_path.exists():
        raise CovidPreprocessError(f"COVID table missing: {covid_path}")

    iz_master = load_iz_master(area_code=area_code)
    centroids = _load_edinburgh_centroids(iz_master)
    covid = read_table(covid_path)
    if "IntZone" not in covid.columns:
        raise CovidPreprocessError("COVID table must contain IntZone.")
    covid["IntZone"] = covid["IntZone"].astype("string").str.strip()

    geo_cols = ["IntZone", "Easting", "Northing"]
    out = covid.merge(centroids[geo_cols], on="IntZone", how="left", validate="many_to_one")
    unmatched = sorted(out.loc[out["Easting"].isna(), "IntZone"].dropna().unique().tolist())
    if unmatched:
        raise CovidPreprocessError(
            f"{len(unmatched)} COVID IntZone codes have no 2011 centroid: {unmatched[:10]}."
        )

    out_path = Path(output_path) if output_path is not None else _results_dir() / "fill1_geo.csv"
    write_table(out, out_path)
    LOGGER.info("Wrote %s rows with Easting/Northing to %s", len(out), out_path)
    return {
        "input_path": str(covid_path),
        "output_path": str(out_path),
        "n_rows": int(len(out)),
        "n_iz": int(out["IntZone"].nunique()),
        "crs": "EPSG:27700",
        "centroid_kind": "population_weighted_2011_intermediate_zone",
    }


def _load_edinburgh_centroids(iz_master: pd.DataFrame) -> pd.DataFrame:
    """Read the official 2011 IZ centroid dbf and keep Edinburgh nodes only."""
    path = project_root() / CENTROID_RELATIVE_PATH
    if not path.exists():
        raise CovidPreprocessError(f"IZ centroid table missing: {path}")
    raw = _read_dbf(path)
    if "InterZone" not in raw.columns:
        raise CovidPreprocessError("Centroid table must contain InterZone.")
    cents = raw.rename(columns={"InterZone": "IntZone"}).copy()
    cents["IntZone"] = cents["IntZone"].astype("string").str.strip()
    for column in ("Easting", "Northing"):
        if column not in cents.columns:
            raise CovidPreprocessError(f"Centroid table must contain {column}.")
        cents[column] = pd.to_numeric(cents[column], errors="coerce")
    if cents["IntZone"].duplicated().any():
        raise CovidPreprocessError("Centroid table has duplicate InterZone codes.")

    master = iz_master[["IntZone"]].copy()
    master["IntZone"] = master["IntZone"].astype("string").str.strip()
    edin = master.merge(cents, on="IntZone", how="left", validate="one_to_one")
    missing = edin.loc[edin["Easting"].isna() | edin["Northing"].isna(), "IntZone"].tolist()
    if missing:
        raise CovidPreprocessError(
            f"{len(missing)} IZs are absent from the 2011 centroid file: {missing[:10]}."
        )
    if len(edin) != len(master):
        raise CovidPreprocessError("Centroid join changed the IZ count.")
    LOGGER.info("Matched %s IZ centroids (EPSG:27700).", len(edin))
    return edin


def _read_dbf(path: Path) -> pd.DataFrame:
    """Read a DBF attribute table without geopandas. Raw shapefiles are not modified."""
    payload = path.read_bytes()
    header_len = int.from_bytes(payload[8:10], "little")
    record_len = int.from_bytes(payload[10:12], "little")
    encoding = "utf-8"
    cpg = path.with_suffix(".cpg")
    if cpg.exists():
        encoding = cpg.read_text(encoding="ascii").strip() or encoding
    fields: list[tuple[str, str, int]] = []
    offset = 32
    while offset < header_len and payload[offset] != 0x0D:
        name = payload[offset : offset + 11].split(b"\x00", 1)[0].decode("ascii")
        kind = chr(payload[offset + 11])
        length = payload[offset + 16]
        fields.append((name, kind, length))
        offset += 32
    rows: list[dict[str, str]] = []
    cursor = header_len
    while cursor + record_len <= len(payload):
        record = payload[cursor : cursor + record_len]
        cursor += record_len
        if not record or record[0:1] == b"\x1a":
            break
        if record[0:1] == b"*":
            continue
        pos = 1
        row: dict[str, str] = {}
        for name, kind, length in fields:
            token = record[pos : pos + length].decode(encoding, errors="replace").strip()
            pos += length
            row[name] = token
        rows.append(row)
    if not rows:
        raise CovidPreprocessError(f"Centroid dbf is empty: {path}")
    return pd.DataFrame(rows)


def preprocess_covid(
    frames: list[pd.DataFrame],
    years: list[int],
    iz_master: pd.DataFrame,
    area_code: str = LOCAL_AUTHORITY_CODE,
    area_name: str = LOCAL_AUTHORITY_NAME,
    suppression_fill_values: tuple[int, ...] = SUPPRESSION_FILL_VALUES,
    primary_scenario_fill: int = PRIMARY_SCENARIO_FILL,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Merge, validate and write fill0/fill1/fill2 COVID scenario tables.

    fill1 is the primary modelling scenario. fill0 and fill2 are sensitivity
    bounds for disclosure-controlled 0-2 counts only.
    """
    fills = _validate_arguments(
        frames=frames,
        years=years,
        iz_master=iz_master,
        suppression_fill_values=suppression_fill_values,
        primary_scenario_fill=primary_scenario_fill,
    )
    warnings: list[str] = []
    tagged = [_attach_source_metadata(frame, index, years) for index, frame in enumerate(frames)]
    _require_columns(tagged)
    combined = pd.concat(tagged, ignore_index=True, sort=False)
    n_before_area = int(len(combined))
    area_before = _area_uniques(combined)

    combined = _parse_dates(combined)
    year_report = _validate_years(combined, years, warnings)
    combined, area_report = _validate_and_filter_area(
        combined, area_code=area_code, area_name=area_name, n_before=n_before_area, before=area_before
    )
    iz_report = _join_iz_master(combined, iz_master, warnings)
    combined = iz_report.pop("frame")
    combined, duplicate_report = _resolve_duplicates(combined, years)
    qf_report = _quality_flag_frequencies(combined)
    _reject_unknown_quality_flags(combined)
    combined = _classify_response(combined)
    _validate_observed_counts(combined)
    combined, population_report = _validate_population(combined)
    combined = combined.sort_values(["Date", "node_index"]).reset_index(drop=True)
    gap_report = _calendar_gaps(combined, iz_master)
    if gap_report["n_missing_iz_date_cells"]:
        warnings.append(
            f"{gap_report['n_missing_iz_date_cells']} Date x IntZone cells are absent and were not invented."
        )
    rate_report = _published_rate_discrepancy(combined)

    scenarios = _build_scenarios(
        combined,
        fills=fills,
        primary_fill=primary_scenario_fill,
    )
    _assert_cross_scenario_consistency(scenarios, fills=fills)
    scenario_stats = {
        label: _scenario_statistics(frame) for label, frame in scenarios.items()
    }
    suppression_stats = _suppression_diagnostics(combined, scenarios)

    output_paths = _write_scenario_files(scenarios, output_dir=output_dir)
    diagnostic_paths = _write_diagnostics(combined, gap_report, output_dir=output_dir)
    primary_label = SCENARIO_LABELS[primary_scenario_fill]
    sensitivity = [SCENARIO_LABELS[fill] for fill in fills if fill != primary_scenario_fill]
    report = {
        "input_years": list(years),
        "actual_years": year_report["actual_years"],
        "row_counts_by_actual_year": year_report["row_counts_by_actual_year"],
        "area_code": area_code,
        "area_name": area_name,
        "source_row_counts": [int(len(frame)) for frame in tagged],
        "n_rows_before_area_filter": area_report["n_before"],
        "n_rows_after_area_filter": area_report["n_after"],
        "ca_values_before": area_report["ca_before"],
        "caname_values_before": area_report["caname_before"],
        "ca_values_after": area_report["ca_after"],
        "caname_values_after": area_report["caname_after"],
        "n_iz_after_filter": area_report["n_iz"],
        "date_min": None if combined.empty else str(combined["Date"].min().date()),
        "date_max": None if combined.empty else str(combined["Date"].max().date()),
        "date_formats_supported": ["YYYYMMDD", "YYYY-MM-DD"],
        "n_invalid_dates": 0,
        "iz": iz_report,
        "quality_flag_frequencies": qf_report,
        "response_status_counts": combined["response_status"].value_counts(dropna=False).to_dict(),
        "population": population_report,
        "duplicates": duplicate_report,
        "calendar_gaps": gap_report,
        "published_rate_discrepancy": rate_report,
        "scenario_statistics": scenario_stats,
        "suppression_diagnostics": suppression_stats,
        "primary_scenario": primary_label,
        "sensitivity_scenarios": sensitivity,
        "output_paths": output_paths,
        "diagnostic_paths": diagnostic_paths,
        "warnings": warnings,
        "preprocessing_configuration": {
            "suppression_flag": SUPPRESSION_FLAG,
            "suppression_fill_values": list(fills),
            "primary_scenario_fill": primary_scenario_fill,
            "known_quality_flags": sorted(KNOWN_QUALITY_FLAGS),
            "rate_definition": f"{MODELLING_RATE_COLUMN} = case_count_used / Population * 100000",
            "rate_discrepancy_tolerance": RATE_DISCREPANCY_TOLERANCE,
            "model_geography": "2011 Intermediate Zones",
            "response": "rolling_seven_day_positive_case_count_and_rate",
        },
    }
    report_folder = Path(output_dir) if output_dir is not None else _results_dir()
    report_folder.mkdir(parents=True, exist_ok=True)
    report_path = report_folder / "report.json"
    write_json(report, report_path)
    LOGGER.info("Wrote primary scenario %s to %s", primary_label, output_paths[primary_label])
    return {
        "base_frame": combined,
        "scenario_frames": scenarios,
        "primary_scenario": primary_label,
        "sensitivity_scenarios": sensitivity,
        "output_paths": output_paths,
        "report_path": str(report_path),
        "report": report,
        "warnings": warnings,
    }


def _validate_arguments(
    frames: list[pd.DataFrame],
    years: list[int],
    iz_master: pd.DataFrame,
    suppression_fill_values: tuple[int, ...],
    primary_scenario_fill: int,
) -> tuple[int, ...]:
    if not frames:
        raise CovidPreprocessError("frames is empty.")
    if not years:
        raise CovidPreprocessError("years is empty.")
    fills = tuple(int(value) for value in suppression_fill_values)
    if len(set(fills)) != len(fills):
        raise CovidPreprocessError("suppression_fill_values must be unique.")
    if any(value not in {0, 1, 2} for value in fills):
        raise CovidPreprocessError("suppression_fill_values may contain only 0, 1 and 2.")
    if primary_scenario_fill not in fills:
        raise CovidPreprocessError("primary_scenario_fill must be in suppression_fill_values.")
    if "IntZone" not in iz_master.columns or "node_index" not in iz_master.columns:
        raise CovidPreprocessError("iz_master must contain IntZone and node_index.")
    zones = iz_master["IntZone"].astype("string").str.strip()
    if zones.duplicated().any() or iz_master["node_index"].duplicated().any():
        raise CovidPreprocessError("IntZone and node_index must be unique in iz_master.")
    return fills


def _attach_source_metadata(frame: pd.DataFrame, index: int, years: list[int]) -> pd.DataFrame:
    out = frame.copy()
    out["source_frame_index"] = index
    out["source_year_argument"] = years[index] if index < len(years) else pd.NA
    if "source_file" not in out.columns:
        source_file = out.attrs.get("source_file")
        if source_file:
            out["source_file"] = str(source_file)
    return out


def _require_columns(frames: list[pd.DataFrame]) -> None:
    for index, frame in enumerate(frames):
        missing = [column for column in COVID_COLUMNS if column not in frame.columns]
        if missing:
            raise CovidPreprocessError(
                f"Input frame {index} is missing required columns: {missing}."
            )


def _parse_dates(frame: pd.DataFrame) -> pd.DataFrame:
    """Parse only the documented PHS date formats; do not infer other layouts."""
    out = frame.copy()
    original = out["Date"].astype("string")
    out["Date_original"] = original
    text = original.str.strip().str.replace(r"\.0$", "", regex=True)
    parsed = pd.to_datetime(text, format="%Y%m%d", errors="coerce")
    iso_mask = parsed.isna() & text.ne("") & text.notna()
    parsed.loc[iso_mask] = pd.to_datetime(text.loc[iso_mask], format="%Y-%m-%d", errors="coerce")
    unparsed = parsed.isna() & text.ne("") & text.notna()
    if bool(unparsed.any()):
        samples = text.loc[unparsed].head(5).tolist()
        raise CovidPreprocessError(
            f"{int(unparsed.sum())} non-empty Date values are not YYYYMMDD or YYYY-MM-DD: {samples}."
        )
    out["Date"] = parsed
    empty_key = out["Date"].isna() | out["IntZone"].astype("string").str.strip().eq("")
    if bool(empty_key.any()):
        LOGGER.info("Dropping %s rows with empty Date or IntZone", int(empty_key.sum()))
        out = out.loc[~empty_key].copy()
    return out


def _validate_years(frame: pd.DataFrame, years: list[int], warnings: list[str]) -> dict[str, Any]:
    requested = [int(year) for year in years]
    actual = frame["Date"].dt.year.astype("int64")
    actual_years = sorted({int(year) for year in actual.unique()})
    unexpected = sorted(set(actual_years) - set(requested))
    if unexpected:
        raise CovidPreprocessError(
            f"Data contain years {unexpected} that were not requested ({requested})."
        )
    missing_requested = [year for year in requested if year not in actual_years]
    for year in missing_requested:
        warnings.append(f"Requested year {year} contains no observations.")
    counts = actual.value_counts().sort_index()
    return {
        "actual_years": actual_years,
        "row_counts_by_actual_year": {int(year): int(count) for year, count in counts.items()},
    }


def _area_uniques(frame: pd.DataFrame) -> dict[str, list[str]]:
    ca = frame["CA"].astype("string").str.strip() if "CA" in frame.columns else pd.Series(dtype="string")
    name = frame["CAName"].astype("string").str.strip() if "CAName" in frame.columns else pd.Series(dtype="string")
    return {
        "ca": sorted({value for value in ca.dropna().tolist() if value != ""}),
        "caname": sorted({value for value in name.dropna().tolist() if value != ""}),
    }


def _validate_and_filter_area(
    frame: pd.DataFrame,
    area_code: str,
    area_name: str,
    n_before: int,
    before: dict[str, list[str]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Keep rows by official CA code, never by postcode or IntZoneName."""
    if area_code == LOCAL_AUTHORITY_CODE and area_name != LOCAL_AUTHORITY_NAME:
        raise CovidPreprocessError(
            f"area_code {area_code} maps to {LOCAL_AUTHORITY_NAME!r}, not {area_name!r}."
        )
    out = frame.copy()
    out["CA"] = out["CA"].astype("string").str.strip()
    out["CAName"] = out["CAName"].astype("string").str.strip()
    matched = out.loc[out["CA"] == area_code]
    unexpected_names = sorted(
        {
            name
            for name in matched["CAName"].dropna().tolist()
            if name not in {"", area_name}
        }
    )
    if unexpected_names:
        raise CovidPreprocessError(
            f"CA={area_code} has unexpected CAName values {unexpected_names}; expected {area_name!r}."
        )
    out = matched.copy()
    after = _area_uniques(out)
    return out, {
        "n_before": n_before,
        "n_after": int(len(out)),
        "ca_before": before["ca"],
        "caname_before": before["caname"],
        "ca_after": after["ca"],
        "caname_after": after["caname"],
        "n_iz": int(out["IntZone"].astype("string").str.strip().nunique()),
    }


def _join_iz_master(
    frame: pd.DataFrame,
    iz_master: pd.DataFrame,
    warnings: list[str],
) -> dict[str, Any]:
    """Use iz_master as the node list; unexpected COVID IZs are a geography error."""
    master = iz_master.copy()
    master["IntZone"] = master["IntZone"].astype("string").str.strip()
    expected = set(master["IntZone"].tolist())
    out = frame.copy()
    out["IntZone"] = out["IntZone"].astype("string").str.strip()
    observed = set(out["IntZone"].tolist())
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    if unexpected:
        raise CovidPreprocessError(
            f"COVID data contain IntZone codes absent from iz_master: {unexpected[:8]}."
        )
    if missing:
        warnings.append(f"{len(missing)} expected IZs are absent from the COVID extract.")
    out = out.merge(master[["IntZone", "node_index"]], on="IntZone", how="left")
    if out["node_index"].isna().any():
        raise CovidPreprocessError("node_index join failed for one or more IntZone rows.")
    out["node_index"] = out["node_index"].astype("int64")
    return {
        "frame": out,
        "expected_iz_count": int(len(expected)),
        "observed_iz_count": int(len(observed)),
        "missing_expected_iz_codes": missing,
        "unexpected_iz_codes": unexpected,
    }


def _normalise_compare(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.strip()


def _resolve_duplicates(frame: pd.DataFrame, years: list[int]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Drop only exact copies. Conflicting Date+IntZone rows must not be resolved by keep=last."""
    key = ["Date", "IntZone"]
    duplicated = frame.duplicated(key, keep=False)
    n_duplicated_keys = int(frame.loc[duplicated, key].drop_duplicates().shape[0]) if duplicated.any() else 0
    if not duplicated.any():
        return frame.reset_index(drop=True), {
            "n_duplicated_keys": 0,
            "n_exact_duplicate_rows_removed": 0,
            "n_conflicting_duplicate_keys": 0,
            "conflicting_duplicates_path": None,
        }
    compare_cols = [column for column in DUPLICATE_COMPARE_FIELDS if column in frame.columns]
    work = frame.copy()
    signature = work[compare_cols].apply(_normalise_compare).agg("|".join, axis=1)
    work["_signature"] = signature
    n_signatures = work.groupby(key, dropna=False)["_signature"].nunique()
    conflicting_index = n_signatures[n_signatures > 1].index
    if len(conflicting_index) > 0:
        keyed = work.set_index(key)
        conflict_rows = keyed.loc[keyed.index.isin(conflicting_index)].reset_index()
        path = _results_dir() / "duplicates.csv"
        write_table(conflict_rows.drop(columns=["_signature"], errors="ignore"), path)
        raise CovidPreprocessError(
            f"{int(len(conflicting_index))} Date+IntZone keys have conflicting values. "
            f"Conflicting rows written to {path}."
        )
    before = len(work)
    deduped = work.drop_duplicates(key, keep="first").drop(columns=["_signature"])
    removed = before - len(deduped)
    LOGGER.info("Removed %s exact duplicate Date+IntZone rows", removed)
    return deduped.reset_index(drop=True), {
        "n_duplicated_keys": n_duplicated_keys,
        "n_exact_duplicate_rows_removed": int(removed),
        "n_conflicting_duplicate_keys": 0,
        "conflicting_duplicates_path": None,
    }


def _quality_flag_frequencies(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    report: dict[str, dict[str, int]] = {}
    for column in ("Positive7DayQF", "CrudeRate7DayPositiveQF"):
        counts = frame[column].astype("string").fillna("").str.strip().value_counts()
        report[column] = {str(key): int(value) for key, value in counts.items()}
    return report


def _reject_unknown_quality_flags(frame: pd.DataFrame) -> None:
    """Unknown QF codes must not be guessed as suppression or as ordinary missing."""
    for column in ("Positive7DayQF", "CrudeRate7DayPositiveQF"):
        flags = frame[column].astype("string").fillna("").str.strip().str.lower()
        unknown = sorted({flag for flag in flags.unique() if flag not in KNOWN_QUALITY_FLAGS})
        if unknown:
            raise CovidPreprocessError(
                f"Unknown {column} values {unknown}. Known flags are {sorted(KNOWN_QUALITY_FLAGS)}."
            )


def _count_token(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.strip()


def _classify_response(frame: pd.DataFrame) -> pd.DataFrame:
    """Separate observed, disclosure-controlled, true missing and invalid counts."""
    out = frame.copy()
    out["Positive7Day_original"] = out["Positive7Day"].astype("string")
    out["CrudeRate7DayPositive_original"] = out["CrudeRate7DayPositive"].astype("string")
    qf = out["Positive7DayQF"].astype("string").fillna("").str.strip().str.lower()
    token = _count_token(out["Positive7Day"])
    suppressed = qf.eq(SUPPRESSION_FLAG) | token.str.lower().eq(SUPPRESSION_FLAG)
    empty = token.eq("")
    # Coerce only non-empty, non-suppressed tokens so true missing stays distinct from invalid.
    candidates = ~suppressed & ~empty
    numeric = pd.to_numeric(token.where(candidates, pd.NA), errors="coerce")
    invalid = candidates & (numeric.isna() | numeric.isin([float("inf"), float("-inf")]))
    true_missing = ~suppressed & empty
    status = pd.Series("observed", index=out.index, dtype="string")
    status.loc[suppressed] = "disclosure_controlled_0_2"
    status.loc[true_missing] = "true_missing_not_suppressed"
    status.loc[invalid] = "invalid_non_numeric_value"
    out["response_status"] = status
    out["suppression_flag"] = suppressed.astype("int64")
    out["missing_flag"] = true_missing.astype("int64")
    out["invalid_value_flag"] = invalid.astype("int64")
    out["missing_reason"] = pd.Series(pd.NA, index=out.index, dtype="string")
    out.loc[suppressed, "missing_reason"] = "disclosure_controlled_0_2"
    out.loc[true_missing, "missing_reason"] = "true_missing_not_suppressed"
    out.loc[invalid, "missing_reason"] = "invalid_non_numeric_value"
    out["_numeric_count"] = numeric
    flag_sum = out["suppression_flag"] + out["missing_flag"] + out["invalid_value_flag"]
    if (flag_sum > 1).any():
        raise CovidPreprocessError("A row has more than one of suppression, missing and invalid flags.")
    if not set(out["response_status"].unique()).issubset(RESPONSE_STATUSES):
        raise CovidPreprocessError("Unrecognised response_status values.")
    return out


def _validate_observed_counts(frame: pd.DataFrame) -> None:
    observed = frame["response_status"].eq("observed")
    if not bool(observed.any()):
        return
    values = frame.loc[observed, "_numeric_count"].astype("float64")
    if bool((values < 0).any()):
        raise CovidPreprocessError("Negative observed Positive7Day counts are not allowed.")
    non_integer = values.dropna().map(lambda value: not float(value).is_integer())
    if bool(non_integer.any()):
        raise CovidPreprocessError(
            f"{int(non_integer.sum())} observed Positive7Day values are not integers; stopping for review."
        )


def _validate_population(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = frame.copy()
    out["Population_original"] = out["Population"].astype("string")
    token = out["Population"].astype("string").fillna("").str.strip()
    parsed = pd.to_numeric(token.replace("", pd.NA), errors="coerce")
    finite = parsed.notna() & parsed.map(lambda value: bool(math.isfinite(float(value))) if pd.notna(value) else False)
    invalid = token.eq("") | ~finite | (parsed <= 0)
    parsed = parsed.mask(invalid, pd.NA)
    out["Population"] = parsed.astype("Float64")
    out["invalid_population_flag"] = invalid.astype("int64")
    used = out["response_status"].isin(["observed", "disclosure_controlled_0_2"])
    if bool((used & invalid).any()):
        n_bad = int((used & invalid).sum())
        raise CovidPreprocessError(
            f"{n_bad} observed or disclosure-controlled rows have missing, non-positive or non-finite Population."
        )
    numeric = pd.to_numeric(token.replace("", pd.NA), errors="coerce")
    non_positive = int(((numeric.notna()) & (numeric <= 0)).sum())
    return out, {
        "n_missing_population": int(token.eq("").sum()),
        "n_non_positive_population": non_positive,
        "n_invalid_population": int(invalid.sum()),
        "affected_iz_sample": sorted(out.loc[invalid, "IntZone"].astype(str).unique())[:20],
        "affected_date_sample": [
            str(value.date()) for value in out.loc[invalid, "Date"].dropna().unique()[:20]
        ],
    }


def _published_rate_discrepancy(frame: pd.DataFrame) -> dict[str, Any]:
    """Compare reconstructed rates with CrudeRate7DayPositive only when that field is numeric.

    PHS neighbourhood extracts publish CrudeRate7DayPositive as disclosure bands
    such as '50 to 99', not as a numeric rate. Band midpoints are not invented.
    """
    published = pd.to_numeric(frame["CrudeRate7DayPositive_original"], errors="coerce")
    eligible = (
        frame["response_status"].eq("observed")
        & frame["invalid_population_flag"].eq(0)
        & published.notna()
        & frame["_numeric_count"].notna()
        & frame["Population"].notna()
    )
    unique_published = sorted(
        {str(value) for value in frame["CrudeRate7DayPositive_original"].fillna("").unique() if str(value) != ""}
    )
    if not bool(eligible.any()):
        return {
            "n_eligible": 0,
            "mean_absolute_discrepancy": None,
            "max_absolute_discrepancy": None,
            "n_beyond_tolerance": 0,
            "pct_beyond_tolerance": None,
            "largest_discrepancies_sample": [],
            "published_rate_kind": "disclosure_band",
            "published_rate_labels": unique_published,
            "note": (
                "CrudeRate7DayPositive is a PHS disclosure band, not a numeric rate. "
                "Numeric discrepancy comparison was skipped rather than inventing band midpoints."
            ),
        }
    reconstructed = (frame.loc[eligible, "_numeric_count"] / frame.loc[eligible, "Population"]) * 100000
    abs_diff = (reconstructed.astype("float64") - published.loc[eligible].astype("float64")).abs()
    beyond = abs_diff > RATE_DISCREPANCY_TOLERANCE
    sample = abs_diff.sort_values(ascending=False).head(10)
    return {
        "n_eligible": int(eligible.sum()),
        "mean_absolute_discrepancy": float(abs_diff.mean()),
        "max_absolute_discrepancy": float(abs_diff.max()),
        "n_beyond_tolerance": int(beyond.sum()),
        "pct_beyond_tolerance": float(100 * beyond.mean()),
        "largest_discrepancies_sample": [float(value) for value in sample.tolist()],
        "published_rate_kind": "numeric",
        "published_rate_labels": unique_published,
        "tolerance": RATE_DISCREPANCY_TOLERANCE,
    }


def _calendar_gaps(frame: pd.DataFrame, iz_master: pd.DataFrame) -> dict[str, Any]:
    """Report missing Date x IZ cells without inserting fabricated case records."""
    expected_iz = iz_master["IntZone"].astype("string").str.strip().tolist()
    dates = pd.to_datetime(frame["Date"])
    if dates.empty:
        return {
            "n_expected_iz_date_cells": 0,
            "n_observed_iz_date_cells": 0,
            "n_missing_iz_date_cells": 0,
            "n_dates_with_partial_iz_coverage": 0,
            "partial_coverage_dates_sample": [],
            "n_fully_missing_dates": 0,
            "fully_missing_dates_sample": [],
            "n_iz_with_missing_dates": 0,
            "missing_cells_by_iz_summary": {},
        }
    all_days = pd.date_range(dates.min(), dates.max(), freq="D")
    expected = pd.MultiIndex.from_product(
        [all_days, expected_iz], names=["Date", "IntZone"]
    )
    observed = pd.MultiIndex.from_arrays(
        [dates, frame["IntZone"].astype("string").str.strip()], names=["Date", "IntZone"]
    ).unique()
    missing = expected.difference(observed)
    missing_by_date: dict[str, int] = {}
    missing_by_iz: dict[str, int] = {}
    for day, zone in missing:
        day_key = pd.Timestamp(day).strftime("%Y-%m-%d")
        missing_by_date[day_key] = missing_by_date.get(day_key, 0) + 1
        missing_by_iz[str(zone)] = missing_by_iz.get(str(zone), 0) + 1
    n_iz = len(expected_iz)
    partial = sorted(day for day, count in missing_by_date.items() if 0 < count < n_iz)
    full = sorted(day for day, count in missing_by_date.items() if count == n_iz)
    expected_days = {pd.Timestamp(day).strftime("%Y-%m-%d") for day in all_days}
    observed_days = {pd.Timestamp(day).strftime("%Y-%m-%d") for day in dates}
    fully_missing_from_span = sorted(expected_days - observed_days)
    fully_missing = sorted(set(full) | set(fully_missing_from_span))
    return {
        "n_expected_iz_date_cells": int(len(expected)),
        "n_observed_iz_date_cells": int(len(observed)),
        "n_missing_iz_date_cells": int(len(missing)),
        "n_dates_with_partial_iz_coverage": int(len(partial)),
        "partial_coverage_dates_sample": partial[:20],
        "n_fully_missing_dates": int(len(fully_missing)),
        "fully_missing_dates_sample": fully_missing[:20],
        "n_iz_with_missing_dates": int(len(missing_by_iz)),
        "missing_cells_by_iz_summary": dict(sorted(missing_by_iz.items())[:111]),
    }


def _build_scenarios(
    base: pd.DataFrame,
    fills: tuple[int, ...],
    primary_fill: int,
) -> dict[str, pd.DataFrame]:
    scenarios: dict[str, pd.DataFrame] = {}
    for fill in fills:
        label = SCENARIO_LABELS[fill]
        out = base.copy()
        count = out["_numeric_count"].copy()
        suppressed = out["suppression_flag"].eq(1)
        count.loc[suppressed] = fill
        missing_or_invalid = out["missing_flag"].eq(1) | out["invalid_value_flag"].eq(1)
        count.loc[missing_or_invalid] = pd.NA
        out["suppression_scenario"] = label
        out["suppression_fill"] = fill
        out["is_primary_scenario"] = fill == primary_fill
        out["case_count_used"] = pd.to_numeric(count, errors="coerce").round(0).astype("Int64")
        valid_rate = out["case_count_used"].notna() & out["invalid_population_flag"].eq(0) & out["Population"].notna()
        rate = pd.Series(pd.NA, index=out.index, dtype="Float64")
        rate.loc[valid_rate] = (
            out.loc[valid_rate, "case_count_used"].astype("float64")
            / out.loc[valid_rate, "Population"].astype("float64")
        ) * 100000
        if bool(rate.notna().any()) and bool((rate.dropna().map(lambda value: not math.isfinite(float(value)))).any()):
            raise CovidPreprocessError(f"Infinite {MODELLING_RATE_COLUMN} values were produced.")
        out[MODELLING_RATE_COLUMN] = rate
        out = out.drop(columns=["_numeric_count"])
        scenarios[label] = out
    return scenarios


def _assert_cross_scenario_consistency(scenarios: dict[str, pd.DataFrame], fills: tuple[int, ...]) -> None:
    labels = [SCENARIO_LABELS[fill] for fill in fills]
    reference = scenarios[labels[0]]
    ref_keys = list(zip(reference["Date"].astype(str), reference["IntZone"].astype(str)))
    for label in labels[1:]:
        other = scenarios[label]
        if len(other) != len(reference):
            raise CovidPreprocessError("Scenario row counts differ.")
        other_keys = list(zip(other["Date"].astype(str), other["IntZone"].astype(str)))
        if other_keys != ref_keys:
            raise CovidPreprocessError("Scenario Date+IntZone keys or order differ.")
        if not other["node_index"].equals(reference["node_index"]):
            raise CovidPreprocessError("Scenario node_index order differs.")
        for column in CROSS_SCENARIO_IDENTITY_FIELDS:
            if column not in reference.columns:
                continue
            if not _series_equal(reference[column], other[column]):
                raise CovidPreprocessError(f"Non-suppressed identity field {column} differs in {label}.")
        observed = reference["response_status"].eq("observed")
        if not _series_equal(reference.loc[observed, "case_count_used"], other.loc[observed, "case_count_used"]):
            raise CovidPreprocessError("Observed case_count_used differs across scenarios.")
        missing = reference["missing_flag"].eq(1) | reference["invalid_value_flag"].eq(1)
        if not other.loc[missing, "case_count_used"].isna().all():
            raise CovidPreprocessError("True missing or invalid counts were filled in a scenario.")
        suppressed = reference["suppression_flag"].eq(1)
        expected_fill = int(label.replace("fill", ""))
        if suppressed.any():
            if set(other.loc[suppressed, "case_count_used"].dropna().unique().tolist()) - {expected_fill}:
                raise CovidPreprocessError(f"{label} assigned a fill other than {expected_fill} to suppressed rows.")
            if other.loc[~suppressed & other["case_count_used"].eq(expected_fill) & reference["response_status"].ne("observed")].shape[0]:
                pass
        non_suppressed = ~suppressed
        if not _series_equal(
            reference.loc[non_suppressed, "case_count_used"],
            other.loc[non_suppressed, "case_count_used"],
        ):
            raise CovidPreprocessError("A non-suppressed case_count_used differs across scenarios.")


def _series_equal(left: pd.Series, right: pd.Series) -> bool:
    left_na = left.isna()
    right_na = right.isna()
    if not left_na.equals(right_na):
        return False
    comparable = ~left_na
    return bool((left[comparable].astype(str) == right[comparable].astype(str)).all())


def _scenario_statistics(frame: pd.DataFrame) -> dict[str, Any]:
    rate = pd.to_numeric(frame[MODELLING_RATE_COLUMN], errors="coerce")
    valid = rate.notna()
    stats: dict[str, Any] = {
        "n_rows": int(len(frame)),
        "n_valid_modelling_response_rows": int(valid.sum()),
        "n_suppressed": int(frame["suppression_flag"].sum()),
        "n_ordinary_missing": int(frame["missing_flag"].sum()),
        "n_invalid": int(frame["invalid_value_flag"].sum()),
    }
    if bool(valid.any()):
        values = rate.loc[valid].astype("float64")
        stats.update(
            {
                "mean_infection_rate": float(values.mean()),
                "std_infection_rate": float(values.std(ddof=0)),
                "min_infection_rate": float(values.min()),
                "max_infection_rate": float(values.max()),
                "quantiles": {str(q): float(values.quantile(q)) for q in QUANTILES},
            }
        )
    suppressed = frame["suppression_flag"].eq(1)
    suppressed_rate = rate.loc[suppressed]
    if bool(suppressed_rate.notna().any()):
        values = suppressed_rate.dropna().astype("float64")
        stats["suppressed_substituted_rate"] = {
            "mean": float(values.mean()),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return stats


def _suppression_diagnostics(base: pd.DataFrame, scenarios: dict[str, pd.DataFrame]) -> dict[str, Any]:
    n = len(base)
    suppressed = base["suppression_flag"].eq(1)
    by_iz = (base.assign(_s=suppressed).groupby("IntZone")["_s"].mean() * 100).sort_values(ascending=False)
    by_date = (
        base.assign(_s=suppressed, _d=base["Date"].dt.strftime("%Y-%m-%d"))
        .groupby("_d")["_s"]
        .mean()
        * 100
    ).sort_values(ascending=False)
    return {
        "pct_panel_suppressed": float(100 * suppressed.mean()) if n else 0.0,
        "highest_suppression_iz": [
            {"IntZone": str(zone), "pct": float(pct)} for zone, pct in by_iz.head(10).items()
        ],
        "highest_suppression_dates": [
            {"Date": str(day), "pct": float(pct)} for day, pct in by_date.head(10).items()
        ],
        "note": "These percentages are preprocessing diagnostics, not model results.",
    }


def _write_scenario_files(
    scenarios: dict[str, pd.DataFrame],
    output_dir: Path | None = None,
) -> dict[str, str]:
    folder = Path(output_dir) if output_dir is not None else _results_dir()
    folder.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for label, frame in scenarios.items():
        out = frame.copy()
        if "Date_original" in out.columns:
            out["Date"] = out["Date_original"]
        else:
            out["Date"] = pd.to_datetime(out["Date"]).dt.strftime("%Y%m%d")
        path = folder / f"{label}.csv"
        write_table(out, path)
        paths[label] = str(path)
    return paths


def _write_diagnostics(
    base: pd.DataFrame,
    gap_report: dict[str, Any],
    output_dir: Path | None = None,
) -> dict[str, str | None]:
    folder = Path(output_dir) if output_dir is not None else _results_dir()
    folder.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str | None] = {
        "invalid_records": None,
        "missing_panel_cells_by_iz": None,
    }
    invalid = base.loc[base["invalid_value_flag"].eq(1)]
    if not invalid.empty:
        path = folder / "invalid.csv"
        write_table(invalid, path)
        paths["invalid_records"] = str(path)
    summary = gap_report.get("missing_cells_by_iz_summary") or {}
    if summary:
        table = pd.DataFrame(
            [{"IntZone": zone, "n_missing_dates": count} for zone, count in summary.items()]
        )
        path = folder / "missing.csv"
        write_table(table, path)
        paths["missing_panel_cells_by_iz"] = str(path)
    return paths
