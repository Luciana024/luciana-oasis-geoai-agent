"""Aggregate SIMD 2020v2 rates from council Data Zones to 2011 Intermediate Zones.

One indicator per domain:
- income: income_rate
- employment: employment_rate
- education: university_rate (17-21 year olds entering university)
- housing: overcrowded_rate
- crime: crime_rate (proportion of population; fill1 is primary)
- GP access: pt_gp_min (public-transport minutes; SIMD has no GP rate)

SIMD technical notes: Data Zone crime counts below 3 are published as '*'.
Those cells are filled as 0, 1 and 2 (fill0 / fill1 / fill2). Raw files are not modified.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from data.covid import load_iz_master
from common.utils import LOCAL_AUTHORITY_CODE, PANEL_CSV, get_logger, project_root, read_table, write_table

LOGGER = get_logger("deprivation")

INDICATORS_XLSX = Path("data") / "raw" / "deprivation" / "SIMD+2020v2+-+indicators.xlsx"
LOOKUP_CSV = Path("data") / "raw" / "boundaries" / "Code lookup.csv"

COUNT_RATE_SPECS = (
    ("income_rate", "Income_count", "Total_population"),
    ("employment_rate", "Employment_count", "Working_age_population"),
    ("overcrowded_rate", "overcrowded_count", "Total_population"),
)
WEIGHTED_RATE_SPECS = (
    ("university_rate", "University", "Total_population"),
)
WEIGHTED_TIME_SPECS = (
    ("pt_gp_min", "PT_GP", "Total_population"),
)
CRIME_FILLS = (0, 1, 2)
PRIMARY_CRIME_FILL = 1
SIMD_JOIN_COLUMNS = [
    "IntZone",
    "income_rate",
    "employment_rate",
    "university_rate",
    "overcrowded_rate",
    "crime_rate",
    "crime_rate_fill0",
    "crime_rate_fill1",
    "crime_rate_fill2",
    "crime_incomplete",
    "n_dz_crime_suppressed",
    "pt_gp_min",
]


class DeprivationError(ValueError):
    """SIMD cannot be aggregated to IZ without guessing."""


def aggregate_simd_to_iz(
    area_code: str = LOCAL_AUTHORITY_CODE,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Build one IZ row from SIMD Data Zone rates and write simd_iz.csv."""
    iz_master = load_iz_master(area_code=area_code)
    lookup = _edinburgh_dz_lookup(area_code=area_code)
    dz = _load_indicator_rows()
    dz = _join_intzone(dz, lookup)
    dz = _to_numeric_indicators(dz)
    iz = _aggregate_to_iz(dz, iz_master)
    out_path = Path(output_path) if output_path is not None else _results_dir() / "simd_iz.csv"
    write_table(iz, out_path)
    LOGGER.info("Wrote %s Intermediate Zone SIMD rows to %s", len(iz), out_path)
    return {
        "output_path": str(out_path),
        "n_data_zones": int(len(dz)),
        "n_iz": int(len(iz)),
        "n_iz_crime_incomplete": int(iz["crime_incomplete"].sum()),
        "primary_crime_scenario": f"fill{PRIMARY_CRIME_FILL}",
        "source": str(project_root() / INDICATORS_XLSX),
    }


def attach_simd_to_panel(
    covid_path: Path | None = None,
    out_path: Path | None = None,
    area_code: str = LOCAL_AUTHORITY_CODE,
) -> dict[str, Any]:
    """Join IZ-level SIMD rates onto fill1_geo.csv and write panel.csv.

    SIMD attributes are constant within an Intermediate Zone, so one IZ row is
    copied onto every COVID date for that IZ. Unmatched IntZone codes fail;
    they are not invented. fill1.csv and fill1_geo.csv are not modified.
    """
    covid_path = Path(covid_path) if covid_path is not None else _results_dir() / "fill1_geo.csv"
    out_path = Path(out_path) if out_path is not None else _results_dir() / PANEL_CSV
    if not covid_path.exists():
        raise DeprivationError(f"COVID table missing: {covid_path}. Run attach_iz_centroids first.")
    covid = read_table(covid_path)
    if "IntZone" not in covid.columns:
        raise DeprivationError("COVID table must contain IntZone.")
    covid["IntZone"] = covid["IntZone"].astype("string").str.strip()
    already = [column for column in SIMD_JOIN_COLUMNS if column != "IntZone" and column in covid.columns]
    if already:
        covid = covid.drop(columns=already)

    aggregated = aggregate_simd_to_iz(area_code=area_code)
    simd = read_table(Path(aggregated["output_path"]))
    missing_cols = [column for column in SIMD_JOIN_COLUMNS if column not in simd.columns]
    if missing_cols:
        raise DeprivationError(f"simd_iz.csv missing columns {missing_cols}.")
    simd = simd[SIMD_JOIN_COLUMNS].copy()
    simd["IntZone"] = simd["IntZone"].astype("string").str.strip()
    if simd["IntZone"].duplicated().any():
        raise DeprivationError("simd_iz.csv has duplicate IntZone codes.")

    out = covid.merge(simd, on="IntZone", how="left", validate="many_to_one")
    unmatched = sorted(out.loc[out["income_rate"].isna(), "IntZone"].dropna().unique().tolist())
    if unmatched:
        raise DeprivationError(
            f"{len(unmatched)} COVID IntZone codes have no SIMD row: {unmatched[:10]}."
        )
    write_table(out, out_path)
    LOGGER.info("Joined SIMD rates onto %s COVID rows at %s", len(out), out_path)
    return {
        "covid_path": str(covid_path),
        "output_path": str(out_path),
        "simd_path": aggregated["output_path"],
        "n_rows": int(len(out)),
        "n_iz": int(out["IntZone"].nunique()),
        "n_iz_crime_incomplete": aggregated["n_iz_crime_incomplete"],
        "joined_columns": [column for column in SIMD_JOIN_COLUMNS if column != "IntZone"],
    }


def _results_dir() -> Path:
    return project_root() / "data" / "results"


def _edinburgh_dz_lookup(area_code: str) -> pd.DataFrame:
    path = project_root() / LOOKUP_CSV
    if not path.exists():
        raise DeprivationError(f"IZ lookup missing: {path}")
    lookup = read_table(path)
    needed = {"DataZone", "IntZone", "CA"}
    missing = sorted(needed - set(lookup.columns))
    if missing:
        raise DeprivationError(f"Lookup missing columns {missing}.")
    out = lookup.loc[lookup["CA"].astype("string").str.strip() == area_code].copy()
    out["DataZone"] = out["DataZone"].astype("string").str.strip()
    out["IntZone"] = out["IntZone"].astype("string").str.strip()
    out = out.loc[out["DataZone"].ne("") & out["IntZone"].ne(""), ["DataZone", "IntZone"]]
    out = out.drop_duplicates()
    if out["DataZone"].duplicated().any():
        raise DeprivationError(f"Lookup has duplicate DataZone codes for CA={area_code}.")
    if out.empty:
        raise DeprivationError(f"No Data Zones found for CA={area_code}.")
    return out.reset_index(drop=True)


def _load_indicator_rows() -> pd.DataFrame:
    path = project_root() / INDICATORS_XLSX
    if not path.exists():
        raise DeprivationError(f"SIMD indicators workbook missing: {path}")
    frame = pd.read_excel(path, sheet_name="Data", dtype=str)
    frame.columns = [str(col).strip() for col in frame.columns]
    if "Data_Zone" not in frame.columns:
        raise DeprivationError("SIMD Data sheet must contain Data_Zone.")
    frame = frame.rename(columns={"Data_Zone": "DataZone"})
    frame["DataZone"] = frame["DataZone"].astype("string").str.strip()
    return frame


def _join_intzone(dz: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
    """Keep this council's Data Zones by official lookup code, not by IZ name."""
    merged = lookup.merge(dz, on="DataZone", how="left", indicator=True)
    missing = merged.loc[merged["_merge"].eq("left_only"), "DataZone"].tolist()
    if missing:
        raise DeprivationError(
            f"{len(missing)} Data Zones are absent from SIMD indicators: {missing[:10]}."
        )
    extra = set(dz["DataZone"]) - set(lookup["DataZone"])
    # Extra rows are other councils; they are dropped by the left join on lookup.
    merged = merged.drop(columns=["_merge"])
    LOGGER.info(
        "Joined %s Data Zones to %s Intermediate Zones (%s SIMD rows unused).",
        len(merged),
        merged["IntZone"].nunique(),
        len(extra),
    )
    return merged


def _to_numeric_indicators(dz: pd.DataFrame) -> pd.DataFrame:
    out = dz.copy()
    columns = [
        "Total_population",
        "Working_age_population",
        "Income_count",
        "Employment_count",
        "overcrowded_count",
        "crime_count",
        "University",
        "PT_GP",
    ]
    missing = [column for column in columns if column not in out.columns]
    if missing:
        raise DeprivationError(f"SIMD Data sheet missing columns {missing}.")
    out["crime_suppressed"] = out["crime_count"].astype("string").str.strip().eq("*")
    for column in columns:
        out[column] = _numeric_or_missing(out[column])
    missing_pop = out["Total_population"].isna()
    if missing_pop.any():
        codes = out.loc[missing_pop, "DataZone"].astype(str).head(10).tolist() if "DataZone" in out.columns else []
        raise DeprivationError(
            f"{int(missing_pop.sum())} Data Zones have missing Total_population: {codes}."
        )
    missing_wa = out["Working_age_population"].isna()
    if missing_wa.any():
        codes = out.loc[missing_wa, "DataZone"].astype(str).head(10).tolist() if "DataZone" in out.columns else []
        raise DeprivationError(
            f"{int(missing_wa.sum())} Data Zones have missing Working_age_population: {codes}."
        )
    return out


def _numeric_or_missing(series: pd.Series) -> pd.Series:
    """Convert SIMD cells to numbers. '*' and blanks stay missing."""
    text = series.astype("string").str.strip()
    suppressed = text.isin({"*", "None", "<null>", "."}) | text.eq("") | text.isna()
    values = pd.to_numeric(text.mask(suppressed), errors="coerce")
    invalid = values.isna() & ~suppressed
    if bool(invalid.any()):
        samples = text.loc[invalid].head(5).tolist()
        raise DeprivationError(f"Non-numeric SIMD values that are not suppression flags: {samples}.")
    return values


def _inhabited_data_zones(dz: pd.DataFrame) -> pd.DataFrame:
    """Keep DZs with positive SIMD population. Observed zeros are omitted, not filled."""
    zero = dz["Total_population"].le(0) | dz["Working_age_population"].le(0)
    if bool(zero.any()):
        codes = dz.loc[zero, "DataZone"].astype(str).tolist() if "DataZone" in dz.columns else []
        LOGGER.warning(
            "%s Data Zones have SIMD population 0 and are omitted from IZ rates: %s",
            int(zero.sum()),
            codes,
        )
    inhabited = dz.loc[~zero].copy()
    if inhabited.empty:
        raise DeprivationError("No Data Zones with positive SIMD population remain.")
    return inhabited


def _aggregate_to_iz(dz: pd.DataFrame, iz_master: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = _inhabited_data_zones(dz).groupby("IntZone", sort=True)
    for zone, part in grouped:
        row: dict[str, Any] = {
            "IntZone": zone,
            "n_data_zones": int(len(part)),
            "total_population": float(part["Total_population"].sum()),
            "working_age_population": float(part["Working_age_population"].sum()),
        }
        for out_name, count_col, pop_col in COUNT_RATE_SPECS:
            row[out_name] = _rate_from_counts(part[count_col], part[pop_col])
        row.update(_crime_rate_scenarios(part))
        for out_name, value_col, pop_col in (*WEIGHTED_RATE_SPECS, *WEIGHTED_TIME_SPECS):
            row[out_name] = _weighted_mean(part[value_col], part[pop_col])
        rows.append(row)
    iz = pd.DataFrame(rows)
    master = iz_master[["IntZone", "node_index"]].copy()
    master["IntZone"] = master["IntZone"].astype("string").str.strip()
    out = master.merge(iz, on="IntZone", how="left", validate="one_to_one")
    missing_iz = out.loc[out["n_data_zones"].isna(), "IntZone"].tolist()
    if missing_iz:
        raise DeprivationError(
            f"{len(missing_iz)} IZs have no SIMD Data Zones with positive population: {missing_iz[:10]}."
        )
    extra_iz = sorted(set(iz["IntZone"]) - set(master["IntZone"]))
    if extra_iz:
        raise DeprivationError(f"SIMD produced unexpected IZs: {extra_iz[:10]}.")
    if len(out) != len(master):
        raise DeprivationError("IZ aggregation changed the Intermediate Zone count.")
    return out.sort_values("node_index").reset_index(drop=True)


def _crime_rate_scenarios(part: pd.DataFrame) -> dict[str, Any]:
    """Fill SIMD crime '*' as 0/1/2. Official notes: counts below 3 are suppressed."""
    suppressed = part["crime_suppressed"].fillna(False).astype(bool)
    payload: dict[str, Any] = {
        "n_dz_crime_suppressed": int(suppressed.sum()),
        "crime_incomplete": int(bool(suppressed.any())),
    }
    for fill in CRIME_FILLS:
        counts = part["crime_count"].copy()
        counts.loc[suppressed] = fill
        rate = _rate_from_counts(counts, part["Total_population"])
        payload[f"crime_rate_fill{fill}"] = rate
    payload["crime_rate"] = payload[f"crime_rate_fill{PRIMARY_CRIME_FILL}"]
    return payload


def _rate_from_counts(counts: pd.Series, population: pd.Series) -> float | pd.NA:
    """IZ rate = sum(count) / sum(population). Missing if any Data Zone count is missing."""
    if counts.isna().any() or population.isna().any():
        return pd.NA
    denom = float(population.sum())
    if denom <= 0:
        return pd.NA
    return float(counts.sum()) / denom


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float | pd.NA:
    """Population-weighted mean. Missing if any Data Zone value is missing."""
    if values.isna().any() or weights.isna().any():
        return pd.NA
    denom = float(weights.sum())
    if denom <= 0:
        return pd.NA
    return float((values * weights).sum()) / denom
