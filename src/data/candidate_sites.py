"""Candidate intervention sites from oasis-v4 (GP, pharmacy, mobile car parks).

Load existing tables with load_candidate_sites. Build them with
prepare_candidate_sites (city is a parameter). Coordinates are not invented:
official lists plus optional OSM / postcodes.io. Outputs go to
data/results/candidate_sites/<area_code>/.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pandas as pd

from common.errors import ModelError
from common.utils import (
    LOCAL_AUTHORITY_CODE,
    get_logger,
    project_root,
    write_json,
    write_table,
)

LOGGER = get_logger("data.candidate_sites")

SITE_ID_CANDIDATES = ("site_id", "siteId", "SiteID", "SITE_ID")
EASTING_CANDIDATES = ("easting", "Easting", "iz_easting")
NORTHING_CANDIDATES = ("northing", "Northing", "iz_northing")
PREFERRED_NAMES = (
    "merged_candidate_sites.shp",
    "merged_candidate_sites.csv",
    "edinburgh_merged_candidate_sites.shp",
    "edinburgh_merged_candidate_sites.csv",
    "candidate_sites.shp",
    "candidate_sites.csv",
)
SPATIAL_SUFFIXES = {".shp", ".geojson", ".gpkg"}
TABLE_SUFFIXES = {".csv", ".parquet"}


def candidate_sites_dir() -> Path:
    return project_root() / "data" / "raw" / "candidate_sites"


def _pick_site_file_in_folder(folder: Path) -> Path | None:
    if not folder.exists():
        return None
    for name in PREFERRED_NAMES:
        path = folder / name
        if path.exists():
            return path
    files = [
        path
        for path in sorted(folder.iterdir())
        if path.is_file()
        and path.name not in {".gitkeep", ".DS_Store"}
        and path.suffix.lower() in (SPATIAL_SUFFIXES | TABLE_SUFFIXES)
    ]
    if len(files) == 1:
        return files[0]
    if len(files) > 1:
        raise ModelError(
            "Multiple candidate-site files found; pass an explicit path.",
            code="invalid_config",
            details={"folder": str(folder), "files": [path.name for path in files]},
        )
    return None


def find_candidate_site_file(
    directory: Path | None = None,
    area_code: str | None = None,
) -> Path:
    """Pick a site file without guessing among several unrelated tables.

    Default search: data/raw/candidate_sites, then data/results/candidate_sites/<CA>/.
    """
    if directory is not None:
        folder = Path(directory)
        hit = _pick_site_file_in_folder(folder)
        if hit is not None:
            return hit
        raise ModelError(
            f"No candidate-site table in {folder}. Do not invent vaccination sites.",
            code="missing_dataset",
        )

    searched = [candidate_sites_dir()]
    results_root = project_root() / "data" / "results" / "candidate_sites"
    if area_code:
        searched.append(candidate_sites_results_dir(str(area_code).strip()))
    elif results_root.exists():
        searched.extend(sorted(path for path in results_root.iterdir() if path.is_dir()))

    found: list[Path] = []
    for folder in searched:
        hit = _pick_site_file_in_folder(folder)
        if hit is not None:
            found.append(hit)
    if len(found) == 1:
        return found[0]
    if len(found) > 1:
        merged = [path for path in found if path.name.startswith("merged_candidate_sites")]
        if merged:
            return merged[0]
        return found[0]
    raise ModelError(
        "No candidate-site table in data/raw/candidate_sites or "
        "data/results/candidate_sites. Do not invent vaccination sites.",
        code="missing_dataset",
        details={"searched": [str(folder) for folder in searched]},
    )


def load_candidate_sites(
    path: str | Path | None = None,
    directory: str | Path | None = None,
    area_code: str | None = None,
):
    """Return a GeoDataFrame with site_id and point geometry in EPSG:27700."""
    import geopandas as gpd

    file_path = Path(path) if path is not None else find_candidate_site_file(
        Path(directory) if directory is not None else None,
        area_code=area_code,
    )
    if not file_path.exists():
        raise ModelError(
            f"Candidate-site table missing: {file_path}. Do not invent vaccination sites.",
            code="missing_dataset",
        )
    suffix = file_path.suffix.lower()
    if suffix in SPATIAL_SUFFIXES:
        frame = gpd.read_file(file_path)
    elif suffix == ".csv":
        frame = gpd.GeoDataFrame(pd.read_csv(file_path))
    elif suffix == ".parquet":
        frame = gpd.GeoDataFrame(pd.read_parquet(file_path))
    else:
        raise ModelError(
            f"Unsupported candidate-site format: {file_path}",
            code="invalid_config",
        )
    return _normalise_sites(frame, source=str(file_path))


def _normalise_sites(frame, source: str):
    import geopandas as gpd
    from shapely.geometry import Point

    if frame is None or len(frame) == 0:
        raise ModelError(
            "Candidate-site table is empty. Do not invent vaccination sites.",
            code="missing_dataset",
            details={"source": source},
        )
    site_col = next((name for name in SITE_ID_CANDIDATES if name in frame.columns), None)
    if site_col is None:
        raise ModelError(
            f"Candidate-site table must contain site_id. Columns: {list(frame.columns)}",
            code="invalid_config",
            details={"source": source},
        )
    out = frame.rename(columns={site_col: "site_id"}).copy()
    out["site_id"] = out["site_id"].astype("string").str.strip()
    if out["site_id"].eq("").any() or out["site_id"].isna().any():
        raise ModelError("Candidate-site table contains empty site_id values.", code="invalid_config")
    duplicated = out.loc[out["site_id"].duplicated(), "site_id"].astype(str).tolist()
    if duplicated:
        raise ModelError(
            f"Candidate-site table has duplicate site_id values: {duplicated[:10]}.",
            code="invalid_config",
        )

    geometry_col = getattr(out, "_geometry_column_name", "geometry")
    has_geometry = geometry_col in out.columns and out[geometry_col].notna().any()
    if has_geometry:
        gdf = gpd.GeoDataFrame(out, geometry=geometry_col)
        if gdf.crs is None:
            raise ModelError(
                "Candidate-site geometries have no CRS; refusing to guess.",
                code="invalid_config",
                details={"source": source},
            )
        gdf = gdf.to_crs(epsg=27700)
    else:
        easting_col = next((name for name in EASTING_CANDIDATES if name in out.columns), None)
        northing_col = next((name for name in NORTHING_CANDIDATES if name in out.columns), None)
        if easting_col is None or northing_col is None:
            raise ModelError(
                "Candidate-site table needs geometry or easting/northing. Do not invent coordinates.",
                code="invalid_config",
                details={"source": source},
            )
        easting = pd.to_numeric(out[easting_col], errors="coerce")
        northing = pd.to_numeric(out[northing_col], errors="coerce")
        if easting.isna().any() or northing.isna().any():
            raise ModelError(
                "Candidate-site easting/northing contain missing values.",
                code="invalid_config",
            )
        gdf = gpd.GeoDataFrame(
            out,
            geometry=[Point(x, y) for x, y in zip(easting.tolist(), northing.tolist())],
            crs="EPSG:27700",
        )
    LOGGER.info("Loaded %s candidate sites from %s.", len(gdf), source)
    return gdf


def site_manifest(frame) -> dict[str, Any]:
    types = None
    if "site_type" in frame.columns:
        types = frame["site_type"].astype(str).value_counts().to_dict()
    return {
        "n_sites": int(len(frame)),
        "site_types": types,
        "site_id_sample": frame["site_id"].astype(str).head(5).tolist(),
    }


ALLOWED_SOURCES = ("local", "api", "osm")
CORE_COLUMNS = [
    "site_id",
    "site_name",
    "site_type",
    "easting",
    "northing",
    "iz_code",
    "iz_easting",
    "iz_northing",
    "suitability",
    "coord_source",
]
CAR_PARK_OSM_TAGS = {"amenity": "parking"}
RESTRICTED_ACCESS = (
    "private",
    "no",
    "customers",
    "permit",
    "restricted",
    "members",
    "delivery",
)
EXCLUDED_PARKING = ("underground", "garages", "sheds")
AIRPORT_NAME_PATTERN = r"airport|premair|plane parking"


def candidate_sites_results_dir(area_code: str) -> Path:
    token = str(area_code).strip() or "unknown_area"
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in token)
    return project_root() / "data" / "results" / "candidate_sites" / safe


def _sites_to_geodataframe(frame: pd.DataFrame):
    import geopandas as gpd

    valid = frame.dropna(subset=["easting", "northing"]).copy()
    if valid.empty:
        raise ModelError(
            "No candidate sites have coordinates. Do not invent them.",
            code="missing_dataset",
        )
    return gpd.GeoDataFrame(
        valid,
        geometry=gpd.points_from_xy(valid["easting"], valid["northing"]),
        crs="EPSG:27700",
    )


def _harmonise_core(frame) -> pd.DataFrame:
    import geopandas as gpd

    out = frame.copy()
    rename = {
        "iz_northin": "iz_northing",
        "suitabilit": "suitability",
        "coord_sour": "coord_source",
    }
    out = out.rename(columns={old: new for old, new in rename.items() if old in out.columns})
    for column in CORE_COLUMNS:
        if column not in out.columns:
            out[column] = None
    cols = CORE_COLUMNS + [name for name in out.columns if name not in CORE_COLUMNS]
    out = out[cols]
    if isinstance(out, gpd.GeoDataFrame):
        return out
    return out


def filter_public_car_parks(osm_gdf):
    """oasis-v4 car-park rules, plus drop airport lots (not vaccination sites).

    access, parking and capacity tags are required. Geofabrik shapefiles cannot
    be passed here because they do not carry those tags. Airport parking is
    excluded here so it never enters the merged candidate table.
    """
    gdf = osm_gdf.copy()
    missing = [name for name in ("access", "parking", "capacity") if name not in gdf.columns]
    if missing:
        raise ModelError(
            "Car-park OSM tags are missing: "
            f"{missing}. oasis-v4 filters cannot run. "
            "Fetch parking from the Overpass API, not the Geofabrik shapefile.",
            code="invalid_config",
            details={"missing_tags": missing},
        )
    access = gdf["access"].fillna("").astype(str).str.lower()
    gdf = gdf[~access.isin(RESTRICTED_ACCESS)].copy()
    parking_type = gdf["parking"].fillna("").astype(str).str.lower()
    gdf = gdf[~parking_type.isin(EXCLUDED_PARKING)].copy()
    is_airport = pd.Series(False, index=gdf.index)
    if "name" in gdf.columns:
        is_airport = is_airport | gdf["name"].fillna("").astype(str).str.contains(
            AIRPORT_NAME_PATTERN, case=False, regex=True, na=False
        )
    if "operator" in gdf.columns:
        is_airport = is_airport | gdf["operator"].fillna("").astype(str).str.contains(
            "airport", case=False, na=False
        )
    gdf = gdf[~is_airport].copy()
    gdf["capacity_num"] = pd.to_numeric(gdf["capacity"], errors="coerce")
    is_large = gdf["capacity_num"] >= 20
    is_p_and_r = (
        gdf["park_and_ride"].astype(str).str.lower().isin(["yes", "park_and_ride"])
        if "park_and_ride" in gdf.columns
        else pd.Series(False, index=gdf.index)
    )
    has_name = (
        gdf["name"].notna()
        & (gdf["name"].astype(str).str.strip() != "")
        & (gdf["name"].astype(str) != "Public Car Park")
        if "name" in gdf.columns
        else pd.Series(False, index=gdf.index)
    )
    gdf = gdf[is_large | is_p_and_r | has_name].copy()
    if gdf.empty:
        return gdf
    if gdf.crs is None:
        raise ModelError("Parking geometries have no CRS; refusing to guess.", code="invalid_config")
    gdf = gdf.to_crs(epsg=27700)
    gdf["geometry"] = gdf.geometry.centroid
    gdf["easting"] = gdf.geometry.x.round(2)
    gdf["northing"] = gdf.geometry.y.round(2)
    gdf["x_round"] = (gdf["easting"] / 100).round() * 100
    gdf["y_round"] = (gdf["northing"] / 100).round() * 100
    gdf = gdf.drop_duplicates(subset=["x_round", "y_round"]).drop(columns=["x_round", "y_round"])
    return gdf.reset_index(drop=True)


def prepare_mobile_sites(
    area_code: str = LOCAL_AUTHORITY_CODE,
    osm_place: str | None = None,
    *,
    osm_features=None,
    fetch_fn: Callable | None = None,
) -> pd.DataFrame:
    """Car parks as provisional mobile_stop sites. Not official vaccination clinics.

    Parking must come from OSM Overpass (tagged). Geofabrik shapefiles are rejected
    because they cannot apply oasis-v4 access/capacity filters.
    """
    import geopandas as gpd

    from data.travel_time import load_iz_origins

    if osm_features is None:
        if fetch_fn is not None:
            from data.healthcare import resolve_osm_place

            place = resolve_osm_place(area_code, osm_place)
            raw = fetch_fn(osm_place=place, tags=CAR_PARK_OSM_TAGS)
        else:
            from data.osm_extract import clip_to_area, fetch_overpass_parking

            raw = fetch_overpass_parking(area_code)
            raw = clip_to_area(raw, area_code)
    else:
        raw = osm_features
    if raw is None or len(raw) == 0:
        raise ModelError(
            "OSM returned no parking features. Do not invent mobile sites.",
            code="missing_dataset",
        )
    filtered = filter_public_car_parks(raw)
    if filtered.empty:
        raise ModelError(
            "No public car parks remain after oasis-v4 access/capacity filters.",
            code="missing_dataset",
        )
    origins = load_iz_origins(area_code=area_code)
    origins = origins.copy()
    origins["iz_easting"] = origins.geometry.x.round(2)
    origins["iz_northing"] = origins.geometry.y.round(2)
    joined = gpd.sjoin_nearest(
        filtered,
        origins[["iz_code", "iz_easting", "iz_northing", "geometry"]],
        how="left",
        distance_col="dist_to_iz",
    )
    joined["site_id"] = "MS_" + (joined.index + 1).astype(str)
    if "name" in joined.columns:
        joined["site_name"] = joined["name"].fillna("Public Car Park")
    else:
        joined["site_name"] = "Public Car Park"
    joined["site_type"] = "mobile_stop"
    joined["suitability"] = "provisional"
    joined["coord_source"] = "OSM_overpass_parking"
    useful = [name for name in ("capacity", "parking", "fee", "access", "operator", "surface", "park_and_ride") if name in joined.columns]
    LOGGER.info("Prepared %s mobile_stop sites for CA=%s.", len(joined), area_code)
    return _harmonise_core(joined[CORE_COLUMNS + useful + ["geometry"]])


def merge_candidate_sites(*layers) -> pd.DataFrame:
    frames = []
    for layer in layers:
        if layer is None or len(layer) == 0:
            continue
        frames.append(_harmonise_core(layer))
    if not frames:
        raise ModelError("No candidate-site layers to merge.", code="missing_dataset")
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop(columns=["geometry"], errors="ignore")
    if combined["site_id"].duplicated().any():
        raise ModelError("Merged candidate sites have duplicate site_id values.", code="invalid_config")
    return _sites_to_geodataframe(combined)


def _write_site_layer(frame, directory: Path, stem: str) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(frame.drop(columns=["geometry"], errors="ignore"))
    csv_path = write_table(table, directory / f"{stem}.csv")
    try:
        gdf = frame if "geometry" in getattr(frame, "columns", []) else _sites_to_geodataframe(table)
        gdf.to_file(directory / f"{stem}.geojson")
    except Exception as exc:
        LOGGER.warning("Could not write %s.geojson: %s", stem, exc)
    return str(csv_path)


LAYER_STEMS = (
    "gp_candidate_sites",
    "pharmacy_candidate_sites",
    "car_parks_candidate_sites",
    "merged_candidate_sites",
)


def remove_site_ids(
    site_ids: list[str] | tuple[str, ...],
    directory: str | Path,
    *,
    reason: str | None = None,
) -> dict[str, Any]:
    """Drop named sites from an existing candidate-site results folder. IDs are not reused."""
    import json

    folder = Path(directory)
    dropped = {str(item).strip() for item in site_ids if str(item).strip()}
    if not dropped or not folder.exists():
        return {"dropped": sorted(dropped), "rewritten": [], "n_remaining": None}

    rewritten: list[str] = []
    remaining = None
    for stem in LAYER_STEMS:
        csv_path = folder / f"{stem}.csv"
        if csv_path.exists():
            table = pd.read_csv(csv_path)
            if "site_id" in table.columns:
                table = table.loc[~table["site_id"].astype(str).isin(dropped)].copy()
                write_table(table, csv_path)
                rewritten.append(str(csv_path))
                if stem == "merged_candidate_sites":
                    remaining = int(len(table))
        geo_path = folder / f"{stem}.geojson"
        if geo_path.exists():
            payload = json.loads(geo_path.read_text(encoding="utf-8"))
            features = [
                feature
                for feature in payload.get("features") or []
                if str((feature.get("properties") or {}).get("site_id", "")).strip() not in dropped
            ]
            payload["features"] = features
            geo_path.write_text(json.dumps(payload), encoding="utf-8")
            rewritten.append(str(geo_path))

    provenance_path = folder / "provenance.json"
    if provenance_path.exists():
        record = json.loads(provenance_path.read_text(encoding="utf-8"))
        record["dropped_site_ids"] = sorted(set(record.get("dropped_site_ids") or []) | dropped)
        if reason:
            record["dropped_reason"] = reason
        if remaining is not None:
            record["n_merged"] = remaining
        parks_csv = folder / "car_parks_candidate_sites.csv"
        if parks_csv.exists():
            record["n_mobile"] = int(len(pd.read_csv(parks_csv)))
        gp_csv = folder / "gp_candidate_sites.csv"
        if gp_csv.exists():
            record["n_gp"] = int(len(pd.read_csv(gp_csv)))
        pharm_csv = folder / "pharmacy_candidate_sites.csv"
        if pharm_csv.exists():
            record["n_pharmacy"] = int(len(pd.read_csv(pharm_csv)))
        write_json(record, provenance_path)
        rewritten.append(str(provenance_path))
    LOGGER.info("Removed %s from %s.", sorted(dropped), folder)
    return {"dropped": sorted(dropped), "rewritten": rewritten, "n_remaining": remaining}


def prepare_candidate_sites(
    area_code: str = LOCAL_AUTHORITY_CODE,
    source: str = "osm",
    osm_place: str | None = None,
    gp_path: str | Path | None = None,
    pharmacy_path: str | Path | None = None,
    *,
    gp_sites=None,
    pharmacy_sites=None,
    mobile_sites=None,
    include_mobile: bool = True,
    output_dir: str | Path | None = None,
    geocode_fn: Callable | None = None,
    fetch_fn: Callable | None = None,
    use_osm: bool = True,
    ckan_get: Callable | None = None,
) -> dict[str, Any]:
    """Build or load the merged candidate-site table for one CA."""
    from datetime import datetime, timezone

    from data.healthcare import prepare_gp_sites, prepare_pharmacy_sites

    source_key = str(source).strip().lower()
    if source_key not in ALLOWED_SOURCES:
        raise ModelError(
            "Candidate-site source must be 'local' (existing table), "
            "'api' (PHS CKAN lists + Geofabrik OSM), or 'osm' (local lists + Geofabrik OSM).",
            code="invalid_config",
        )
    area = str(area_code).strip() or LOCAL_AUTHORITY_CODE
    if source_key == "local":
        loaded = load_candidate_sites()
        return {
            "status": "ok",
            "source": "local",
            "area_code": area,
            "n_sites": int(len(loaded)),
            "site_types": loaded["site_type"].astype(str).value_counts().to_dict() if "site_type" in loaded.columns else None,
            "output_path": str(find_candidate_site_file()),
            "warnings": [],
        }

    list_source = "api" if source_key == "api" else "local"
    gp_layer = gp_sites if gp_sites is not None else prepare_gp_sites(
        area_code=area,
        gp_path=gp_path,
        osm_place=osm_place,
        list_source=list_source,
        geocode_fn=geocode_fn,
        fetch_fn=fetch_fn,
        use_osm=use_osm,
        ckan_get=ckan_get,
    )
    pharm_layer = pharmacy_sites if pharmacy_sites is not None else prepare_pharmacy_sites(
        area_code=area,
        pharmacy_path=pharmacy_path,
        osm_place=osm_place,
        list_source=list_source,
        geocode_fn=geocode_fn,
        fetch_fn=fetch_fn,
        use_osm=use_osm,
        ckan_get=ckan_get,
    )
    mobile_layer = mobile_sites
    warnings: list[str] = []
    if include_mobile and mobile_layer is None:
        try:
            mobile_layer = prepare_mobile_sites(
                area_code=area,
                osm_place=osm_place,
                fetch_fn=fetch_fn,
            )
        except ModelError as error:
            warnings.append(str(error))
            mobile_layer = None
    layers = [gp_layer, pharm_layer]
    if mobile_layer is not None and len(mobile_layer):
        layers.append(mobile_layer)
    merged = merge_candidate_sites(*layers)
    out_dir = Path(output_dir) if output_dir is not None else candidate_sites_results_dir(area)
    paths = {
        "gp": _write_site_layer(gp_layer, out_dir, "gp_candidate_sites"),
        "pharmacy": _write_site_layer(pharm_layer, out_dir, "pharmacy_candidate_sites"),
        "merged": _write_site_layer(merged, out_dir, "merged_candidate_sites"),
    }
    if mobile_layer is not None and len(mobile_layer):
        paths["mobile"] = _write_site_layer(mobile_layer, out_dir, "car_parks_candidate_sites")
    provenance = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "area_code": area,
        "source": source_key,
        "osm_place": osm_place,
        "gp_list_source": list_source,
        "geofabrik_url": "https://download.geofabrik.de/europe/united-kingdom/scotland-latest-free.shp.zip",
        "parking_source": "overpass",
        "parking_filters": "oasis-v4 access/parking-type/capacity; drop airport lots",
        "gp_ckan": "https://www.opendata.nhs.scot/dataset/gp-practice-contact-details-and-list-sizes/resource/993422a6-c64f-4c57-ba41-9279ad5a7c89",
        "pharmacy_ckan": "https://www.opendata.nhs.scot/dataset/dispenser-location-contact-details/resource/f44e6a10-4f1f-4ffd-9205-956944bacf95",
        "n_gp": int(len(gp_layer)),
        "n_pharmacy": int(len(pharm_layer)),
        "n_mobile": int(len(mobile_layer)) if mobile_layer is not None else 0,
        "n_merged": int(len(merged)),
        "join_key": "iz_code",
        "geography": "2011 Intermediate Zones via DataZone→CA",
        "not_2024_lad_clip": True,
        "mobile_sites_are_provisional": True,
        "raw_not_modified": True,
    }
    provenance_path = write_json(provenance, out_dir / "provenance.json")
    warnings.append("GP/pharmacy coordinates use postcodes.io unless an OSM building name matches (≥0.50 within 500 m).")
    warnings.append(
        "mobile_stop car parks are provisional OSM points after oasis-v4 access/capacity filters, not confirmed clinics."
    )
    return {
        "status": "ok",
        "source": source_key,
        "area_code": area,
        "n_sites": int(len(merged)),
        "n_gp": int(len(gp_layer)),
        "n_pharmacy": int(len(pharm_layer)),
        "n_mobile": provenance["n_mobile"],
        "output_path": paths["merged"],
        "layer_paths": paths,
        "provenance_path": str(provenance_path),
        "warnings": warnings,
    }


def main(argv: list[str] | None = None) -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Load or build candidate intervention sites. City is a parameter.")
    parser.add_argument("--area-code", default=LOCAL_AUTHORITY_CODE)
    parser.add_argument("--source", choices=list(ALLOWED_SOURCES), required=True)
    parser.add_argument("--osm-place", default=None)
    parser.add_argument("--gp-path", default=None)
    parser.add_argument("--pharmacy-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-mobile", action="store_true")
    args = parser.parse_args(argv)
    result = prepare_candidate_sites(
        area_code=args.area_code,
        source=args.source,
        osm_place=args.osm_place,
        gp_path=args.gp_path,
        pharmacy_path=args.pharmacy_path,
        include_mobile=not args.no_mobile,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()

