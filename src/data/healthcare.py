"""GP and pharmacy candidate layers from oasis-v4, with city as a parameter.

Official lists stay local. OSM and postcodes.io are optional geocoding
helpers; they do not invent practices. Area filter uses 2011 DataZone → CA
from Code lookup.csv, not an EH-postcode heuristic.
"""

from __future__ import annotations

import io
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from common.errors import ModelError
from common.utils import LOCAL_AUTHORITY_CODE, get_logger, load_yaml, project_root, read_table, write_json, write_table

LOGGER = get_logger("data.healthcare")

PAGE_SIZE = 10000
USER_AGENT = "oasis-geoai-agent/0.1 (OASIS 2026 Track B)"
ALLOWED_LIST_SOURCES = ("api", "local")
CKAN_ATTEMPTS = 4

BNG_CRS = "EPSG:27700"
LOOKUP_PATH = Path("data") / "raw" / "boundaries" / "Code lookup.csv"
GP_DIR = Path("data") / "raw" / "gp"
OSM_MATCH_RADIUS_M = 500.0
OSM_MATCH_MIN_SCORE = 0.50
GP_NAME_DROP = ("practice", "medical", "centre", "center", "surgery")
PHARM_NAME_DROP = ("pharmacy", "chemist", "ltd", "limited")
GP_PREFERRED = (
    "GP Practices and List sizes January 2023 .csv",
    "gp_practices.csv",
    "gp.csv",
)
PHARMACY_PREFERRED = (
    "dispenser_contactdetails_jan2023.csv",
    "pharmacy.csv",
    "pharmacies.csv",
)
GP_OSM_TAGS = {"amenity": "doctors", "healthcare": ["doctor", "physician"]}
PHARMACY_OSM_TAGS = {"amenity": "pharmacy", "healthcare": "pharmacy"}
SITE_CORE = (
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
)


def gp_dir() -> Path:
    return project_root() / GP_DIR


def load_code_lookup() -> pd.DataFrame:
    path = project_root() / LOOKUP_PATH
    if not path.exists():
        raise ModelError(f"2011 code lookup missing: {path}", code="missing_dataset")
    lookup = read_table(path)
    needed = {"DataZone", "IntZone", "CA", "CAName", "HSCP", "HSCPName"}
    missing = needed - set(lookup.columns)
    if missing:
        raise ModelError(f"Code lookup missing columns {sorted(missing)}.", code="invalid_config")
    out = lookup.copy()
    for column in needed:
        out[column] = out[column].astype("string").str.strip()
    return out


def area_lookup_slice(area_code: str) -> pd.DataFrame:
    lookup = load_code_lookup()
    area = str(area_code).strip()
    sliced = lookup.loc[lookup["CA"] == area].copy()
    if sliced.empty:
        raise ModelError(
            f"No 2011 Data Zones found for CA={area}. Do not invent a city filter.",
            code="missing_dataset",
        )
    return sliced


def resolve_osm_place(area_code: str, osm_place: str | None) -> str:
    from data.osm_extract import geofabrik_ready
    from data.travel_time import OSM_PLACE_BY_AREA

    place = str(osm_place).strip() if osm_place else ""
    if place:
        return place
    mapped = OSM_PLACE_BY_AREA.get(str(area_code).strip())
    if mapped:
        return mapped
    if geofabrik_ready():
        return f"CA={str(area_code).strip()}"
    raise ModelError(
        "osm_place is required for this area_code; a city name will not be invented.",
        code="invalid_config",
        details={"area_code": area_code},
    )


def _find_named_file(directory: Path, preferred: Sequence[str], kind: str) -> Path:
    if not directory.exists():
        raise ModelError(
            f"{kind} folder missing: {directory}. Do not invent site lists.",
            code="missing_dataset",
        )
    for name in preferred:
        path = directory / name
        if path.exists():
            return path
    files = [
        path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() == ".csv" and path.name not in {".gitkeep", ".DS_Store"}
    ]
    if len(files) == 1:
        return files[0]
    if not files:
        raise ModelError(
            f"No {kind} CSV in {directory}. Do not invent site lists.",
            code="missing_dataset",
        )
    raise ModelError(
        f"Multiple {kind} CSVs found; pass an explicit path.",
        code="invalid_config",
        details={"files": [path.name for path in files]},
    )


def find_gp_file(path: str | Path | None = None) -> Path:
    if path is not None:
        file_path = Path(path)
        if not file_path.exists():
            raise ModelError(f"GP table missing: {file_path}", code="missing_dataset")
        return file_path
    return _find_named_file(gp_dir(), GP_PREFERRED, "GP")


def find_pharmacy_file(path: str | Path | None = None) -> Path:
    if path is not None:
        file_path = Path(path)
        if not file_path.exists():
            raise ModelError(f"Pharmacy table missing: {file_path}", code="missing_dataset")
        return file_path
    return _find_named_file(gp_dir(), PHARMACY_PREFERRED, "pharmacy")


def _urlopen_with_retry(request: urllib.request.Request, timeout: int = 60, attempts: int = CKAN_ATTEMPTS):
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return urllib.request.urlopen(request, timeout=timeout)
        except (ConnectionResetError, TimeoutError, urllib.error.URLError, OSError) as exc:
            last = exc
            LOGGER.warning("HTTP attempt %s/%s failed: %s", attempt, attempts, exc)
            if attempt < attempts:
                time.sleep(1.5 * attempt)
    assert last is not None
    raise last


def _ckan_get(url: str, params: dict[str, Any]) -> dict[str, Any]:
    query = urllib.parse.urlencode(params)
    request = urllib.request.Request(
        f"{url}?{query}",
        headers={"User-Agent": USER_AGENT},
    )
    with _urlopen_with_retry(request) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_ckan_csv(download_url: str, *, csv_get: Callable | None = None) -> pd.DataFrame:
    """Download the registered PHS resource CSV. Same official extract as datastore_search."""
    if csv_get is not None:
        return csv_get(download_url)
    request = urllib.request.Request(download_url, headers={"User-Agent": USER_AGENT})
    with _urlopen_with_retry(request, timeout=120) as response:
        payload = response.read()
    return pd.read_csv(io.BytesIO(payload))


def fetch_ckan_resource(
    resource_id: str,
    *,
    base_url: str | None = None,
    ckan_get: Callable | None = None,
) -> pd.DataFrame:
    """Page through a PHS CKAN datastore. Does not invent missing rows."""
    sources = load_yaml("configs/data.yaml")
    endpoint = (base_url or sources["ckan"]["base_url"]).rstrip("/") + "/datastore_search"
    getter = ckan_get or _ckan_get
    rows: list[dict[str, Any]] = []
    offset = 0
    total = None
    while True:
        payload = getter(
            endpoint,
            {
                "resource_id": resource_id,
                "limit": PAGE_SIZE,
                "offset": offset,
            },
        )
        if not payload.get("success"):
            raise ModelError(
                f"CKAN datastore_search unsuccessful for {resource_id}.",
                code="missing_dataset",
            )
        result = payload["result"]
        records = result.get("records") or []
        total = result.get("total", total)
        rows.extend(records)
        offset += len(records)
        if not records or (total is not None and offset >= total) or len(records) < PAGE_SIZE:
            break
    if not rows:
        raise ModelError(f"CKAN returned no rows for {resource_id}.", code="missing_dataset")
    frame = pd.DataFrame(rows)
    if "_id" in frame.columns:
        frame = frame.drop(columns=["_id"])
    return frame


def acquire_healthcare_table(
    kind: str,
    source: str = "api",
    *,
    path: str | Path | None = None,
    ckan_get: Callable | None = None,
    csv_get: Callable | None = None,
) -> dict[str, Any]:
    """Fetch GP or pharmacy contact details. api = PHS CKAN; local = data/raw/gp."""
    kind_key = str(kind).strip().lower()
    if kind_key not in {"gp", "pharmacy"}:
        raise ModelError("Healthcare kind must be 'gp' or 'pharmacy'.", code="invalid_config")
    source_key = str(source).strip().lower()
    if source_key not in ALLOWED_LIST_SOURCES:
        raise ModelError(
            "Healthcare source must be 'api' (PHS CKAN) or 'local' (data/raw/gp). "
            "The source is not switched silently.",
            code="invalid_config",
        )
    spec = load_yaml("configs/data.yaml")["healthcare_resources"][kind_key]
    if source_key == "local":
        file_path = find_gp_file(path) if kind_key == "gp" else find_pharmacy_file(path)
        frame = pd.read_csv(file_path)
        retrieval = "local_raw"
        output_path = str(file_path)
    else:
        retrieval = "ckan_datastore_search"
        try:
            frame = fetch_ckan_resource(spec["resource_id"], ckan_get=ckan_get)
        except Exception as exc:
            LOGGER.warning("CKAN datastore_search failed for %s (%s); trying resource CSV.", kind_key, exc)
            try:
                frame = fetch_ckan_csv(spec["download_url"], csv_get=csv_get)
                retrieval = "ckan_csv_download"
            except Exception as csv_exc:
                raise ModelError(
                    f"API retrieval failed for {kind_key}. Local files were not used because "
                    f"source='api'. Re-run with source='local'. Original error: {csv_exc}",
                    code="missing_dataset",
                ) from csv_exc
        out_dir = project_root() / "data" / "results" / "healthcare"
        output_path = str(write_table(frame, out_dir / spec["local_filename"]))
    record = {
        "kind": kind_key,
        "chosen_source": source_key,
        "retrieval": retrieval,
        "resource_id": spec["resource_id"],
        "resource_name": spec["resource_name"],
        "dataset_url": spec["dataset_url"],
        "resource_url": spec["resource_url"],
        "licence": "Open Government Licence",
        "query_date": datetime.now(timezone.utc).date().isoformat(),
        "n_rows": int(len(frame)),
        "output_path": output_path,
        "raw_not_modified": True,
    }
    provenance_path = write_json(
        {"recorded_at": datetime.now(timezone.utc).isoformat(), "record": record},
        project_root() / "data" / "results" / "healthcare" / f"{kind_key}_provenance.json",
    )
    LOGGER.info("Acquired %s %s rows via %s.", len(frame), kind_key, retrieval)
    return {"frame": frame, "provenance": record, "provenance_path": str(provenance_path)}


def attach_iz_from_datazone(
    frame: pd.DataFrame,
    datazone_column: str,
    area_code: str,
) -> pd.DataFrame:
    """Join 2011 DataZone → IntZone and keep rows in the requested CA."""
    if datazone_column not in frame.columns:
        raise ModelError(
            f"Table must contain {datazone_column} to join 2011 IZs.",
            code="invalid_config",
        )
    sliced = area_lookup_slice(area_code)
    dz_to_iz = sliced[["DataZone", "IntZone"]].drop_duplicates()
    out = frame.copy()
    out["_dz"] = out[datazone_column].astype("string").str.strip()
    out = out.merge(dz_to_iz, left_on="_dz", right_on="DataZone", how="left")
    kept = out.loc[out["IntZone"].notna()].copy()
    dropped = int(len(out) - len(kept))
    if kept.empty:
        raise ModelError(
            f"No rows remain after filtering {datazone_column} to CA={area_code}.",
            code="missing_dataset",
        )
    if dropped:
        LOGGER.info("Dropped %s rows whose DataZone is outside CA=%s.", dropped, area_code)
    kept["iz_code"] = kept["IntZone"].astype("string").str.strip()
    return kept.drop(columns=["_dz"], errors="ignore")


def attach_iz_centroid_xy(frame: pd.DataFrame, area_code: str) -> pd.DataFrame:
    from data.travel_time import load_iz_origins

    origins = load_iz_origins(area_code=area_code)
    coords = pd.DataFrame(
        {
            "iz_code": origins["iz_code"].astype("string").str.strip(),
            "iz_easting": origins.geometry.x.round(2),
            "iz_northing": origins.geometry.y.round(2),
        }
    )
    out = frame.copy()
    out["iz_code"] = out["iz_code"].astype("string").str.strip()
    return out.merge(coords, on="iz_code", how="left")


def clean_site_name(text: Any, drop_tokens: Iterable[str]) -> str:
    cleaned = re.sub(r"[^\w\s]", "", str(text)).lower()
    for token in drop_tokens:
        cleaned = cleaned.replace(token, "")
    return cleaned.strip()


def geocode_postcode_uk(postcode: str, get_json: Callable | None = None) -> tuple[float, float]:
    """postcodes.io easting/northing. Failures stay missing; they are not invented."""
    import requests

    clean_pc = str(postcode).replace(" ", "")
    if not clean_pc or clean_pc.lower() == "nan":
        return np.nan, np.nan
    url = f"https://api.postcodes.io/postcodes/{clean_pc}"
    try:
        if get_json is not None:
            payload = get_json(clean_pc)
        else:
            response = requests.get(url, timeout=5)
            if response.status_code != 200:
                return np.nan, np.nan
            payload = response.json()
        result = payload.get("result") or {}
        easting, northing = result.get("eastings"), result.get("northings")
        if easting is None or northing is None:
            return np.nan, np.nan
        return float(easting), float(northing)
    except Exception:
        return np.nan, np.nan


def geocode_postcode_column(
    series: pd.Series,
    geocode_fn: Callable[[str], tuple[float, float]] | None = None,
) -> tuple[pd.Series, pd.Series]:
    coder = geocode_fn or geocode_postcode_uk
    unique_pcs = [pc for pc in series.dropna().astype(str).unique() if pc.strip()]
    LOGGER.info("Geocoding %s unique postcodes.", len(unique_pcs))
    mapping = {pc: coder(pc) for pc in unique_pcs}
    easting = series.map(lambda pc: mapping.get(str(pc), (np.nan, np.nan))[0] if pd.notna(pc) else np.nan)
    northing = series.map(lambda pc: mapping.get(str(pc), (np.nan, np.nan))[1] if pd.notna(pc) else np.nan)
    return easting, northing


def fetch_osm_named_points(
    osm_place: str,
    tags: dict[str, Any],
    fetch_fn: Callable | None = None,
):
    import geopandas as gpd

    if fetch_fn is not None:
        raw = fetch_fn(osm_place=osm_place, tags=tags)
        if raw is None:
            return gpd.GeoDataFrame(columns=["name", "geometry"], crs=BNG_CRS)
        return raw

    amenity = tags.get("amenity")
    fclasses = None
    if amenity == "doctors":
        from data.osm_extract import GP_FCLASSES, load_geofabrik_pois

        fclasses = GP_FCLASSES
        loader = load_geofabrik_pois
    elif amenity == "pharmacy":
        from data.osm_extract import PHARMACY_FCLASSES, load_geofabrik_pois

        fclasses = PHARMACY_FCLASSES
        loader = load_geofabrik_pois
    else:
        loader = None
    if loader is not None:
        try:
            return loader(fclasses)
        except ModelError as error:
            LOGGER.warning("Geofabrik OSM extract unavailable (%s); not calling osmnx silently.", error)

    try:
        import osmnx as ox
    except ImportError as exc:
        raise ModelError(
            "OSM features need the Geofabrik Scotland extract under data/raw/roads/scotland-free.shp "
            "or osmnx. Install osmnx only if the Geofabrik shapefile is absent.",
            code="missing_dependency",
        ) from exc
    LOGGER.info("Fetching OSM features for %s tags=%s.", osm_place, tags)
    if hasattr(ox, "features_from_place"):
        gdf = ox.features_from_place(osm_place, tags=tags)
    else:
        gdf = ox.geometries_from_place(osm_place, tags=tags)
    gdf = gdf.copy()
    gdf["geometry"] = gdf.geometry.centroid
    if "name" not in gdf.columns:
        gdf["name"] = ""
    else:
        gdf["name"] = gdf["name"].fillna("")
    return gdf[["name", "geometry"]].to_crs(epsg=27700)


def match_osm_buildings(
    names: Sequence[str],
    pc_easting: Sequence[Any],
    pc_northing: Sequence[Any],
    osm_points,
    drop_tokens: Iterable[str],
    radius_m: float = OSM_MATCH_RADIUS_M,
    min_score: float = OSM_MATCH_MIN_SCORE,
) -> tuple[list[Any], list[Any], list[str]]:
    """Notebook rule: OSM building if name similarity ≥ 0.50 within 500 m, else postcode."""
    empty = osm_points is None or len(osm_points) == 0
    eastings: list[Any] = []
    northings: list[Any] = []
    sources: list[str] = []
    for name, pc_east, pc_north in zip(names, pc_easting, pc_northing):
        best_coords = None
        best_score = 0.0
        pc_east_f = pd.to_numeric(pd.Series([pc_east]), errors="coerce").iloc[0]
        pc_north_f = pd.to_numeric(pd.Series([pc_north]), errors="coerce").iloc[0]
        if not empty and pd.notna(pc_east_f) and pd.notna(pc_north_f):
            dx = osm_points.geometry.x - float(pc_east_f)
            dy = osm_points.geometry.y - float(pc_north_f)
            nearby = osm_points.loc[np.sqrt(dx**2 + dy**2) <= radius_m]
            query = clean_site_name(name, drop_tokens)
            for _, osm_row in nearby.iterrows():
                osm_name = clean_site_name(osm_row.get("name", ""), drop_tokens)
                if not osm_name:
                    continue
                score = SequenceMatcher(None, query, osm_name).ratio()
                if score > best_score:
                    best_score = score
                    best_coords = (osm_row.geometry.x, osm_row.geometry.y)
        if best_score >= min_score and best_coords is not None:
            eastings.append(round(float(best_coords[0]), 2))
            northings.append(round(float(best_coords[1]), 2))
            sources.append("OSM_exact_building")
        else:
            eastings.append(pc_east_f)
            northings.append(pc_north_f)
            sources.append("Postcode_centroid")
    return eastings, northings, sources


def _finalise_sites(
    frame: pd.DataFrame,
    site_id_prefix: str,
    id_column: str,
    name_column: str,
    site_type: str,
) -> pd.DataFrame:
    out = frame.copy()
    out["site_id"] = site_id_prefix + out[id_column].astype(str)
    out["site_name"] = out[name_column]
    out["site_type"] = site_type
    out["suitability"] = "confirmed"
    extra = [column for column in out.columns if column not in SITE_CORE]
    return out[list(SITE_CORE) + extra]


def prepare_gp_sites(
    area_code: str = LOCAL_AUTHORITY_CODE,
    gp_path: str | Path | None = None,
    osm_place: str | None = None,
    *,
    gp_table: pd.DataFrame | None = None,
    list_source: str = "local",
    osm_points=None,
    geocode_fn: Callable | None = None,
    fetch_fn: Callable | None = None,
    use_osm: bool = True,
    ckan_get: Callable | None = None,
) -> pd.DataFrame:
    if gp_table is not None:
        raw = gp_table
    elif str(list_source).strip().lower() == "api":
        raw = acquire_healthcare_table("gp", source="api", ckan_get=ckan_get)["frame"]
    else:
        raw = pd.read_csv(find_gp_file(gp_path))
    for column in ("PracticeCode", "GPPracticeName", "Postcode", "DataZone"):
        if column not in raw.columns:
            raise ModelError(f"GP table must contain {column}.", code="invalid_config")
    kept = attach_iz_from_datazone(raw, "DataZone", area_code)
    kept = attach_iz_centroid_xy(kept, area_code)
    easting, northing = geocode_postcode_column(kept["Postcode"], geocode_fn=geocode_fn)
    kept["pc_easting"] = easting
    kept["pc_northing"] = northing
    osm = osm_points
    if use_osm and osm is None:
        place = resolve_osm_place(area_code, osm_place)
        osm = fetch_osm_named_points(place, GP_OSM_TAGS, fetch_fn=fetch_fn)
    eastings, northings, sources = match_osm_buildings(
        kept["GPPracticeName"].tolist(),
        kept["pc_easting"].tolist(),
        kept["pc_northing"].tolist(),
        osm,
        GP_NAME_DROP,
    )
    kept["easting"] = eastings
    kept["northing"] = northings
    kept["coord_source"] = sources
    LOGGER.info("Prepared %s GP sites for CA=%s.", len(kept), area_code)
    return _finalise_sites(kept, "GP_", "PracticeCode", "GPPracticeName", "gp")


def prepare_pharmacy_sites(
    area_code: str = LOCAL_AUTHORITY_CODE,
    pharmacy_path: str | Path | None = None,
    osm_place: str | None = None,
    *,
    pharmacy_table: pd.DataFrame | None = None,
    list_source: str = "local",
    osm_points=None,
    geocode_fn: Callable | None = None,
    fetch_fn: Callable | None = None,
    use_osm: bool = True,
    ckan_get: Callable | None = None,
) -> pd.DataFrame:
    if pharmacy_table is not None:
        raw = pharmacy_table
    elif str(list_source).strip().lower() == "api":
        raw = acquire_healthcare_table("pharmacy", source="api", ckan_get=ckan_get)["frame"]
    else:
        raw = pd.read_csv(find_pharmacy_file(pharmacy_path))
    dz_col = "datazone2011" if "datazone2011" in raw.columns else "DataZone"
    for column in ("DispCode", "DispLocationName", "DispLocationPostcode", dz_col):
        if column not in raw.columns:
            raise ModelError(f"Pharmacy table must contain {column}.", code="invalid_config")
    kept = attach_iz_from_datazone(raw, dz_col, area_code)
    kept = attach_iz_centroid_xy(kept, area_code)
    easting, northing = geocode_postcode_column(kept["DispLocationPostcode"], geocode_fn=geocode_fn)
    kept["pc_easting"] = easting
    kept["pc_northing"] = northing
    osm = osm_points
    if use_osm and osm is None:
        place = resolve_osm_place(area_code, osm_place)
        osm = fetch_osm_named_points(place, PHARMACY_OSM_TAGS, fetch_fn=fetch_fn)
    eastings, northings, sources = match_osm_buildings(
        kept["DispLocationName"].tolist(),
        kept["pc_easting"].tolist(),
        kept["pc_northing"].tolist(),
        osm,
        PHARM_NAME_DROP,
    )
    kept["easting"] = eastings
    kept["northing"] = northings
    kept["coord_source"] = sources
    LOGGER.info("Prepared %s pharmacy sites for CA=%s.", len(kept), area_code)
    return _finalise_sites(kept, "PH_", "DispCode", "DispLocationName", "pharmacy")


def load_healthcare_layers(
    area_code: str = LOCAL_AUTHORITY_CODE,
    gp_path: str | Path | None = None,
    pharmacy_path: str | Path | None = None,
) -> dict[str, Any]:
    """Read official GP/pharmacy extracts. Does not invent catchments."""
    payload: dict[str, Any] = {"area_code": area_code, "layers": {}}
    try:
        gp_file = find_gp_file(gp_path)
        payload["layers"]["gp"] = {"path": str(gp_file), "n_rows": int(len(pd.read_csv(gp_file)))}
    except ModelError as error:
        payload["layers"]["gp"] = {"error": str(error)}
    try:
        pharm_file = find_pharmacy_file(pharmacy_path)
        payload["layers"]["pharmacy"] = {
            "path": str(pharm_file),
            "n_rows": int(len(pd.read_csv(pharm_file))),
        }
    except ModelError as error:
        payload["layers"]["pharmacy"] = {"error": str(error)}
    return payload
