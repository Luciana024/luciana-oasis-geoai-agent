"""OpenStreetMap extracts for candidate sites.

Doctors/pharmacies: Geofabrik Scotland shapefile already unpacked under
data/raw/roads/scotland-free.shp
(https://download.geofabrik.de/europe/united-kingdom/scotland-latest-free.shp.zip).

Parking: Overpass API with OSM tags (access, parking, capacity). The Geofabrik
shapefile cannot be used for oasis-v4 public-car-park filters.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterable

from common.errors import ModelError
from common.utils import LOCAL_AUTHORITY_CODE, get_logger, project_root

LOGGER = get_logger("data.osm_extract")

GEOFABRIK_URL = (
    "https://download.geofabrik.de/europe/united-kingdom/scotland-latest-free.shp.zip"
)
GEOFABRIK_DIR = (
    Path("data") / "raw" / "roads" / "scotland-free.shp"
)
BOUNDARY_SHP = (
    Path("data")
    / "raw"
    / "boundaries"
    / "SG_IntermediateZoneBdry_2011"
    / "SG_IntermediateZone_Bdry_2011.shp"
)
POI_LAYERS = ("gis_osm_pois_free_1.shp", "gis_osm_pois_a_free_1.shp")
PARKING_LAYERS = ("gis_osm_traffic_free_1.shp", "gis_osm_traffic_a_free_1.shp")
GP_FCLASSES = frozenset({"doctors"})
PHARMACY_FCLASSES = frozenset({"pharmacy"})
PARKING_FCLASSES = frozenset({"parking"})
BNG_CRS = "EPSG:27700"
WGS84_CRS = "EPSG:4326"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OVERPASS_MIRRORS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
)
USER_AGENT = "oasis-geoai-agent/0.1 (OASIS 2026 Track B)"
PARKING_TAG_FIELDS = (
    "name",
    "access",
    "parking",
    "capacity",
    "park_and_ride",
    "fee",
    "operator",
    "surface",
)


def geofabrik_dir() -> Path:
    return project_root() / GEOFABRIK_DIR


def geofabrik_ready() -> bool:
    folder = geofabrik_dir()
    return folder.exists() and (folder / "gis_osm_pois_free_1.shp").exists()


def _require_geofabrik() -> Path:
    folder = geofabrik_dir()
    if not (folder / "gis_osm_pois_free_1.shp").exists():
        raise ModelError(
            "Geofabrik Scotland OSM shapefile is missing. "
            f"Place scotland-latest-free.shp.zip extract under {folder}. "
            f"Source: {GEOFABRIK_URL}. The archive is not downloaded automatically.",
            code="missing_dataset",
            details={"url": GEOFABRIK_URL, "expected": str(folder)},
        )
    return folder


def _read_fclass_layers(layer_names: Iterable[str], fclasses: Iterable[str]):
    import geopandas as gpd

    folder = _require_geofabrik()
    wanted = set(fclasses)
    frames = []
    for name in layer_names:
        path = folder / name
        if not path.exists():
            continue
        frame = gpd.read_file(path)
        if "fclass" not in frame.columns:
            continue
        kept = frame.loc[frame["fclass"].astype(str).isin(wanted)].copy()
        if kept.empty:
            continue
        if kept.crs is None:
            raise ModelError("Geofabrik layer has no CRS; refusing to guess.", code="invalid_config")
        kept = kept.to_crs(BNG_CRS)
        kept["geometry"] = kept.geometry.centroid
        if "name" not in kept.columns:
            kept["name"] = ""
        else:
            kept["name"] = kept["name"].fillna("")
        frames.append(kept)
    if not frames:
        raise ModelError(
            f"Geofabrik layers have no features in {sorted(wanted)}.",
            code="missing_dataset",
        )
    import pandas as pd

    out = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=BNG_CRS)
    LOGGER.info("Loaded %s Geofabrik features for %s.", len(out), sorted(wanted))
    return out[["name", "geometry"]].copy()


def load_geofabrik_pois(fclasses: Iterable[str]):
    """Named POI points (doctors, pharmacy, …) in British National Grid."""
    return _read_fclass_layers(POI_LAYERS, fclasses)


def load_geofabrik_parking():
    """Parking geometry only. Lacks access/capacity — do not use for oasis-v4 filters."""
    raise ModelError(
        "Geofabrik shapefiles have no access/capacity/parking-type tags. "
        "Car parks must be fetched from the OSM Overpass API so oasis-v4 filters can run.",
        code="invalid_config",
        details={"url": GEOFABRIK_URL},
    )


def load_area_iz_polygons(area_code: str = LOCAL_AUTHORITY_CODE):
    """2011 IZ polygons for one CA. Not a 2024 LAD clip."""
    import geopandas as gpd

    from data.covid import load_iz_master

    path = project_root() / BOUNDARY_SHP
    if not path.exists():
        raise ModelError(f"2011 IZ boundary shapefile missing: {path}", code="missing_dataset")
    raw = gpd.read_file(path)
    code_col = next(
        (name for name in ("IntZone", "InterZone", "IZ_CODE", "iz_code") if name in raw.columns),
        None,
    )
    if code_col is None:
        raise ModelError("IZ boundary shapefile has no IntZone column.", code="invalid_config")
    master = load_iz_master(area_code=area_code)
    needed = set(master["IntZone"].astype("string").str.strip())
    polys = raw.rename(columns={code_col: "iz_code"}).copy()
    polys["iz_code"] = polys["iz_code"].astype("string").str.strip()
    polys = polys.loc[polys["iz_code"].isin(needed)].copy()
    if polys.empty:
        raise ModelError(f"No 2011 IZ polygons for CA={area_code}.", code="missing_dataset")
    if polys.crs is None:
        raise ModelError("IZ polygons have no CRS; refusing to guess.", code="invalid_config")
    return gpd.GeoDataFrame(polys, geometry="geometry").to_crs(BNG_CRS)[["iz_code", "geometry"]]


def clip_to_area(frame, area_code: str = LOCAL_AUTHORITY_CODE):
    """Keep features that intersect the CA's 2011 IZ polygons."""
    import geopandas as gpd

    polys = load_area_iz_polygons(area_code)
    gdf = gpd.GeoDataFrame(frame).to_crs(BNG_CRS)
    try:
        union = polys.geometry.union_all()
    except AttributeError:
        union = polys.geometry.unary_union
    clipped = gdf.loc[gdf.intersects(union)].copy()
    LOGGER.info("Clipped %s → %s features for CA=%s.", len(gdf), len(clipped), area_code)
    return clipped


def area_wgs84_bbox(area_code: str = LOCAL_AUTHORITY_CODE, buffer_deg: float = 0.02) -> tuple[float, float, float, float]:
    """south, west, north, east for Overpass, from 2011 IZ polygons."""
    polys = load_area_iz_polygons(area_code).to_crs(WGS84_CRS)
    minx, miny, maxx, maxy = polys.total_bounds
    return (
        float(miny - buffer_deg),
        float(minx - buffer_deg),
        float(maxy + buffer_deg),
        float(maxx + buffer_deg),
    )


def parking_overpass_query(south: float, west: float, north: float, east: float) -> str:
    return (
        "[out:json][timeout:180];\n"
        f'(nwr["amenity"="parking"]({south},{west},{north},{east}););\n'
        "out center tags;\n"
    )


def _overpass_post(query: str) -> dict[str, Any]:
    last: Exception | None = None
    for attempt, url in enumerate(OVERPASS_MIRRORS * 2, start=1):
        request = urllib.request.Request(
            url,
            data=urllib.parse.urlencode({"data": query}).encode("utf-8"),
            headers={"User-Agent": USER_AGENT, "Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ConnectionResetError, OSError) as exc:
            last = exc
            LOGGER.warning("Overpass attempt %s at %s failed: %s", attempt, url, exc)
            time.sleep(2.0 * attempt)
    assert last is not None
    raise last


def _element_point(element: dict[str, Any]) -> tuple[float, float] | None:
    if element.get("type") == "node" and "lat" in element and "lon" in element:
        return float(element["lat"]), float(element["lon"])
    center = element.get("center") or {}
    if "lat" in center and "lon" in center:
        return float(center["lat"]), float(center["lon"])
    return None


def overpass_elements_to_parking_gdf(payload: dict[str, Any]):
    """Build a tagged parking GeoDataFrame from an Overpass JSON payload."""
    import geopandas as gpd
    from shapely.geometry import Point

    elements = payload.get("elements") or []
    rows: list[dict[str, Any]] = []
    geoms = []
    for element in elements:
        point = _element_point(element)
        if point is None:
            continue
        tags = element.get("tags") or {}
        if tags.get("amenity") not in (None, "parking"):
            continue
        row = {field: tags.get(field) for field in PARKING_TAG_FIELDS}
        row["osm_id"] = element.get("id")
        row["osm_type"] = element.get("type")
        rows.append(row)
        geoms.append(Point(point[1], point[0]))
    if not rows:
        raise ModelError("Overpass returned no parking geometries.", code="missing_dataset")
    gdf = gpd.GeoDataFrame(rows, geometry=geoms, crs=WGS84_CRS)
    LOGGER.info("Parsed %s tagged OSM parking features from Overpass.", len(gdf))
    return gdf


def fetch_overpass_parking(
    area_code: str = LOCAL_AUTHORITY_CODE,
    *,
    overpass_post: Callable[[str], dict[str, Any]] | None = None,
    bbox: tuple[float, float, float, float] | None = None,
):
    """Parking with OSM tags (access, parking, capacity). Not the Geofabrik shapefile."""
    south, west, north, east = bbox if bbox is not None else area_wgs84_bbox(area_code)
    query = parking_overpass_query(south, west, north, east)
    LOGGER.info("Querying Overpass parking for CA=%s bbox=%s.", area_code, (south, west, north, east))
    try:
        payload = (overpass_post or _overpass_post)(query)
    except Exception as exc:
        raise ModelError(
            f"Overpass parking query failed. Geofabrik shapefiles cannot substitute because they lack tags. {exc}",
            code="missing_dataset",
        ) from exc
    return overpass_elements_to_parking_gdf(payload)
