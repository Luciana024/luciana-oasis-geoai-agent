"""IZ-to-site travel-time matrix. Not the road-graph kilometre adjacency.

Extracted from oasis-v4 (OSMnx + NetworkX shortest path) with the city made
a parameter. Speeds are the notebook's assumed urban averages, not official
TTW or GTFS.

    source='local'  read an existing matrix
    source='osm'    download the OSM graph for this city and compute once

Outputs go to data/results/travel_time/<area_code>/. data/raw is not modified.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from common.errors import ModelError
from common.utils import (
    EXPECTED_IZ_COUNT,
    LOCAL_AUTHORITY_CODE,
    get_logger,
    project_root,
    write_json,
    write_table,
)
from data.covid import load_iz_master

LOGGER = get_logger("data.travel_time")

MATRIX_COLUMNS = ("iz_code", "site_id", "mode", "travel_time_min")
ALLOWED_SOURCES = ("local", "osm")
BNG_CRS = "EPSG:27700"
WGS84_CRS = "EPSG:4326"
NETWORK_BUFFER_M = 2000.0
DRIVE_SPEED_KMH = 30.0
WALK_SPEED_KMH = 4.5
DEFAULT_MODES: dict[str, dict[str, Any]] = {
    "drive": {"network_type": "drive", "speed_kmh": DRIVE_SPEED_KMH},
    "walk": {"network_type": "walk", "speed_kmh": WALK_SPEED_KMH},
}
OSM_PLACE_BY_AREA = {
    LOCAL_AUTHORITY_CODE: "City of Edinburgh, UK",
}
IZ_CODE_CANDIDATES = ("iz_code", "IntZone", "InterZone", "IZ_CODE")
CENTROID_SHP = (
    Path("data")
    / "raw"
    / "boundaries"
    / "SG_IntermediateZoneCent_2011"
    / "SG_IntermediateZone_Cent_2011.shp"
)
RAW_MATRIX_NAMES = ("travel_time_matrix.csv", "travel_time_matrix.parquet")


def travel_time_raw_dir() -> Path:
    return project_root() / "data" / "raw" / "travel_time"


def travel_time_results_dir(area_code: str) -> Path:
    return project_root() / "data" / "results" / "travel_time" / _safe_area_token(area_code)


def _safe_area_token(area_code: str) -> str:
    token = str(area_code).strip() or "unknown_area"
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in token)


def load_iz_origins(
    area_code: str = LOCAL_AUTHORITY_CODE,
    iz_path: str | Path | None = None,
):
    """Origins are official 2011 IZ centroids filtered by CA, not a 2024 LAD clip."""
    import geopandas as gpd
    from shapely.geometry import Point

    if iz_path is not None:
        path = Path(iz_path)
        if not path.exists():
            raise ModelError(f"IZ origin file missing: {path}", code="missing_dataset")
        frame = gpd.read_file(path)
        return _normalise_origins(frame, source=str(path), area_code=area_code)

    master = load_iz_master(area_code=area_code)
    shp = project_root() / CENTROID_SHP
    if not shp.exists():
        raise ModelError(f"2011 IZ centroid shapefile missing: {shp}", code="missing_dataset")
    raw = gpd.read_file(shp)
    code_col = next((name for name in IZ_CODE_CANDIDATES if name in raw.columns), None)
    if code_col is None:
        raise ModelError(
            f"Centroid shapefile must contain an IZ code among {list(IZ_CODE_CANDIDATES)}.",
            code="invalid_config",
        )
    points = raw.rename(columns={code_col: "iz_code"}).copy()
    points["iz_code"] = points["iz_code"].astype("string").str.strip()
    needed = master.rename(columns={"IntZone": "iz_code"})[["iz_code"]].copy()
    needed["iz_code"] = needed["iz_code"].astype("string").str.strip()
    merged = needed.merge(points, on="iz_code", how="left", validate="one_to_one")
    gdf = gpd.GeoDataFrame(merged, geometry="geometry")
    missing = gdf.loc[gdf.geometry.isna() | gdf.geometry.is_empty, "iz_code"].astype(str).tolist()
    if missing:
        raise ModelError(
            f"{len(missing)} IZs for CA={area_code} are absent from the 2011 centroid file: {missing[:10]}.",
            code="missing_dataset",
        )
    if gdf.crs is None:
        if "Easting" in gdf.columns and "Northing" in gdf.columns:
            gdf = gpd.GeoDataFrame(
                gdf,
                geometry=[
                    Point(x, y)
                    for x, y in zip(
                        pd.to_numeric(gdf["Easting"], errors="coerce"),
                        pd.to_numeric(gdf["Northing"], errors="coerce"),
                    )
                ],
                crs=BNG_CRS,
            )
        else:
            raise ModelError("IZ centroids have no CRS; refusing to guess.", code="invalid_config")
    gdf = gdf.to_crs(BNG_CRS)
    if area_code == LOCAL_AUTHORITY_CODE and len(gdf) != EXPECTED_IZ_COUNT:
        raise ModelError(
            f"Edinburgh IZ origin count is {len(gdf)}, expected {EXPECTED_IZ_COUNT}.",
            code="node_order_mismatch",
        )
    LOGGER.info("Loaded %s IZ origins for CA=%s.", len(gdf), area_code)
    return gdf[["iz_code", "geometry"]].copy()


def export_iz_origins(
    area_code: str = LOCAL_AUTHORITY_CODE,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Write 2011 IZ centroids for one CA. Replaces the notebook's 2024 LAD clip."""
    origins = load_iz_origins(area_code=area_code)
    token = _safe_area_token(area_code)
    out_dir = Path(output_dir) if output_dir is not None else (
        project_root() / "data" / "results" / "boundaries" / token
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(
        {
            "iz_code": origins["iz_code"].astype(str),
            "iz_easting": origins.geometry.x.round(2),
            "iz_northing": origins.geometry.y.round(2),
        }
    )
    csv_path = write_table(table, out_dir / "iz_origins.csv")
    geojson_path = out_dir / "iz_origins.geojson"
    origins.to_file(geojson_path)
    LOGGER.info("Wrote %s IZ origins for CA=%s.", len(origins), area_code)
    return {
        "status": "ok",
        "area_code": area_code,
        "n_iz": int(len(origins)),
        "output_path": str(csv_path),
        "geojson_path": str(geojson_path),
        "not_2024_lad_clip": True,
        "geography": "2011 Intermediate Zones",
    }


def _normalise_origins(frame, source: str, area_code: str):
    import geopandas as gpd

    code_col = next((name for name in IZ_CODE_CANDIDATES if name in frame.columns), None)
    if code_col is None:
        raise ModelError(
            f"IZ origin table must contain iz_code. Columns: {list(frame.columns)}",
            code="invalid_config",
            details={"source": source},
        )
    gdf = gpd.GeoDataFrame(frame.rename(columns={code_col: "iz_code"}).copy())
    gdf["iz_code"] = gdf["iz_code"].astype("string").str.strip()
    if gdf["iz_code"].eq("").any() or gdf["iz_code"].isna().any():
        raise ModelError("IZ origin table contains empty iz_code values.", code="invalid_config")
    duplicated = gdf.loc[gdf["iz_code"].duplicated(), "iz_code"].astype(str).tolist()
    if duplicated:
        raise ModelError(
            f"IZ origin table has duplicate iz_code values: {duplicated[:10]}.",
            code="invalid_config",
        )
    if gdf.crs is None:
        raise ModelError("IZ origin geometries have no CRS; refusing to guess.", code="invalid_config")
    gdf = gdf.to_crs(BNG_CRS)
    if gdf.geometry.isna().any() or gdf.geometry.is_empty.any():
        raise ModelError("IZ origin table contains missing geometries.", code="invalid_config")
    if area_code == LOCAL_AUTHORITY_CODE and len(gdf) != EXPECTED_IZ_COUNT:
        raise ModelError(
            f"Edinburgh IZ origin count is {len(gdf)}, expected {EXPECTED_IZ_COUNT}.",
            code="node_order_mismatch",
            details={"source": source},
        )
    return gdf[["iz_code", "geometry"]].copy()


def resolve_osm_query(
    area_code: str,
    osm_place: str | None,
    iz_origins,
    buffer_m: float = NETWORK_BUFFER_M,
) -> dict[str, Any]:
    """Choose how to download the OSM graph. Do not invent a place name."""
    place = str(osm_place).strip() if osm_place else ""
    if place:
        return {
            "method": "place",
            "osm_place": place,
            "area_code": area_code,
            "buffer_m": None,
        }
    mapped = OSM_PLACE_BY_AREA.get(str(area_code).strip())
    if mapped:
        return {
            "method": "place",
            "osm_place": mapped,
            "area_code": area_code,
            "buffer_m": None,
        }
    import geopandas as gpd

    buffered = gpd.GeoSeries(iz_origins.geometry, crs=iz_origins.crs).to_crs(BNG_CRS).buffer(buffer_m)
    try:
        union = buffered.union_all()
    except AttributeError:
        union = buffered.unary_union
    polygon_wgs = gpd.GeoSeries([union], crs=BNG_CRS).to_crs(WGS84_CRS).iloc[0]
    return {
        "method": "polygon",
        "osm_place": None,
        "area_code": area_code,
        "buffer_m": buffer_m,
        "polygon_wgs": polygon_wgs,
    }


def shortest_path_matrix(
    iz_codes: Sequence[str],
    site_ids: Sequence[str],
    origin_nodes: Sequence[Any],
    dest_nodes: Sequence[Any],
    path_lengths_by_origin: dict[Any, dict[Any, float]],
    mode_name: str,
) -> pd.DataFrame:
    """Build one mode of the IZ×site matrix. Unreachable pairs stay NaN."""
    if len(iz_codes) != len(origin_nodes):
        raise ModelError("Origin node list does not match iz_code list.", code="invalid_tensor_shape")
    if len(site_ids) != len(dest_nodes):
        raise ModelError("Destination node list does not match site_id list.", code="invalid_tensor_shape")
    rows: list[dict[str, Any]] = []
    for iz_code, orig_node in zip(iz_codes, origin_nodes):
        path_lengths = path_lengths_by_origin.get(orig_node, {})
        for site_id, dest_node in zip(site_ids, dest_nodes):
            travel_time = path_lengths.get(dest_node, np.nan)
            if travel_time is None or (isinstance(travel_time, float) and np.isnan(travel_time)):
                value = np.nan
            else:
                value = round(float(travel_time), 2)
            rows.append(
                {
                    "iz_code": str(iz_code),
                    "site_id": str(site_id),
                    "mode": mode_name,
                    "travel_time_min": value,
                }
            )
    return pd.DataFrame(rows, columns=list(MATRIX_COLUMNS))


def site_ids_unreachable_from_all_origins(matrix: pd.DataFrame) -> list[str]:
    """Sites that have NaN travel time from every IZ on at least one mode."""
    if matrix is None or matrix.empty:
        return []
    n_iz = matrix["iz_code"].astype(str).nunique()
    flagged: set[str] = set()
    for (site_id, _mode), part in matrix.groupby(["site_id", "mode"], sort=False):
        if int(part["travel_time_min"].isna().sum()) >= n_iz:
            flagged.add(str(site_id))
    return sorted(flagged)


def validate_travel_time_matrix(
    frame: pd.DataFrame,
    iz_codes: Sequence[str] | None = None,
    site_ids: Sequence[str] | None = None,
    modes: Sequence[str] | None = None,
) -> pd.DataFrame:
    missing = [name for name in MATRIX_COLUMNS if name not in frame.columns]
    if missing:
        raise ModelError(
            f"Travel-time matrix missing columns {missing}.",
            code="invalid_config",
        )
    out = frame.loc[:, list(MATRIX_COLUMNS)].copy()
    out["iz_code"] = out["iz_code"].astype("string").str.strip()
    out["site_id"] = out["site_id"].astype("string").str.strip()
    out["mode"] = out["mode"].astype("string").str.strip()
    out["travel_time_min"] = pd.to_numeric(out["travel_time_min"], errors="coerce")
    if out["iz_code"].eq("").any() or out["site_id"].eq("").any() or out["mode"].eq("").any():
        raise ModelError("Travel-time matrix contains empty join keys.", code="invalid_config")
    duplicated = out.duplicated(subset=["iz_code", "site_id", "mode"])
    if duplicated.any():
        raise ModelError(
            "Travel-time matrix has duplicate iz_code/site_id/mode rows.",
            code="invalid_config",
        )
    finite = out["travel_time_min"].dropna()
    if (finite < 0).any():
        raise ModelError("Travel-time minutes must be non-negative or NaN.", code="invalid_config")
    if iz_codes is not None:
        expected = set(map(str, iz_codes))
        unknown = sorted(set(out["iz_code"].astype(str)) - expected)
        omitted = sorted(expected - set(out["iz_code"].astype(str)))
        if unknown or omitted:
            raise ModelError(
                "Travel-time matrix IZ codes do not match the origin set.",
                code="node_order_mismatch",
                details={"unknown": unknown[:10], "omitted": omitted[:10]},
            )
    if site_ids is not None:
        expected_sites = set(map(str, site_ids))
        unknown_sites = sorted(set(out["site_id"].astype(str)) - expected_sites)
        omitted_sites = sorted(expected_sites - set(out["site_id"].astype(str)))
        if unknown_sites or omitted_sites:
            raise ModelError(
                "Travel-time matrix site_id values do not match the candidate sites.",
                code="invalid_config",
                details={"unknown": unknown_sites[:10], "omitted": omitted_sites[:10]},
            )
    if modes is not None:
        unknown_modes = sorted(set(out["mode"].astype(str)) - set(modes))
        if unknown_modes:
            raise ModelError(
                f"Travel-time matrix has unknown modes {unknown_modes}.",
                code="invalid_config",
            )
    return out.reset_index(drop=True)


def find_travel_time_file(area_code: str = LOCAL_AUTHORITY_CODE, path: str | Path | None = None) -> Path:
    if path is not None:
        file_path = Path(path)
        if not file_path.exists():
            raise ModelError(f"Travel-time matrix missing: {file_path}", code="missing_dataset")
        return file_path
    candidates: list[Path] = []
    results = travel_time_results_dir(area_code)
    raw = travel_time_raw_dir()
    for folder in (results, raw, raw / _safe_area_token(area_code)):
        for name in RAW_MATRIX_NAMES:
            candidate = folder / name
            if candidate.exists():
                candidates.append(candidate)
    if candidates:
        return candidates[0]
    raise ModelError(
        "Travel-time matrix not found. Road-graph kilometres are not travel time. "
        "Place travel_time_matrix.csv under data/raw/travel_time or run source='osm'.",
        code="missing_dataset",
        details={"area_code": area_code},
    )


def load_travel_time(
    path: str | Path | None = None,
    area_code: str = LOCAL_AUTHORITY_CODE,
    iz_codes: Sequence[str] | None = None,
    site_ids: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Read a previously computed matrix. Does not call OSM."""
    file_path = find_travel_time_file(area_code=area_code, path=path)
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(file_path)
    elif suffix == ".parquet":
        frame = pd.read_parquet(file_path)
    else:
        raise ModelError(f"Unsupported travel-time format: {file_path}", code="invalid_config")
    LOGGER.info("Loaded travel-time matrix %s (%s rows).", file_path, len(frame))
    return validate_travel_time_matrix(frame, iz_codes=iz_codes, site_ids=site_ids)


def _edge_travel_time_minutes(length_m: float, speed_kmh: float) -> float:
    speed_m_per_min = (float(speed_kmh) * 1000.0) / 60.0
    if speed_m_per_min <= 0:
        raise ModelError("Travel speed_kmh must be positive.", code="invalid_config")
    return float(length_m) / speed_m_per_min


def _assign_edge_times(graph, speed_kmh: float) -> None:
    try:
        edge_iter = graph.edges(keys=True, data=True)
    except TypeError:
        edge_iter = ((u, v, 0, data) for u, v, data in graph.edges(data=True))
    for _u, _v, _k, data in edge_iter:
        length_m = data.get("length", 1.0)
        data["travel_time_min"] = _edge_travel_time_minutes(length_m, speed_kmh)


def _require_osmnx():
    try:
        import networkx as nx
        import osmnx as ox
    except ImportError as exc:  # pragma: no cover - optional extra
        raise ModelError(
            "Travel-time computation requires osmnx and networkx. "
            "Install them or use source='local' with an existing matrix.",
            code="missing_dependency",
        ) from exc
    return ox, nx


def fetch_mode_graph(query: dict[str, Any], network_type: str, fetch_graph_fn: Callable | None = None):
    if fetch_graph_fn is not None:
        return fetch_graph_fn(network_type=network_type, query=query)
    ox, _nx = _require_osmnx()
    if query["method"] == "place":
        LOGGER.info("Fetching OSM '%s' graph for %s.", network_type, query["osm_place"])
        try:
            return ox.graph_from_place(query["osm_place"], network_type=network_type)
        except TypeError as exc:
            raise ModelError(
                f"Nominatim did not return a polygon for {query['osm_place']!r}. "
                "Use the 2011 IZ polygon download instead of a place name.",
                code="invalid_config",
                details={"osm_place": query.get("osm_place"), "area_code": query.get("area_code")},
            ) from exc
    LOGGER.info("Fetching OSM '%s' graph from IZ polygon (buffer=%sm).", network_type, query.get("buffer_m"))
    return ox.graph_from_polygon(query["polygon_wgs"], network_type=network_type)


def compute_mode_matrix(
    iz_origins,
    sites,
    mode_name: str,
    network_type: str,
    speed_kmh: float,
    query: dict[str, Any],
    *,
    fetch_graph_fn: Callable | None = None,
    nearest_nodes_fn: Callable | None = None,
    path_length_fn: Callable | None = None,
) -> pd.DataFrame:
    """One mode: snap IZ centroids and sites, then shortest-path minutes."""
    ox, nx = (None, None)
    if nearest_nodes_fn is None or path_length_fn is None:
        ox, nx = _require_osmnx()

    graph = fetch_mode_graph(query, network_type=network_type, fetch_graph_fn=fetch_graph_fn)
    _assign_edge_times(graph, speed_kmh)

    iz_wgs = iz_origins.to_crs(WGS84_CRS)
    sites_wgs = sites.to_crs(WGS84_CRS)
    snap = nearest_nodes_fn or ox.distance.nearest_nodes
    origin_nodes = list(snap(graph, X=iz_wgs.geometry.x, Y=iz_wgs.geometry.y))
    dest_nodes = list(snap(graph, X=sites_wgs.geometry.x, Y=sites_wgs.geometry.y))

    dijkstra = path_length_fn or (
        lambda G, source: nx.single_source_dijkstra_path_length(G, source=source, weight="travel_time_min")
    )
    path_lengths_by_origin: dict[Any, dict[Any, float]] = {}
    unique_origins = list(dict.fromkeys(origin_nodes))
    for orig_node in unique_origins:
        path_lengths_by_origin[orig_node] = dijkstra(graph, orig_node)

    return shortest_path_matrix(
        iz_codes=iz_origins["iz_code"].astype(str).tolist(),
        site_ids=sites["site_id"].astype(str).tolist(),
        origin_nodes=origin_nodes,
        dest_nodes=dest_nodes,
        path_lengths_by_origin=path_lengths_by_origin,
        mode_name=mode_name,
    )


def prepare_travel_time(
    area_code: str = LOCAL_AUTHORITY_CODE,
    source: str = "osm",
    osm_place: str | None = None,
    iz_path: str | Path | None = None,
    sites_path: str | Path | None = None,
    modes: dict[str, dict[str, Any]] | None = None,
    *,
    iz_origins=None,
    sites=None,
    fetch_graph_fn: Callable | None = None,
    nearest_nodes_fn: Callable | None = None,
    path_length_fn: Callable | None = None,
    compute_mode_matrix_fn: Callable | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Compute or load the travel-time matrix for one study area."""
    source_key = str(source).strip().lower()
    if source_key not in ALLOWED_SOURCES:
        raise ModelError(
            "Travel-time source must be 'local' (existing CSV) or 'osm' (compute once).",
            code="invalid_config",
            details={"source": source},
        )
    area = str(area_code).strip() or LOCAL_AUTHORITY_CODE
    mode_cfg = modes or DEFAULT_MODES

    if source_key == "local":
        matrix = load_travel_time(area_code=area)
        output_path = find_travel_time_file(area_code=area)
        return {
            "status": "ok",
            "source": "local",
            "area_code": area,
            "n_rows": int(len(matrix)),
            "n_iz": int(matrix["iz_code"].nunique()),
            "n_sites": int(matrix["site_id"].nunique()),
            "modes": sorted(matrix["mode"].astype(str).unique().tolist()),
            "output_path": str(output_path),
            "provenance_path": None,
            "warnings": [
                "Assumed speeds if this file came from oasis-v4: drive 30 km/h, walk 4.5 km/h."
            ],
        }

    from data.candidate_sites import load_candidate_sites

    origins = iz_origins if iz_origins is not None else load_iz_origins(area_code=area, iz_path=iz_path)
    destinations = sites if sites is not None else load_candidate_sites(
        path=sites_path,
        area_code=area,
    )
    query = resolve_osm_query(area_code=area, osm_place=osm_place, iz_origins=origins)

    frames: list[pd.DataFrame] = []
    for mode_name, cfg in mode_cfg.items():
        network_type = str(cfg.get("network_type") or mode_name)
        speed_kmh = float(cfg.get("speed_kmh"))
        LOGGER.info("Computing %s matrix at %s km/h.", mode_name, speed_kmh)
        if compute_mode_matrix_fn is not None:
            part = compute_mode_matrix_fn(
                iz_origins=origins,
                sites=destinations,
                mode_name=mode_name,
                network_type=network_type,
                speed_kmh=speed_kmh,
                query=query,
            )
        else:
            part = compute_mode_matrix(
                iz_origins=origins,
                sites=destinations,
                mode_name=mode_name,
                network_type=network_type,
                speed_kmh=speed_kmh,
                query=query,
                fetch_graph_fn=fetch_graph_fn,
                nearest_nodes_fn=nearest_nodes_fn,
                path_length_fn=path_length_fn,
            )
        frames.append(part)

    matrix = validate_travel_time_matrix(
        pd.concat(frames, ignore_index=True),
        iz_codes=origins["iz_code"].astype(str).tolist(),
        site_ids=destinations["site_id"].astype(str).tolist(),
        modes=list(mode_cfg),
    )
    dropped_ids = site_ids_unreachable_from_all_origins(matrix)
    if dropped_ids:
        LOGGER.warning(
            "Dropping %s sites unreachable from every IZ on at least one mode: %s",
            len(dropped_ids),
            dropped_ids,
        )
        matrix = matrix.loc[~matrix["site_id"].astype(str).isin(dropped_ids)].copy()
        kept_sites = [
            str(site_id)
            for site_id in destinations["site_id"].astype(str).tolist()
            if str(site_id) not in set(dropped_ids)
        ]
        matrix = validate_travel_time_matrix(
            matrix,
            iz_codes=origins["iz_code"].astype(str).tolist(),
            site_ids=kept_sites,
            modes=list(mode_cfg),
        )
        from data.candidate_sites import candidate_sites_results_dir, remove_site_ids

        sites_dir = candidate_sites_results_dir(area)
        if sites_dir.exists():
            remove_site_ids(
                dropped_ids,
                sites_dir,
                reason="unreachable from all IZ origins on the OSM graph; not filled",
            )
    n_sites = int(matrix["site_id"].nunique())
    n_unreachable = int(matrix["travel_time_min"].isna().sum())
    out_dir = Path(output_dir) if output_dir is not None else travel_time_results_dir(area)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = write_table(matrix, out_dir / "travel_time_matrix.csv")
    provenance = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "area_code": area,
        "source": "osm",
        "osm_method": query["method"],
        "osm_place": query.get("osm_place"),
        "buffer_m": query.get("buffer_m"),
        "n_iz": int(len(origins)),
        "n_sites": n_sites,
        "n_rows": int(len(matrix)),
        "n_unreachable": n_unreachable,
        "dropped_site_ids": dropped_ids,
        "modes": {
            name: {
                "network_type": cfg.get("network_type"),
                "speed_kmh": cfg.get("speed_kmh"),
                "speed_is_assumed_urban_average": True,
            }
            for name, cfg in mode_cfg.items()
        },
        "join_key": "iz_code",
        "crs_origins": BNG_CRS,
        "not_official_ttw": True,
        "not_road_graph_km": True,
        "not_gtfs": True,
        "raw_not_modified": True,
    }
    provenance_path = write_json(provenance, out_dir / "provenance.json")
    LOGGER.info("Wrote %s (%s rows).", csv_path, len(matrix))
    warnings = [
        "Drive 30 km/h and walk 4.5 km/h are assumed urban averages from oasis-v4, not official travel-to-work times.",
        "Sites unreachable from every IZ on a mode are dropped entirely; remaining NaN cells are not filled.",
    ]
    if dropped_ids:
        warnings.append(
            f"Dropped {len(dropped_ids)} sites unreachable from all IZ origins: {dropped_ids}."
        )
    if n_unreachable:
        warnings.append(f"{n_unreachable} IZ–site–mode cells are unreachable on the OSM graph.")
    return {
        "status": "ok",
        "source": "osm",
        "area_code": area,
        "n_rows": int(len(matrix)),
        "n_iz": int(len(origins)),
        "n_sites": n_sites,
        "dropped_site_ids": dropped_ids,
        "modes": list(mode_cfg),
        "n_unreachable": n_unreachable,
        "osm_method": query["method"],
        "osm_place": query.get("osm_place"),
        "output_path": str(csv_path),
        "provenance_path": str(provenance_path),
        "warnings": warnings,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Load or compute an IZ-to-site travel-time matrix. City is a parameter."
    )
    parser.add_argument("--area-code", default=LOCAL_AUTHORITY_CODE, help="Local authority code, e.g. S12000036")
    parser.add_argument("--source", choices=list(ALLOWED_SOURCES), required=True)
    parser.add_argument("--osm-place", default=None, help="OSMnx place string; omit to use the area map or IZ polygon")
    parser.add_argument("--iz-path", default=None, help="Optional IZ origin shapefile/GeoJSON")
    parser.add_argument("--sites-path", default=None, help="Candidate-site shapefile or CSV")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)
    result = prepare_travel_time(
        area_code=args.area_code,
        source=args.source,
        osm_place=args.osm_place,
        iz_path=args.iz_path,
        sites_path=args.sites_path,
        output_dir=args.output_dir,
    )
    print(json.dumps({k: v for k, v in result.items() if k != "matrix"}, indent=2, default=str))


if __name__ == "__main__":
    main()
