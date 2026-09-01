"""Road network-distance graph for Edinburgh 2011 Intermediate Zones.

Five modes are kept: road, bicycle, walking, road+rail, walking+rail.
Edge length is geometric length in EPSG:27700 (metres), converted to kilometres.
This is not travel time, real-time traffic, observed mobility, demand, or
mode choice. OSM maxspeed is not used. dodgr d_weighted is not used.

Unreachable pairs stay Inf; Inf is never replaced with 0. The diagonal is 0.
Node order is the existing COVID node_index. Easting/Northing are British
National Grid metres, not longitude/latitude.

IZ-to-station and station-to-IZ distances are shortest paths on the relevant
network. Euclidean nearest-station assignment is not used. Tram tracks and
tram_stop points are excluded so stations are not snapped onto the tram
system. Total distance includes Euclidean snap-to-network at both endpoints
of each leg (origin snap + network path + destination snap). Grade-separated
lines (layer/bridge/tunnel) are not joined at 2D interior intersections.
Defaulted layer/bridge/tunnel values are not treated as observed grade data.

    PYTHONPATH=src python -m graph.road
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csgraph
from scipy.spatial import cKDTree

from data.covid import _load_edinburgh_centroids
from graph.geo import (
    BOUNDARY_RELATIVE_PATH,
    GraphError,
    TARGET_CRS,
    _align_polygons,
    _load_boundaries,
    _load_nodes,
)
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

try:
    import geopandas as gpd
    import shapely
    from shapely import STRtree
    from shapely.geometry.base import BaseGeometry
    from shapely.ops import unary_union
except ImportError as exc:  # pragma: no cover - environment missing GIS stack
    raise ImportError(
        "road graph construction requires geopandas and shapely>=2.0."
    ) from exc

if int(str(shapely.__version__).split(".", 1)[0]) < 2:
    raise ImportError(
        "road graph construction requires shapely>=2.0; "
        "STRtree.query must return integer indices."
    )

LOGGER = get_logger("graph.road")

# Clip once to the buffered city union. Do not clip to the city and then buffer.
DEFAULT_BUFFER_M = 2000.0
DEFAULT_K = 5
DEFAULT_SNAP_MAX_M = 500.0
# Accepted snaps longer than this (capped at 0.5 * snap_max_m) are flagged as ok_long.
LONG_ACCEPTED_SNAP_CAP_M = 100.0
LINE_LENGTH_EPS = 1e-6
NODE_COORD_DIGITS = 2
# Midpoint of a noded segment must lie this close to an original line (metres).
SEGMENT_MATCH_MAX_M = 0.5

OSM_DIR = Path("data") / "raw" / "roads" / "scotland-free.shp"
ROADS_RELATIVE_PATH = OSM_DIR / "gis_osm_roads_free_1.shp"
RAILWAYS_RELATIVE_PATH = OSM_DIR / "gis_osm_railways_free_1.shp"
STATIONS_RELATIVE_PATH = OSM_DIR / "gis_osm_transport_free_1.shp"

MODE_ORDER = ("road", "bicycle", "walking", "road+rail", "walking+rail")
STREET_MODES = ("road", "bicycle", "walking")
UNKNOWN_FCLASS = frozenset({"", "unknown", "nan", "none", "null"})
DEDICATED_BIKE_FCLASS = frozenset({"cycleway", "path"})

# Per-mode allow lists. A class unused by cars is not deleted from the other modes.
ROAD_FCLASS = frozenset(
    {
        "motorway",
        "motorway_link",
        "trunk",
        "trunk_link",
        "primary",
        "primary_link",
        "secondary",
        "secondary_link",
        "tertiary",
        "tertiary_link",
        "unclassified",
        "residential",
        "living_street",
        "service",
        "track",
        "track_grade1",
        "track_grade2",
        "track_grade3",
        "track_grade4",
        "track_grade5",
    }
)
BICYCLE_FCLASS = frozenset(
    {
        "trunk",
        "trunk_link",
        "primary",
        "primary_link",
        "secondary",
        "secondary_link",
        "tertiary",
        "tertiary_link",
        "unclassified",
        "residential",
        "living_street",
        "service",
        "track",
        "track_grade1",
        "track_grade2",
        "track_grade3",
        "track_grade4",
        "track_grade5",
        "cycleway",
        "path",
        "bridleway",
        "footway",
        "pedestrian",
    }
)
WALKING_FCLASS = frozenset(
    {
        "primary",
        "primary_link",
        "secondary",
        "secondary_link",
        "tertiary",
        "tertiary_link",
        "unclassified",
        "residential",
        "living_street",
        "service",
        "track",
        "track_grade1",
        "track_grade2",
        "track_grade3",
        "track_grade4",
        "track_grade5",
        "cycleway",
        "path",
        "bridleway",
        "footway",
        "pedestrian",
        "steps",
    }
)
RAIL_FCLASS = frozenset({"rail", "light_rail", "subway", "narrow_gauge"})
STATION_FCLASS = frozenset({"railway_station", "railway_halt"})
COORD_DUP_EPS_M = 0.01
AMBIGUOUS_CROSSING_EXAMPLE_CAP = 25
MISSING_ONEWAY = frozenset({"", "NAN", "NONE", "<NA>"})
FCLASS_BY_STREET_MODE = {
    "road": ROAD_FCLASS,
    "bicycle": BICYCLE_FCLASS,
    "walking": WALKING_FCLASS,
}

DISTANCE_CSV_FILENAMES = (
    "D_road_km.csv",
    "D_bicycle_km.csv",
    "D_walking_km.csv",
    "D_road_rail_km.csv",
    "D_walking_rail_km.csv",
    "D_multimodal_km.csv",
)
SNAP_FILENAMES = ("iz_snaps.csv", "station_snaps.csv")
OUTPUT_FILENAMES = (
    "nodes.csv",
    "edges.csv",
    "adjacency_road.npz",
    "distances.npz",
    "shortest_mode.csv",
    "validation_report.json",
    *DISTANCE_CSV_FILENAMES,
    *SNAP_FILENAMES,
)


@dataclass
class RouteGraph:
    """Directed network: node coordinates in EPSG:27700 and edge lengths in metres."""

    node_xy: np.ndarray
    length_m: sparse.csr_matrix

    @property
    def n_nodes(self) -> int:
        return int(self.node_xy.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.length_m.nnz)


@dataclass
class SnapResult:
    """Point-to-network snap for one mode. node_index is -1 when snap_ok is False."""

    node_index: np.ndarray
    distance_m: np.ndarray
    snap_x: np.ndarray
    snap_y: np.ndarray

    @property
    def ok(self) -> np.ndarray:
        return self.node_index >= 0


def _empty_snap(n: int) -> SnapResult:
    return SnapResult(
        node_index=np.full(n, -1, dtype="int64"),
        distance_m=np.full(n, np.inf, dtype="float64"),
        snap_x=np.full(n, np.nan, dtype="float64"),
        snap_y=np.full(n, np.nan, dtype="float64"),
    )


def _empty_crossing_report() -> dict[str, Any]:
    return {
        "n_grade_separated_crossings_ignored": 0,
        "n_ambiguous_crossings": 0,
        "ambiguous_crossing_examples": [],
        "source_has_layer": False,
        "source_has_bridge": False,
        "source_has_tunnel": False,
        "n_records_without_source_layer": 0,
        "n_records_without_source_bridge": 0,
        "n_records_without_source_tunnel": 0,
        "n_records_missing_layer_value": 0,
        "n_records_missing_bridge_value": 0,
        "n_records_missing_tunnel_value": 0,
        "defaulted_grade_not_treated_as_complete": True,
    }


def _require_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise GraphError(f"{name} must be an integer.")
    parsed = int(value)
    if parsed < 1:
        raise GraphError(f"{name} must be a positive integer.")
    return parsed


def _long_snap_threshold_m(snap_max_m: float) -> float:
    return min(LONG_ACCEPTED_SNAP_CAP_M, 0.5 * float(snap_max_m))


def construct_road_graph(
    nodes: pd.DataFrame | Path | None = None,
    coords: pd.DataFrame | Path | None = None,
    boundaries_path: Path | None = None,
    polygons: gpd.GeoDataFrame | None = None,
    roads: gpd.GeoDataFrame | Path | None = None,
    railways: gpd.GeoDataFrame | Path | None = None,
    stations: gpd.GeoDataFrame | Path | None = None,
    output_dir: Path | None = None,
    overwrite: bool = True,
    buffer_m: float = DEFAULT_BUFFER_M,
    k: int = DEFAULT_K,
    snap_max_m: float = DEFAULT_SNAP_MAX_M,
    area_code: str = LOCAL_AUTHORITY_CODE,
) -> dict[str, Any]:
    """Build a directed k-NN graph from five shortest network-distance modes.

    Does not train a model and does not change COVID, forecast, or rook-graph
    outputs. Do not run this on the full Scotland OSM extract from unit tests.
    """
    k = _require_positive_int(k, "k")
    if buffer_m < 0:
        raise GraphError("buffer_m must be >= 0.")
    if snap_max_m <= 0:
        raise GraphError("snap_max_m must be positive.")

    node_table, node_source = _load_nodes(nodes, area_code=area_code)
    coord_table, coord_source = _load_iz_coordinates(node_table, coords)
    aligned, unused, repaired, boundary_source = _load_aligned_polygons(
        node_table, polygons=polygons, boundaries_path=boundaries_path
    )
    clip_geom = unary_union(list(aligned.geometry)).buffer(float(buffer_m))

    roads_gdf, roads_source = _load_line_layer(roads, ROADS_RELATIVE_PATH, "roads")
    railways_gdf, railways_source = _load_line_layer(
        railways, RAILWAYS_RELATIVE_PATH, "railways"
    )
    stations_gdf, stations_source = _load_station_layer(stations)

    roads_clipped = _clip_lines(roads_gdf, clip_geom)
    railways_clipped = _clip_lines(railways_gdf, clip_geom)
    stations_clipped = _clip_points(stations_gdf, clip_geom)
    stations_clipped = _deduplicate_stations(stations_clipped)
    if "osm_id" not in stations_clipped.columns:
        stations_clipped = stations_clipped.copy()
        stations_clipped["osm_id"] = [f"station_{i}" for i in range(len(stations_clipped))]

    street_graphs: dict[str, RouteGraph] = {}
    crossing_reports: dict[str, dict[str, Any]] = {}
    iz_snaps: dict[str, SnapResult] = {}
    for mode in STREET_MODES:
        filtered = _filter_fclass(roads_clipped, FCLASS_BY_STREET_MODE[mode], layer="roads")
        graph, crossings = _lines_to_graph(filtered, mode=mode)
        street_graphs[mode] = graph
        crossing_reports[mode] = crossings
        iz_snaps[mode] = _snap_points(coord_table, graph, snap_max_m=snap_max_m)

    rail_graph, rail_crossings = _lines_to_graph(
        _filter_fclass(railways_clipped, RAIL_FCLASS, layer="railways"),
        mode="rail",
    )
    crossing_reports["rail"] = rail_crossings
    station_xy = _point_frame(stations_clipped)
    station_road_snaps = _snap_xy(station_xy, street_graphs["road"], snap_max_m)
    station_walk_snaps = _snap_xy(station_xy, street_graphs["walking"], snap_max_m)
    station_rail_snaps = _snap_xy(station_xy, rail_graph, snap_max_m)

    distances: dict[str, np.ndarray] = {
        "road": _pairwise_distances(street_graphs["road"], iz_snaps["road"]),
        "bicycle": _pairwise_distances(street_graphs["bicycle"], iz_snaps["bicycle"]),
        "walking": _pairwise_distances(street_graphs["walking"], iz_snaps["walking"]),
    }
    road_access = _od_distances(
        street_graphs["road"],
        iz_snaps["road"].node_index,
        station_road_snaps.node_index,
        source_snap_m=iz_snaps["road"].distance_m,
        target_snap_m=station_road_snaps.distance_m,
    )
    road_egress = _od_distances(
        street_graphs["road"],
        station_road_snaps.node_index,
        iz_snaps["road"].node_index,
        source_snap_m=station_road_snaps.distance_m,
        target_snap_m=iz_snaps["road"].distance_m,
    )
    walk_access = _od_distances(
        street_graphs["walking"],
        iz_snaps["walking"].node_index,
        station_walk_snaps.node_index,
        source_snap_m=iz_snaps["walking"].distance_m,
        target_snap_m=station_walk_snaps.distance_m,
    )
    walk_egress = _od_distances(
        street_graphs["walking"],
        station_walk_snaps.node_index,
        iz_snaps["walking"].node_index,
        source_snap_m=station_walk_snaps.distance_m,
        target_snap_m=iz_snaps["walking"].distance_m,
    )
    rail_station = _pairwise_distances(rail_graph, station_rail_snaps)
    distances["road+rail"] = intermodal_distance(road_access, rail_station, road_egress)
    distances["walking+rail"] = intermodal_distance(walk_access, rail_station, walk_egress)

    n_nodes = len(coord_table)
    for name, matrix in distances.items():
        assert_mode_distance_matrix(matrix, n_nodes=n_nodes, name=name)

    multimodal, shortest_mode = combine_mode_distances(distances)
    edges, tau = knn_road_edges(multimodal, shortest_mode, coord_table, k=k)
    adjacency = directed_adjacency_matrix(coord_table, edges)
    _assert_directed_knn_invariants(
        adjacency, n_nodes=n_nodes, n_edges=len(edges), k=k
    )
    iz_snap_table = _iz_snap_table(coord_table, iz_snaps, snap_max_m=snap_max_m)
    station_snap_table = _station_snap_table(
        stations_clipped,
        {"road": station_road_snaps, "walking": station_walk_snaps, "rail": station_rail_snaps},
        snap_max_m=snap_max_m,
    )
    long_snap_m = _long_snap_threshold_m(snap_max_m)

    report = _validation_report(
        nodes_out=coord_table,
        edges=edges,
        adjacency=adjacency,
        distances=distances,
        multimodal=multimodal,
        shortest_mode=shortest_mode,
        tau=tau,
        k=k,
        buffer_m=buffer_m,
        snap_max_m=snap_max_m,
        unused=unused,
        repaired=repaired,
        node_source=node_source,
        coord_source=coord_source,
        boundary_source=boundary_source,
        roads_source=roads_source,
        railways_source=railways_source,
        stations_source=stations_source,
        n_road_edges=street_graphs["road"].n_edges,
        n_bicycle_edges=street_graphs["bicycle"].n_edges,
        n_walking_edges=street_graphs["walking"].n_edges,
        n_rail_edges=rail_graph.n_edges,
        n_stations=int(len(stations_clipped)),
        n_iz_unsnapped={mode: int((~iz_snaps[mode].ok).sum()) for mode in STREET_MODES},
        n_stations_unsnapped={
            "road": int((~station_road_snaps.ok).sum()),
            "walking": int((~station_walk_snaps.ok).sum()),
            "rail": int((~station_rail_snaps.ok).sum()),
        },
        n_iz_long_accepted_snap={
            mode: _count_long_snaps(iz_snaps[mode], long_snap_m) for mode in STREET_MODES
        },
        n_station_long_accepted_snap={
            "road": _count_long_snaps(station_road_snaps, long_snap_m),
            "walking": _count_long_snaps(station_walk_snaps, long_snap_m),
            "rail": _count_long_snaps(station_rail_snaps, long_snap_m),
        },
        long_accepted_snap_threshold_m=long_snap_m,
        crossing_reports=crossing_reports,
    )

    output_dir = (
        Path(output_dir) if output_dir is not None else results_dir() / "graph" / "road"
    )
    _assert_overwrite_allowed(output_dir, overwrite=overwrite)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = _write_outputs(
        output_dir,
        coord_table,
        edges,
        adjacency,
        distances,
        multimodal,
        shortest_mode,
        report,
        iz_snap_table,
        station_snap_table,
    )

    summary = {
        "status": "ok",
        "n_nodes": report["n_nodes"],
        "n_edges": report["n_edges"],
        "k": k,
        "tau_km": tau,
        "adjacency_shape": report["adjacency_matrix_shape"],
        "is_symmetric": report["is_symmetric"],
        "diagonal_is_zero": report["diagonal_is_zero"],
        "n_isolated": report["n_isolated"],
        "isolated_iz": report["isolated_iz"],
        "n_weakly_connected_components": report["n_weakly_connected_components"],
        "shortest_mode_share": report["shortest_mode_share"],
        "crs": TARGET_CRS,
        "output_paths": paths,
        "validation_report": report,
    }
    write_run_log(
        {
            "event": "graph_road_prepare_complete",
            **{key: value for key, value in summary.items() if key != "validation_report"},
        },
        filename="graph_road_prepare.jsonl",
    )
    LOGGER.info(
        "Wrote road graph with %s nodes and %s directed edges to %s",
        report["n_nodes"],
        report["n_edges"],
        output_dir,
    )
    return summary


def assert_mode_distance_matrix(matrix: np.ndarray, *, n_nodes: int, name: str) -> None:
    """Require a square node_index-ordered distance matrix with Inf for unreachable pairs."""
    values = np.asarray(matrix, dtype="float64")
    if values.shape != (n_nodes, n_nodes):
        raise GraphError(
            f"{name} matrix shape is {list(values.shape)}, expected [{n_nodes}, {n_nodes}]."
        )
    if np.isnan(values).any():
        raise GraphError(f"{name} matrix contains NaN.")
    if not np.all(values.diagonal() == 0.0):
        raise GraphError(f"{name} matrix diagonal is not zero.")
    offdiag = values.copy()
    np.fill_diagonal(offdiag, np.nan)
    finite = np.isfinite(offdiag)
    if np.any(offdiag[finite] < 0.0):
        raise GraphError(f"{name} matrix contains negative distances.")
    unreachable = ~finite & ~np.isnan(offdiag)
    if unreachable.any() and not np.isposinf(offdiag[unreachable]).all():
        raise GraphError(f"{name} unreachable pairs must be Inf, not a finite filler.")


def combine_mode_distances(matrices: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Elementwise min over the five modes. Ties keep MODE_ORDER (not travel time)."""
    missing = [name for name in MODE_ORDER if name not in matrices]
    if missing:
        raise GraphError(f"Missing mode matrices: {missing}.")
    first = np.asarray(matrices[MODE_ORDER[0]], dtype="float64")
    if first.ndim != 2 or first.shape[0] != first.shape[1]:
        raise GraphError("Mode distance matrices must be square.")
    n = first.shape[0]
    for name in MODE_ORDER:
        assert_mode_distance_matrix(np.asarray(matrices[name], dtype="float64"), n_nodes=n, name=name)
    combined = np.full((n, n), np.inf, dtype="float64")
    shortest_mode = np.full((n, n), "unreachable", dtype=object)
    for name in MODE_ORDER:
        matrix = np.asarray(matrices[name], dtype="float64")
        better = np.isfinite(matrix) & (matrix < combined)
        combined = np.where(better, matrix, combined)
        shortest_mode = np.where(better, name, shortest_mode)
    np.fill_diagonal(combined, 0.0)
    np.fill_diagonal(shortest_mode, "self")
    return combined, shortest_mode


def intermodal_distance(
    access_iz_to_station: np.ndarray,
    rail_station: np.ndarray,
    egress_station_to_iz: np.ndarray,
    *,
    require_distinct_stations: bool = True,
) -> np.ndarray:
    """D[i,j] = min_{a,b} access[i,a] + rail[a,b] + egress[b,j].

    a==b is excluded by default so the path cannot collapse to walking into a
    station and out again without using the railway.
    """
    access = np.asarray(access_iz_to_station, dtype="float64")
    rail = np.asarray(rail_station, dtype="float64")
    egress = np.asarray(egress_station_to_iz, dtype="float64")
    n_iz, n_st = access.shape
    if rail.shape != (n_st, n_st):
        raise GraphError(
            f"Rail station matrix shape {list(rail.shape)} does not match [{n_st}, {n_st}]."
        )
    if egress.shape != (n_st, n_iz):
        raise GraphError(
            f"Egress matrix shape {list(egress.shape)} does not match [{n_st}, {n_iz}]."
        )
    out = np.full((n_iz, n_iz), np.inf, dtype="float64")
    for origin_station in range(n_st):
        for dest_station in range(n_st):
            if require_distinct_stations and origin_station == dest_station:
                continue
            rail_ab = rail[origin_station, dest_station]
            if not np.isfinite(rail_ab):
                continue
            candidate = (
                access[:, origin_station][:, None] + rail_ab + egress[dest_station, :][None, :]
            )
            np.minimum(out, candidate, out=out)
    np.fill_diagonal(out, 0.0)
    return out


def knn_road_edges(
    distance_km: np.ndarray,
    shortest_mode: np.ndarray,
    nodes: pd.DataFrame,
    k: int = DEFAULT_K,
) -> tuple[pd.DataFrame, float]:
    """Keep the k nearest finite destinations per origin. No self-loops. No 1/(D+eps)."""
    matrix = np.asarray(distance_km, dtype="float64")
    modes = np.asarray(shortest_mode, dtype=object)
    n = matrix.shape[0]
    if matrix.shape != (n, n) or modes.shape != (n, n):
        raise GraphError("Distance and shortest_mode must be square and aligned.")
    k = _require_positive_int(k, "k")
    if len(nodes) != n:
        raise GraphError("Node table length does not match the distance matrix.")

    codes = nodes[NODE_KEY].astype(str).tolist()
    rows: list[dict[str, Any]] = []
    selected: list[float] = []
    for source in range(n):
        distances = matrix[source].copy()
        distances[source] = np.inf
        reachable = np.flatnonzero(np.isfinite(distances))
        if reachable.size == 0:
            continue
        order = reachable[np.argsort(distances[reachable], kind="stable")][:k]
        for rank, target in enumerate(order, start=1):
            distance = float(distances[target])
            selected.append(distance)
            rows.append(
                {
                    "source_node_index": int(source),
                    "target_node_index": int(target),
                    "source_iz_code": codes[source],
                    "target_iz_code": codes[target],
                    "multimodal_distance_km": distance,
                    "shortest_mode": str(modes[source, target]),
                    "neighbour_rank": int(rank),
                }
            )

    if not selected:
        raise GraphError("No finite off-diagonal road-graph distances; cannot build k-NN edges.")
    tau = float(np.median(np.asarray(selected, dtype="float64")))
    if not np.isfinite(tau) or tau <= 0.0:
        raise GraphError(f"tau must be finite and positive, got {tau}.")
    edges = pd.DataFrame(rows)
    edges["weight"] = np.exp(-edges["multimodal_distance_km"].to_numpy(dtype="float64") / tau)
    if not np.all((edges["weight"] > 0.0) & (edges["weight"] <= 1.0)):
        raise GraphError("Edge weights must lie in (0, 1].")
    return (
        edges.sort_values(["source_node_index", "neighbour_rank"]).reset_index(drop=True),
        tau,
    )


def directed_adjacency_matrix(nodes: pd.DataFrame, edges: pd.DataFrame) -> sparse.csr_matrix:
    """Directed sparse weights. Not symmetrised. Diagonal stays 0."""
    n = int(len(nodes))
    if edges.empty:
        return sparse.csr_matrix((n, n), dtype="float64")
    sources = edges["source_node_index"].to_numpy(dtype="int64")
    targets = edges["target_node_index"].to_numpy(dtype="int64")
    if np.any(sources == targets):
        raise GraphError("Directed adjacency cannot contain self-loops.")
    weights = edges["weight"].to_numpy(dtype="float64")
    matrix = sparse.csr_matrix((weights, (sources, targets)), shape=(n, n), dtype="float64")
    matrix.setdiag(0)
    matrix.eliminate_zeros()
    return matrix


def _assert_directed_knn_invariants(
    adjacency: sparse.csr_matrix,
    n_nodes: int,
    n_edges: int,
    k: int,
) -> None:
    if adjacency.shape != (n_nodes, n_nodes):
        raise GraphError(
            f"Adjacency shape is {list(adjacency.shape)}, expected [{n_nodes}, {n_nodes}]."
        )
    if not np.all(adjacency.diagonal() == 0):
        raise GraphError("Adjacency diagonal is not zero.")
    if int(adjacency.nnz) != int(n_edges):
        raise GraphError(
            f"Adjacency nnz is {int(adjacency.nnz)}, expected {int(n_edges)} directed edges."
        )
    if adjacency.nnz and not np.all((adjacency.data > 0.0) & (adjacency.data <= 1.0)):
        raise GraphError("Adjacency weights must lie in (0, 1].")
    out_degree = np.diff(adjacency.indptr)
    if np.any(out_degree > k):
        raise GraphError(f"An origin has more than k={k} outgoing edges.")


def _load_iz_coordinates(
    nodes: pd.DataFrame,
    coords: pd.DataFrame | Path | None,
) -> tuple[pd.DataFrame, str]:
    """Join Easting/Northing onto the existing node_index order.

    Accepts InterZone/X/Y as aliases. Does not invent a new node order.
    """
    if coords is None:
        raw = _load_edinburgh_centroids(nodes)
        source = "2011 IZ population-weighted centroids (Easting/Northing, EPSG:27700)"
    elif isinstance(coords, (str, Path)):
        path = Path(coords)
        if not path.exists():
            raise GraphError(f"IZ coordinate table missing: {path}")
        raw = read_table(path)
        source = str(path)
    else:
        raw = coords.copy()
        source = "supplied DataFrame"

    frame = _normalise_coord_columns(raw)
    frame["Easting"] = pd.to_numeric(frame["Easting"], errors="coerce")
    frame["Northing"] = pd.to_numeric(frame["Northing"], errors="coerce")
    frame = _collapse_iz_coordinates(frame)
    merged = nodes.merge(frame, on=NODE_KEY, how="left", validate="one_to_one")
    missing = merged.loc[merged["Easting"].isna() | merged["Northing"].isna(), NODE_KEY]
    if not missing.empty:
        raise GraphError(
            f"{len(missing)} node-order IZs have no Easting/Northing: {missing.astype(str).tolist()[:10]}."
        )
    easting = merged["Easting"].to_numpy(dtype="float64")
    northing = merged["Northing"].to_numpy(dtype="float64")
    if not np.isfinite(easting).all() or not np.isfinite(northing).all():
        raise GraphError("Easting/Northing contain NaN or infinite values.")
    merged = merged.sort_values("node_index").reset_index(drop=True)
    if merged["node_index"].tolist() != list(range(len(merged))):
        raise GraphError("Coordinate table is not in node_index order.")
    return merged[[NODE_KEY, "node_index", "Easting", "Northing"]], source


def _normalise_coord_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rename: dict[str, str] = {}
    if NODE_KEY not in frame.columns:
        for alias in ("InterZone", "IZ_CODE", "iz_code"):
            if alias in frame.columns:
                rename[alias] = NODE_KEY
                break
    if "Easting" not in frame.columns:
        for alias in ("X", "x"):
            if alias in frame.columns:
                rename[alias] = "Easting"
                break
    if "Northing" not in frame.columns:
        for alias in ("Y", "y"):
            if alias in frame.columns:
                rename[alias] = "Northing"
                break
    out = frame.rename(columns=rename).copy()
    if NODE_KEY not in out.columns or "Easting" not in out.columns or "Northing" not in out.columns:
        raise GraphError("Coordinate table must contain IntZone (or InterZone) and Easting/Northing (or X/Y).")
    out[NODE_KEY] = out[NODE_KEY].astype("string").str.strip()
    return out[[NODE_KEY, "Easting", "Northing"]]


def _collapse_iz_coordinates(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse identical IZ repeats. Conflicting duplicates are an error."""
    if not frame[NODE_KEY].duplicated().any():
        return frame.reset_index(drop=True)
    rows: list[pd.Series] = []
    for code, part in frame.groupby(NODE_KEY, sort=False):
        easting = part["Easting"].to_numpy(dtype="float64")
        northing = part["Northing"].to_numpy(dtype="float64")
        if not np.isfinite(easting).all() or not np.isfinite(northing).all():
            raise GraphError(f"Duplicate IntZone {code} has NaN or infinite coordinates.")
        if float(np.ptp(easting)) > COORD_DUP_EPS_M or float(np.ptp(northing)) > COORD_DUP_EPS_M:
            raise GraphError(f"Duplicate IntZone {code} has conflicting Easting/Northing.")
        rows.append(part.iloc[0])
    return pd.DataFrame(rows).reset_index(drop=True)


def _load_aligned_polygons(
    nodes: pd.DataFrame,
    polygons: gpd.GeoDataFrame | None,
    boundaries_path: Path | None,
) -> tuple[gpd.GeoDataFrame, list[str], int, str]:
    if polygons is None:
        path = (
            Path(boundaries_path)
            if boundaries_path is not None
            else project_root() / BOUNDARY_RELATIVE_PATH
        )
        polygons, source = _load_boundaries(path)
    else:
        polygons = polygons.copy()
        source = "supplied GeoDataFrame"
        from graph.geo import _normalise_polygon_frame

        polygons = _normalise_polygon_frame(polygons)
    aligned, unused, repaired = _align_polygons(nodes, polygons)
    return aligned, unused, repaired, source


def _load_line_layer(
    supplied: gpd.GeoDataFrame | Path | None,
    relative_path: Path,
    label: str,
) -> tuple[gpd.GeoDataFrame, str]:
    if supplied is None:
        path = project_root() / relative_path
        if not path.exists():
            raise GraphError(f"{label} shapefile missing: {path}")
        frame = gpd.read_file(path)
        source = str(path)
    elif isinstance(supplied, (str, Path)):
        path = Path(supplied)
        if not path.exists():
            raise GraphError(f"{label} shapefile missing: {path}")
        frame = gpd.read_file(path)
        source = str(path)
    else:
        frame = supplied.copy()
        source = f"supplied {label} GeoDataFrame"
    return _normalise_line_frame(frame, label), source


def _load_station_layer(
    supplied: gpd.GeoDataFrame | Path | None,
) -> tuple[gpd.GeoDataFrame, str]:
    if supplied is None:
        path = project_root() / STATIONS_RELATIVE_PATH
        if not path.exists():
            raise GraphError(f"Station shapefile missing: {path}")
        frame = gpd.read_file(path)
        source = str(path)
    elif isinstance(supplied, (str, Path)):
        path = Path(supplied)
        if not path.exists():
            raise GraphError(f"Station shapefile missing: {path}")
        frame = gpd.read_file(path)
        source = str(path)
    else:
        frame = supplied.copy()
        source = "supplied stations GeoDataFrame"
    return _normalise_station_frame(frame), source


def _require_crs(frame: gpd.GeoDataFrame, label: str) -> gpd.GeoDataFrame:
    if not isinstance(frame, gpd.GeoDataFrame):
        raise GraphError(f"{label} input must be a GeoDataFrame.")
    if frame.crs is None:
        raise GraphError(f"{label} geometries have no CRS; refusing to guess.")
    return frame.to_crs(TARGET_CRS)


def _normalise_line_frame(frame: gpd.GeoDataFrame, label: str) -> gpd.GeoDataFrame:
    out = _require_crs(frame, label)
    if "fclass" not in out.columns:
        raise GraphError(f"{label} table must contain fclass.")
    source_has_layer = "layer" in out.columns
    source_has_bridge = "bridge" in out.columns
    source_has_tunnel = "tunnel" in out.columns
    if "oneway" not in out.columns:
        out["oneway"] = "B"
    else:
        oneway = out["oneway"].astype("string").str.strip().str.upper()
        oneway = oneway.mask(oneway.isin(MISSING_ONEWAY) | oneway.isna(), "B")
        out["oneway"] = oneway
    if not source_has_layer:
        out["layer"] = 0
    if not source_has_bridge:
        out["bridge"] = "F"
    if not source_has_tunnel:
        out["tunnel"] = "F"
    out["source_has_layer"] = source_has_layer
    out["source_has_bridge"] = source_has_bridge
    out["source_has_tunnel"] = source_has_tunnel
    out["fclass"] = out["fclass"].astype("string").str.strip().str.lower()
    out["oneway"] = out["oneway"].astype("string").str.strip().str.upper()
    out["bridge"] = out["bridge"].astype("string").str.strip().str.upper()
    out["tunnel"] = out["tunnel"].astype("string").str.strip().str.upper()
    if out.geometry.isna().any():
        raise GraphError(f"{label} table contains missing geometries.")
    return out


def _normalise_station_frame(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = _require_crs(frame, "stations")
    if "fclass" not in out.columns:
        raise GraphError("Station table must contain fclass.")
    out["fclass"] = out["fclass"].astype("string").str.strip().str.lower()
    stations = out.loc[out["fclass"].isin(STATION_FCLASS)].copy()
    if stations.geometry.isna().any():
        raise GraphError("Station table contains missing geometries.")
    for geom in stations.geometry:
        if geom.geom_type != "Point":
            raise GraphError("Railway stations must be Point geometries.")
    return stations.reset_index(drop=True)


def _filter_fclass(frame: gpd.GeoDataFrame, allowed: frozenset[str], *, layer: str) -> gpd.GeoDataFrame:
    fclass = frame["fclass"].astype(str).str.strip().str.lower()
    keep = fclass.isin(allowed) & ~fclass.isin(UNKNOWN_FCLASS)
    return frame.loc[keep].copy()


def _clip_lines(frame: gpd.GeoDataFrame, clip_geom: BaseGeometry) -> gpd.GeoDataFrame:
    if frame.empty:
        return frame.iloc[0:0].copy()
    clip_frame = gpd.GeoDataFrame(geometry=[clip_geom], crs=TARGET_CRS)
    clipped = gpd.clip(frame, clip_frame)
    if clipped.empty:
        return clipped
    clipped = clipped.explode(index_parts=False)
    clipped = clipped.loc[~clipped.geometry.is_empty].copy()
    return clipped.reset_index(drop=True)


def _clip_points(frame: gpd.GeoDataFrame, clip_geom: BaseGeometry) -> gpd.GeoDataFrame:
    if frame.empty:
        return frame.iloc[0:0].copy()
    keep = frame.geometry.intersects(clip_geom)
    return frame.loc[keep].reset_index(drop=True)


def _deduplicate_stations(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if frame.empty:
        return frame
    keys = [_coord_key(geom.x, geom.y) for geom in frame.geometry]
    out = frame.copy()
    out["_xy"] = keys
    out = out.drop_duplicates("_xy", keep="first").drop(columns=["_xy"])
    return out.reset_index(drop=True)


def _coord_key(x: float, y: float) -> tuple[float, float]:
    return (round(float(x), NODE_COORD_DIGITS), round(float(y), NODE_COORD_DIGITS))


def _iter_line_parts(geom: BaseGeometry | None):
    if geom is None or geom.is_empty:
        return
    if geom.geom_type == "LineString":
        yield geom
        return
    if geom.geom_type in {"MultiLineString", "GeometryCollection"}:
        for part in geom.geoms:
            yield from _iter_line_parts(part)


def _directions(mode: str, fclass: str, oneway: str) -> tuple[bool, bool]:
    """Return (forward, backward) along the digitised direction.

    Geofabrik shapefile oneway: F = forward only, T = reverse only, B = both.
    Missing oneway defaults to B. Any other token is an error. Walking is
    two-way. Bicycle on dedicated cycleway/path is two-way; bicycle on the
    carriageway follows oneway.
    """
    token = _oneway_token(oneway)
    if mode == "walking":
        return True, True
    if mode == "bicycle" and fclass in DEDICATED_BIKE_FCLASS:
        return True, True
    if token == "B":
        return True, True
    if token == "F":
        return True, False
    return False, True


def _oneway_token(oneway: str) -> str:
    token = str(oneway if oneway is not None else "B").strip().upper()
    if token in MISSING_ONEWAY:
        return "B"
    if token not in {"F", "T", "B"}:
        raise GraphError(
            f"Unsupported Geofabrik oneway value {oneway!r}; expected F, T or B."
        )
    return token


def _empty_graph() -> RouteGraph:
    return RouteGraph(
        node_xy=np.zeros((0, 2), dtype="float64"),
        length_m=sparse.csr_matrix((0, 0), dtype="float64"),
    )


def _is_missing_grade_value(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return True
    token = str(value).strip().upper()
    return token in MISSING_ONEWAY


def _parse_grade(
    row: Any,
    *,
    source_has_layer: bool,
    source_has_bridge: bool,
    source_has_tunnel: bool,
) -> dict[str, Any]:
    """Working grade key plus whether the source actually supplied each field.

    Missing columns are filled with layer=0 / bridge=F / tunnel=F for grouping
    only. Those defaults are never treated as complete observed grade data.
    """
    layer_val = getattr(row, "layer", 0)
    bridge_val = getattr(row, "bridge", "F")
    tunnel_val = getattr(row, "tunnel", "F")
    missing_layer_value = source_has_layer and _is_missing_grade_value(layer_val)
    missing_bridge_value = source_has_bridge and _is_missing_grade_value(bridge_val)
    missing_tunnel_value = source_has_tunnel and _is_missing_grade_value(tunnel_val)
    layer = 0
    if source_has_layer and not missing_layer_value:
        try:
            layer = int(layer_val)
        except (TypeError, ValueError):
            missing_layer_value = True
            layer = 0
    bridge = False
    if source_has_bridge and not missing_bridge_value:
        token = str(bridge_val).strip().upper()
        if token not in {"T", "F"}:
            missing_bridge_value = True
        else:
            bridge = token == "T"
    tunnel = False
    if source_has_tunnel and not missing_tunnel_value:
        token = str(tunnel_val).strip().upper()
        if token not in {"T", "F"}:
            missing_tunnel_value = True
        else:
            tunnel = token == "T"
    complete = (
        source_has_layer
        and source_has_bridge
        and source_has_tunnel
        and not missing_layer_value
        and not missing_bridge_value
        and not missing_tunnel_value
    )
    return {
        "grade": (layer, bridge, tunnel),
        "complete": complete,
        "source_has_layer": source_has_layer,
        "source_has_bridge": source_has_bridge,
        "source_has_tunnel": source_has_tunnel,
        "missing_layer_value": missing_layer_value,
        "missing_bridge_value": missing_bridge_value,
        "missing_tunnel_value": missing_tunnel_value,
    }


def _intersection_is_endpoint_only(left: BaseGeometry, right: BaseGeometry, inter: BaseGeometry) -> bool:
    if inter.is_empty:
        return True
    if inter.geom_type in {"LineString", "MultiLineString"} and inter.length > LINE_LENGTH_EPS:
        return False
    points: list[BaseGeometry] = []
    if inter.geom_type == "Point":
        points = [inter]
    elif inter.geom_type == "MultiPoint":
        points = list(inter.geoms)
    elif inter.geom_type == "GeometryCollection":
        for part in inter.geoms:
            if part.geom_type in {"LineString", "MultiLineString"} and part.length > LINE_LENGTH_EPS:
                return False
            if part.geom_type == "Point":
                points.append(part)
            elif part.geom_type == "MultiPoint":
                points.extend(list(part.geoms))
    else:
        return False
    ends = {
        _coord_key(*left.coords[0]),
        _coord_key(*left.coords[-1]),
        _coord_key(*right.coords[0]),
        _coord_key(*right.coords[-1]),
    }
    return all(_coord_key(point.x, point.y) in ends for point in points)


def _line_records(frame: gpd.GeoDataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    flag_layer = "source_has_layer" in frame.columns
    flag_bridge = "source_has_bridge" in frame.columns
    flag_tunnel = "source_has_tunnel" in frame.columns
    inferred_layer = "layer" in frame.columns
    inferred_bridge = "bridge" in frame.columns
    inferred_tunnel = "tunnel" in frame.columns
    for row in frame.itertuples(index=False):
        parsed = _parse_grade(
            row,
            source_has_layer=bool(row.source_has_layer) if flag_layer else inferred_layer,
            source_has_bridge=bool(row.source_has_bridge) if flag_bridge else inferred_bridge,
            source_has_tunnel=bool(row.source_has_tunnel) if flag_tunnel else inferred_tunnel,
        )
        for part in _iter_line_parts(row.geometry):
            records.append(
                {
                    "geom": part,
                    "fclass": str(row.fclass),
                    "oneway": str(row.oneway),
                    **parsed,
                }
            )
    return records


def _crossing_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    report = _empty_crossing_report()
    if records:
        report.update(_source_grade_fields_report(records))
    if len(records) < 2:
        return report
    geoms = [rec["geom"] for rec in records]
    tree = STRtree(geoms)
    examples: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for i, rec in enumerate(records):
        hits = np.asarray(tree.query(rec["geom"]), dtype="int64")
        for raw in hits:
            j = int(raw)
            if j <= i:
                continue
            pair = (i, j)
            if pair in seen:
                continue
            seen.add(pair)
            other = records[j]
            inter = rec["geom"].intersection(other["geom"])
            if _intersection_is_endpoint_only(rec["geom"], other["geom"], inter):
                continue
            same_grade = rec["grade"] == other["grade"]
            both_complete = rec["complete"] and other["complete"]
            if same_grade and both_complete:
                continue
            if (not same_grade) and both_complete:
                report["n_grade_separated_crossings_ignored"] += 1
                continue
            report["n_ambiguous_crossings"] += 1
            if len(examples) < AMBIGUOUS_CROSSING_EXAMPLE_CAP:
                point = inter.representative_point() if not inter.is_empty else rec["geom"].centroid
                examples.append(
                    {
                        "x": float(point.x),
                        "y": float(point.y),
                        "left_grade": list(rec["grade"]),
                        "right_grade": list(other["grade"]),
                        "left_complete": rec["complete"],
                        "right_complete": other["complete"],
                    }
                )
    report["ambiguous_crossing_examples"] = examples
    report.update(_source_grade_fields_report(records))
    return report


def _source_grade_fields_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    return {
        "source_has_layer": n > 0 and all(rec["source_has_layer"] for rec in records),
        "source_has_bridge": n > 0 and all(rec["source_has_bridge"] for rec in records),
        "source_has_tunnel": n > 0 and all(rec["source_has_tunnel"] for rec in records),
        "n_records_without_source_layer": int(sum(not rec["source_has_layer"] for rec in records)),
        "n_records_without_source_bridge": int(sum(not rec["source_has_bridge"] for rec in records)),
        "n_records_without_source_tunnel": int(sum(not rec["source_has_tunnel"] for rec in records)),
        "n_records_missing_layer_value": int(sum(rec["missing_layer_value"] for rec in records)),
        "n_records_missing_bridge_value": int(sum(rec["missing_bridge_value"] for rec in records)),
        "n_records_missing_tunnel_value": int(sum(rec["missing_tunnel_value"] for rec in records)),
        "defaulted_grade_not_treated_as_complete": True,
    }


def _lines_to_graph(frame: gpd.GeoDataFrame, *, mode: str) -> tuple[RouteGraph, dict[str, Any]]:
    """Node at-grade intersections only. Edge weight is geometric length in metres."""
    if frame.empty:
        return _empty_graph(), _empty_crossing_report()
    records = _line_records(frame)
    if not records:
        return _empty_graph(), _empty_crossing_report()
    crossings = _crossing_report(records)

    groups: dict[tuple[int, bool, bool], list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        groups[rec["grade"]].append(rec)

    node_to_i: dict[tuple[float, float], int] = {}
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    def node_id(xy: tuple[float, float]) -> int:
        index = node_to_i.get(xy)
        if index is None:
            index = len(node_to_i)
            node_to_i[xy] = index
        return index

    def add_edge(source_xy: tuple[float, float], target_xy: tuple[float, float], length_m: float) -> None:
        if source_xy == target_xy:
            return
        rows.append(node_id(source_xy))
        cols.append(node_id(target_xy))
        data.append(length_m)

    for recs in groups.values():
        originals = [rec["geom"] for rec in recs]
        attributes = [(rec["fclass"], rec["oneway"]) for rec in recs]
        noded_parts = list(_iter_line_parts(unary_union(originals)))
        tree = STRtree(originals)
        for segment in noded_parts:
            if segment.length <= LINE_LENGTH_EPS:
                continue
            midpoint = segment.interpolate(0.5)
            hits = np.asarray(tree.query(midpoint), dtype="int64")
            if hits.size == 0:
                continue
            best_i = int(hits[np.argmin([originals[int(i)].distance(midpoint) for i in hits])])
            if originals[best_i].distance(midpoint) > SEGMENT_MATCH_MAX_M:
                continue
            fclass, oneway = attributes[best_i]
            start = _coord_key(*segment.coords[0])
            end = _coord_key(*segment.coords[-1])
            length_m = float(segment.length)
            forward, backward = _directions(mode, fclass, oneway)
            if forward:
                add_edge(start, end, length_m)
            if backward:
                add_edge(end, start, length_m)

    if not node_to_i:
        return _empty_graph(), crossings
    n = len(node_to_i)
    node_xy = np.zeros((n, 2), dtype="float64")
    for xy, index in node_to_i.items():
        node_xy[index] = xy
    if not data:
        return RouteGraph(node_xy=node_xy, length_m=sparse.csr_matrix((n, n), dtype="float64")), crossings
    edges = pd.DataFrame({"row": rows, "col": cols, "length_m": data})
    edges = edges.groupby(["row", "col"], as_index=False)["length_m"].min()
    length = sparse.csr_matrix(
        (
            edges["length_m"].to_numpy(dtype="float64"),
            (edges["row"].to_numpy(dtype="int64"), edges["col"].to_numpy(dtype="int64")),
        ),
        shape=(n, n),
        dtype="float64",
    )
    return RouteGraph(node_xy=node_xy, length_m=length), crossings


def _point_frame(stations: gpd.GeoDataFrame) -> np.ndarray:
    if stations.empty:
        return np.zeros((0, 2), dtype="float64")
    return np.column_stack(
        [
            stations.geometry.x.to_numpy(dtype="float64"),
            stations.geometry.y.to_numpy(dtype="float64"),
        ]
    )


def _snap_points(coords: pd.DataFrame, graph: RouteGraph, snap_max_m: float) -> SnapResult:
    xy = coords[["Easting", "Northing"]].to_numpy(dtype="float64")
    return _snap_xy(xy, graph, snap_max_m)


def _snap_xy(xy: np.ndarray, graph: RouteGraph, snap_max_m: float) -> SnapResult:
    n = int(xy.shape[0])
    snapped = _empty_snap(n)
    if n == 0 or graph.n_nodes == 0:
        return snapped
    distance, index = cKDTree(graph.node_xy).query(xy, k=1)
    for i, (dist, node_i) in enumerate(
        zip(np.atleast_1d(distance), np.atleast_1d(index), strict=True)
    ):
        if np.isfinite(dist) and float(dist) <= snap_max_m:
            snapped.node_index[i] = int(node_i)
            snapped.distance_m[i] = float(dist)
            snapped.snap_x[i] = float(graph.node_xy[int(node_i), 0])
            snapped.snap_y[i] = float(graph.node_xy[int(node_i), 1])
    return snapped


def _od_distances(
    graph: RouteGraph,
    sources: np.ndarray,
    targets: np.ndarray,
    source_snap_m: np.ndarray | None = None,
    target_snap_m: np.ndarray | None = None,
) -> np.ndarray:
    """Shortest-path kilometres plus endpoint snap distances. Unreachable stays Inf."""
    source_idx = np.asarray(sources, dtype="int64")
    target_idx = np.asarray(targets, dtype="int64")
    n_s = int(source_idx.size)
    n_t = int(target_idx.size)
    out = np.full((n_s, n_t), np.inf, dtype="float64")
    if graph.n_nodes == 0:
        return out
    valid_sources = np.flatnonzero(source_idx >= 0)
    if valid_sources.size == 0:
        return out
    dist = csgraph.dijkstra(
        graph.length_m,
        directed=True,
        indices=source_idx[valid_sources],
        unweighted=False,
    )
    if dist.ndim == 1:
        dist = dist[None, :]
    src_snap = None if source_snap_m is None else np.asarray(source_snap_m, dtype="float64")
    tgt_snap = None if target_snap_m is None else np.asarray(target_snap_m, dtype="float64")
    for row, origin in enumerate(valid_sources):
        origin_extra = 0.0 if src_snap is None else float(src_snap[int(origin)])
        if src_snap is not None and not np.isfinite(origin_extra):
            continue
        for j, target in enumerate(target_idx):
            if target < 0:
                continue
            metres = dist[row, int(target)]
            if not np.isfinite(metres):
                continue
            dest_extra = 0.0 if tgt_snap is None else float(tgt_snap[j])
            if tgt_snap is not None and not np.isfinite(dest_extra):
                continue
            out[int(origin), j] = (float(metres) + origin_extra + dest_extra) / 1000.0
    return out


def _pairwise_distances(graph: RouteGraph, snaps: SnapResult) -> np.ndarray:
    matrix = _od_distances(
        graph,
        snaps.node_index,
        snaps.node_index,
        source_snap_m=snaps.distance_m,
        target_snap_m=snaps.distance_m,
    )
    np.fill_diagonal(matrix, 0.0)
    return matrix


def _snap_status(ok: bool, distance_m: float, long_snap_m: float) -> str:
    if not ok:
        return "unsnapped"
    if np.isfinite(distance_m) and float(distance_m) > long_snap_m:
        return "ok_long"
    return "ok"


def _count_long_snaps(snap: SnapResult, long_snap_m: float) -> int:
    return int(np.sum(snap.ok & np.isfinite(snap.distance_m) & (snap.distance_m > long_snap_m)))


def _iz_snap_table(
    nodes: pd.DataFrame,
    snaps: dict[str, SnapResult],
    *,
    snap_max_m: float,
) -> pd.DataFrame:
    long_snap_m = _long_snap_threshold_m(snap_max_m)
    rows: list[dict[str, Any]] = []
    for mode, snap in snaps.items():
        for i, node_row in nodes.iterrows():
            del i
            idx = int(node_row["node_index"])
            ok = bool(snap.ok[idx])
            distance = float(snap.distance_m[idx]) if ok else np.inf
            rows.append(
                {
                    "IntZone": str(node_row[NODE_KEY]),
                    "node_index": idx,
                    "mode": mode,
                    "snap_ok": ok,
                    "snap_status": _snap_status(ok, distance, long_snap_m),
                    "snap_distance_m": distance,
                    "snap_node_index": int(snap.node_index[idx]),
                    "snap_x": float(snap.snap_x[idx]) if ok else np.nan,
                    "snap_y": float(snap.snap_y[idx]) if ok else np.nan,
                    "Easting": float(node_row["Easting"]),
                    "Northing": float(node_row["Northing"]),
                }
            )
    return pd.DataFrame(rows)


def _station_snap_table(
    stations: gpd.GeoDataFrame,
    snaps: dict[str, SnapResult],
    *,
    snap_max_m: float,
) -> pd.DataFrame:
    long_snap_m = _long_snap_threshold_m(snap_max_m)
    rows: list[dict[str, Any]] = []
    if stations.empty:
        return pd.DataFrame(
            columns=[
                "station_id",
                "fclass",
                "mode",
                "snap_ok",
                "snap_status",
                "snap_distance_m",
                "snap_node_index",
                "snap_x",
                "snap_y",
                "Easting",
                "Northing",
            ]
        )
    ids = stations["osm_id"].astype(str).tolist()
    fclass = stations["fclass"].astype(str).tolist()
    xs = stations.geometry.x.to_numpy(dtype="float64")
    ys = stations.geometry.y.to_numpy(dtype="float64")
    for mode, snap in snaps.items():
        for i, station_id in enumerate(ids):
            ok = bool(snap.ok[i])
            distance = float(snap.distance_m[i]) if ok else np.inf
            rows.append(
                {
                    "station_id": station_id,
                    "fclass": fclass[i],
                    "mode": mode,
                    "snap_ok": ok,
                    "snap_status": _snap_status(ok, distance, long_snap_m),
                    "snap_distance_m": distance,
                    "snap_node_index": int(snap.node_index[i]),
                    "snap_x": float(snap.snap_x[i]) if ok else np.nan,
                    "snap_y": float(snap.snap_y[i]) if ok else np.nan,
                    "Easting": float(xs[i]),
                    "Northing": float(ys[i]),
                }
            )
    return pd.DataFrame(rows)


def _finite_offdiag(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype="float64").copy()
    np.fill_diagonal(values, np.nan)
    finite = values[np.isfinite(values)]
    return finite


def _distance_stats(matrix: np.ndarray) -> dict[str, Any]:
    n = int(matrix.shape[0])
    offdiag = n * n - n
    finite = _finite_offdiag(matrix)
    unreachable = int(offdiag - int(finite.size))
    if finite.size == 0:
        return {
            "n_unreachable_pairs": unreachable,
            "n_finite_pairs": 0,
            "min_km": None,
            "median_km": None,
            "max_km": None,
        }
    return {
        "n_unreachable_pairs": unreachable,
        "n_finite_pairs": int(finite.size),
        "min_km": float(np.min(finite)),
        "median_km": float(np.median(finite)),
        "max_km": float(np.max(finite)),
    }


def _validation_report(
    *,
    nodes_out: pd.DataFrame,
    edges: pd.DataFrame,
    adjacency: sparse.csr_matrix,
    distances: dict[str, np.ndarray],
    multimodal: np.ndarray,
    shortest_mode: np.ndarray,
    tau: float,
    k: int,
    buffer_m: float,
    snap_max_m: float,
    unused: list[str],
    repaired: int,
    node_source: str,
    coord_source: str,
    boundary_source: str,
    roads_source: str,
    railways_source: str,
    stations_source: str,
    n_road_edges: int,
    n_bicycle_edges: int,
    n_walking_edges: int,
    n_rail_edges: int,
    n_stations: int,
    n_iz_unsnapped: dict[str, int],
    n_stations_unsnapped: dict[str, int],
    n_iz_long_accepted_snap: dict[str, int],
    n_station_long_accepted_snap: dict[str, int],
    long_accepted_snap_threshold_m: float,
    crossing_reports: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    n = int(len(nodes_out))
    binary = adjacency.copy()
    binary.data = np.ones_like(binary.data)
    undirected = binary.maximum(binary.T)
    n_components = int(csgraph.connected_components(undirected, directed=False)[0]) if n else 0
    in_degree = np.asarray(adjacency.sum(axis=0)).ravel()
    out_degree = np.asarray((adjacency > 0).sum(axis=1)).ravel()
    isolated_mask = (in_degree == 0) & (out_degree == 0)
    isolated = nodes_out.loc[isolated_mask, NODE_KEY].astype(str).tolist()
    offdiag = np.isfinite(multimodal) & ~np.eye(n, dtype=bool)
    finite_mode = shortest_mode[offdiag]
    share = {
        name: float(np.mean(finite_mode == name)) if finite_mode.size else 0.0
        for name in MODE_ORDER
    }
    return {
        "study_area": LOCAL_AUTHORITY_NAME,
        "local_authority_code": LOCAL_AUTHORITY_CODE,
        "geography_type": "2011 Intermediate Zone",
        "geography_vintage": GEOGRAPHY_VINTAGE,
        "graph_kind": "road_network_distance",
        "not_travel_time": True,
        "not_realtime_traffic": True,
        "not_observed_mobility": True,
        "not_demand_or_mode_choice": True,
        "maxspeed_not_used": True,
        "dodgr_d_weighted_not_used": True,
        "euclidean_nearest_station_not_used": True,
        "snap_distances_included_in_total": True,
        "snap_distance_rule": "origin_snap_m + network_path_m + destination_snap_m",
        "grade_separated_2d_intersections_not_connected": True,
        "inf_not_filled_with_zero": True,
        "coordinates_are_epsg27700_metres": True,
        "clip": "st_buffer(st_union(city_boundary), buffer_m) once",
        "crs": TARGET_CRS,
        "buffer_m": float(buffer_m),
        "snap_max_m": float(snap_max_m),
        "k": int(k),
        "tau_km": float(tau),
        "weight": "exp(-d_km / tau)",
        "self_loops": False,
        "directed": True,
        "symmetrised": False,
        "n_nodes": n,
        "expected_iz_count": EXPECTED_IZ_COUNT,
        "n_edges": int(len(edges)),
        "n_adjacency_nonzero": int(adjacency.nnz),
        "adjacency_matrix_shape": list(adjacency.shape),
        "is_symmetric": bool((adjacency - adjacency.T).nnz == 0),
        "diagonal_is_zero": bool(np.all(adjacency.diagonal() == 0)),
        "n_isolated": int(len(isolated)),
        "isolated_iz": isolated,
        "n_weakly_connected_components": n_components,
        "out_degree_min": int(out_degree.min()) if n else 0,
        "out_degree_max": int(out_degree.max()) if n else 0,
        "n_boundary_polygons_not_in_node_order": int(len(unused)),
        "n_geometries_repaired": int(repaired),
        "node_order_source": node_source,
        "coordinate_source": coord_source,
        "boundary_source": boundary_source,
        "roads_source": roads_source,
        "railways_source": railways_source,
        "stations_source": stations_source,
        "n_road_network_edges": n_road_edges,
        "n_bicycle_network_edges": n_bicycle_edges,
        "n_walking_network_edges": n_walking_edges,
        "n_rail_network_edges": n_rail_edges,
        "n_stations": n_stations,
        "n_iz_unsnapped": n_iz_unsnapped,
        "n_stations_unsnapped": n_stations_unsnapped,
        "n_iz_long_accepted_snap": n_iz_long_accepted_snap,
        "n_station_long_accepted_snap": n_station_long_accepted_snap,
        "long_accepted_snap_threshold_m": float(long_accepted_snap_threshold_m),
        "tram_and_tram_stop_excluded": True,
        "defaulted_grade_not_treated_as_complete": True,
        "crossing_reports": crossing_reports,
        "distance_stats": {name: _distance_stats(distances[name]) for name in MODE_ORDER},
        "multimodal_distance_stats": _distance_stats(multimodal),
        "shortest_mode_share": share,
        "node_index_sequence": nodes_out["node_index"].tolist(),
        "iz_codes": nodes_out[NODE_KEY].astype(str).tolist(),
        "processing_timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _planned_output_paths(output_dir: Path) -> list[Path]:
    return [output_dir / name for name in OUTPUT_FILENAMES]


def _assert_overwrite_allowed(output_dir: Path, overwrite: bool) -> None:
    existing = [path for path in _planned_output_paths(output_dir) if path.exists()]
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise GraphError(
            f"Output already exists ({names}). Pass overwrite=True or omit --no-overwrite to replace "
            "only these road graph files."
        )


def _write_distance_csv(matrix: np.ndarray, nodes: pd.DataFrame, path: Path) -> Path:
    labels = nodes[NODE_KEY].astype(str).tolist()
    frame = pd.DataFrame(matrix, index=labels, columns=labels)
    formatted = frame.where(np.isfinite(matrix), "Inf")
    path.parent.mkdir(parents=True, exist_ok=True)
    formatted.to_csv(path, index_label=NODE_KEY)
    return path


def _write_outputs(
    output_dir: Path,
    nodes_out: pd.DataFrame,
    edges: pd.DataFrame,
    adjacency: sparse.csr_matrix,
    distances: dict[str, np.ndarray],
    multimodal: np.ndarray,
    shortest_mode: np.ndarray,
    report: dict[str, Any],
    iz_snap_table: pd.DataFrame,
    station_snap_table: pd.DataFrame,
) -> dict[str, str]:
    node_path = write_table(nodes_out, output_dir / "nodes.csv")
    edge_path = write_table(edges, output_dir / "edges.csv")
    adj_path = output_dir / "adjacency_road.npz"
    sparse.save_npz(adj_path, adjacency)
    dist_path = output_dir / "distances.npz"
    np.savez_compressed(
        dist_path,
        D_road=distances["road"],
        D_bicycle=distances["bicycle"],
        D_walking=distances["walking"],
        D_road_rail=distances["road+rail"],
        D_walking_rail=distances["walking+rail"],
        D_multimodal=multimodal,
        shortest_mode=np.asarray(shortest_mode, dtype="U32"),
        iz_codes=nodes_out[NODE_KEY].astype(str).to_numpy(),
        node_index=nodes_out["node_index"].to_numpy(dtype="int64"),
    )
    labels = nodes_out[NODE_KEY].astype(str).tolist()
    mode_path = output_dir / "shortest_mode.csv"
    pd.DataFrame(shortest_mode, index=labels, columns=labels).to_csv(
        mode_path, index_label=NODE_KEY
    )
    for name, filename in (
        ("road", "D_road_km.csv"),
        ("bicycle", "D_bicycle_km.csv"),
        ("walking", "D_walking_km.csv"),
        ("road+rail", "D_road_rail_km.csv"),
        ("walking+rail", "D_walking_rail_km.csv"),
    ):
        _write_distance_csv(distances[name], nodes_out, output_dir / filename)
    _write_distance_csv(multimodal, nodes_out, output_dir / "D_multimodal_km.csv")
    iz_snap_path = write_table(iz_snap_table, output_dir / "iz_snaps.csv")
    station_snap_path = write_table(station_snap_table, output_dir / "station_snaps.csv")
    report_path = write_json(report, output_dir / "validation_report.json")
    return {
        "nodes": str(node_path),
        "edges": str(edge_path),
        "adjacency": str(adj_path),
        "distances": str(dist_path),
        "shortest_mode": str(mode_path),
        "iz_snaps": str(iz_snap_path),
        "station_snaps": str(station_snap_path),
        "validation_report": str(report_path),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build the Edinburgh 2011 IZ road network-distance graph."
    )
    parser.add_argument(
        "--nodes",
        type=Path,
        help="Node-order CSV with IntZone and node_index (default: COVID IZ master).",
    )
    parser.add_argument(
        "--coords",
        type=Path,
        help="IZ coordinates with Easting/Northing (default: 2011 population-weighted centroids).",
    )
    parser.add_argument(
        "--boundaries",
        type=Path,
        help="2011 IZ polygon shapefile (used only to build the clip buffer).",
    )
    parser.add_argument("--roads", type=Path, help="OSM roads shapefile (gis_osm_roads_free_1).")
    parser.add_argument("--railways", type=Path, help="OSM railways shapefile.")
    parser.add_argument("--stations", type=Path, help="OSM transport points shapefile.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output folder (default: data/results/graph/road).",
    )
    parser.add_argument("--buffer-m", type=float, default=DEFAULT_BUFFER_M)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--snap-max-m", type=float, default=DEFAULT_SNAP_MAX_M)
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Replace existing road graph output files in the chosen directory (default: True).",
    )
    args = parser.parse_args(argv)
    result = construct_road_graph(
        nodes=args.nodes,
        coords=args.coords,
        boundaries_path=args.boundaries,
        roads=args.roads,
        railways=args.railways,
        stations=args.stations,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        buffer_m=args.buffer_m,
        k=args.k,
        snap_max_m=args.snap_max_m,
    )
    printable = {key: value for key, value in result.items() if key != "validation_report"}
    print(json.dumps(printable, indent=2, default=str))


if __name__ == "__main__":
    main()
