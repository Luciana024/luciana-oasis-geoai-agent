"""Geographic rook-adjacency graph for Edinburgh 2011 Intermediate Zones.

The node order is taken from the existing COVID IZ master (or an explicit
node-order table). Shapefile row order is never used as node_index.

Two IZs are neighbours only when they share a boundary line (rook
contiguity). Point-only contact is not an edge. Isolated IZs are reported
and left isolated. Raw boundary files are not modified.

    PYTHONPATH=src python -m graph.geo
    PYTHONPATH=src python -m graph geo --nodes data/results/forecast/L7_H7_S1_20200308_20230225/node_order.csv
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csgraph

from data.covid import load_iz_master
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
    from shapely.geometry.base import BaseGeometry
    from shapely.validation import make_valid
except ImportError as exc:  # pragma: no cover - environment missing GIS stack
    raise ImportError(
        "graph construction requires geopandas and shapely. "
        "Install them before running this module."
    ) from exc

LOGGER = get_logger("graph.geo")

TARGET_CRS = "EPSG:27700"
POLYGON_TYPES = frozenset({"Polygon", "MultiPolygon"})
# Shared-boundary length below this (metres in EPSG:27700) is treated as a point.
LINE_LENGTH_EPS = 1e-6
# Interior overlap above this (square metres) is a topology error, not a missing edge.
AREA_OVERLAP_EPS = 0.01
BOUNDARY_RELATIVE_PATH = (
    Path("data")
    / "raw"
    / "boundaries"
    / "SG_IntermediateZoneBdry_2011"
    / "SG_IntermediateZone_Bdry_2011.shp"
)
OUTPUT_FILENAMES = (
    "nodes.csv",
    "edges.csv",
    "adjacency_geo.npz",
    "validation_report.json",
)
IZ_CODE_CANDIDATES = (NODE_KEY, "InterZone", "IZ_CODE", "iz_code")


class GraphError(ValueError):
    """Adjacency-graph construction cannot continue without guessing."""


def construct_adjacency_graph(
    nodes: pd.DataFrame | Path | None = None,
    boundaries_path: Path | None = None,
    polygons: gpd.GeoDataFrame | None = None,
    output_dir: Path | None = None,
    overwrite: bool = True,
    area_code: str = LOCAL_AUTHORITY_CODE,
) -> dict[str, Any]:
    """Build an undirected rook-contiguity graph aligned to COVID node_index.

    Does not train a model and does not change COVID or forecast outputs.
    """
    node_table, node_source = _load_nodes(nodes, area_code=area_code)
    if polygons is None:
        boundaries_path = (
            Path(boundaries_path)
            if boundaries_path is not None
            else project_root() / BOUNDARY_RELATIVE_PATH
        )
        polygons, boundary_source = _load_boundaries(boundaries_path)
    else:
        polygons = polygons.copy()
        boundary_source = "supplied GeoDataFrame"
        polygons = _normalise_polygon_frame(polygons)

    aligned, unused, repaired = _align_polygons(node_table, polygons)
    edges = _rook_edges(aligned)
    adjacency = _adjacency_matrix(node_table, edges)
    _assert_adjacency_invariants(
        adjacency,
        n_nodes=len(node_table),
        n_undirected_edges=len(edges),
    )
    nodes_out = _node_degrees(node_table, edges)
    report = _validation_report(
        nodes_out=nodes_out,
        edges=edges,
        adjacency=adjacency,
        aligned=aligned,
        unused=unused,
        repaired=repaired,
        node_source=node_source,
        boundary_source=boundary_source,
    )

    output_dir = Path(output_dir) if output_dir is not None else results_dir() / "graph" / "geo"
    _assert_overwrite_allowed(output_dir, overwrite=overwrite)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = _write_outputs(output_dir, nodes_out, edges, adjacency, report)

    summary = {
        "status": "ok",
        "n_nodes": report["n_nodes"],
        "n_edges": report["n_edges"],
        "adjacency_shape": report["adjacency_matrix_shape"],
        "is_symmetric": report["is_symmetric"],
        "diagonal_is_zero": report["diagonal_is_zero"],
        "n_isolated": report["n_isolated"],
        "isolated_iz": report["isolated_iz"],
        "n_connected_components": report["n_connected_components"],
        "crs": report["crs"],
        "output_paths": paths,
        "validation_report": report,
    }
    write_run_log({"event": "graph_prepare_complete", **{k: v for k, v in summary.items() if k != "validation_report"}}, filename="graph_prepare.jsonl")
    LOGGER.info(
        "Wrote rook graph with %s nodes and %s undirected edges to %s",
        report["n_nodes"],
        report["n_edges"],
        output_dir,
    )
    return summary


def _load_nodes(
    nodes: pd.DataFrame | Path | None,
    area_code: str,
) -> tuple[pd.DataFrame, str]:
    """Use the existing node-order table. Do not invent order from the shapefile."""
    if nodes is None:
        frame = load_iz_master(area_code=area_code)
        source = "covid.load_iz_master"
    elif isinstance(nodes, (str, Path)):
        path = Path(nodes)
        if not path.exists():
            raise GraphError(f"Node-order table missing: {path}")
        frame = read_table(path)
        source = str(path)
    else:
        frame = nodes.copy()
        source = "supplied DataFrame"
    return _validate_nodes(frame), source


def _validate_nodes(iz_master: pd.DataFrame) -> pd.DataFrame:
    if NODE_KEY not in iz_master.columns or "node_index" not in iz_master.columns:
        raise GraphError("Node-order table must contain IntZone and node_index.")
    nodes = iz_master[[NODE_KEY, "node_index"]].copy()
    nodes[NODE_KEY] = nodes[NODE_KEY].astype("string").str.strip()
    index = pd.to_numeric(nodes["node_index"], errors="coerce")
    if nodes[NODE_KEY].eq("").any() or nodes[NODE_KEY].isna().any():
        raise GraphError("Node-order table contains empty IntZone codes.")
    values = index.to_numpy(dtype="float64")
    if index.isna().any() or (not np.isfinite(values).all()) or (not np.equal(values, np.floor(values)).all()):
        raise GraphError("Node-order table contains non-integer node_index values.")
    nodes["node_index"] = index.astype("int64")
    duplicated = nodes.loc[nodes[NODE_KEY].duplicated(), NODE_KEY].astype(str).tolist()
    if duplicated:
        raise GraphError(f"Node-order table has duplicate IntZone codes: {duplicated[:10]}.")
    if nodes["node_index"].duplicated().any():
        raise GraphError("Node-order table has duplicate node_index values.")
    nodes = nodes.sort_values("node_index").reset_index(drop=True)
    expected = list(range(len(nodes)))
    if nodes["node_index"].tolist() != expected:
        raise GraphError("node_index must be contiguous from 0 to N-1.")
    return nodes


def _load_boundaries(path: Path) -> tuple[gpd.GeoDataFrame, str]:
    if not path.exists():
        raise GraphError(f"IZ boundary shapefile missing: {path}")
    frame = gpd.read_file(path)
    return _normalise_polygon_frame(frame), str(path)


def _normalise_polygon_frame(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if not isinstance(frame, gpd.GeoDataFrame):
        raise GraphError("Boundary input must be a GeoDataFrame.")
    if frame.empty:
        raise GraphError("Boundary table is empty.")
    code_col = next((name for name in IZ_CODE_CANDIDATES if name in frame.columns), None)
    if code_col is None:
        raise GraphError(
            f"Boundary table must contain an IZ code column among {list(IZ_CODE_CANDIDATES)}."
        )
    out = frame.rename(columns={code_col: NODE_KEY}).copy()
    out[NODE_KEY] = out[NODE_KEY].astype("string").str.strip()
    if out.crs is None:
        raise GraphError("Boundary geometries have no CRS; refusing to guess.")
    out = out.to_crs(TARGET_CRS)
    if out.geometry.isna().any():
        raise GraphError("Boundary table contains missing geometries.")
    return out[[NODE_KEY, "geometry"]]


def _align_polygons(
    nodes: pd.DataFrame,
    polygons: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, list[str], int]:
    """Match one polygon to each node-order IZ. Missing IZs are not dropped."""
    poly = polygons.copy()
    duplicated = poly.loc[poly[NODE_KEY].duplicated(), NODE_KEY].astype(str).tolist()
    if duplicated:
        raise GraphError(f"Boundary table has duplicate IntZone codes: {sorted(set(duplicated))[:10]}.")

    unused = sorted(set(poly[NODE_KEY].astype(str)) - set(nodes[NODE_KEY].astype(str)))
    needed = nodes.merge(poly, on=NODE_KEY, how="left", validate="one_to_one")
    missing = needed.loc[needed.geometry.isna(), NODE_KEY].astype(str).tolist()
    if missing:
        raise GraphError(
            f"{len(missing)} node-order IZs have no polygon and were not removed: {missing[:10]}."
        )

    repaired = 0
    geometries: list[BaseGeometry] = []
    for iz_code, geom in zip(needed[NODE_KEY].astype(str), needed.geometry, strict=True):
        geom = _require_polygonal(geom, iz_code, after_repair=False)
        if geom.is_empty or not geom.is_valid:
            geom = make_valid(geom)
            repaired += 1
        geom = _require_polygonal(geom, iz_code, after_repair=True)
        geometries.append(geom)
    needed = needed.copy()
    needed["geometry"] = geometries
    needed = gpd.GeoDataFrame(needed, geometry="geometry", crs=TARGET_CRS)
    needed = needed.sort_values("node_index").reset_index(drop=True)
    if needed["node_index"].tolist() != list(range(len(needed))):
        raise GraphError("Aligned polygons are not in node_index order.")
    return needed, unused, repaired


def _require_polygonal(geom: BaseGeometry | None, iz_code: str, *, after_repair: bool) -> BaseGeometry:
    """Require Polygon or MultiPolygon. Emptiness and validity are enforced after repair."""
    where = "after make_valid" if after_repair else "before make_valid"
    if geom is None:
        raise GraphError(f"IZ {iz_code} geometry is missing {where}.")
    if geom.geom_type not in POLYGON_TYPES:
        raise GraphError(
            f"IZ {iz_code} geometry type {geom.geom_type} is not Polygon or MultiPolygon {where}."
        )
    if after_repair and geom.is_empty:
        raise GraphError(f"IZ {iz_code} geometry is empty {where}.")
    if after_repair and not geom.is_valid:
        raise GraphError(f"IZ {iz_code} geometry is still invalid after make_valid.")
    return geom


def _interior_overlap_area(left: BaseGeometry, right: BaseGeometry) -> float:
    inter = left.intersection(right)
    if inter.is_empty:
        return 0.0
    return float(inter.area)


def _line_length(geom: BaseGeometry) -> float:
    if geom.is_empty:
        return 0.0
    if geom.geom_type in {"LineString", "LinearRing", "MultiLineString"}:
        return float(geom.length)
    if geom.geom_type == "GeometryCollection":
        return float(sum(_line_length(part) for part in geom.geoms))
    return 0.0


def _rook_edges(aligned: gpd.GeoDataFrame) -> pd.DataFrame:
    """Undirected edges for polygon pairs that share a positive-length boundary."""
    geoms = list(aligned.geometry)
    codes = aligned[NODE_KEY].astype(str).tolist()
    indices = aligned["node_index"].astype(int).tolist()
    tree = aligned.sindex
    seen: set[tuple[int, int]] = set()
    rows: list[dict[str, Any]] = []
    for pos, geom in enumerate(geoms):
        hits = list(tree.query(geom, predicate="intersects"))
        for other in hits:
            if other == pos:
                continue
            i, j = indices[pos], indices[other]
            if i == j:
                continue
            source, target = (i, j) if i < j else (j, i)
            pair = (source, target)
            if pair in seen:
                continue
            overlap_area = _interior_overlap_area(geoms[pos], geoms[other])
            if overlap_area > AREA_OVERLAP_EPS:
                left, right = sorted((codes[pos], codes[other]))
                raise GraphError(
                    f"IZ polygons overlap in area: {left} and {right}; "
                    f"overlap_area_m2={overlap_area}."
                )
            shared_boundary = geoms[pos].boundary.intersection(geoms[other].boundary)
            shared_length = _line_length(shared_boundary)
            if shared_length <= LINE_LENGTH_EPS:
                continue
            seen.add(pair)
            src_pos, tgt_pos = (pos, other) if i < j else (other, pos)
            rows.append(
                {
                    "source_node_index": source,
                    "target_node_index": target,
                    "source_iz_code": codes[src_pos],
                    "target_iz_code": codes[tgt_pos],
                    "weight": 1,
                    "shared_boundary_length_m": shared_length,
                }
            )
    edges = pd.DataFrame(
        rows,
        columns=[
            "source_node_index",
            "target_node_index",
            "source_iz_code",
            "target_iz_code",
            "weight",
            "shared_boundary_length_m",
        ],
    )
    if edges.empty:
        return edges
    return edges.sort_values(["source_node_index", "target_node_index"]).reset_index(drop=True)


def _adjacency_matrix(nodes: pd.DataFrame, edges: pd.DataFrame) -> sparse.csr_matrix:
    n = int(len(nodes))
    if edges.empty:
        return sparse.csr_matrix((n, n), dtype="float64")
    sources = edges["source_node_index"].to_numpy(dtype="int64")
    targets = edges["target_node_index"].to_numpy(dtype="int64")
    weights = edges["weight"].to_numpy(dtype="float64")
    rows = np.concatenate([sources, targets])
    cols = np.concatenate([targets, sources])
    data = np.concatenate([weights, weights])
    matrix = sparse.csr_matrix((data, (rows, cols)), shape=(n, n), dtype="float64")
    matrix.setdiag(0)
    matrix.eliminate_zeros()
    return matrix


def _assert_adjacency_invariants(
    adjacency: sparse.csr_matrix,
    n_nodes: int,
    n_undirected_edges: int,
) -> None:
    """Raise if the sparse adjacency matrix is not a binary undirected graph."""
    if adjacency.shape != (n_nodes, n_nodes):
        raise GraphError(
            f"Adjacency shape is {list(adjacency.shape)}, expected [{n_nodes}, {n_nodes}]."
        )
    if (adjacency - adjacency.T).nnz != 0:
        raise GraphError("Adjacency matrix is not symmetric.")
    if not np.all(adjacency.diagonal() == 0):
        raise GraphError("Adjacency diagonal is not zero.")
    expected_nnz = 2 * int(n_undirected_edges)
    if int(adjacency.nnz) != expected_nnz:
        raise GraphError(
            f"Adjacency nnz is {int(adjacency.nnz)}, expected {expected_nnz} "
            f"(2 times {int(n_undirected_edges)} undirected edges)."
        )
    if adjacency.nnz and not np.all(adjacency.data == 1.0):
        raise GraphError("Adjacency weights must all equal 1.0.")


def _node_degrees(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    out = nodes[[NODE_KEY, "node_index"]].copy()
    if edges.empty:
        out["degree"] = 0
        return out.sort_values("node_index").reset_index(drop=True)
    counts = pd.concat(
        [edges["source_node_index"], edges["target_node_index"]],
        ignore_index=True,
    ).value_counts()
    out["degree"] = out["node_index"].map(counts).fillna(0).astype("int64")
    return out.sort_values("node_index").reset_index(drop=True)


def _validation_report(
    nodes_out: pd.DataFrame,
    edges: pd.DataFrame,
    adjacency: sparse.csr_matrix,
    aligned: gpd.GeoDataFrame,
    unused: list[str],
    repaired: int,
    node_source: str,
    boundary_source: str,
) -> dict[str, Any]:
    dense_ok = adjacency - adjacency.T
    n_components = int(csgraph.connected_components(adjacency, directed=False)[0]) if adjacency.shape[0] else 0
    isolated = nodes_out.loc[nodes_out["degree"].eq(0), NODE_KEY].astype(str).tolist()
    return {
        "study_area": LOCAL_AUTHORITY_NAME,
        "local_authority_code": LOCAL_AUTHORITY_CODE,
        "geography_type": "2011 Intermediate Zone",
        "geography_vintage": GEOGRAPHY_VINTAGE,
        "contiguity": "rook",
        "point_only_contact_is_edge": False,
        "self_loops": False,
        "isolated_iz_get_artificial_edges": False,
        "crs": TARGET_CRS,
        "n_nodes": int(len(nodes_out)),
        "expected_iz_count": EXPECTED_IZ_COUNT,
        "n_edges": int(len(edges)),
        "n_adjacency_nonzero": int(adjacency.nnz),
        "adjacency_matrix_shape": list(adjacency.shape),
        "is_symmetric": bool(dense_ok.nnz == 0),
        "diagonal_is_zero": bool(np.all(adjacency.diagonal() == 0)),
        "n_isolated": int(len(isolated)),
        "isolated_iz": isolated,
        "n_connected_components": n_components,
        "missing_iz_codes": [],
        "duplicated_iz_codes": [],
        "n_boundary_polygons_not_in_node_order": int(len(unused)),
        "n_geometries_repaired": int(repaired),
        "node_order_source": node_source,
        "boundary_source": boundary_source,
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
            "only these graph files."
        )


def _write_outputs(
    output_dir: Path,
    nodes_out: pd.DataFrame,
    edges: pd.DataFrame,
    adjacency: sparse.csr_matrix,
    report: dict[str, Any],
) -> dict[str, str]:
    edge_table = edges[
        [
            "source_node_index",
            "target_node_index",
            "source_iz_code",
            "target_iz_code",
            "weight",
            "shared_boundary_length_m",
        ]
    ].copy()
    node_path = write_table(nodes_out, output_dir / "nodes.csv")
    edge_path = write_table(edge_table, output_dir / "edges.csv")
    adj_path = output_dir / "adjacency_geo.npz"
    sparse.save_npz(adj_path, adjacency)
    report_path = write_json(report, output_dir / "validation_report.json")
    return {
        "nodes": str(node_path),
        "edges": str(edge_path),
        "adjacency": str(adj_path),
        "validation_report": str(report_path),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build the Edinburgh 2011 IZ rook-adjacency graph."
    )
    parser.add_argument(
        "--nodes",
        type=Path,
        help="Node-order CSV with IntZone and node_index (default: COVID IZ master).",
    )
    parser.add_argument(
        "--boundaries",
        type=Path,
        help="2011 IZ polygon shapefile (default: SG_IntermediateZone_Bdry_2011.shp).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output folder (default: data/results/graph/geo).",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Replace existing graph output files in the chosen directory (default: True).",
    )
    args = parser.parse_args(argv)
    result = construct_adjacency_graph(
        nodes=args.nodes,
        boundaries_path=args.boundaries,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    printable = {k: v for k, v in result.items() if k != "validation_report"}
    print(json.dumps(printable, indent=2, default=str))


if __name__ == "__main__":
    main()
