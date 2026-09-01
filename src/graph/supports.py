"""Load the three existing adjacency graphs.

See docs/model.md sections 5 and 14. This module does not rebuild geo, road,
or mobility graphs and does not overwrite adjacency files.

A[i, j] > 0 means a directed edge i -> j. geo may be symmetric.
transport and mobility stay directed; they are not symmetrised, not kNN-ed,
and not pre-fused.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from model.constants import (
    GRAPH_GEO,
    GRAPH_MOBILITY,
    GRAPH_TRANSPORT,
    THREE_GRAPH_SET,
    TWO_GRAPH_SET,
)
from common.errors import LEVEL_ACCEPTED, LEVEL_CRITICAL, LEVEL_REVIEW, ModelError, ModelWarning
from data.node_order import NodeOrder, assert_same_node_order, load_node_order, sha256_file
from common.utils import NODE_KEY

GRAPH_FILE_KEYS = {
    GRAPH_GEO: "geo",
    GRAPH_TRANSPORT: "transport",
    GRAPH_MOBILITY: "mobility",
}


@dataclass
class LoadedGraph:
    name: str
    adjacency: np.ndarray
    node_order: NodeOrder
    path: Path
    file_sha256: str
    n_edges: int
    n_zero_out: int
    n_zero_in: int
    isolated_iz: tuple[str, ...]
    is_symmetric: bool
    warnings: list[ModelWarning] = field(default_factory=list)
    report: dict[str, Any] = field(default_factory=dict)


def _dense_adjacency(path: Path) -> np.ndarray:
    matrix = sparse.load_npz(path).tocsr()
    dense = np.asarray(matrix.toarray(), dtype=np.float64)
    return dense


def _validate_adjacency(name: str, adjacency: np.ndarray, n_nodes: int) -> list[ModelWarning]:
    warnings: list[ModelWarning] = []
    if adjacency.shape != (n_nodes, n_nodes):
        raise ModelError(
            f"{name} adjacency shape is {adjacency.shape}, expected [{n_nodes}, {n_nodes}].",
            code="node_order_mismatch",
        )
    if not np.isfinite(adjacency).all():
        raise ModelError(f"{name} adjacency contains NaN or Inf.", code="illegal_graph_weights")
    if np.any(adjacency < 0):
        raise ModelError(f"{name} adjacency contains negative weights.", code="illegal_graph_weights")
    if not np.allclose(np.diag(adjacency), 0.0):
        raise ModelError(f"{name} adjacency diagonal must be 0.", code="illegal_graph_weights")
    return warnings


def _degree_stats(adjacency: np.ndarray, codes: tuple[str, ...]) -> tuple[int, int, tuple[str, ...]]:
    out_deg = adjacency.sum(axis=1)
    in_deg = adjacency.sum(axis=0)
    zero_out = out_deg <= 0
    zero_in = in_deg <= 0
    isolated = tuple(codes[i] for i in range(len(codes)) if zero_out[i] and zero_in[i])
    return int(zero_out.sum()), int(zero_in.sum()), isolated


def _mobility_warnings(report: dict[str, Any], isolated_iz: tuple[str, ...]) -> list[ModelWarning]:
    warnings: list[ModelWarning] = []
    warnings.append(
        ModelWarning(
            code="mobility_pre_averaged_od",
            level=LEVEL_ACCEPTED,
            message="Mobility is a pre-averaged 2019-2023 OD matrix, not real-time pandemic flow.",
            details={"source_years": report.get("source_years") or report.get("od_source_years")},
        )
    )
    excluded = report.get("excluded_non_covid_iz") or report.get("excluded_external_iz")
    if not excluded:
        for warning in report.get("warnings", []):
            if "Excluded non-COVID" in str(warning) or "external OD" in str(warning):
                excluded = True
                break
    if excluded:
        warnings.append(
            ModelWarning(
                code="excluded_external_od",
                level=LEVEL_ACCEPTED,
                message="OD pairs involving IZs outside the study area were excluded.",
                details={"report_warnings": report.get("warnings", [])},
            )
        )
    warnings.append(
        ModelWarning(
            code="sparse_od",
            level=LEVEL_ACCEPTED,
            message="OD tables may omit IZ-IZ pairs. A complete N x N directed graph is not required.",
        )
    )
    if isolated_iz:
        warnings.append(
            ModelWarning(
                code="mobility_isolated_nodes",
                level=LEVEL_REVIEW,
                message="Some canonical IZs have both in-degree and out-degree 0 on the mobility graph.",
                details={"n_isolated": len(isolated_iz), "isolated_iz": list(isolated_iz)},
            )
        )
    return warnings


def load_graph(
    name: str,
    adjacency_path: Path,
    *,
    canonical: NodeOrder,
    report_path: Path | None = None,
) -> LoadedGraph:
    if not adjacency_path.is_file():
        raise ModelError(f"Missing adjacency file: {adjacency_path}", code="missing_graph")
    nodes_path = adjacency_path.parent / "nodes.csv"
    if not nodes_path.is_file():
        raise ModelError(f"Missing graph node table: {nodes_path}", code="missing_graph")
    graph_order = load_node_order(nodes_path)
    assert_same_node_order(canonical, graph_order, left_name="covid", right_name=name)

    adjacency = _dense_adjacency(adjacency_path)
    _validate_adjacency(name, adjacency, canonical.n_nodes)
    n_zero_out, n_zero_in, isolated = _degree_stats(adjacency, canonical.codes)
    report: dict[str, Any] = {}
    if report_path is not None and report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    warnings: list[ModelWarning] = []
    if name == GRAPH_MOBILITY:
        warnings.extend(_mobility_warnings(report, isolated))
    is_symmetric = bool(np.allclose(adjacency, adjacency.T))
    return LoadedGraph(
        name=name,
        adjacency=adjacency,
        node_order=graph_order,
        path=adjacency_path,
        file_sha256=sha256_file(adjacency_path),
        n_edges=int((adjacency > 0).sum()),
        n_zero_out=n_zero_out,
        n_zero_in=n_zero_in,
        isolated_iz=isolated,
        is_symmetric=is_symmetric,
        warnings=warnings,
        report=report,
    )


def normalise_graph_set(graph_set: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    names = tuple(str(name) for name in graph_set)
    if names == THREE_GRAPH_SET or names == TWO_GRAPH_SET:
        return names
    raise ModelError(
        f"Unsupported graph_set {list(names)}. Allowed: {list(THREE_GRAPH_SET)} or {list(TWO_GRAPH_SET)}.",
        code="graph_set_mismatch",
    )


def load_graph_bundle(
    paths: dict[str, Path],
    *,
    canonical: NodeOrder,
    graph_set: list[str] | tuple[str, ...] | None = None,
    reports: dict[str, Path] | None = None,
) -> dict[str, LoadedGraph]:
    """Load the requested graph_set. Do not drop mobility from a three-graph set here."""
    requested = normalise_graph_set(graph_set or THREE_GRAPH_SET)
    reports = reports or {}
    loaded: dict[str, LoadedGraph] = {}
    for name in requested:
        key = GRAPH_FILE_KEYS[name]
        loaded[name] = load_graph(
            name,
            paths[key],
            canonical=canonical,
            report_path=reports.get(key),
        )
    return loaded


def load_projected_centroids(road_nodes_path: Path, canonical: NodeOrder) -> np.ndarray:
    """EPSG:27700 Easting/Northing from road/nodes.csv. Not converted from lon/lat."""
    table = pd.read_csv(road_nodes_path)
    if "Easting" not in table.columns or "Northing" not in table.columns:
        raise ModelError(
            f"{road_nodes_path} must contain Easting and Northing.",
            code="missing_coordinates",
        )
    table = table.sort_values("node_index").reset_index(drop=True)
    codes = tuple(table[NODE_KEY].astype(str).tolist())
    if codes != canonical.codes:
        raise ModelError(
            "Centroid table IZ sequence does not match canonical node order.",
            code="node_order_mismatch",
        )
    coords = table[["Easting", "Northing"]].to_numpy(dtype=np.float64)
    if not np.isfinite(coords).all():
        raise ModelError("Centroid coordinates contain NaN or Inf.", code="missing_coordinates")
    return coords
