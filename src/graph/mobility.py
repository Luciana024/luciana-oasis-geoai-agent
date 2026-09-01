"""Observed mobility graph for Edinburgh 2011 Intermediate Zones.

The study window is 2020–2023. The only local OD extract is already averaged
over 2019–2023; 2019 cannot be removed from that file. The graph therefore
uses a pre-averaged matrix whose source years are wider than the study
window. It is not a 2020–2023-only mobility graph.

Edge weight is the recorded global flow share (`percentage`). This is not a
count, not an origin-conditional probability, not travel time, and not a
road-network distance.

The node order is the existing COVID node_index. Intermediate Zones outside
that list are excluded and reported; they are not added as nodes. Missing
COVID IZs are an error. An IZ that appears only as origin or only as
destination is reported, not invented. Observed zeros stay zeros; missing
OD records are not imputed. The adjacency diagonal is 0; within-IZ flow is
stored on the node table.

The three graphs are not merged.

    PYTHONPATH=src python -m graph.mobility
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csgraph

from graph.geo import GraphError, _load_nodes
from allocation.contracts import EDINBURGH_CA
from common.utils import (
    ALLOWED_YEARS,
    EXPECTED_IZ_COUNT,
    GEOGRAPHY_VINTAGE,
    LOCAL_AUTHORITY_CODE,
    LOCAL_AUTHORITY_NAME,
    NODE_KEY,
    get_logger,
    project_root,
    results_dir,
    write_json,
    write_run_log,
    write_table,
)

LOGGER = get_logger("graph.mobility")

OD_FOLDERS = {
    LOCAL_AUTHORITY_CODE: "Edinburgh",
    EDINBURGH_CA: "Edinburgh",
    "S12000049": "Glasgow",
}
OD_PREFERRED_NAME = "averaged_od_matrix_2019_2023.csv"
FORBIDDEN_OD_NAMES = {
    "final_od_matrix_Edinburgh_averaged.csv",
    "final_od_matrix_Glasgow_averaged.csv",
}
ORIGIN_COL = "origin_geo_code"
DEST_COL = "destination_geo_code"
FLOW_COL = "percentage"
STUDY_YEARS = ALLOWED_YEARS
OD_SOURCE_YEARS = (2019, 2020, 2021, 2022, 2023)
OUTPUT_FILENAMES = (
    "nodes.csv",
    "edges.csv",
    "adjacency_mobility.npz",
    "od_matrix.csv",
    "od_matrix.npz",
    "validation_report.json",
)


def _validate_mobility_nodes(iz_master: pd.DataFrame) -> pd.DataFrame:
    """Require a sorted, contiguous node_index before the graph is built."""
    if NODE_KEY not in iz_master.columns or "node_index" not in iz_master.columns:
        raise GraphError("Node-order table must contain IntZone and node_index.")
    nodes = iz_master[[NODE_KEY, "node_index"]].copy()
    nodes[NODE_KEY] = nodes[NODE_KEY].astype("string").str.strip()
    if nodes[NODE_KEY].eq("").any() or nodes[NODE_KEY].isna().any():
        raise GraphError("Node-order table contains empty IntZone codes.")
    duplicated = nodes.loc[nodes[NODE_KEY].duplicated(), NODE_KEY].astype(str).tolist()
    if duplicated:
        raise GraphError(f"Node-order table has duplicate IntZone codes: {duplicated[:10]}.")
    index = pd.to_numeric(nodes["node_index"], errors="coerce")
    values = index.to_numpy(dtype="float64")
    if (
        index.isna().any()
        or (not np.isfinite(values).all())
        or (not np.equal(values, np.floor(values)).all())
    ):
        raise GraphError("node_index must be non-null integers.")
    nodes["node_index"] = index.astype("int64")
    if nodes["node_index"].duplicated().any():
        raise GraphError("Node-order table has duplicate node_index values.")
    expected = list(range(len(nodes)))
    if nodes["node_index"].tolist() != expected:
        raise GraphError("node_index must be unique, consecutive from 0 to N-1, and sorted.")
    return nodes.reset_index(drop=True)


def _node_order_hash(codes: list[str]) -> dict[str, Any]:
    payload = "\n".join(str(code) for code in codes).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return {
        "node_order_hash": digest,
        "node_order_hash_algorithm": "SHA256",
        "node_order_n": int(len(codes)),
        "node_order_first_iz": codes[0] if codes else None,
        "node_order_last_iz": codes[-1] if codes else None,
    }


def construct_mobility_graph(
    nodes: pd.DataFrame | Path | None = None,
    od: pd.DataFrame | Path | None = None,
    output_dir: Path | None = None,
    overwrite: bool = True,
    area_code: str = LOCAL_AUTHORITY_CODE,
) -> dict[str, Any]:
    """Build a directed observed-OD graph aligned to COVID node_index.

    Study years are 2020–2023. The source file is a 2019–2023 average and is
    not subset to 2020–2023. Does not train a model and does not change COVID,
    forecast, geo, or road outputs. Does not read
    final_od_matrix_Edinburgh_averaged.csv. Does not row-normalise, symmetrise,
    or apply kNN. One adjacency matrix uses the original percentage.
    """
    if isinstance(nodes, pd.DataFrame):
        _validate_mobility_nodes(nodes)
    elif isinstance(nodes, (str, Path)):
        path = Path(nodes)
        if not path.exists():
            raise GraphError(f"Node-order table missing: {path}")
        _validate_mobility_nodes(pd.read_csv(path))
    node_table, node_source = _load_nodes(nodes, area_code=area_code)
    node_table = _validate_mobility_nodes(node_table)
    od_table, od_source = _load_od_table(od, area_code=area_code)
    matrix, extras = _align_od_matrix(node_table, od_table)
    nodes_out, edges = _graph_from_matrix(node_table, matrix)
    adjacency = _directed_adjacency(node_table, edges)
    _assert_mobility_invariants(
        adjacency, matrix, n_nodes=len(node_table), n_edges=len(edges)
    )
    order = _node_order_hash(node_table[NODE_KEY].astype(str).tolist())
    report = _validation_report(
        nodes_out=nodes_out,
        edges=edges,
        adjacency=adjacency,
        matrix=matrix,
        extras=extras,
        node_source=node_source,
        od_source=od_source,
        order=order,
        area_code=area_code,
    )

    output_dir = (
        Path(output_dir)
        if output_dir is not None
        else _default_mobility_output_dir(area_code)
    )
    _assert_overwrite_allowed(output_dir, overwrite=overwrite)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = _write_outputs(output_dir, nodes_out, edges, adjacency, matrix, report, order)

    summary = {
        "status": report["status"],
        "n_nodes": report["n_nodes"],
        "n_edges": report["n_edges"],
        "adjacency_shape": report["adjacency_matrix_shape"],
        "is_symmetric": report["is_symmetric"],
        "diagonal_is_zero": report["diagonal_is_zero"],
        "n_isolated": report["n_isolated"],
        "isolated_iz": report["isolated_iz"],
        "n_excluded_iz": report["n_excluded_iz"],
        "excluded_iz": report["excluded_iz"],
        "output_paths": paths,
        "validation_report": report,
    }
    write_run_log(
        {
            "event": "graph_mobility_prepare_complete",
            **{k: v for k, v in summary.items() if k != "validation_report"},
        },
        filename="graph_mobility_prepare.jsonl",
    )
    LOGGER.info(
        "Wrote mobility graph with %s nodes and %s directed edges to %s",
        report["n_nodes"],
        report["n_edges"],
        output_dir,
    )
    return summary


def default_od_path(area_code: str = LOCAL_AUTHORITY_CODE) -> Path:
    """City-specific averaged OD. Does not use final_od_matrix_* files."""
    root = project_root() / "data" / "raw" / "mobility"
    folder_name = OD_FOLDERS.get(str(area_code).strip(), str(area_code).strip())
    candidates = [
        root / folder_name / OD_PREFERRED_NAME,
        root / str(area_code).strip() / OD_PREFERRED_NAME,
    ]
    for path in candidates:
        if path.exists() and path.stat().st_size > 0:
            return path
    raise GraphError(
        f"Averaged OD matrix missing for CA={area_code}. "
        f"Expected a non-empty {OD_PREFERRED_NAME} under data/raw/mobility/{folder_name}/."
    )


def _default_mobility_output_dir(area_code: str) -> Path:
    if str(area_code).strip() in {LOCAL_AUTHORITY_CODE, EDINBURGH_CA}:
        return results_dir() / "graph" / "mobility"
    return results_dir() / "regions" / str(area_code).strip() / "graph" / "mobility"


def _load_od_table(
    supplied: pd.DataFrame | Path | None,
    area_code: str = LOCAL_AUTHORITY_CODE,
) -> tuple[pd.DataFrame, str]:
    if supplied is None:
        path = default_od_path(area_code)
        frame = pd.read_csv(path)
        return _normalise_od_frame(frame), str(path)
    if isinstance(supplied, (str, Path)):
        path = Path(supplied)
        if not path.exists():
            raise GraphError(f"Averaged OD matrix missing: {path}")
        if path.name in FORBIDDEN_OD_NAMES or path.name.startswith("final_od_matrix_"):
            raise GraphError(
                f"{path.name} is not the mobility source; "
                "use averaged_od_matrix_2019_2023.csv."
            )
        frame = pd.read_csv(path)
        return _normalise_od_frame(frame), str(path)
    return _normalise_od_frame(supplied.copy()), "supplied OD DataFrame"


def _normalise_od_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [name for name in (ORIGIN_COL, DEST_COL, FLOW_COL) if name not in frame.columns]
    if missing:
        raise GraphError(f"OD table must contain {ORIGIN_COL}, {DEST_COL} and {FLOW_COL}.")
    out = frame[[ORIGIN_COL, DEST_COL, FLOW_COL]].copy()
    out[ORIGIN_COL] = out[ORIGIN_COL].astype("string").str.strip()
    out[DEST_COL] = out[DEST_COL].astype("string").str.strip()
    if out[ORIGIN_COL].eq("").any() or out[ORIGIN_COL].isna().any():
        raise GraphError("OD table contains empty origin_geo_code values.")
    if out[DEST_COL].eq("").any() or out[DEST_COL].isna().any():
        raise GraphError("OD table contains empty destination_geo_code values.")
    flow = pd.to_numeric(out[FLOW_COL], errors="coerce")
    values = flow.to_numpy(dtype="float64")
    if flow.isna().any() or (not np.isfinite(values).all()):
        raise GraphError("OD percentage contains NaN or infinite values.")
    if np.any(values < 0.0):
        raise GraphError("OD percentage contains negative values.")
    out[FLOW_COL] = flow.astype("float64")
    duplicated = out.duplicated([ORIGIN_COL, DEST_COL], keep=False)
    if duplicated.any():
        sample = (
            out.loc[duplicated, [ORIGIN_COL, DEST_COL]]
            .astype(str)
            .drop_duplicates()
            .head(10)
            .to_dict("records")
        )
        raise GraphError(f"OD table has duplicate origin-destination pairs: {sample}.")
    return out.reset_index(drop=True)


def _align_od_matrix(
    nodes: pd.DataFrame,
    od: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Place observed OD shares onto the COVID node_index grid.

    Extra IZs are not nodes. Missing internal pairs stay missing (NaN), not 0.
    """
    codes = nodes[NODE_KEY].astype(str).tolist()
    index_of = {code: i for i, code in enumerate(codes)}
    needed = set(codes)
    origins = set(od[ORIGIN_COL].astype(str))
    dests = set(od[DEST_COL].astype(str))
    present = origins.union(dests)
    missing_required = sorted(needed - present)
    if missing_required:
        raise GraphError(
            f"{len(missing_required)} node-order IZs are absent from the OD matrix "
            f"and were not invented: {missing_required[:10]}."
        )
    missing_origin_iz = sorted(needed - origins)
    missing_destination_iz = sorted(needed - dests)
    excluded = sorted(present - needed)
    n = len(codes)
    matrix = np.full((n, n), np.nan, dtype="float64")
    origin = od[ORIGIN_COL].astype(str).to_numpy()
    dest = od[DEST_COL].astype(str).to_numpy()
    flow = od[FLOW_COL].to_numpy(dtype="float64")
    kept = np.zeros(len(od), dtype=bool)
    for i, (o_code, d_code, share) in enumerate(zip(origin, dest, flow, strict=True)):
        source = index_of.get(o_code)
        target = index_of.get(d_code)
        if source is None or target is None:
            continue
        matrix[source, target] = float(share)
        kept[i] = True
    dropped = flow[~kept]
    observed_internal = int(np.isfinite(matrix).sum())
    expected_internal = int(n * n)
    missing_internal = expected_internal - observed_internal
    return matrix, {
        "excluded_iz": excluded,
        "n_od_iz": int(len(present)),
        "n_dropped_pairs": int((~kept).sum()),
        "dropped_flow_share": float(dropped.sum()) if dropped.size else 0.0,
        "kept_flow_share": float(np.nansum(flow[kept])) if kept.any() else 0.0,
        "source_flow_share": float(flow.sum()),
        "missing_origin_iz": missing_origin_iz,
        "missing_destination_iz": missing_destination_iz,
        "expected_internal_od_pairs": expected_internal,
        "observed_internal_od_pairs": observed_internal,
        "missing_internal_od_pairs": missing_internal,
        "is_complete_internal_od_matrix": missing_internal == 0,
    }


def _graph_from_matrix(
    nodes: pd.DataFrame,
    matrix: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = int(len(nodes))
    codes = nodes[NODE_KEY].astype(str).tolist()
    eye = np.eye(n, dtype=bool)
    offdiag = np.where(eye, np.nan, matrix)
    positive = np.isfinite(offdiag) & (offdiag > 0.0)
    self_flow = np.diag(matrix).astype("float64")
    out_strength = np.nansum(offdiag, axis=1)
    in_strength = np.nansum(offdiag, axis=0)
    out_degree = positive.sum(axis=1).astype("int64")
    in_degree = positive.sum(axis=0).astype("int64")
    nodes_out = nodes[[NODE_KEY, "node_index"]].copy()
    nodes_out["self_percentage"] = self_flow
    nodes_out["out_strength"] = out_strength
    nodes_out["in_strength"] = in_strength
    nodes_out["out_degree"] = out_degree
    nodes_out["in_degree"] = in_degree
    nodes_out = nodes_out.sort_values("node_index").reset_index(drop=True)

    rows: list[dict[str, Any]] = []
    for source in range(n):
        targets = np.flatnonzero(positive[source])
        if targets.size == 0:
            continue
        order = targets[np.argsort(-offdiag[source, targets], kind="stable")]
        for rank, target in enumerate(order, start=1):
            share = float(offdiag[source, int(target)])
            rows.append(
                {
                    "source_node_index": int(source),
                    "target_node_index": int(target),
                    "source_iz_code": codes[source],
                    "target_iz_code": codes[int(target)],
                    "percentage": share,
                    "weight": share,
                    "neighbour_rank": int(rank),
                }
            )
    edges = pd.DataFrame(rows)
    if edges.empty:
        edges = pd.DataFrame(
            columns=[
                "source_node_index",
                "target_node_index",
                "source_iz_code",
                "target_iz_code",
                "percentage",
                "weight",
                "neighbour_rank",
            ]
        )
        return nodes_out, edges
    return (
        nodes_out,
        edges.sort_values(["source_node_index", "neighbour_rank"]).reset_index(drop=True),
    )


def _directed_adjacency(nodes: pd.DataFrame, edges: pd.DataFrame) -> sparse.csr_matrix:
    n = int(len(nodes))
    if edges.empty:
        return sparse.csr_matrix((n, n), dtype="float64")
    sources = edges["source_node_index"].to_numpy(dtype="int64")
    targets = edges["target_node_index"].to_numpy(dtype="int64")
    if np.any(sources == targets):
        raise GraphError("Mobility adjacency cannot contain self-loops.")
    weights = edges["weight"].to_numpy(dtype="float64")
    if np.any(weights <= 0.0):
        raise GraphError("Mobility edge weights must be positive.")
    matrix = sparse.csr_matrix((weights, (sources, targets)), shape=(n, n), dtype="float64")
    matrix.setdiag(0)
    matrix.eliminate_zeros()
    return matrix


def _assert_mobility_invariants(
    adjacency: sparse.csr_matrix,
    matrix: np.ndarray,
    n_nodes: int,
    n_edges: int,
) -> None:
    if adjacency.shape != (n_nodes, n_nodes):
        raise GraphError(
            f"Adjacency shape is {list(adjacency.shape)}, expected [{n_nodes}, {n_nodes}]."
        )
    if matrix.shape != (n_nodes, n_nodes):
        raise GraphError(
            f"OD matrix shape is {list(matrix.shape)}, expected [{n_nodes}, {n_nodes}]."
        )
    if not np.all(adjacency.diagonal() == 0):
        raise GraphError("Adjacency diagonal is not zero.")
    if int(adjacency.nnz) != int(n_edges):
        raise GraphError(
            f"Adjacency nnz is {int(adjacency.nnz)}, expected {int(n_edges)} directed edges."
        )
    eye = np.eye(n_nodes, dtype=bool)
    positive = np.isfinite(matrix) & (matrix > 0.0) & ~eye
    expected = int(positive.sum())
    if expected != int(n_edges):
        raise GraphError(
            f"Positive off-diagonal OD cells are {expected}, but there are {int(n_edges)} edges."
        )
    expected_adj = np.where(positive, matrix, 0.0)
    if not np.allclose(adjacency.toarray(), expected_adj, atol=0.0, rtol=0.0, equal_nan=False):
        raise GraphError("Adjacency is inconsistent with the observed OD percentages.")


def _validation_report(
    *,
    nodes_out: pd.DataFrame,
    edges: pd.DataFrame,
    adjacency: sparse.csr_matrix,
    matrix: np.ndarray,
    extras: dict[str, Any],
    node_source: str,
    od_source: str,
    order: dict[str, Any],
    area_code: str,
) -> dict[str, Any]:
    n = int(len(nodes_out))
    binary = adjacency.copy()
    binary.data = np.ones_like(binary.data)
    undirected = binary.maximum(binary.T)
    n_weak = int(csgraph.connected_components(undirected, directed=False)[0]) if n else 0
    n_strong = (
        int(csgraph.connected_components(binary, directed=True, connection="strong")[0])
        if n
        else 0
    )
    zero_outflow = nodes_out.loc[nodes_out["out_degree"].eq(0), NODE_KEY].astype(str).tolist()
    zero_inflow = nodes_out.loc[nodes_out["in_degree"].eq(0), NODE_KEY].astype(str).tolist()
    isolated = nodes_out.loc[
        nodes_out["out_degree"].eq(0) & nodes_out["in_degree"].eq(0), NODE_KEY
    ].astype(str).tolist()
    eye = np.eye(n, dtype=bool) if n else np.zeros((0, 0), dtype=bool)
    self_flow = np.diag(matrix) if n else np.array([], dtype="float64")
    observed_offdiag_zero = np.isfinite(matrix) & (matrix == 0.0) & ~eye
    positive = np.isfinite(matrix) & (matrix > 0.0) & ~eye
    positive_values = matrix[positive]
    warnings = _mobility_warnings(
        extras=extras,
        zero_outflow=zero_outflow,
        zero_inflow=zero_inflow,
        isolated=isolated,
    )
    status = _mobility_status(
        extras=extras,
        isolated=isolated,
        zero_outflow=zero_outflow,
        zero_inflow=zero_inflow,
    )
    return {
        "status": status,
        "warnings": warnings,
        "study_area": _study_area_name(area_code),
        "local_authority_code": str(area_code).strip(),
        "geography_type": "2011 Intermediate Zone",
        "geography_vintage": GEOGRAPHY_VINTAGE,
        "graph_kind": "observed_od_mobility",
        "mobility_graph_is_static": True,
        "mobility_graph_is_directed": True,
        "source_is_pre_averaged": True,
        "source_years": list(OD_SOURCE_YEARS),
        "covid_study_years": list(STUDY_YEARS),
        "study_years": list(STUDY_YEARS),
        "od_source_years": list(OD_SOURCE_YEARS),
        "od_includes_year_outside_study_window": True,
        "od_not_subset_to_study_years": True,
        "od_file": "averaged_od_matrix_2019_2023.csv",
        "final_od_matrix_not_used": True,
        "weight_field": FLOW_COL,
        "weight_meaning": "recorded global flow share",
        "weight": "observed_global_flow_share",
        "adjacency_is_row_normalised": False,
        "model_stage_will_normalise": True,
        "self_flow_excluded_from_adjacency": True,
        "geographic_and_road_graphs_not_merged": True,
        "not_origin_conditional_probability": True,
        "not_travel_time": True,
        "not_road_network_distance": True,
        "not_symmetrised": True,
        "zeros_not_filled": True,
        "missing_od_records_not_imputed": True,
        "extra_iz_not_added_as_nodes": True,
        "self_loops": False,
        "directed": True,
        "symmetrised": False,
        "k_nn_not_applied": True,
        "n_nodes": n,
        "expected_iz_count": EXPECTED_IZ_COUNT,
        "n_edges": int(len(edges)),
        "n_adjacency_nonzero": int(adjacency.nnz),
        "adjacency_matrix_shape": list(adjacency.shape),
        "is_symmetric": bool((adjacency - adjacency.T).nnz == 0),
        "diagonal_is_zero": bool(np.all(adjacency.diagonal() == 0)),
        "n_isolated": int(len(isolated)),
        "isolated_iz": isolated,
        "zero_outflow_iz": zero_outflow,
        "zero_inflow_iz": zero_inflow,
        "n_zero_outflow": int(len(zero_outflow)),
        "n_zero_inflow": int(len(zero_inflow)),
        "n_weakly_connected_components": n_weak,
        "n_strongly_connected_components": n_strong,
        "out_degree_min": int(nodes_out["out_degree"].min()) if n else 0,
        "out_degree_max": int(nodes_out["out_degree"].max()) if n else 0,
        "in_degree_min": int(nodes_out["in_degree"].min()) if n else 0,
        "in_degree_max": int(nodes_out["in_degree"].max()) if n else 0,
        "n_zero_offdiag": int(observed_offdiag_zero.sum()),
        "n_positive_offdiag": int(positive_values.size),
        "self_flow_share": float(np.nansum(self_flow)),
        "offdiag_flow_share": float(np.nansum(np.where(eye, np.nan, matrix))) if n else 0.0,
        "kept_flow_share": float(extras["kept_flow_share"]),
        "dropped_flow_share": float(extras["dropped_flow_share"]),
        "source_flow_share": float(extras["source_flow_share"]),
        "renormalised_after_excluding_extra_iz": False,
        "n_od_iz": int(extras["n_od_iz"]),
        "n_excluded_iz": int(len(extras["excluded_iz"])),
        "excluded_iz": extras["excluded_iz"],
        "n_dropped_pairs": int(extras["n_dropped_pairs"]),
        "missing_origin_iz": extras["missing_origin_iz"],
        "missing_destination_iz": extras["missing_destination_iz"],
        "expected_internal_od_pairs": int(extras["expected_internal_od_pairs"]),
        "observed_internal_od_pairs": int(extras["observed_internal_od_pairs"]),
        "missing_internal_od_pairs": int(extras["missing_internal_od_pairs"]),
        "is_complete_internal_od_matrix": bool(extras["is_complete_internal_od_matrix"]),
        "node_order_source": node_source,
        "od_source": od_source,
        "node_order_hash": order["node_order_hash"],
        "node_order_hash_algorithm": order["node_order_hash_algorithm"],
        "node_order_n": order["node_order_n"],
        "node_order_first_iz": order["node_order_first_iz"],
        "node_order_last_iz": order["node_order_last_iz"],
        "edge_weight_min": float(positive_values.min()) if positive_values.size else None,
        "edge_weight_median": float(np.median(positive_values)) if positive_values.size else None,
        "edge_weight_max": float(positive_values.max()) if positive_values.size else None,
        "node_index_sequence": nodes_out["node_index"].tolist(),
        "iz_codes": nodes_out[NODE_KEY].astype(str).tolist(),
        "processing_timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _mobility_warnings(
    *,
    extras: dict[str, Any],
    zero_outflow: list[str],
    zero_inflow: list[str],
    isolated: list[str],
) -> list[str]:
    warnings: list[str] = [
        "OD source is a pre-averaged 2019-2023 dataset; it is not subset to COVID study years 2020-2023."
    ]
    if not extras["is_complete_internal_od_matrix"]:
        warnings.append(
            f"Internal OD table is incomplete: observed {extras['observed_internal_od_pairs']} "
            f"of {extras['expected_internal_od_pairs']} pairs; missing records were not imputed."
        )
    if extras["missing_origin_iz"]:
        warnings.append(f"COVID IZs missing as origin: {extras['missing_origin_iz']}.")
    if extras["missing_destination_iz"]:
        warnings.append(f"COVID IZs missing as destination: {extras['missing_destination_iz']}.")
    if isolated:
        warnings.append(f"Isolated IZs (no off-diagonal inflow or outflow): {isolated}.")
    else:
        if zero_outflow:
            warnings.append(f"Zero-outflow IZs: {zero_outflow}.")
        if zero_inflow:
            warnings.append(f"Zero-inflow IZs: {zero_inflow}.")
    if extras["excluded_iz"]:
        warnings.append(f"Excluded non-COVID IZs: {extras['excluded_iz']}.")
    if extras["n_dropped_pairs"]:
        warnings.append(
            f"Dropped {extras['n_dropped_pairs']} external OD pairs "
            f"(flow share {extras['dropped_flow_share']})."
        )
    return warnings


def _mobility_status(
    *,
    extras: dict[str, Any],
    isolated: list[str],
    zero_outflow: list[str],
    zero_inflow: list[str],
) -> str:
    if (
        (not extras["is_complete_internal_od_matrix"])
        or extras["missing_origin_iz"]
        or extras["missing_destination_iz"]
        or isolated
        or zero_outflow
        or zero_inflow
        or extras["excluded_iz"]
        or extras["n_dropped_pairs"]
    ):
        return "ok_with_warnings"
    return "ok"


def _study_area_name(area_code: str) -> str:
    """Resolve a council area name without defaulting every code to Edinburgh."""
    code = str(area_code).strip()
    if code == LOCAL_AUTHORITY_CODE:
        return LOCAL_AUTHORITY_NAME
    path = project_root() / "data" / "raw" / "boundaries" / "Code lookup.csv"
    if path.exists():
        lookup = pd.read_csv(path, dtype="string", encoding="utf-8-sig")
        if "CA" in lookup.columns and "CAName" in lookup.columns:
            matched = lookup.loc[lookup["CA"].astype("string").str.strip().eq(code), "CAName"]
            names = sorted(
                {str(name).strip() for name in matched.dropna().tolist() if str(name).strip()}
            )
            if len(names) == 1:
                return names[0]
    return f"unknown ({code})"


def _planned_output_paths(output_dir: Path) -> list[Path]:
    return [output_dir / name for name in OUTPUT_FILENAMES]


def _assert_overwrite_allowed(output_dir: Path, overwrite: bool) -> None:
    existing = [path for path in _planned_output_paths(output_dir) if path.exists()]
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise GraphError(
            f"Output already exists ({names}). Pass overwrite=True or omit --no-overwrite to replace "
            "only these mobility graph files."
        )


def _write_od_csv(matrix: np.ndarray, nodes: pd.DataFrame, path: Path) -> Path:
    labels = nodes[NODE_KEY].astype(str).tolist()
    frame = pd.DataFrame(matrix, index=labels, columns=labels)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index_label=NODE_KEY)
    return path


def _write_outputs(
    output_dir: Path,
    nodes_out: pd.DataFrame,
    edges: pd.DataFrame,
    adjacency: sparse.csr_matrix,
    matrix: np.ndarray,
    report: dict[str, Any],
    order: dict[str, Any],
) -> dict[str, str]:
    node_path = write_table(nodes_out, output_dir / "nodes.csv")
    edge_path = write_table(edges, output_dir / "edges.csv")
    adj_path = output_dir / "adjacency_mobility.npz"
    sparse.save_npz(adj_path, adjacency)
    od_csv = _write_od_csv(matrix, nodes_out, output_dir / "od_matrix.csv")
    od_npz = output_dir / "od_matrix.npz"
    np.savez_compressed(
        od_npz,
        od_percentage=matrix,
        node_index=nodes_out["node_index"].to_numpy(dtype="int64"),
        iz_codes=nodes_out[NODE_KEY].astype(str).to_numpy(),
        node_order_hash=np.asarray(order["node_order_hash"]),
    )
    report_path = write_json(report, output_dir / "validation_report.json")
    return {
        "nodes": str(node_path),
        "edges": str(edge_path),
        "adjacency": str(adj_path),
        "od_matrix": str(od_csv),
        "od_matrix_npz": str(od_npz),
        "validation_report": str(report_path),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build the Edinburgh 2011 IZ observed-mobility graph."
    )
    parser.add_argument(
        "--nodes",
        type=Path,
        help="Node-order CSV with IntZone and node_index (default: COVID IZ master).",
    )
    parser.add_argument(
        "--od",
        type=Path,
        help="Averaged OD CSV (default: averaged_od_matrix_2019_2023.csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output folder (default: data/results/graph/mobility).",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Replace existing mobility graph output files in the chosen directory (default: True).",
    )
    parser.add_argument(
        "--area-code",
        default=LOCAL_AUTHORITY_CODE,
        help="Council area code. Selects data/raw/mobility/<city>/averaged_od_matrix_2019_2023.csv.",
    )
    args = parser.parse_args(argv)
    result = construct_mobility_graph(
        nodes=args.nodes,
        od=args.od,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        area_code=args.area_code,
    )
    printable = {k: v for k, v in result.items() if k != "validation_report"}
    print(json.dumps(printable, indent=2, default=str))


if __name__ == "__main__":
    main()
