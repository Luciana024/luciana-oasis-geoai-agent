from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from shapely.geometry import LineString, Polygon
from shapely.validation import make_valid

from graph.geo import GraphError, _assert_adjacency_invariants, construct_adjacency_graph


def _square(x0: float, y0: float, size: float = 1.0) -> Polygon:
    return Polygon(
        [
            (x0, y0),
            (x0 + size, y0),
            (x0 + size, y0 + size),
            (x0, y0 + size),
        ]
    )


def _toy_polygons() -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
    """Four IZs. A shares a side with B; A touches C only at a point; D is isolated.

    Shapefile-like row order is deliberately not node_index order.
    """
    nodes = pd.DataFrame(
        {
            "IntZone": ["S020AA002", "S020CC003", "S020AA001", "S020DD004"],
            "node_index": [0, 1, 2, 3],
        }
    )
    # node_index: B=0, C=1, A=2, D=3
    polygons = gpd.GeoDataFrame(
        {
            "InterZone": ["S020DD004", "S020CC003", "S020AA001", "S020AA002"],
            "geometry": [
                _square(10, 10),  # D isolated
                _square(1, 1),  # C touches A at (1,1) only; shares a side with B
                _square(0, 0),  # A
                _square(1, 0),  # B
            ],
        },
        crs="EPSG:27700",
    )
    return nodes, polygons


def test_shared_boundary_creates_a_rook_edge(tmp_path: Path):
    nodes, polygons = _toy_polygons()
    result = construct_adjacency_graph(
        nodes=nodes,
        polygons=polygons,
        output_dir=tmp_path / "graph",
    )
    edges = pd.read_csv(result["output_paths"]["edges"])
    pairs = set(
        zip(edges["source_node_index"].astype(int), edges["target_node_index"].astype(int))
    )
    assert (0, 2) in pairs  # B-A share a side
    assert (0, 1) in pairs  # B-C share a side
    assert result["n_edges"] == 2
    assert (edges["weight"] == 1).all()
    assert "shared_boundary_length_m" in edges.columns
    assert (edges["shared_boundary_length_m"] > 0).all()
    assert not (edges["source_node_index"] == edges["target_node_index"]).any()


def test_point_only_contact_does_not_create_an_edge(tmp_path: Path):
    nodes, polygons = _toy_polygons()
    result = construct_adjacency_graph(
        nodes=nodes,
        polygons=polygons,
        output_dir=tmp_path / "graph",
    )
    edges = pd.read_csv(result["output_paths"]["edges"])
    pairs = set(
        zip(edges["source_node_index"].astype(int), edges["target_node_index"].astype(int))
    )
    assert (1, 2) not in pairs  # A-C touch only at a point


def test_adjacency_invariants(tmp_path: Path):
    nodes, polygons = _toy_polygons()
    result = construct_adjacency_graph(
        nodes=nodes,
        polygons=polygons,
        output_dir=tmp_path / "graph",
    )
    matrix = sparse.load_npz(result["output_paths"]["adjacency"])
    n_edges = int(result["n_edges"])
    _assert_adjacency_invariants(matrix, n_nodes=4, n_undirected_edges=n_edges)
    assert matrix.shape == (4, 4)
    assert (matrix - matrix.T).nnz == 0
    assert np.all(matrix.diagonal() == 0)
    assert int(matrix.nnz) == 2 * n_edges
    assert np.all(matrix.data == 1.0)
    assert result["is_symmetric"] is True
    assert result["diagonal_is_zero"] is True
    dense = matrix.toarray()
    assert dense[0, 2] == 1
    assert dense[2, 0] == 1
    assert dense[1, 2] == 0
    assert dense[2, 1] == 0


def test_adjacency_invariants_reject_violations():
    n = 3
    bad_shape = sparse.csr_matrix((n, n + 1), dtype="float64")
    with pytest.raises(GraphError, match="Adjacency shape"):
        _assert_adjacency_invariants(bad_shape, n_nodes=n, n_undirected_edges=0)

    asymmetric = sparse.csr_matrix(([1.0], ([0], [1])), shape=(n, n), dtype="float64")
    with pytest.raises(GraphError, match="not symmetric"):
        _assert_adjacency_invariants(asymmetric, n_nodes=n, n_undirected_edges=1)

    with_diag = sparse.csr_matrix(([1.0], ([0], [0])), shape=(n, n), dtype="float64")
    with pytest.raises(GraphError, match="diagonal"):
        _assert_adjacency_invariants(with_diag, n_nodes=n, n_undirected_edges=0)

    one_edge = sparse.csr_matrix(
        ([1.0, 1.0], ([0, 1], [1, 0])), shape=(n, n), dtype="float64"
    )
    with pytest.raises(GraphError, match="nnz"):
        _assert_adjacency_invariants(one_edge, n_nodes=n, n_undirected_edges=2)

    weighted = sparse.csr_matrix(
        ([2.0, 2.0], ([0, 1], [1, 0])), shape=(n, n), dtype="float64"
    )
    with pytest.raises(GraphError, match="1.0"):
        _assert_adjacency_invariants(weighted, n_nodes=n, n_undirected_edges=1)


def test_matrix_order_follows_node_index_not_shapefile_order(tmp_path: Path):
    nodes, polygons = _toy_polygons()
    result = construct_adjacency_graph(
        nodes=nodes,
        polygons=polygons,
        output_dir=tmp_path / "graph",
    )
    node_table = pd.read_csv(result["output_paths"]["nodes"])
    assert node_table["IntZone"].tolist() == [
        "S020AA002",
        "S020CC003",
        "S020AA001",
        "S020DD004",
    ]
    assert node_table["node_index"].tolist() == [0, 1, 2, 3]
    assert node_table["degree"].tolist() == [2, 1, 1, 0]
    assert result["isolated_iz"] == ["S020DD004"]
    assert result["n_connected_components"] == 2


def test_missing_polygon_is_not_silently_dropped(tmp_path: Path):
    nodes, polygons = _toy_polygons()
    polygons = polygons.iloc[:-1].copy()
    with pytest.raises(GraphError, match="have no polygon"):
        construct_adjacency_graph(
            nodes=nodes,
            polygons=polygons,
            output_dir=tmp_path / "graph",
        )


def test_polygon_overlap_raises(tmp_path: Path):
    nodes = pd.DataFrame(
        {"IntZone": ["S020AA001", "S020BB002"], "node_index": [0, 1]}
    )
    polygons = gpd.GeoDataFrame(
        {
            "InterZone": ["S020AA001", "S020BB002"],
            "geometry": [_square(0, 0, 2), _square(1, 1, 2)],
        },
        crs="EPSG:27700",
    )
    with pytest.raises(GraphError, match=r"S020AA001 and S020BB002; overlap_area_m2="):
        construct_adjacency_graph(
            nodes=nodes,
            polygons=polygons,
            output_dir=tmp_path / "graph",
        )


def test_non_integer_node_index_raises(tmp_path: Path):
    nodes = pd.DataFrame(
        {"IntZone": ["S020AA001", "S020BB002"], "node_index": [0, 1.5]}
    )
    polygons = gpd.GeoDataFrame(
        {
            "InterZone": ["S020AA001", "S020BB002"],
            "geometry": [_square(0, 0), _square(1, 0)],
        },
        crs="EPSG:27700",
    )
    with pytest.raises(GraphError, match="non-integer node_index"):
        construct_adjacency_graph(
            nodes=nodes,
            polygons=polygons,
            output_dir=tmp_path / "graph",
        )


def test_unsupported_geometry_type_raises(tmp_path: Path):
    nodes = pd.DataFrame(
        {"IntZone": ["S020AA001", "S020BB002"], "node_index": [0, 1]}
    )
    polygons = gpd.GeoDataFrame(
        {
            "InterZone": ["S020AA001", "S020BB002"],
            "geometry": [LineString([(0, 0), (1, 0)]), _square(10, 10)],
        },
        crs="EPSG:27700",
    )
    with pytest.raises(GraphError, match="not Polygon or MultiPolygon"):
        construct_adjacency_graph(
            nodes=nodes,
            polygons=polygons,
            output_dir=tmp_path / "graph",
        )


def test_invalid_geometry_that_cannot_become_polygon_fails(tmp_path: Path):
    nodes = pd.DataFrame(
        {"IntZone": ["S020AA001", "S020BB002"], "node_index": [0, 1]}
    )
    # Collinear ring: invalid or empty polygon; make_valid typically yields a LineString.
    collapsed = Polygon([(0, 0), (1, 0), (2, 0), (0, 0)])
    repaired = make_valid(collapsed)
    assert repaired.geom_type not in {"Polygon", "MultiPolygon"} or repaired.is_empty or collapsed.is_empty
    polygons = gpd.GeoDataFrame(
        {
            "InterZone": ["S020AA001", "S020BB002"],
            "geometry": [collapsed, _square(10, 10)],
        },
        crs="EPSG:27700",
    )
    with pytest.raises(GraphError, match="S020AA001"):
        construct_adjacency_graph(
            nodes=nodes,
            polygons=polygons,
            output_dir=tmp_path / "graph",
        )
