from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from shapely.geometry import LineString, Point, Polygon

from graph.geo import GraphError
from graph.road import (
    MODE_ORDER,
    RAIL_FCLASS,
    ROAD_FCLASS,
    STATION_FCLASS,
    _assert_overwrite_allowed,
    _directions,
    _lines_to_graph,
    _load_iz_coordinates,
    _normalise_line_frame,
    _normalise_station_frame,
    _od_distances,
    _snap_status,
    _snap_xy,
    assert_mode_distance_matrix,
    combine_mode_distances,
    construct_road_graph,
    directed_adjacency_matrix,
    intermodal_distance,
    knn_road_edges,
)


def _square(x0: float, y0: float, size: float = 1.0) -> Polygon:
    return Polygon(
        [
            (x0, y0),
            (x0 + size, y0),
            (x0 + size, y0 + size),
            (x0, y0 + size),
        ]
    )


def _nodes():
    return pd.DataFrame(
        {
            "IntZone": ["S020AA001", "S020BB002", "S020CC003"],
            "node_index": [0, 1, 2],
        }
    )


def _inf_matrix(n: int) -> np.ndarray:
    matrix = np.full((n, n), np.inf, dtype="float64")
    np.fill_diagonal(matrix, 0.0)
    return matrix


def test_directions_follow_geofabrik_oneway_rules():
    assert _directions("walking", "residential", "F") == (True, True)
    assert _directions("road", "residential", "F") == (True, False)
    assert _directions("road", "residential", "T") == (False, True)
    assert _directions("road", "residential", "B") == (True, True)
    assert _directions("road", "residential", "") == (True, True)
    assert _directions("bicycle", "cycleway", "F") == (True, True)
    assert _directions("bicycle", "residential", "F") == (True, False)
    assert _directions("rail", "rail", "F") == (True, False)
    assert _directions("rail", "rail", "T") == (False, True)
    assert _directions("rail", "rail", "B") == (True, True)
    with pytest.raises(GraphError, match="Unsupported Geofabrik oneway"):
        _directions("road", "residential", "YES")
    with pytest.raises(GraphError, match="Unsupported Geofabrik oneway"):
        _directions("walking", "residential", "YES")


def test_combine_keeps_node_index_order_and_five_modes():
    n = 3
    matrices = {name: _inf_matrix(n) for name in MODE_ORDER}
    matrices["road"][0, 1] = 4.0
    matrices["bicycle"][0, 1] = 3.0
    matrices["walking"][0, 1] = 5.0
    combined, shortest_mode = combine_mode_distances(matrices)
    assert combined.shape == (3, 3)
    assert shortest_mode.shape == (3, 3)
    assert list(MODE_ORDER) == [
        "road",
        "bicycle",
        "walking",
        "road+rail",
        "walking+rail",
    ]
    assert combined[0, 1] == 3.0
    assert shortest_mode[0, 1] == "bicycle"
    assert np.all(np.diag(combined) == 0.0)
    assert shortest_mode[0, 0] == "self"
    assert shortest_mode[1, 2] == "unreachable"


def test_inf_is_not_replaced_with_zero():
    matrices = {name: _inf_matrix(2) for name in MODE_ORDER}
    combined, shortest_mode = combine_mode_distances(matrices)
    assert np.isposinf(combined[0, 1])
    assert np.isposinf(combined[1, 0])
    assert combined[0, 1] != 0.0
    assert shortest_mode[0, 1] == "unreachable"
    assert shortest_mode[0, 0] == "self"


def test_min_selects_shortest_finite_mode():
    matrices = {name: _inf_matrix(2) for name in MODE_ORDER}
    matrices["road"][0, 1] = np.inf
    matrices["bicycle"][0, 1] = 3.0
    matrices["walking"][0, 1] = 2.0
    matrices["road+rail"][0, 1] = 4.0
    combined, shortest_mode = combine_mode_distances(matrices)
    assert combined[0, 1] == 2.0
    assert shortest_mode[0, 1] == "walking"


def test_tie_break_follows_mode_order_not_travel_time():
    matrices = {name: _inf_matrix(2) for name in MODE_ORDER}
    for name in MODE_ORDER:
        matrices[name][0, 1] = 5.0
    combined, shortest_mode = combine_mode_distances(matrices)
    assert combined[0, 1] == 5.0
    assert shortest_mode[0, 1] == "road"
    assert shortest_mode[0, 1] != "fastest_mode"


def test_intermodal_is_directed_and_excludes_same_station():
    # IZ0 can only enter station A; IZ1 can only leave station B.
    access = np.array([[0.1, np.inf], [np.inf, 0.1]], dtype="float64")
    rail = np.array([[0.0, 1.0], [np.inf, 0.0]], dtype="float64")
    egress = np.array([[0.1, np.inf], [np.inf, 0.1]], dtype="float64")
    road_rail = intermodal_distance(access, rail, egress)
    assert road_rail[0, 1] == pytest.approx(1.2)
    assert np.isposinf(road_rail[1, 0])
    assert road_rail[0, 0] == 0.0

    # Same-station "rail" would otherwise become walk-in / walk-out with no train.
    same_station = intermodal_distance(
        np.array([[0.0, np.inf], [np.inf, 0.0]], dtype="float64"),
        np.array([[0.0, np.inf], [np.inf, 0.0]], dtype="float64"),
        np.array([[0.0, 0.5], [0.5, 0.0]], dtype="float64"),
        require_distinct_stations=True,
    )
    assert np.isposinf(same_station[0, 1])
    allowed = intermodal_distance(
        np.array([[0.0, np.inf], [np.inf, 0.0]], dtype="float64"),
        np.array([[0.0, np.inf], [np.inf, 0.0]], dtype="float64"),
        np.array([[0.0, 0.5], [0.5, 0.0]], dtype="float64"),
        require_distinct_stations=False,
    )
    assert allowed[0, 1] == pytest.approx(0.5)


def test_intermodal_does_not_use_euclidean_nearest_station():
    # Station 0 is the Euclidean-near station but is unreachable on the network.
    access = np.array([[np.inf, 2.0], [2.0, np.inf]], dtype="float64")
    rail = np.array([[0.0, 1.0], [1.0, 0.0]], dtype="float64")
    egress = np.array([[0.1, 2.0], [2.0, 0.1]], dtype="float64")
    distance = intermodal_distance(access, rail, egress)
    assert distance[0, 1] == pytest.approx(5.0)
    euclidean_if_near_station_used = 0.01 + 1.0 + 0.1
    assert distance[0, 1] != pytest.approx(euclidean_if_near_station_used)


def test_knn_k_cap_no_self_loop_and_kernel_weights():
    nodes = _nodes()
    distance = np.array(
        [
            [0.0, 1.0, 2.0],
            [4.0, 0.0, 3.0],
            [np.inf, 5.0, 0.0],
        ],
        dtype="float64",
    )
    mode = np.array(
        [
            ["", "road", "bicycle"],
            ["walking", "", "road+rail"],
            ["", "walking+rail", ""],
        ],
        dtype=object,
    )
    edges, tau = knn_road_edges(distance, mode, nodes, k=2)
    assert tau == pytest.approx(3.0)
    assert (edges["source_node_index"] != edges["target_node_index"]).all()
    assert edges.groupby("source_node_index").size().max() <= 2
    assert set(edges.columns) >= {
        "source_node_index",
        "target_node_index",
        "source_iz_code",
        "target_iz_code",
        "multimodal_distance_km",
        "shortest_mode",
        "neighbour_rank",
        "weight",
    }
    assert (edges["weight"] > 0).all()
    assert (edges["weight"] <= 1).all()
    # Kernel is exp(-d/tau), not 1/(D+eps).
    row = edges.loc[
        (edges["source_node_index"] == 0) & (edges["target_node_index"] == 1)
    ].iloc[0]
    assert row["weight"] == pytest.approx(np.exp(-1.0 / tau))
    assert row["weight"] != pytest.approx(1.0 / (1.0 + 1e-6))
    assert row["shortest_mode"] == "road"
    assert row["neighbour_rank"] == 1

    adjacency = directed_adjacency_matrix(nodes, edges)
    assert adjacency.shape == (3, 3)
    assert np.all(adjacency.diagonal() == 0)
    assert int(adjacency.nnz) == len(edges)
    dense = adjacency.toarray()
    assert dense[2, 0] == 0.0
    assert dense[0, 1] > 0.0
    assert dense[1, 0] > 0.0
    assert dense[0, 1] != dense[1, 0]


def test_construct_follows_node_index_and_road_rail_direction(tmp_path: Path):
    nodes = pd.DataFrame(
        {"IntZone": ["S020AA001", "S020BB002"], "node_index": [0, 1]}
    )
    # Row order is deliberately not node_index order; X/Y aliases must map to Easting/Northing.
    coords = pd.DataFrame(
        {
            "InterZone": ["S020BB002", "S020AA001"],
            "X": [1000.0, 0.0],
            "Y": [50.0, 50.0],
        }
    )
    polygons = gpd.GeoDataFrame(
        {
            "InterZone": ["S020AA001", "S020BB002"],
            "geometry": [_square(-100, -100, 200), _square(900, -100, 200)],
        },
        crs="EPSG:27700",
    )
    roads = gpd.GeoDataFrame(
        {
            "fclass": ["residential", "residential"],
            "oneway": ["B", "B"],
            "geometry": [
                LineString([(0, 50), (0, 0)]),
                LineString([(1000, 50), (1000, 0)]),
            ],
        },
        crs="EPSG:27700",
    )
    railways = gpd.GeoDataFrame(
        {
            "fclass": ["rail"],
            "oneway": ["F"],
            "geometry": [LineString([(0, 0), (1000, 0)])],
        },
        crs="EPSG:27700",
    )
    stations = gpd.GeoDataFrame(
        {
            "fclass": ["railway_station", "railway_halt", "tram_stop"],
            "osm_id": ["1", "2", "3"],
            "geometry": [Point(0, 0), Point(1000, 0), Point(500, 0)],
        },
        crs="EPSG:27700",
    )
    result = construct_road_graph(
        nodes=nodes,
        coords=coords,
        polygons=polygons,
        roads=roads,
        railways=railways,
        stations=stations,
        output_dir=tmp_path / "road",
        k=1,
        buffer_m=2000.0,
        snap_max_m=100.0,
    )
    node_table = pd.read_csv(result["output_paths"]["nodes"])
    assert node_table["IntZone"].tolist() == ["S020AA001", "S020BB002"]
    assert node_table["node_index"].tolist() == [0, 1]
    packed = np.load(result["output_paths"]["distances"])
    for key in ("D_road", "D_bicycle", "D_walking", "D_road_rail", "D_walking_rail", "D_multimodal"):
        assert packed[key].shape == (2, 2)
        assert packed[key][0, 0] == 0.0
        assert packed[key][1, 1] == 0.0
    assert np.isposinf(packed["D_road"][0, 1])
    assert packed["D_road"][0, 1] != 0.0
    assert packed["D_road_rail"][0, 1] == pytest.approx(1.1, abs=1e-6)
    assert np.isposinf(packed["D_road_rail"][1, 0])
    assert packed["D_walking_rail"][0, 1] == pytest.approx(1.1, abs=1e-6)
    assert packed["D_multimodal"][0, 1] == pytest.approx(1.1, abs=1e-6)
    assert str(packed["shortest_mode"][0, 1]) == "road+rail"
    assert str(packed["shortest_mode"][0, 0]) == "self"
    assert "iz_snaps" in result["output_paths"]
    snaps = pd.read_csv(result["output_paths"]["iz_snaps"])
    assert {"mode", "snap_ok", "snap_distance_m", "snap_node_index", "snap_status"} <= set(snaps.columns)
    assert set(snaps["mode"]) >= {"road", "bicycle", "walking"}
    assert result["validation_report"]["buffer_m"] == 2000.0
    assert result["validation_report"]["snap_max_m"] == 100.0
    assert result["validation_report"]["n_stations"] == 2
    assert result["validation_report"]["tram_and_tram_stop_excluded"] is True
    assert result["validation_report"]["defaulted_grade_not_treated_as_complete"] is True
    assert "n_iz_long_accepted_snap" in result["validation_report"]
    assert "n_station_long_accepted_snap" in result["validation_report"]
    stations = pd.read_csv(result["output_paths"]["station_snaps"])
    assert set(stations["fclass"]) <= {"railway_station", "railway_halt"}
    assert "tram_stop" not in set(stations["fclass"])

    matrix = sparse.load_npz(result["output_paths"]["adjacency"])
    dense = matrix.toarray()
    assert dense.shape == (2, 2)
    assert np.all(np.diag(dense) == 0)
    assert dense[0, 1] > 0.0
    assert dense[0, 1] <= 1.0
    assert dense[1, 0] == 0.0
    edges = pd.read_csv(result["output_paths"]["edges"])
    assert not (edges["source_node_index"] == edges["target_node_index"]).any()
    assert edges["neighbour_rank"].max() <= 1
    assert "fastest_mode" not in edges.columns
    csv_text = Path(result["output_paths"]["distances"]).with_name("D_road_km.csv").read_text()
    assert "Inf" in csv_text


def test_construct_ignores_disconnected_euclidean_nearest_station(tmp_path: Path):
    nodes = pd.DataFrame(
        {"IntZone": ["S020AA001", "S020BB002"], "node_index": [0, 1]}
    )
    coords = pd.DataFrame(
        {"IntZone": ["S020AA001", "S020BB002"], "Easting": [0.0, 200.0], "Northing": [0.0, 0.0]}
    )
    polygons = gpd.GeoDataFrame(
        {
            "IntZone": ["S020AA001", "S020BB002"],
            "geometry": [_square(-20, -20, 40), _square(180, -20, 40)],
        },
        crs="EPSG:27700",
    )
    roads = gpd.GeoDataFrame(
        {
            "fclass": ["residential", "residential"],
            "oneway": ["B", "B"],
            "geometry": [
                LineString([(0, 0), (200, 0)]),
                LineString([(10, 100), (11, 100)]),
            ],
        },
        crs="EPSG:27700",
    )
    railways = gpd.GeoDataFrame(
        {
            "fclass": ["rail"],
            "oneway": ["F"],
            "geometry": [LineString([(0, 0), (200, 0)])],
        },
        crs="EPSG:27700",
    )
    stations = gpd.GeoDataFrame(
        {
            "fclass": ["railway_station", "railway_station"],
            "geometry": [Point(10, 100), Point(200, 0)],
        },
        crs="EPSG:27700",
    )
    result = construct_road_graph(
        nodes=nodes,
        coords=coords,
        polygons=polygons,
        roads=roads,
        railways=railways,
        stations=stations,
        output_dir=tmp_path / "road",
        k=1,
        buffer_m=2000.0,
        snap_max_m=20.0,
    )
    packed = np.load(result["output_paths"]["distances"])
    # Direct road is 0.2 km. Using the Euclidean-near isolated station cannot be shorter
    # because that station is not on the IZ road component.
    assert packed["D_road"][0, 1] == pytest.approx(0.2, abs=1e-6)
    assert packed["D_multimodal"][0, 1] == pytest.approx(0.2, abs=1e-6)
    assert str(packed["shortest_mode"][0, 1]) == "road"


def test_missing_mode_matrix_raises():
    with pytest.raises(GraphError, match="Missing mode matrices"):
        combine_mode_distances({"road": _inf_matrix(2)})


def test_geofabrik_track_grades_are_in_street_allow_lists():
    for grade in ("track", "track_grade1", "track_grade2", "track_grade3", "track_grade4", "track_grade5"):
        assert grade in ROAD_FCLASS


def test_stations_exclude_tram_stop():
    frame = gpd.GeoDataFrame(
        {
            "fclass": ["railway_station", "railway_halt", "tram_stop", "bus_stop"],
            "geometry": [Point(0, 0), Point(1, 0), Point(2, 0), Point(3, 0)],
        },
        crs="EPSG:27700",
    )
    kept = _normalise_station_frame(frame)
    assert set(kept["fclass"]) == {"railway_station", "railway_halt"}
    assert STATION_FCLASS == {"railway_station", "railway_halt"}
    assert "tram" not in RAIL_FCLASS
    assert "funicular" not in RAIL_FCLASS
    assert "tram_stop" not in STATION_FCLASS
    assert RAIL_FCLASS == {"rail", "light_rail", "subway", "narrow_gauge"}


def test_identical_iz_coordinate_repeats_are_collapsed():
    nodes = pd.DataFrame({"IntZone": ["S020AA001", "S020BB002"], "node_index": [0, 1]})
    coords = pd.DataFrame(
        {
            "IntZone": ["S020AA001", "S020AA001", "S020BB002"],
            "Easting": [1.0, 1.0, 2.0],
            "Northing": [3.0, 3.0, 4.0],
        }
    )
    table, _ = _load_iz_coordinates(nodes, coords)
    assert len(table) == 2
    assert table["Easting"].tolist() == [1.0, 2.0]


def test_conflicting_duplicate_iz_coordinates_raise():
    nodes = pd.DataFrame({"IntZone": ["S020AA001", "S020BB002"], "node_index": [0, 1]})
    coords = pd.DataFrame(
        {
            "IntZone": ["S020AA001", "S020AA001", "S020BB002"],
            "Easting": [1.0, 9.0, 2.0],
            "Northing": [3.0, 3.0, 4.0],
        }
    )
    with pytest.raises(GraphError, match="conflicting Easting/Northing"):
        _load_iz_coordinates(nodes, coords)


def test_non_finite_easting_northing_raise():
    nodes = pd.DataFrame({"IntZone": ["S020AA001", "S020BB002"], "node_index": [0, 1]})
    coords = pd.DataFrame(
        {
            "IntZone": ["S020AA001", "S020BB002"],
            "Easting": [1.0, np.inf],
            "Northing": [3.0, 4.0],
        }
    )
    with pytest.raises(GraphError, match="NaN or infinite"):
        _load_iz_coordinates(nodes, coords)


def test_k_must_be_an_integer():
    nodes = pd.DataFrame({"IntZone": ["S020AA001", "S020BB002"], "node_index": [0, 1]})
    distance = np.array([[0.0, 1.0], [1.0, 0.0]], dtype="float64")
    mode = np.array([["self", "road"], ["road", "self"]], dtype=object)
    with pytest.raises(GraphError, match="k must be an integer"):
        knn_road_edges(distance, mode, nodes, k=1.5)
    with pytest.raises(GraphError, match="k must be an integer"):
        knn_road_edges(distance, mode, nodes, k=True)


def test_tau_must_be_positive():
    nodes = pd.DataFrame({"IntZone": ["S020AA001", "S020BB002"], "node_index": [0, 1]})
    distance = np.zeros((2, 2), dtype="float64")
    mode = np.array([["self", "road"], ["road", "self"]], dtype=object)
    with pytest.raises(GraphError, match="tau must be finite and positive"):
        knn_road_edges(distance, mode, nodes, k=1)


def test_mode_matrix_validation_rejects_nan_and_negative():
    good = _inf_matrix(2)
    good[0, 1] = 1.5
    assert_mode_distance_matrix(good, n_nodes=2, name="road")
    bad_nan = good.copy()
    bad_nan[0, 1] = np.nan
    with pytest.raises(GraphError, match="NaN"):
        assert_mode_distance_matrix(bad_nan, n_nodes=2, name="road")
    bad_neg = good.copy()
    bad_neg[0, 1] = -0.1
    with pytest.raises(GraphError, match="negative"):
        assert_mode_distance_matrix(bad_neg, n_nodes=2, name="road")
    bad_diag = good.copy()
    bad_diag[1, 1] = 3.0
    with pytest.raises(GraphError, match="diagonal"):
        assert_mode_distance_matrix(bad_diag, n_nodes=2, name="road")
    with pytest.raises(GraphError, match="shape"):
        assert_mode_distance_matrix(np.zeros((2, 3)), n_nodes=2, name="road")


def test_distance_csvs_are_in_overwrite_protection(tmp_path: Path):
    output_dir = tmp_path / "road"
    output_dir.mkdir()
    (output_dir / "D_road_km.csv").write_text("x", encoding="utf-8")
    with pytest.raises(GraphError, match="D_road_km.csv"):
        _assert_overwrite_allowed(output_dir, overwrite=False)
    (output_dir / "D_multimodal_km.csv").write_text("x", encoding="utf-8")
    with pytest.raises(GraphError, match="D_multimodal_km.csv"):
        _assert_overwrite_allowed(output_dir, overwrite=False)


def _path_km(graph, a_xy, b_xy, snap_max_m=50.0):
    points = np.asarray([a_xy, b_xy], dtype="float64")
    snap = _snap_xy(points, graph, snap_max_m)
    dist = _od_distances(
        graph,
        snap.node_index,
        snap.node_index,
        source_snap_m=np.zeros(2),
        target_snap_m=np.zeros(2),
    )
    return float(dist[0, 1])


def test_grade_separated_crossing_is_not_connected():
    roads = gpd.GeoDataFrame(
        {
            "fclass": ["residential", "residential"],
            "oneway": ["B", "B"],
            "layer": [0, 1],
            "bridge": ["F", "T"],
            "tunnel": ["F", "F"],
            "geometry": [
                LineString([(0, 0), (10, 0)]),
                LineString([(5, -5), (5, 5)]),
            ],
        },
        crs="EPSG:27700",
    )
    graph, report = _lines_to_graph(roads, mode="road")
    assert report["n_grade_separated_crossings_ignored"] >= 1
    assert report["n_ambiguous_crossings"] == 0
    assert report["source_has_layer"] is True
    assert report["source_has_bridge"] is True
    assert report["source_has_tunnel"] is True
    assert report["defaulted_grade_not_treated_as_complete"] is True
    assert np.isposinf(_path_km(graph, (0.0, 0.0), (5.0, 5.0)))


def test_at_grade_crossing_is_noded():
    roads = gpd.GeoDataFrame(
        {
            "fclass": ["residential", "residential"],
            "oneway": ["B", "B"],
            "layer": [0, 0],
            "bridge": ["F", "F"],
            "tunnel": ["F", "F"],
            "geometry": [
                LineString([(0, 0), (10, 0)]),
                LineString([(5, -5), (5, 5)]),
            ],
        },
        crs="EPSG:27700",
    )
    graph, report = _lines_to_graph(roads, mode="road")
    assert report["n_grade_separated_crossings_ignored"] == 0
    assert _path_km(graph, (0.0, 0.0), (5.0, 5.0)) == pytest.approx(0.01, abs=1e-6)


def test_ambiguous_crossing_is_reported():
    roads = gpd.GeoDataFrame(
        {
            "fclass": ["residential", "residential"],
            "oneway": ["B", "B"],
            "layer": [0, np.nan],
            "bridge": ["F", "F"],
            "tunnel": ["F", "F"],
            "geometry": [
                LineString([(0, 0), (10, 0)]),
                LineString([(5, -5), (5, 5)]),
            ],
        },
        crs="EPSG:27700",
    )
    _, report = _lines_to_graph(roads, mode="road")
    assert report["n_ambiguous_crossings"] >= 1
    assert report["n_grade_separated_crossings_ignored"] == 0
    assert report["n_records_missing_layer_value"] >= 1
    assert report["ambiguous_crossing_examples"]
    assert report["defaulted_grade_not_treated_as_complete"] is True


def test_missing_grade_source_fields_are_not_treated_as_complete():
    roads = gpd.GeoDataFrame(
        {
            "fclass": ["residential", "residential"],
            "oneway": ["B", "B"],
            "geometry": [
                LineString([(0, 0), (10, 0)]),
                LineString([(5, -5), (5, 5)]),
            ],
        },
        crs="EPSG:27700",
    )
    normalised = _normalise_line_frame(roads, "roads")
    graph, report = _lines_to_graph(normalised, mode="road")
    assert report["source_has_layer"] is False
    assert report["source_has_bridge"] is False
    assert report["source_has_tunnel"] is False
    assert report["n_records_without_source_layer"] == 2
    assert report["n_records_without_source_bridge"] == 2
    assert report["n_records_without_source_tunnel"] == 2
    assert report["defaulted_grade_not_treated_as_complete"] is True
    assert report["n_grade_separated_crossings_ignored"] == 0
    assert report["n_ambiguous_crossings"] >= 1
    assert _path_km(graph, (0.0, 0.0), (5.0, 5.0)) == pytest.approx(0.01, abs=1e-6)


def test_snap_distances_are_included_in_total_network_distance(tmp_path: Path):
    nodes = pd.DataFrame({"IntZone": ["S020AA001", "S020BB002"], "node_index": [0, 1]})
    coords = pd.DataFrame(
        {"IntZone": ["S020AA001", "S020BB002"], "Easting": [0.0, 1000.0], "Northing": [50.0, 50.0]}
    )
    polygons = gpd.GeoDataFrame(
        {
            "IntZone": ["S020AA001", "S020BB002"],
            "geometry": [_square(-100, -100, 200), _square(900, -100, 200)],
        },
        crs="EPSG:27700",
    )
    roads = gpd.GeoDataFrame(
        {
            "fclass": ["residential"],
            "oneway": ["B"],
            "geometry": [LineString([(0, 0), (1000, 0)])],
        },
        crs="EPSG:27700",
    )
    railways = gpd.GeoDataFrame(
        {"fclass": ["rail"], "oneway": ["B"], "geometry": [LineString([(0, -1000), (1, -1000)])]},
        crs="EPSG:27700",
    )
    stations = gpd.GeoDataFrame(
        {"fclass": ["railway_station"], "geometry": [Point(0, -1000)]},
        crs="EPSG:27700",
    )
    result = construct_road_graph(
        nodes=nodes,
        coords=coords,
        polygons=polygons,
        roads=roads,
        railways=railways,
        stations=stations,
        output_dir=tmp_path / "road",
        k=1,
        buffer_m=2000.0,
        snap_max_m=500.0,
    )
    packed = np.load(result["output_paths"]["distances"])
    # 50 m + 1000 m + 50 m = 1.1 km, not the 1.0 km path between snap nodes.
    assert packed["D_road"][0, 1] == pytest.approx(1.1, abs=1e-6)
    assert packed["D_road"][0, 1] != pytest.approx(1.0, abs=1e-6)
    snaps = pd.read_csv(result["output_paths"]["iz_snaps"])
    road_snaps = snaps.loc[snaps["mode"].eq("road")]
    assert np.allclose(road_snaps["snap_distance_m"].to_numpy(dtype="float64"), 50.0, atol=1e-6)
    assert set(road_snaps["snap_status"]) == {"ok"}
    assert result["validation_report"]["snap_distances_included_in_total"] is True
    assert result["validation_report"]["buffer_m"] == 2000.0
    assert result["validation_report"]["snap_max_m"] == 500.0
    assert result["validation_report"]["long_accepted_snap_threshold_m"] == 100.0
    assert result["validation_report"]["n_iz_long_accepted_snap"]["road"] == 0


def test_snap_status_distinguishes_long_accepted_snaps():
    assert _snap_status(False, np.inf, 100.0) == "unsnapped"
    assert _snap_status(True, 50.0, 100.0) == "ok"
    assert _snap_status(True, 150.0, 100.0) == "ok_long"
    assert _snap_status(True, 500.0, 100.0) == "ok_long"


def test_quality_report_counts_long_accepted_snaps(tmp_path: Path):
    nodes = pd.DataFrame({"IntZone": ["S020AA001", "S020BB002"], "node_index": [0, 1]})
    coords = pd.DataFrame(
        {"IntZone": ["S020AA001", "S020BB002"], "Easting": [0.0, 1000.0], "Northing": [150.0, 150.0]}
    )
    polygons = gpd.GeoDataFrame(
        {
            "IntZone": ["S020AA001", "S020BB002"],
            "geometry": [_square(-100, 50, 200), _square(900, 50, 200)],
        },
        crs="EPSG:27700",
    )
    roads = gpd.GeoDataFrame(
        {
            "fclass": ["residential"],
            "oneway": ["B"],
            "geometry": [LineString([(0, 0), (1000, 0)])],
        },
        crs="EPSG:27700",
    )
    railways = gpd.GeoDataFrame(
        {"fclass": ["rail"], "oneway": ["B"], "geometry": [LineString([(0, -1000), (1, -1000)])]},
        crs="EPSG:27700",
    )
    stations = gpd.GeoDataFrame(
        {"fclass": ["railway_station"], "geometry": [Point(0, 0)]},
        crs="EPSG:27700",
    )
    result = construct_road_graph(
        nodes=nodes,
        coords=coords,
        polygons=polygons,
        roads=roads,
        railways=railways,
        stations=stations,
        output_dir=tmp_path / "road",
        k=1,
        buffer_m=2000.0,
        snap_max_m=500.0,
    )
    snaps = pd.read_csv(result["output_paths"]["iz_snaps"])
    road_snaps = snaps.loc[snaps["mode"].eq("road")]
    assert np.allclose(road_snaps["snap_distance_m"].to_numpy(dtype="float64"), 150.0, atol=1e-6)
    assert set(road_snaps["snap_status"]) == {"ok_long"}
    assert result["validation_report"]["long_accepted_snap_threshold_m"] == 100.0
    assert result["validation_report"]["n_iz_long_accepted_snap"]["road"] == 2
    assert result["validation_report"]["n_iz_unsnapped"]["road"] == 0
