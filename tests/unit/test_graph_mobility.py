from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from graph.geo import GraphError
from graph.mobility import (
    _assert_overwrite_allowed,
    _node_order_hash,
    _validate_mobility_nodes,
    construct_mobility_graph,
)


@pytest.mark.external_data
def test_glasgow_od_defaults_to_city_folder():
    from graph.mobility import default_od_path

    path = default_od_path("S12000049")
    assert path.name == "averaged_od_matrix_2019_2023.csv"
    assert "Glasgow" in str(path)
    assert path.exists()
    assert path.stat().st_size > 0
    edinburgh = default_od_path("S12000036")
    assert "Edinburgh" in str(edinburgh)


def _nodes():
    return pd.DataFrame(
        {
            "IntZone": ["S020AA001", "S020BB002", "S020CC003"],
            "node_index": [0, 1, 2],
        }
    )


def _od_rows():
    # File order is deliberately not node_index order. Extra IZ is not a COVID node.
    return pd.DataFrame(
        {
            "origin_geo_code": [
                "S020CC003",
                "S020CC003",
                "S020CC003",
                "S020AA001",
                "S020AA001",
                "S020AA001",
                "S020BB002",
                "S020BB002",
                "S020BB002",
                "S020XX999",
                "S020AA001",
            ],
            "destination_geo_code": [
                "S020CC003",
                "S020AA001",
                "S020BB002",
                "S020AA001",
                "S020BB002",
                "S020CC003",
                "S020AA001",
                "S020BB002",
                "S020CC003",
                "S020AA001",
                "S020XX999",
            ],
            "percentage": [
                0.10,
                0.01,
                0.02,
                0.20,
                0.08,
                0.00,
                0.03,
                0.15,
                0.04,
                0.05,
                0.06,
            ],
        }
    )


def _complete_internal_od():
    codes = ["S020AA001", "S020BB002", "S020CC003"]
    rows = []
    for origin in codes:
        for dest in codes:
            if origin == dest:
                share = 0.20
            elif origin < dest:
                share = 0.08
            else:
                share = 0.03
            rows.append(
                {
                    "origin_geo_code": origin,
                    "destination_geo_code": dest,
                    "percentage": share,
                }
            )
    return pd.DataFrame(rows)


def test_construct_keeps_covid_node_order_and_drops_extra_iz(tmp_path: Path):
    result = construct_mobility_graph(
        nodes=_nodes(),
        od=_od_rows(),
        output_dir=tmp_path / "mobility",
    )
    assert result["status"] == "ok_with_warnings"
    assert result["status"] == result["validation_report"]["status"]
    nodes = pd.read_csv(result["output_paths"]["nodes"])
    assert nodes["IntZone"].tolist() == ["S020AA001", "S020BB002", "S020CC003"]
    assert nodes["node_index"].tolist() == [0, 1, 2]
    assert nodes["self_percentage"].tolist() == pytest.approx([0.20, 0.15, 0.10])
    assert "S020XX999" not in set(nodes["IntZone"])
    report = result["validation_report"]
    assert report["excluded_iz"] == ["S020XX999"]
    assert report["renormalised_after_excluding_extra_iz"] is False
    assert report["final_od_matrix_not_used"] is True
    assert report["k_nn_not_applied"] is True
    assert report["covid_study_years"] == [2020, 2021, 2022, 2023]
    assert report["source_years"] == [2019, 2020, 2021, 2022, 2023]
    assert report["mobility_graph_is_static"] is True
    assert report["mobility_graph_is_directed"] is True
    assert report["adjacency_is_row_normalised"] is False
    assert report["model_stage_will_normalise"] is True
    assert report["self_flow_excluded_from_adjacency"] is True
    assert report["geographic_and_road_graphs_not_merged"] is True
    assert report["weight_field"] == "percentage"
    assert report["weight_meaning"] == "recorded global flow share"
    assert report["is_complete_internal_od_matrix"] is True
    assert report["expected_internal_od_pairs"] == 9
    assert report["observed_internal_od_pairs"] == 9
    assert report["missing_internal_od_pairs"] == 0
    assert report["missing_origin_iz"] == []
    assert report["missing_destination_iz"] == []
    assert set(result["output_paths"]) == {
        "nodes",
        "edges",
        "adjacency",
        "od_matrix",
        "od_matrix_npz",
        "validation_report",
    }

    packed = np.load(result["output_paths"]["od_matrix_npz"])
    matrix = packed["od_percentage"]
    assert matrix.shape == (3, 3)
    assert matrix[0, 0] == pytest.approx(0.20)
    assert matrix[0, 1] == pytest.approx(0.08)
    assert matrix[0, 2] == pytest.approx(0.0)
    assert matrix[1, 0] == pytest.approx(0.03)
    assert matrix[2, 0] == pytest.approx(0.01)
    assert str(packed["node_order_hash"].item()) == report["node_order_hash"]

    adjacency = sparse.load_npz(result["output_paths"]["adjacency"]).toarray()
    assert np.all(np.diag(adjacency) == 0)
    assert adjacency[0, 1] == pytest.approx(0.08)
    assert adjacency[1, 0] == pytest.approx(0.03)
    assert adjacency[0, 2] == 0.0
    assert adjacency[0, 1] != adjacency[1, 0]

    edges = pd.read_csv(result["output_paths"]["edges"])
    assert not (edges["source_node_index"] == edges["target_node_index"]).any()
    pairs = set(
        zip(edges["source_node_index"].astype(int), edges["target_node_index"].astype(int))
    )
    assert (0, 2) not in pairs
    assert result["n_edges"] == 5
    first = edges.loc[edges["source_node_index"].eq(0)].iloc[0]
    assert int(first["target_node_index"]) == 1
    assert int(first["neighbour_rank"]) == 1


def test_node_order_must_be_sorted_unique_and_contiguous():
    ok = _validate_mobility_nodes(_nodes())
    assert ok["node_index"].tolist() == [0, 1, 2]
    unsorted = pd.DataFrame(
        {"IntZone": ["S020BB002", "S020AA001", "S020CC003"], "node_index": [1, 0, 2]}
    )
    with pytest.raises(GraphError, match="consecutive from 0 to N-1, and sorted"):
        _validate_mobility_nodes(unsorted)
    gapped = pd.DataFrame(
        {"IntZone": ["S020AA001", "S020BB002", "S020CC003"], "node_index": [0, 1, 3]}
    )
    with pytest.raises(GraphError, match="consecutive from 0 to N-1, and sorted"):
        _validate_mobility_nodes(gapped)
    duplicate_iz = pd.DataFrame(
        {"IntZone": ["S020AA001", "S020AA001", "S020CC003"], "node_index": [0, 1, 2]}
    )
    with pytest.raises(GraphError, match="duplicate IntZone"):
        _validate_mobility_nodes(duplicate_iz)
    null_index = pd.DataFrame(
        {"IntZone": ["S020AA001", "S020BB002"], "node_index": [0, np.nan]}
    )
    with pytest.raises(GraphError, match="non-null integers"):
        _validate_mobility_nodes(null_index)


def test_node_order_hash_is_deterministic_and_order_sensitive():
    first = _node_order_hash(["S020AA001", "S020BB002", "S020CC003"])
    second = _node_order_hash(["S020AA001", "S020BB002", "S020CC003"])
    swapped = _node_order_hash(["S020BB002", "S020AA001", "S020CC003"])
    assert first["node_order_hash"] == second["node_order_hash"]
    assert first["node_order_hash_algorithm"] == "SHA256"
    assert len(first["node_order_hash"]) == 64
    assert first["node_order_n"] == 3
    assert first["node_order_first_iz"] == "S020AA001"
    assert first["node_order_last_iz"] == "S020CC003"
    assert first["node_order_hash"] != swapped["node_order_hash"]


def test_unsorted_nodes_raise_before_graph_construction(tmp_path: Path):
    unsorted = pd.DataFrame(
        {"IntZone": ["S020BB002", "S020AA001", "S020CC003"], "node_index": [1, 0, 2]}
    )
    with pytest.raises(GraphError, match="consecutive from 0 to N-1, and sorted"):
        construct_mobility_graph(
            nodes=unsorted, od=_complete_internal_od(), output_dir=tmp_path / "mobility"
        )


def test_incomplete_internal_od_is_not_imputed(tmp_path: Path):
    od = _complete_internal_od()
    od = od.loc[~((od["origin_geo_code"].eq("S020AA001") & od["destination_geo_code"].eq("S020CC003")))]
    result = construct_mobility_graph(
        nodes=_nodes(), od=od, output_dir=tmp_path / "mobility"
    )
    report = result["validation_report"]
    assert report["is_complete_internal_od_matrix"] is False
    assert report["expected_internal_od_pairs"] == 9
    assert report["observed_internal_od_pairs"] == 8
    assert report["missing_internal_od_pairs"] == 1
    assert result["status"] == "ok_with_warnings"
    packed = np.load(result["output_paths"]["od_matrix_npz"])
    assert np.isnan(packed["od_percentage"][0, 2])
    adjacency = sparse.load_npz(result["output_paths"]["adjacency"]).toarray()
    assert adjacency[0, 2] == 0.0


def test_missing_origin_or_destination_is_reported(tmp_path: Path):
    od = _complete_internal_od()
    od = od.loc[~od["origin_geo_code"].eq("S020CC003")]
    result = construct_mobility_graph(
        nodes=_nodes(), od=od, output_dir=tmp_path / "mobility"
    )
    report = result["validation_report"]
    assert report["missing_origin_iz"] == ["S020CC003"]
    assert report["missing_destination_iz"] == []
    assert result["status"] == "ok_with_warnings"
    assert report["zero_outflow_iz"] == ["S020CC003"]


def test_missing_covid_iz_raises(tmp_path: Path):
    od = _od_rows()
    od = od.loc[~od["origin_geo_code"].eq("S020CC003") & ~od["destination_geo_code"].eq("S020CC003")]
    with pytest.raises(GraphError, match="absent from the OD matrix"):
        construct_mobility_graph(nodes=_nodes(), od=od, output_dir=tmp_path / "mobility")


def test_directionality_and_zero_diagonal(tmp_path: Path):
    result = construct_mobility_graph(
        nodes=_nodes(), od=_complete_internal_od(), output_dir=tmp_path / "mobility"
    )
    adjacency = sparse.load_npz(result["output_paths"]["adjacency"]).toarray()
    assert np.all(np.diag(adjacency) == 0)
    assert adjacency[0, 1] == pytest.approx(0.08)
    assert adjacency[1, 0] == pytest.approx(0.03)
    assert adjacency[0, 1] != adjacency[1, 0]
    nodes = pd.read_csv(result["output_paths"]["nodes"])
    assert nodes["self_percentage"].tolist() == pytest.approx([0.20, 0.20, 0.20])
    assert result["validation_report"]["self_flow_excluded_from_adjacency"] is True
    assert result["validation_report"]["n_strongly_connected_components"] == 1
    assert result["validation_report"]["n_weakly_connected_components"] == 1


def test_zero_inflow_and_outflow_ignore_self_flow(tmp_path: Path):
    od = pd.DataFrame(
        {
            "origin_geo_code": [
                "S020AA001",
                "S020AA001",
                "S020BB002",
                "S020BB002",
                "S020CC003",
            ],
            "destination_geo_code": [
                "S020AA001",
                "S020BB002",
                "S020BB002",
                "S020AA001",
                "S020CC003",
            ],
            "percentage": [0.2, 0.05, 0.2, 0.05, 0.2],
        }
    )
    result = construct_mobility_graph(nodes=_nodes(), od=od, output_dir=tmp_path / "mobility")
    report = result["validation_report"]
    assert report["zero_outflow_iz"] == ["S020CC003"]
    assert report["zero_inflow_iz"] == ["S020CC003"]
    assert report["isolated_iz"] == ["S020CC003"]
    assert result["status"] == "ok_with_warnings"


def test_invalid_percentage_values_raise(tmp_path: Path):
    negative = _complete_internal_od()
    negative.loc[0, "percentage"] = -0.1
    with pytest.raises(GraphError, match="negative"):
        construct_mobility_graph(nodes=_nodes(), od=negative, output_dir=tmp_path / "neg")
    infinite = _complete_internal_od()
    infinite.loc[0, "percentage"] = np.inf
    with pytest.raises(GraphError, match="NaN or infinite"):
        construct_mobility_graph(nodes=_nodes(), od=infinite, output_dir=tmp_path / "inf")
    missing = _complete_internal_od()
    missing.loc[0, "percentage"] = np.nan
    with pytest.raises(GraphError, match="NaN or infinite"):
        construct_mobility_graph(nodes=_nodes(), od=missing, output_dir=tmp_path / "nan")


def test_adjacency_matches_positive_offdiag_edges(tmp_path: Path):
    result = construct_mobility_graph(
        nodes=_nodes(), od=_od_rows(), output_dir=tmp_path / "mobility"
    )
    edges = pd.read_csv(result["output_paths"]["edges"])
    adjacency = sparse.load_npz(result["output_paths"]["adjacency"])
    dense = adjacency.toarray()
    assert int(adjacency.nnz) == len(edges)
    for row in edges.itertuples(index=False):
        assert dense[int(row.source_node_index), int(row.target_node_index)] == pytest.approx(
            float(row.weight)
        )
    assert np.count_nonzero(dense) == len(edges)
    assert np.all(np.diag(dense) == 0)


def test_status_ok_when_internal_od_is_complete_and_connected(tmp_path: Path):
    result = construct_mobility_graph(
        nodes=_nodes(), od=_complete_internal_od(), output_dir=tmp_path / "mobility"
    )
    assert result["status"] == "ok"
    assert result["validation_report"]["status"] == "ok"
    assert result["validation_report"]["is_complete_internal_od_matrix"] is True
    assert result["validation_report"]["n_isolated"] == 0
    assert result["validation_report"]["excluded_iz"] == []
    assert any("pre-averaged 2019-2023" in item for item in result["validation_report"]["warnings"])
    assert result["validation_report"]["local_authority_code"] == "S12000036"
    assert result["validation_report"]["study_area"] == "City of Edinburgh"


@pytest.mark.external_data
def test_validation_report_uses_supplied_area_code(tmp_path: Path):
    result = construct_mobility_graph(
        nodes=_nodes(),
        od=_complete_internal_od(),
        output_dir=tmp_path / "mobility",
        area_code="S12000019",
    )
    assert result["validation_report"]["local_authority_code"] == "S12000019"
    assert result["validation_report"]["study_area"] == "Midlothian"
    unknown = construct_mobility_graph(
        nodes=_nodes(),
        od=_complete_internal_od(),
        output_dir=tmp_path / "unknown",
        area_code="S99999999",
    )
    assert unknown["validation_report"]["local_authority_code"] == "S99999999"
    assert unknown["validation_report"]["study_area"] == "unknown (S99999999)"


def test_duplicate_od_pairs_raise(tmp_path: Path):
    od = pd.concat([_od_rows(), _od_rows().iloc[[0]]], ignore_index=True)
    with pytest.raises(GraphError, match="duplicate origin-destination"):
        construct_mobility_graph(nodes=_nodes(), od=od, output_dir=tmp_path / "mobility")


def test_final_od_filename_is_rejected(tmp_path: Path):
    path = tmp_path / "final_od_matrix_Edinburgh_averaged.csv"
    _od_rows().to_csv(path, index=False)
    with pytest.raises(GraphError, match="not the mobility source"):
        construct_mobility_graph(nodes=_nodes(), od=path, output_dir=tmp_path / "out")


def test_overwrite_protection(tmp_path: Path):
    output_dir = tmp_path / "mobility"
    output_dir.mkdir()
    (output_dir / "od_matrix.csv").write_text("x", encoding="utf-8")
    with pytest.raises(GraphError, match="od_matrix.csv"):
        _assert_overwrite_allowed(output_dir, overwrite=False)
