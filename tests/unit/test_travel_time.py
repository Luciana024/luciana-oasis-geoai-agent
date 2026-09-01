"""Travel-time matrix: notebook shortest-path logic with a city parameter."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from common.errors import ModelError
from common.utils import LOCAL_AUTHORITY_CODE
from data.candidate_sites import load_candidate_sites
from data.travel_time import (
    DRIVE_SPEED_KMH,
    WALK_SPEED_KMH,
    load_iz_origins,
    load_travel_time,
    prepare_travel_time,
    resolve_osm_query,
    shortest_path_matrix,
    validate_travel_time_matrix,
)


def _origins(codes=("S02001576", "S02001577")):
    return gpd.GeoDataFrame(
        {"iz_code": list(codes)},
        geometry=[Point(325000, 673000), Point(326000, 674000)],
        crs="EPSG:27700",
    )


def _sites(ids=("GP_1", "PH_1")):
    return gpd.GeoDataFrame(
        {"site_id": list(ids)},
        geometry=[Point(325100, 673100), Point(326100, 674100)],
        crs="EPSG:27700",
    )


def test_shortest_path_matrix_keeps_unreachable_nan():
    frame = shortest_path_matrix(
        iz_codes=["A", "B"],
        site_ids=["S1", "S2"],
        origin_nodes=["n0", "n1"],
        dest_nodes=["d0", "d1"],
        path_lengths_by_origin={
            "n0": {"d0": 4.444, "d1": 10.0},
            "n1": {"d0": 8.0},
        },
        mode_name="drive",
    )
    assert list(frame.columns) == ["iz_code", "site_id", "mode", "travel_time_min"]
    assert len(frame) == 4
    row = frame.set_index(["iz_code", "site_id"]).loc[("B", "S2")]
    assert pd.isna(row["travel_time_min"])
    assert frame.loc[(frame.iz_code == "A") & (frame.site_id == "S1"), "travel_time_min"].iloc[0] == 4.44


def test_validate_rejects_negative_minutes():
    frame = pd.DataFrame(
        {
            "iz_code": ["A"],
            "site_id": ["S1"],
            "mode": ["drive"],
            "travel_time_min": [-1.0],
        }
    )
    with pytest.raises(ModelError) as error:
        validate_travel_time_matrix(frame)
    assert error.value.code == "invalid_config"


def test_edinburgh_place_is_mapped_not_hardcoded_for_other_cities():
    origins = _origins()
    edinburgh = resolve_osm_query(LOCAL_AUTHORITY_CODE, osm_place=None, iz_origins=origins)
    assert edinburgh["method"] == "place"
    assert edinburgh["osm_place"] == "City of Edinburgh, UK"

    glasgow = resolve_osm_query("S12000049", osm_place=None, iz_origins=origins)
    assert glasgow["method"] == "polygon"
    assert glasgow["osm_place"] is None
    assert glasgow["polygon_wgs"] is not None

    explicit = resolve_osm_query("S12000049", osm_place="Glasgow, UK", iz_origins=origins)
    assert explicit["method"] == "place"
    assert explicit["osm_place"] == "Glasgow, UK"


def test_prepare_osm_writes_city_parameterised_matrix(tmp_path: Path):
    calls = []

    def fake_compute(**kwargs):
        calls.append(kwargs)
        origins = kwargs["iz_origins"]
        sites = kwargs["sites"]
        rows = []
        for iz in origins["iz_code"]:
            for site in sites["site_id"]:
                rows.append(
                    {
                        "iz_code": iz,
                        "site_id": site,
                        "mode": kwargs["mode_name"],
                        "travel_time_min": 3.0 if kwargs["mode_name"] == "drive" else 12.0,
                    }
                )
        return pd.DataFrame(rows)

    result = prepare_travel_time(
        area_code="S12000049",
        source="osm",
        osm_place="Glasgow, UK",
        iz_origins=_origins(),
        sites=_sites(),
        compute_mode_matrix_fn=fake_compute,
        output_dir=tmp_path,
    )
    assert result["area_code"] == "S12000049"
    assert result["osm_place"] == "Glasgow, UK"
    assert result["n_iz"] == 2
    assert result["n_sites"] == 2
    assert result["n_rows"] == 8
    matrix = pd.read_csv(result["output_path"])
    assert set(matrix["mode"]) == {"drive", "walk"}
    assert {item["query"]["osm_place"] for item in calls} == {"Glasgow, UK"}
    assert {item["speed_kmh"] for item in calls} == {DRIVE_SPEED_KMH, WALK_SPEED_KMH}


def test_unreachable_from_all_origins_is_detected():
    from data.travel_time import site_ids_unreachable_from_all_origins

    frame = pd.DataFrame(
        {
            "iz_code": ["A", "B", "A", "B"],
            "site_id": ["MS_30", "MS_30", "GP_1", "GP_1"],
            "mode": ["drive", "drive", "drive", "drive"],
            "travel_time_min": [pd.NA, pd.NA, 4.0, 5.0],
        }
    )
    assert site_ids_unreachable_from_all_origins(frame) == ["MS_30"]


def test_prepare_drops_site_unreachable_on_every_origin(tmp_path: Path):
    def fake_compute(**kwargs):
        rows = []
        for iz in kwargs["iz_origins"]["iz_code"]:
            for site in kwargs["sites"]["site_id"]:
                value = (
                    float("nan")
                    if site == "GP_1" and kwargs["mode_name"] == "drive"
                    else 8.0
                )
                rows.append(
                    {
                        "iz_code": iz,
                        "site_id": site,
                        "mode": kwargs["mode_name"],
                        "travel_time_min": value,
                    }
                )
        return pd.DataFrame(rows)

    result = prepare_travel_time(
        area_code="S12000049",
        source="osm",
        osm_place="Glasgow, UK",
        iz_origins=_origins(),
        sites=_sites(),
        compute_mode_matrix_fn=fake_compute,
        output_dir=tmp_path,
    )
    matrix = pd.read_csv(result["output_path"])
    assert result["dropped_site_ids"] == ["GP_1"]
    assert set(matrix["site_id"].astype(str)) == {"PH_1"}
    assert result["n_sites"] == 1
    assert result["n_rows"] == 4
    assert int(matrix["travel_time_min"].isna().sum()) == 0


def test_remove_site_ids_rewrites_csv_and_geojson(tmp_path: Path):
    from data.candidate_sites import remove_site_ids

    (tmp_path / "merged_candidate_sites.csv").write_text(
        "site_id,site_name,site_type\nMS_30,Airport Short Stay,mobile_stop\nGP_1,Clinic,gp\n"
    )
    (tmp_path / "car_parks_candidate_sites.csv").write_text(
        "site_id,site_name\nMS_30,Airport Short Stay\nMS_31,St Leonard's\n"
    )
    (tmp_path / "merged_candidate_sites.geojson").write_text(
        '{"type":"FeatureCollection","features":['
        '{"type":"Feature","properties":{"site_id":"MS_30"},"geometry":null},'
        '{"type":"Feature","properties":{"site_id":"GP_1"},"geometry":null}]}'
    )
    write_json = {"n_mobile": 2, "n_merged": 2}
    import json

    (tmp_path / "provenance.json").write_text(json.dumps(write_json))
    result = remove_site_ids(["MS_30"], tmp_path, reason="test drop")
    merged = pd.read_csv(tmp_path / "merged_candidate_sites.csv")
    parks = pd.read_csv(tmp_path / "car_parks_candidate_sites.csv")
    geo = json.loads((tmp_path / "merged_candidate_sites.geojson").read_text())
    prov = json.loads((tmp_path / "provenance.json").read_text())
    assert list(merged["site_id"]) == ["GP_1"]
    assert list(parks["site_id"]) == ["MS_31"]
    assert [f["properties"]["site_id"] for f in geo["features"]] == ["GP_1"]
    assert prov["dropped_site_ids"] == ["MS_30"]
    assert prov["n_merged"] == 1
    assert prov["n_mobile"] == 1
    assert result["n_remaining"] == 1


def test_prepare_osm_does_not_invent_sites(tmp_path: Path):
    missing = tmp_path / "missing_sites.csv"
    with pytest.raises(ModelError) as error:
        prepare_travel_time(
            area_code=LOCAL_AUTHORITY_CODE,
            source="osm",
            iz_origins=_origins(),
            sites_path=missing,
            output_dir=tmp_path,
        )
    assert error.value.code == "missing_dataset"


def test_load_local_matrix(tmp_path: Path, monkeypatch):
    frame = pd.DataFrame(
        {
            "iz_code": ["A", "A"],
            "site_id": ["S1", "S1"],
            "mode": ["drive", "walk"],
            "travel_time_min": [5.0, 20.0],
        }
    )
    path = tmp_path / "travel_time_matrix.csv"
    frame.to_csv(path, index=False)
    monkeypatch.setattr(
        "data.travel_time.find_travel_time_file",
        lambda area_code="S12000036", path=None: Path(path) if path else Path(
            tmp_path / "travel_time_matrix.csv"
        ),
    )
    loaded = load_travel_time(path=path)
    assert len(loaded) == 2
    result = prepare_travel_time(area_code="S12000036", source="local")
    assert result["source"] == "local"
    assert result["n_rows"] == 2


@pytest.mark.external_data
def test_load_iz_origins_filters_edinburgh_2011_centroids():
    origins = load_iz_origins(area_code=LOCAL_AUTHORITY_CODE)
    assert len(origins) == 111
    assert origins["iz_code"].is_unique
    assert origins.crs.to_epsg() == 27700


def test_load_candidate_sites_from_csv(tmp_path: Path):
    path = tmp_path / "merged_candidate_sites.csv"
    pd.DataFrame(
        {
            "site_id": ["GP_1", "PH_1"],
            "easting": [325000, 326000],
            "northing": [673000, 674000],
            "site_type": ["gp", "pharmacy"],
        }
    ).to_csv(path, index=False)
    sites = load_candidate_sites(path=path)
    assert list(sites["site_id"]) == ["GP_1", "PH_1"]
    assert len(sites) == 2
