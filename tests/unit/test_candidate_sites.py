"""oasis-v4 candidate-site merge and car-park filters."""

from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

from common.errors import ModelError
from common.utils import LOCAL_AUTHORITY_CODE
from data.candidate_sites import (
    filter_public_car_parks,
    find_candidate_site_file,
    merge_candidate_sites,
    prepare_candidate_sites,
)
from data.osm_extract import geofabrik_ready, load_geofabrik_parking, load_geofabrik_pois
from data.travel_time import export_iz_origins


def test_car_park_filter_drops_private_and_underground():
    gdf = gpd.GeoDataFrame(
        {
            "name": ["Open Park", "Private Park", "Big Park"],
            "access": ["yes", "private", "yes"],
            "parking": ["surface", "surface", "underground"],
            "capacity": [30, 40, 80],
        },
        geometry=[Point(325000, 673000), Point(325100, 673100), Point(325200, 673200)],
        crs="EPSG:27700",
    )
    kept = filter_public_car_parks(gdf)
    assert list(kept["name"]) == ["Open Park"]


def test_car_park_filter_rejects_geofabrik_without_tags():
    gdf = gpd.GeoDataFrame(
        {"name": ["Some Park"], "fclass": ["parking"]},
        geometry=[Point(325000, 673000)],
        crs="EPSG:27700",
    )
    with pytest.raises(ModelError) as error:
        filter_public_car_parks(gdf)
    assert error.value.code == "invalid_config"
    assert "access" in str(error.value)


def test_car_park_filter_drops_small_unnamed_lots():
    gdf = gpd.GeoDataFrame(
        {
            "name": ["Named Park", ""],
            "access": ["yes", "yes"],
            "parking": ["surface", "surface"],
            "capacity": [5, 5],
        },
        geometry=[Point(325000, 673000), Point(325100, 673100)],
        crs="EPSG:27700",
    )
    kept = filter_public_car_parks(gdf)
    assert list(kept["name"]) == ["Named Park"]


def test_car_park_filter_drops_airport_lots_but_keeps_park_and_ride():
    gdf = gpd.GeoDataFrame(
        {
            "name": ["Airport Short Stay", "Ingliston Park and Ride", "Castle Terrace"],
            "access": ["yes", "yes", "yes"],
            "parking": ["multi-storey", "surface", "surface"],
            "capacity": [800, 5, 40],
            "operator": ["Edinburgh Airport", "", "CEC"],
            "park_and_ride": ["", "yes", ""],
        },
        geometry=[Point(315000, 673500), Point(315370, 672616), Point(324950, 673418)],
        crs="EPSG:27700",
    )
    kept = filter_public_car_parks(gdf)
    assert "Airport Short Stay" not in list(kept["name"])
    assert set(kept["name"]) == {"Ingliston Park and Ride", "Castle Terrace"}


def test_overpass_parser_keeps_access_and_capacity_tags():
    from data.osm_extract import overpass_elements_to_parking_gdf

    payload = {
        "elements": [
            {
                "type": "way",
                "id": 1,
                "center": {"lat": 55.95, "lon": -3.19},
                "tags": {
                    "amenity": "parking",
                    "name": "Castle Terrace",
                    "access": "yes",
                    "parking": "surface",
                    "capacity": "120",
                },
            }
        ]
    }
    gdf = overpass_elements_to_parking_gdf(payload)
    assert gdf.iloc[0]["access"] == "yes"
    assert gdf.iloc[0]["capacity"] == "120"
    assert "parking" in gdf.columns


def test_merge_uses_full_column_names_not_shapefile_truncation():
    gp = pd.DataFrame(
        {
            "site_id": ["GP_1"],
            "site_name": ["Clinic"],
            "site_type": ["gp"],
            "easting": [325000.0],
            "northing": [673000.0],
            "iz_code": ["S02001576"],
            "iz_easting": [325010.0],
            "iz_northing": [673010.0],
            "suitability": ["confirmed"],
            "coord_source": ["Postcode_centroid"],
        }
    )
    pharm = gp.copy()
    pharm["site_id"] = "PH_1"
    pharm["site_type"] = "pharmacy"
    merged = merge_candidate_sites(gp, pharm)
    assert "iz_northing" in merged.columns
    assert "suitability" in merged.columns
    assert "coord_source" in merged.columns
    assert "iz_northin" not in merged.columns
    assert len(merged) == 2


def test_find_candidate_site_file_prefers_merged_in_explicit_folder(tmp_path: Path):
    (tmp_path / "gp_candidate_sites.csv").write_text("site_id\nGP_1\n")
    merged = tmp_path / "merged_candidate_sites.csv"
    merged.write_text("site_id\nGP_1\n")
    assert find_candidate_site_file(tmp_path) == merged


def test_prepare_candidate_sites_writes_merged_table(tmp_path: Path):
    gp = pd.DataFrame(
        {
            "site_id": ["GP_1"],
            "site_name": ["Clinic"],
            "site_type": ["gp"],
            "easting": [325000.0],
            "northing": [673000.0],
            "iz_code": ["S02001576"],
            "iz_easting": [325010.0],
            "iz_northing": [673010.0],
            "suitability": ["confirmed"],
            "coord_source": ["Postcode_centroid"],
        }
    )
    pharm = gp.copy()
    pharm["site_id"] = "PH_1"
    pharm["site_type"] = "pharmacy"
    result = prepare_candidate_sites(
        area_code="S12000049",
        source="osm",
        gp_sites=gp,
        pharmacy_sites=pharm,
        include_mobile=False,
        output_dir=tmp_path,
    )
    assert result["area_code"] == "S12000049"
    assert result["n_sites"] == 2
    merged = pd.read_csv(result["output_path"])
    assert set(merged["site_type"]) == {"gp", "pharmacy"}


@pytest.mark.external_data
def test_export_iz_origins_is_2011_ca_filter_not_lad2024(tmp_path: Path):
    result = export_iz_origins(area_code=LOCAL_AUTHORITY_CODE, output_dir=tmp_path)
    assert result["n_iz"] == 111
    assert result["not_2024_lad_clip"] is True
    table = pd.read_csv(result["output_path"])
    assert list(table.columns) == ["iz_code", "iz_easting", "iz_northing"]
    assert len(table) == 111


@pytest.mark.external_data
def test_geofabrik_scotland_extract_has_doctors_and_pharmacies():
    assert geofabrik_ready()
    doctors = load_geofabrik_pois({"doctors"})
    pharmacies = load_geofabrik_pois({"pharmacy"})
    assert len(doctors) > 0
    assert len(pharmacies) > 0
    assert doctors.crs.to_epsg() == 27700
    with pytest.raises(ModelError) as error:
        load_geofabrik_parking()
    assert "Overpass" in str(error.value)
