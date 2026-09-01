"""oasis-v4 GP/pharmacy extraction with CA-based city filter."""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

from common.errors import ModelError
from common.utils import LOCAL_AUTHORITY_CODE
from data.healthcare import (
    acquire_healthcare_table,
    load_code_lookup,
    match_osm_buildings,
    prepare_gp_sites,
    prepare_pharmacy_sites,
)


def _edin_and_other_datazones():
    lookup = load_code_lookup()
    edin = lookup.loc[lookup["CA"] == LOCAL_AUTHORITY_CODE, "DataZone"].iloc[0]
    other = lookup.loc[lookup["CA"] != LOCAL_AUTHORITY_CODE, "DataZone"].iloc[0]
    return str(edin), str(other)


@pytest.mark.external_data
def test_gp_sites_keep_only_requested_ca():
    edin_dz, other_dz = _edin_and_other_datazones()
    table = pd.DataFrame(
        {
            "PracticeCode": ["111", "222"],
            "GPPracticeName": ["Leith Medical Practice", "Aberdeen Medical Practice"],
            "Postcode": ["EH6 6AA", "AB10 1AA"],
            "DataZone": [edin_dz, other_dz],
        }
    )
    sites = prepare_gp_sites(
        area_code=LOCAL_AUTHORITY_CODE,
        gp_table=table,
        use_osm=False,
        geocode_fn=lambda _pc: (325000.0, 673000.0),
    )
    assert len(sites) == 1
    assert sites.iloc[0]["site_id"] == "GP_111"
    assert sites.iloc[0]["site_type"] == "gp"
    assert sites.iloc[0]["coord_source"] == "Postcode_centroid"


@pytest.mark.external_data
def test_pharmacy_sites_keep_only_requested_ca():
    edin_dz, other_dz = _edin_and_other_datazones()
    table = pd.DataFrame(
        {
            "DispCode": ["P1", "P2"],
            "DispLocationName": ["Leith Pharmacy", "Aberdeen Pharmacy"],
            "DispLocationPostcode": ["EH6 6AA", "AB10 1AA"],
            "datazone2011": [edin_dz, other_dz],
        }
    )
    sites = prepare_pharmacy_sites(
        area_code=LOCAL_AUTHORITY_CODE,
        pharmacy_table=table,
        use_osm=False,
        geocode_fn=lambda _pc: (325100.0, 673100.0),
    )
    assert len(sites) == 1
    assert sites.iloc[0]["site_id"] == "PH_P1"
    assert sites.iloc[0]["site_type"] == "pharmacy"


def test_osm_name_match_uses_building_when_similar():
    osm = gpd.GeoDataFrame(
        {"name": ["Leith Medical Practice"]},
        geometry=[Point(325050, 673050)],
        crs="EPSG:27700",
    )
    eastings, northings, sources = match_osm_buildings(
        names=["Leith Medical Centre"],
        pc_easting=[325000],
        pc_northing=[673000],
        osm_points=osm,
        drop_tokens=("practice", "medical", "centre", "center", "surgery"),
    )
    assert sources == ["OSM_exact_building"]
    assert eastings == [325050.0]
    assert northings == [673050.0]


def test_acquire_gp_from_ckan_does_not_use_local():
    def fake_ckan(_url, params):
        assert params["resource_id"] == "993422a6-c64f-4c57-ba41-9279ad5a7c89"
        return {
            "success": True,
            "result": {
                "total": 1,
                "records": [
                    {
                        "PracticeCode": 10002,
                        "GPPracticeName": "Muirhead Medical Centre",
                        "Postcode": "DD2 5NH",
                        "DataZone": "S01007129",
                        "HSCP": "S37000007",
                    }
                ],
            },
        }

    result = acquire_healthcare_table("gp", source="api", ckan_get=fake_ckan)
    assert result["provenance"]["retrieval"] == "ckan_datastore_search"
    assert len(result["frame"]) == 1
    assert int(result["frame"].iloc[0]["PracticeCode"]) == 10002


def test_acquire_api_failure_does_not_fall_back_to_local():
    def fake_ckan(_url, _params):
        raise RuntimeError("network down")

    def fake_csv(_url):
        raise RuntimeError("csv also down")

    with pytest.raises(ModelError) as error:
        acquire_healthcare_table("pharmacy", source="api", ckan_get=fake_ckan, csv_get=fake_csv)
    assert error.value.code == "missing_dataset"
    assert "Local files were not used" in str(error.value)


def test_acquire_pharmacy_uses_official_csv_when_datastore_resets():
    def fake_ckan(_url, _params):
        raise ConnectionResetError("Connection reset by peer")

    def fake_csv(url):
        assert "f44e6a10-4f1f-4ffd-9205-956944bacf95" in url
        return pd.DataFrame(
            [
                {
                    "DispCode": 1,
                    "DispLocationName": "Test Pharmacy",
                    "Postcode": "EH1 1AA",
                    "DataZone": "S01008417",
                }
            ]
        )

    result = acquire_healthcare_table("pharmacy", source="api", ckan_get=fake_ckan, csv_get=fake_csv)
    assert result["provenance"]["retrieval"] == "ckan_csv_download"
    assert len(result["frame"]) == 1
    assert int(result["frame"].iloc[0]["DispCode"]) == 1
