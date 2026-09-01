from pathlib import Path

import pandas as pd
import pytest
from shapely.geometry import Polygon

from allocation.contracts import N_SITES
from common.errors import ModelError
from presentation.plot_forecast_sites import CityMapSpec, plot_forecast_sites


def _toy_city(tmp_path: Path, key: str, label: str, n_iz: int, origin: float) -> CityMapSpec:
    import geopandas as gpd

    iz_codes = [f"S020{key[:3].upper()}{i:02d}" for i in range(n_iz)]
    polys = []
    for i in range(n_iz):
        x0 = origin + i * 1200
        polys.append(Polygon([(x0, origin), (x0 + 1100, origin), (x0 + 1100, origin + 1100), (x0, origin + 1100)]))
    gdf = gpd.GeoDataFrame({"iz_code": iz_codes, "geometry": polys}, crs="EPSG:27700")
    bounds = tmp_path / f"{key}_iz.geojson"
    gdf.to_file(bounds, driver="GeoJSON")

    forecast = pd.DataFrame(
        {
            "iz_code": iz_codes,
            "target_report_date": "2023-03-04",
            "predicted_rate": [20.0 + i for i in range(n_iz)],
        }
    )
    forecast_path = tmp_path / f"{key}_forecast.csv"
    forecast.to_csv(forecast_path, index=False)

    site_ids = [f"{key}_S{i}" for i in range(N_SITES)]
    types = ["gp", "pharmacy", "mobile_stop", "gp", "pharmacy", "pharmacy"]
    selected = pd.DataFrame(
        {
            "site_id": site_ids,
            "site_name": [f"{label} site {i+1}" for i in range(N_SITES)],
            "site_type": types,
        }
    )
    selected_path = tmp_path / f"{key}_selected.csv"
    selected.to_csv(selected_path, index=False)

    candidates = pd.DataFrame(
        {
            "site_id": site_ids + [f"{key}_extra"],
            "site_name": [f"{label} site {i+1}" for i in range(N_SITES)] + ["unused"],
            "site_type": types + ["gp"],
            "easting": [origin + 200 + i * 150 for i in range(N_SITES)] + [origin],
            "northing": [origin + 400 for _ in range(N_SITES)] + [origin],
        }
    )
    cand_path = tmp_path / f"{key}_candidates.csv"
    candidates.to_csv(cand_path, index=False)

    return CityMapSpec(
        key=key,
        label=label,
        n_iz=n_iz,
        forecast=forecast_path,
        boundaries=bounds,
        selected_sites=selected_path,
        candidates=cand_path,
    )


def test_plot_forecast_sites_writes_pdfs_not_under_article(tmp_path: Path):
    pytest.importorskip("geopandas")
    edi = _toy_city(tmp_path, "edinburgh", "Edinburgh", 3, 320000.0)
    gla = _toy_city(tmp_path, "glasgow", "Glasgow", 4, 250000.0)
    out = tmp_path / "figures"
    written = plot_forecast_sites(specs=(edi, gla), out_dirs=(out,), root=tmp_path)
    pdfs = [Path(path) for path in written if path.endswith(".pdf")]
    stems = {path.name for path in pdfs}
    assert "fig_map_forecast_sites_edinburgh.pdf" in stems
    assert "fig_map_forecast_sites_glasgow.pdf" in stems
    assert "fig_map_forecast_sites_both.pdf" in stems
    for path in pdfs:
        assert path.is_file()
        assert path.stat().st_size > 1000
        assert "website_article_v1" not in str(path)
    sites = pd.read_csv(out / "fig_map_forecast_sites_selected.csv")
    assert len(sites) == 12
    assert set(sites["map_index"].unique()) == {1, 2, 3, 4, 5, 6}


def test_plot_forecast_sites_rejects_wrong_site_count(tmp_path: Path):
    pytest.importorskip("geopandas")
    spec = _toy_city(tmp_path, "edinburgh", "Edinburgh", 3, 320000.0)
    selected = pd.read_csv(spec.selected_sites)
    selected.iloc[:5].to_csv(spec.selected_sites, index=False)
    with pytest.raises(ModelError, match="exactly 6"):
        plot_forecast_sites(specs=(spec,), out_dirs=(tmp_path / "out",), root=tmp_path)
