"""New-region prepare + train. Does not overwrite Edinburgh U10 or rolling_v1."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from allocation.contracts import EDINBURGH_CA
from common.errors import ModelError
from common.utils import ALLOWED_YEARS, LOCAL_AUTHORITY_NAME, get_logger, project_root
from graph.mobility import default_od_path

LOGGER = get_logger("agent.region_training")

GLASGOW_CA = "S12000049"
AREA_NAMES = {
    EDINBURGH_CA: LOCAL_AUTHORITY_NAME,
    GLASGOW_CA: "Glasgow City",
}


def region_output_dir(area_code: str) -> Path:
    return project_root() / "data" / "results" / "regions" / str(area_code).strip()


def region_artefacts_ready(area_code: str) -> bool:
    """True when this city already has a trained checkpoint and allocation forecast."""
    code = str(area_code or "").strip()
    if not code or code in {EDINBURGH_CA, "UNKNOWN"}:
        return False
    out = region_output_dir(code)
    ckpt = out / "model" / "geo_transport_mobility" / "checkpoint.pt"
    forecast = out / "forecast_for_allocation.csv"
    return ckpt.exists() and ckpt.stat().st_size > 0 and forecast.exists()


def region_boundaries_path(area_code: str) -> Path:
    return region_output_dir(area_code) / "planning" / "iz_boundaries.geojson"


def write_region_iz_boundaries(area_code: str, *, overwrite: bool = False) -> Path:
    """City IZ polygons for the dashboard. Does not write website_article_v1."""
    import geopandas as gpd
    from data.covid import load_iz_master

    code = str(area_code or "").strip()
    dest = region_boundaries_path(code)
    _assert_not_protected(dest)
    if dest.is_file() and dest.stat().st_size > 0 and not overwrite:
        return dest
    master = load_iz_master(area_code=code)
    shp = (
        project_root()
        / "data"
        / "raw"
        / "boundaries"
        / "SG_IntermediateZoneBdry_2011"
        / "SG_IntermediateZone_Bdry_2011.shp"
    )
    polygons = gpd.read_file(shp)
    code_col = next(name for name in ("InterZone", "IntZone", "iz_code") if name in polygons.columns)
    polygons["iz_code"] = polygons[code_col].astype(str)
    keep = set(master["IntZone"].astype(str))
    polygons = polygons.loc[polygons["iz_code"].isin(keep)].copy()
    name_col = next((name for name in ("Name", "IntZoneName", "IZ_NAME") if name in polygons.columns), None)
    if name_col:
        polygons["iz_name"] = polygons[name_col].astype(str).str.strip()
    else:
        polygons["iz_name"] = polygons["iz_code"]
    if polygons.crs is None:
        raise ModelError("IZ boundary shapefile has no CRS.", code="missing_crs")
    projected = polygons.to_crs(epsg=27700)
    projected["geometry"] = projected.geometry.simplify(80, preserve_topology=True)
    web = projected.to_crs(epsg=4326)
    dest.parent.mkdir(parents=True, exist_ok=True)
    web[["iz_code", "iz_name", "geometry"]].to_file(dest, driver="GeoJSON")
    LOGGER.info("Wrote %s (%s IZs)", dest, len(web))
    return dest


def _protected_paths() -> tuple[Path, ...]:
    root = project_root()
    return (
        root / "data" / "results" / "fill1.csv",
        root / "data" / "results" / "simd_iz.csv",
        root / "data" / "results" / "model" / "rolling_v1_split65_10_25",
        root / "data" / "results" / "exports" / "website_article_v1",
        root / "data" / "raw",
    )


def _assert_not_protected(path: Path) -> None:
    resolved = path.resolve()
    for banned in _protected_paths():
        banned_res = banned.resolve()
        if resolved == banned_res or banned_res in resolved.parents:
            raise ModelError(
                f"New-region training refused to write {path}; that path is reserved for Edinburgh artefacts.",
                code="overwrite_forbidden",
            )


def run_new_region_training(request: dict[str, Any]) -> dict[str, Any]:
    """Prepare city-specific tables, graphs, then train a new checkpoint.

    Requires confirm_new_region_training. Edinburgh stays on U10.
    """
    area_code = str(request.get("area_code") or "").strip()
    area_name = str(request.get("area_name") or AREA_NAMES.get(area_code, area_code))
    if not area_code or area_code in {"UNKNOWN", EDINBURGH_CA}:
        return {
            "status": "error",
            "mode": "new_region_training",
            "executed": False,
            "message": "This training step is only for a new city, not Edinburgh.",
        }

    # A packaged city checkpoint must be reusable without any restricted raw
    # inputs. Check this before resolving OD, COVID, SIMD, or road sources.
    out = region_output_dir(area_code)
    model_dir = out / "model"
    existing_ckpt = model_dir / "geo_transport_mobility" / "checkpoint.pt"
    existing_forecast = out / "forecast_for_allocation.csv"
    existing_geoshapley = model_dir / "geo_transport_mobility" / "exports" / "geoshapley.csv"
    if (
        existing_ckpt.is_file()
        and existing_ckpt.stat().st_size > 0
        and existing_forecast.is_file()
        and existing_forecast.stat().st_size > 0
        and not request.get("force_retrain")
    ):
        import pandas as pd

        forecast = pd.read_csv(existing_forecast, usecols=["iz_code"])
        n_iz = int(forecast["iz_code"].astype(str).nunique())
        return {
            "status": "ok",
            "mode": "new_region_training",
            "executed": False,
            "retrained": False,
            "area_code": area_code,
            "area_name": area_name,
            "n_iz": n_iz,
            "od_path": None,
            "output_dir": str(out),
            "checkpoint_id": f"{area_code}-trained",
            "checkpoint_path": str(existing_ckpt),
            "forecast_path": str(existing_forecast),
            "geoshapley_path": str(existing_geoshapley) if existing_geoshapley.is_file() else None,
            "config_path": str(out / "model.yaml") if (out / "model.yaml").is_file() else None,
            "steps": [{"step": "load_frozen_region", "reused": True}],
            "message": f"Reused the saved model for {area_name}.",
        }

    try:
        od_path = default_od_path(area_code)
    except Exception as exc:
        return {
            "status": "error",
            "mode": "new_region_training",
            "executed": False,
            "message": str(exc),
        }

    _assert_not_protected(out)
    out.mkdir(parents=True, exist_ok=True)
    covid_dir = out / "covid"
    graph_dir = out / "graph"
    forecast_dir = out / "forecast"
    model_dir = out / "model"
    steps: list[dict[str, Any]] = []

    try:
        from data.covid import (
            acquire_data,
            attach_iz_centroids,
            load_iz_master,
            preprocess_covid,
        )
        from data.dataset import prepare_forecast_dataset
        from data.deprivation import aggregate_simd_to_iz
        from data.candidate_sites import candidate_sites_results_dir, prepare_candidate_sites
        from data.travel_time import prepare_travel_time, travel_time_results_dir
        from graph.geo import construct_adjacency_graph
        from graph.road import construct_road_graph
        from graph.mobility import construct_mobility_graph
        from agent.tools import train_model
        from model.config import load_model_config

        LOGGER.info("New-region training for %s (%s). OD=%s", area_name, area_code, od_path)
        iz_master = load_iz_master(area_code=area_code)
        n_iz = int(len(iz_master))
        fill1 = covid_dir / "fill1.csv"
        if fill1.exists() and fill1.stat().st_size > 0:
            steps.append({"step": "preprocess_covid", "reused": str(fill1), "n_iz": n_iz})
        else:
            acquired = acquire_data(
                years=list(ALLOWED_YEARS),
                area_code=area_code,
                source=str(request.get("data_source") or "api"),
                output_dir=covid_dir,
            )
            steps.append({"step": "acquire_covid", "n_frames": len(acquired.get("frames") or [])})
            prepared = preprocess_covid(
                frames=acquired["frames"],
                years=list(ALLOWED_YEARS),
                iz_master=iz_master,
                area_code=area_code,
                area_name=area_name,
                output_dir=covid_dir,
            )
            fill1 = Path(prepared["output_paths"]["fill1"])
            steps.append({"step": "preprocess_covid", "fill1": str(fill1), "n_iz": n_iz})

        geo_path = covid_dir / "fill1_geo.csv"
        if not geo_path.exists():
            geo_covid = attach_iz_centroids(
                covid_path=fill1,
                area_code=area_code,
                output_path=geo_path,
            )
            steps.append({"step": "centroids", "path": geo_covid.get("output_path"), "n_iz": geo_covid.get("n_iz")})

        simd_path = out / "simd_iz.csv"
        if simd_path.exists():
            steps.append({"step": "simd", "reused": str(simd_path)})
        else:
            simd = aggregate_simd_to_iz(area_code=area_code, output_path=simd_path)
            simd_path = Path(simd["output_path"])

        if (forecast_dir / "node_order.csv").exists():
            steps.append({"step": "forecast_dataset", "reused": str(forecast_dir)})
        else:
            prepare_forecast_dataset(
                area_code=area_code,
                covid_path=fill1,
                iz_master=iz_master,
                simd_path=simd_path,
                output_dir=forecast_dir,
                overwrite=True,
            )
            steps.append({"step": "forecast_dataset", "output_dir": str(forecast_dir)})

        geo_npz = graph_dir / "geo" / "adjacency_geo.npz"
        road_npz = graph_dir / "road" / "adjacency_road.npz"
        mobility_npz = graph_dir / "mobility" / "adjacency_mobility.npz"
        rebuild_graphs = bool(request.get("rebuild_graphs") or request.get("force_retrain"))
        if geo_npz.exists() and road_npz.exists() and mobility_npz.exists() and not rebuild_graphs:
            steps.append({"step": "graphs", "reused": True, "n_iz": n_iz})
        else:
            geo = construct_adjacency_graph(
                area_code=area_code,
                output_dir=graph_dir / "geo",
                overwrite=True,
            )
            road = construct_road_graph(
                area_code=area_code,
                output_dir=graph_dir / "road",
                overwrite=True,
            )
            mobility = construct_mobility_graph(
                area_code=area_code,
                od=od_path,
                output_dir=graph_dir / "mobility",
                overwrite=True,
            )
            steps.append(
                {
                    "step": "graphs",
                    "geo_nodes": geo.get("n_nodes"),
                    "road_nodes": road.get("n_nodes"),
                    "mobility_nodes": mobility.get("n_nodes"),
                    "rebuilt": True,
                }
            )
        write_region_iz_boundaries(area_code, overwrite=rebuild_graphs)

        sites_csv = candidate_sites_results_dir(area_code) / "merged_candidate_sites.csv"
        if sites_csv.exists():
            import pandas as pd

            n_sites = int(len(pd.read_csv(sites_csv)))
            steps.append({"step": "sites", "reused": str(sites_csv), "n_sites": n_sites})
            sites = {"n_sites": n_sites}
        else:
            sites = prepare_candidate_sites(area_code=area_code, source="api")
        travel_csv = travel_time_results_dir(area_code) / "travel_time_matrix.csv"
        if travel_csv.exists() and travel_csv.stat().st_size > 0:
            travel = {"status": "ok", "output_path": str(travel_csv), "reused": True}
            steps.append({"step": "travel", "reused": str(travel_csv)})
        else:
            travel = prepare_travel_time(area_code=area_code, source="osm")
            steps.append(
                {
                    "step": "sites_travel",
                    "n_sites": sites.get("n_sites") or sites.get("n_rows"),
                    "travel": travel.get("output_path") or travel.get("status"),
                }
            )

        cfg = load_model_config()
        cfg["expected_edinburgh_iz_count"] = n_iz
        cfg["rolling_evaluation"]["enabled"] = False
        cfg["rolling_evaluation"]["output_dir"] = str(out / "rolling")
        cfg["rolling_evaluation"]["fixed_s1_dir"] = str(model_dir)
        cfg["paths"]["dataset"] = str(forecast_dir)
        cfg["paths"]["geo"] = str(graph_dir / "geo" / "adjacency_geo.npz")
        cfg["paths"]["transport"] = str(graph_dir / "road" / "adjacency_road.npz")
        cfg["paths"]["mobility"] = str(graph_dir / "mobility" / "adjacency_mobility.npz")
        cfg["paths"]["geo_report"] = str(graph_dir / "geo" / "validation_report.json")
        cfg["paths"]["transport_report"] = str(graph_dir / "road" / "validation_report.json")
        cfg["paths"]["mobility_report"] = str(graph_dir / "mobility" / "validation_report.json")
        cfg["paths"]["road_nodes"] = str(graph_dir / "road" / "nodes.csv")
        cfg["paths"]["checkpoints"] = str(model_dir)
        cfg_path = out / "model.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

        existing_ckpt = model_dir / "geo_transport_mobility" / "checkpoint.pt"
        existing_forecast = out / "forecast_for_allocation.csv"
        existing_geoshapley = model_dir / "geo_transport_mobility" / "exports" / "geoshapley.csv"
        if (
            existing_ckpt.exists()
            and existing_ckpt.stat().st_size > 0
            and existing_forecast.exists()
            and not request.get("force_retrain")
        ):
            steps.append({"step": "train", "reused": str(existing_ckpt)})
            return {
                "status": "ok",
                "mode": "new_region_training",
                "executed": True,
                "retrained": False,
                "area_code": area_code,
                "area_name": area_name,
                "n_iz": n_iz,
                "od_path": str(od_path),
                "output_dir": str(out),
                "checkpoint_id": f"{area_code}-trained",
                "checkpoint_path": str(existing_ckpt),
                "forecast_path": str(existing_forecast),
                "geoshapley_path": str(existing_geoshapley) if existing_geoshapley.exists() else None,
                "config_path": str(cfg_path),
                "steps": steps,
                "message": (
                    f"Reused the saved model for {area_name}."
                ),
            }

        trained = train_model(config_path_override=cfg_path)
        if trained.get("status") == "failed":
            raise ModelError(
                trained.get("message")
                or str((trained.get("warnings") or [{}])[0].get("message") or "Could not build the model."),
                code="training_failed",
            )
        outputs = trained.get("outputs") or trained
        checkpoint_path = outputs.get("checkpoint_path")
        steps.append({"step": "train", "checkpoint": checkpoint_path})

        export_paths = outputs.get("export_paths") or {}
        raw_forecast = export_paths.get("forecast_map")
        geoshapley_path = export_paths.get("geoshapley")
        if not raw_forecast:
            raise ModelError(
                "Training finished but no forecast table was written.",
                code="missing_forecast",
            )
        forecast_path = _write_allocation_forecast(
            raw_forecast,
            out / "forecast_for_allocation.csv",
            forecast_date=str(request.get("forecast_date") or ""),
        )
        steps.append(
            {
                "step": "allocation_forecast",
                "forecast_path": str(forecast_path),
                "source": raw_forecast,
            }
        )

        return {
            "status": "ok",
            "mode": "new_region_training",
            "executed": True,
            "retrained": True,
            "area_code": area_code,
            "area_name": area_name,
            "n_iz": n_iz,
            "od_path": str(od_path),
            "output_dir": str(out),
            "checkpoint_id": f"{area_code}-trained",
            "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
            "forecast_path": str(forecast_path),
            "geoshapley_path": str(geoshapley_path) if geoshapley_path else None,
            "config_path": str(cfg_path),
            "steps": steps,
            "message": (
                f"Built a model for {area_name} only. Edinburgh's files were not changed."
            ),
        }
    except Exception as exc:
        LOGGER.exception("New-region training failed for %s", area_code)
        return {
            "status": "error",
            "mode": "new_region_training",
            "executed": True,
            "retrained": False,
            "area_code": area_code,
            "output_dir": str(out),
            "steps": steps,
            "message": str(exc),
        }


def _write_allocation_forecast(source: str | Path | Any, dest: Path, forecast_date: str = "") -> Path:
    """One row per IZ with predicted_rate. Does not invent missing IZs."""
    import pandas as pd

    frame = source.copy() if isinstance(source, pd.DataFrame) else pd.read_csv(source)
    if "target_report_date" in frame.columns:
        dates = frame["target_report_date"].astype(str)
        if forecast_date and forecast_date in set(dates):
            frame = frame.loc[dates == forecast_date].copy()
        else:
            last = dates.max()
            frame = frame.loc[dates == last].copy()
    if "predicted_rate" not in frame.columns:
        if "predicted_mu_original" in frame.columns:
            frame["predicted_rate"] = frame["predicted_mu_original"]
        elif "predicted_mean" in frame.columns:
            frame["predicted_rate"] = frame["predicted_mean"]
        elif "display_mean" in frame.columns:
            frame["predicted_rate"] = frame["display_mean"]
    if "predicted_sigma" not in frame.columns:
        if "predicted_sigma_original" in frame.columns:
            frame["predicted_sigma"] = frame["predicted_sigma_original"]
    if "iz_code" not in frame.columns or "predicted_rate" not in frame.columns:
        raise ModelError(
            "Train export must contain iz_code and a predicted rate column.",
            code="missing_forecast",
        )
    frame = frame.drop_duplicates("iz_code", keep="last")
    dest.parent.mkdir(parents=True, exist_ok=True)
    _assert_not_protected(dest)
    frame.to_csv(dest, index=False)
    return dest


def write_region_unlabelled_forecast(
    *,
    config_path: Path,
    checkpoint_path: Path,
    calibration_path: Path,
    panel_path: Path,
    forecast_dest: Path,
    geoshapley_dest: Path,
    forecast_date: str,
) -> dict[str, str]:
    """t+7 forecast and GeoShapley after the last panel date. Does not use Edinburgh U10."""
    import numpy as np
    import pandas as pd

    from agent.tools import _explain_all_iz_for_samples, _predict_residual, _restore_runtime
    from model.config import geoshapley_settings, load_model_config, temporal_target
    from model.evaluate import assert_artefact_matches_checkpoint
    from model.operational import build_operational_split
    from model.residual import apply_residual_scalers_to_split
    from presentation.tables import build_forecast_table

    for path in (forecast_dest, geoshapley_dest):
        _assert_not_protected(path)
    cfg = load_model_config(config_path)
    tt = temporal_target(cfg)
    target = pd.Timestamp(forecast_date).normalize()
    issue = target - pd.Timedelta(days=int(tt["target_offset_days"]))
    artefact = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
    assert_artefact_matches_checkpoint(artefact, Path(checkpoint_path))
    runtime = _restore_runtime(cfg, Path(checkpoint_path))
    dataset = runtime["dataset"]
    split = build_operational_split(
        Path(panel_path),
        dataset.node_order,
        issue_date=issue,
        lookback_days=int(tt["lookback_steps"]),
        target_offset_days=int(tt["target_offset_days"]),
    )
    split = apply_residual_scalers_to_split(split, dataset.residual_scalers)
    preds = _predict_residual(runtime, split)
    issue_str = str(issue.date())
    target_str = str(target.date())
    input_start = str((issue - pd.Timedelta(days=int(tt["lookback_steps"]) - 1)).date())
    table = build_forecast_table(
        node_order=dataset.node_order,
        issue_dates=[issue_str],
        input_start_dates=[input_start],
        target_dates=[target_str],
        target_offset_days=int(tt["target_offset_days"]),
        mu_z=preds["mu_delta_z"],
        variance_z=preds["variance_delta_z"],
        sigma_z=preds["sigma_delta_z"],
        mu=preds["mu"],
        variance=preds["variance"],
        sigma=preds["sigma"],
        artefact=artefact,
        checkpoint_id=str(checkpoint_path),
        calibration_artefact_id=str(calibration_path),
        y_anchor=preds["y_anchor"],
        mu_delta=preds["mu_delta"],
    )
    forecast_dest.parent.mkdir(parents=True, exist_ok=True)
    _write_allocation_forecast(table, forecast_dest, forecast_date=target_str)
    geo_cfg = geoshapley_settings(cfg)
    geo_table = _explain_all_iz_for_samples(
        runtime,
        split,
        np.asarray([0], dtype=int),
        additivity_tolerance=geo_cfg["additivity_tolerance"],
    )
    geoshapley_dest.parent.mkdir(parents=True, exist_ok=True)
    geo_table.to_csv(geoshapley_dest, index=False)
    LOGGER.info("Wrote unlabelled %s GeoShapley (%s rows) to %s", target_str, len(geo_table), geoshapley_dest)
    return {
        "forecast_path": str(forecast_dest),
        "geoshapley_path": str(geoshapley_dest),
        "issue_date": issue_str,
        "target_date": target_str,
        "n_iz": int(table["iz_code"].nunique()) if "iz_code" in table.columns else int(len(table)),
    }


def region_rolling_alpha_path(area_code: str) -> Path:
    return region_output_dir(area_code) / "rolling" / "final_test" / "W730" / "rolling_alpha.csv"


def prepare_region_rolling(area_code: str, *, stage: str = "final_test") -> dict[str, Any]:
    """Recut a 65/10/25 copy and write rolling.yaml. Does not train and does not touch rolling_v1."""
    code = str(area_code or "").strip()
    if not code or code in {EDINBURGH_CA, "UNKNOWN"}:
        raise ModelError("Rolling prepare is only for a new city, not Edinburgh.", code="invalid_config")
    if stage not in {"plan", "final_test"}:
        raise ModelError("stage must be plan or final_test.", code="invalid_config")
    out = region_output_dir(code)
    _assert_not_protected(out)
    source = out / "forecast"
    recut = out / "forecast_split65_10_25"
    rolling_dir = out / "rolling"
    _assert_not_protected(recut)
    _assert_not_protected(rolling_dir)
    if not (source / "train.npz").is_file():
        raise ModelError(f"Missing prepared forecast windows at {source}.", code="missing_dataset")

    from data.dataset import load_temporal_dataset, write_chronological_resplit
    from model.config import load_model_config

    n_iz = int(len(_node_codes(source)))
    if not (recut / "test.npz").is_file():
        dataset = load_temporal_dataset(source, expected_iz_count=n_iz)
        summary = write_chronological_resplit(dataset, recut, train_frac=0.65, val_frac=0.10)
    else:
        summary = json.loads((recut / "split_summary.json").read_text(encoding="utf-8"))

    live_cfg = out / "model.yaml"
    cfg = load_model_config(live_cfg if live_cfg.is_file() else None)
    cfg["expected_edinburgh_iz_count"] = n_iz
    cfg["operational_inference"] = dict(cfg.get("operational_inference") or {})
    cfg["operational_inference"]["enabled"] = False
    cfg["rolling_evaluation"] = dict(cfg.get("rolling_evaluation") or {})
    cfg["rolling_evaluation"]["enabled"] = True
    cfg["rolling_evaluation"]["stage"] = stage
    cfg["rolling_evaluation"]["retrain_frequency_days"] = 28
    cfg["rolling_evaluation"]["selection_target_dates"] = 28
    cfg["rolling_evaluation"]["calibration_target_dates"] = 28
    cfg["rolling_evaluation"]["selected_window_days"] = 730
    cfg["rolling_evaluation"]["origin_split"] = "test"
    cfg["rolling_evaluation"]["window_selection_split"] = "validation"
    cfg["rolling_evaluation"]["write_geoshapley"] = False
    cfg["rolling_evaluation"]["output_dir"] = str(rolling_dir)
    cfg["rolling_evaluation"]["fixed_s1_dir"] = str(out / "model")
    cfg.setdefault("paths", {})
    cfg["paths"]["dataset"] = str(recut)
    cfg["paths"]["checkpoints"] = str(out / "model")
    yaml_path = out / "rolling.yaml"
    yaml_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return {
        "area_code": code,
        "config_path": str(yaml_path),
        "dataset": str(recut),
        "output_dir": str(rolling_dir),
        "split_summary": summary,
        "did_not_overwrite_rolling_v1": True,
        "did_not_overwrite_live_checkpoint": True,
    }


def _node_codes(forecast_dir: Path) -> list[str]:
    import pandas as pd

    order = pd.read_csv(forecast_dir / "node_order.csv")
    col = "iz_code" if "iz_code" in order.columns else order.columns[0]
    return [str(v) for v in order[col].tolist()]
