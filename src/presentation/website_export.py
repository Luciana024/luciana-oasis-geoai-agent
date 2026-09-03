"""Assemble website and article-table exports without retraining.

Reads existing rolling W730, U10 operational, and split65_10_25 artefacts.
Writes a new directory; does not overwrite rolling_v1 or fixed checkpoints.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from model.constants import (
    CALIBRATION_METHOD,
    EXCHANGEABILITY_LIMITATION,
    EXPLANATION_SCOPE,
    FEATURE_PLAYER_NAMES,
    INTERACTION_PLAYER_NAMES,
    LOCATION_PLAYER,
    MODEL_NAME,
)
from common.errors import ModelError
from model.heads import RAW80_Z, RAW95_Z
from data.node_order import sha256_file
from common.utils import NODE_KEY, PANEL_CSV, project_relative_path, project_root, results_dir

EXPORT_RELATIVE = "data/results/exports/website_article_v1"
CANONICAL_HASH = "8f625000ca42af45709b4e887a429c93971443f30f2fbddbe07863342ca16d34"
LATE_STABLE_START = "2022-09-20"
EXPECTED_ROLLING = {
    "model_mae": 46.53,
    "persistence_mae": 50.66,
    "mae_skill": 0.081,
    "model_rmse": 69.06,
    "persistence_rmse": 81.74,
    "model_r2": 0.67,
    "persistence_r2": 0.53,
}
LOG_TWO_PI = math.log(2.0 * math.pi)


def _root() -> Path:
    return project_root()


def _out_dir() -> Path:
    path = _root() / EXPORT_RELATIVE
    path.mkdir(parents=True, exist_ok=True)
    (path / "website").mkdir(exist_ok=True)
    (path / "article").mkdir(exist_ok=True)
    (path / "article" / "full_precision").mkdir(exist_ok=True)
    return path


def _assert(condition: bool, message: str, code: str = "export_quality_failed") -> None:
    if not condition:
        raise ModelError(message, code=code)


def _metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    err = pred - y
    abs_err = np.abs(err)
    mse = float(np.mean(err ** 2))
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "mae": float(np.mean(abs_err)),
        "rmse": float(np.sqrt(mse)),
        "mse": mse,
        "bias": float(np.mean(err)),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot else float("nan"),
        "n": int(y.size),
    }


def _skill(model: float, baseline: float) -> float:
    return float("nan") if baseline == 0 else float(1.0 - model / baseline)


def _round_check(value: float, expected: float, digits: int, name: str) -> None:
    got = round(value, digits)
    _assert(
        got == expected,
        f"{name} rounded to {digits} dp is {got}, expected {expected} (raw={value}).",
    )


def _split_stats(npz_path: Path) -> dict[str, Any]:
    with np.load(npz_path, allow_pickle=True) as payload:
        y = np.asarray(payload["y_target_raw"], dtype=np.float64)
        x = np.asarray(payload["X_dynamic_raw"], dtype=np.float64)
        target = pd.to_datetime(payload["target_date"])
    anchor = x[:, -1, :, 0]
    y2 = y[:, :, 0]
    valid = np.isfinite(y2) & np.isfinite(anchor)
    delta = y2 - anchor
    dates = pd.Index(pd.to_datetime(target)).normalize().unique().sort_values()
    return {
        "target_start_date": str(dates.min().date()),
        "target_end_date": str(dates.max().date()),
        "n_unique_target_dates": int(len(dates)),
        "n_valid_iz_date_cells": int(valid.sum()),
        "mean_infection_rate": float(y2[valid].mean()),
        "std_infection_rate": float(y2[valid].std(ddof=1)),
        "mean_target_delta": float(delta[valid].mean()),
        "std_target_delta": float(delta[valid].std(ddof=1)),
    }


def _alpha_from_checkpoint_path(path: Path) -> dict[str, float]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    logits = payload["model_state_dict"]["fusion.logits"].float()
    values = F.softmax(logits, dim=0).detach().cpu().numpy()
    names = list(payload["graph_set"])
    return {name: float(values[i]) for i, name in enumerate(names)}


def _interval_stats(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    width = hi[valid] - lo[valid]
    covered = (y[valid] >= lo[valid]) & (y[valid] <= hi[valid])
    return {
        "observed_coverage": float(np.mean(covered)),
        "mean_interval_width": float(np.mean(width)),
        "median_interval_width": float(np.median(width)),
        "n_valid_cells": int(valid.sum()),
    }


ARTICLE_ROUNDING = {
    "table01_split_summary": {
        "fraction": 4,
        "mean_infection_rate": 2,
        "std_infection_rate": 2,
        "mean_target_delta": 2,
        "std_target_delta": 2,
    },
    "table02_overall_performance": {"MAE": 2, "MAE_skill": 3, "RMSE": 2, "MSE_skill": 3, "R2": 2, "bias": 2},
    "table03_performance_by_period": {
        "model_MAE": 2,
        "persistence_MAE": 2,
        "MAE_skill": 3,
        "model_RMSE": 2,
        "persistence_RMSE": 2,
        "MSE_skill": 3,
        "model_R2": 2,
        "persistence_R2": 2,
        "model_bias": 2,
    },
    "table04_uncertainty_intervals": {
        "nominal_coverage": 2,
        "observed_coverage": 3,
        "mean_interval_width": 2,
        "median_interval_width": 2,
    },
    "table04_uncertainty_sigma": {
        "mean_predicted_sigma": 2,
        "median_predicted_sigma": 2,
        "corr_sigma_absolute_error": 3,
        "gaussian_nll": 3,
    },
    "table05a_alpha_by_checkpoint": {"alpha_geo": 4, "alpha_transport": 4, "alpha_mobility": 4},
    "table05b_alpha_descriptive_summary": {
        "prediction_weighted_mean_alpha": 4,
        "between_update_standard_deviation": 4,
        "minimum_alpha": 4,
        "maximum_alpha": 4,
    },
    "table06_geoshapley_summary": {
        "mean_absolute_main_effect": 3,
        "mean_signed_main_effect": 3,
        "mean_absolute_location_interaction": 3,
        "positive_effect_iz_fraction": 3,
        "negative_effect_iz_fraction": 3,
    },
}


def round_article_table(frame: pd.DataFrame, numeric_dp: dict[str, int]) -> pd.DataFrame:
    out_frame = frame.copy()
    for col, digits in numeric_dp.items():
        if col in out_frame.columns:
            out_frame[col] = pd.to_numeric(out_frame[col], errors="coerce").round(digits)
    return out_frame


def write_article_tables(article_dir: Path, tables: dict[str, pd.DataFrame]) -> Path:
    """Write rounded paper tables plus full-precision copies. Does not invent values."""
    article_dir.mkdir(parents=True, exist_ok=True)
    full_dir = article_dir / "full_precision"
    full_dir.mkdir(parents=True, exist_ok=True)
    for stem, frame in tables.items():
        frame.to_csv(full_dir / f"{stem}.csv", index=False)
        round_article_table(frame, ARTICLE_ROUNDING.get(stem, {})).to_csv(
            article_dir / f"{stem}.csv", index=False
        )
    return article_dir


def export_website_and_article() -> dict[str, Any]:
    root = _root()
    out = _out_dir()
    rolling_dir = root / "data/results/model/rolling_v1_split65_10_25/final_test/W730"
    operational_dir = root / "data/results/model/rolling_v1_split65_10_25/operational_20230225"
    split_dir = root / "data/results/forecast/L7_H7_S1_20200308_20230225_split65_10_25"
    fixed_metrics_path = root / "data/results/model/split65_10_25/geo_transport_mobility/exports/test_metrics.json"
    panel_path = results_dir() / PANEL_CSV
    shp = root / "data/raw/boundaries/SG_IntermediateZoneBdry_2011/SG_IntermediateZone_Bdry_2011.shp"

    pred = pd.read_csv(rolling_dir / "predictions.csv")
    manifest = json.loads((rolling_dir / "updates_manifest.json").read_text(encoding="utf-8"))
    leak = json.loads((rolling_dir / "leakage_audit.json").read_text(encoding="utf-8"))
    iz_ops = pd.read_csv(operational_dir / "iz_outputs.csv")
    geo_ops = pd.read_csv(operational_dir / "geoshapley.csv")
    names = (
        pd.read_csv(panel_path, usecols=[NODE_KEY, "IntZoneName", "node_index"])
        .drop_duplicates(NODE_KEY)
        .rename(columns={NODE_KEY: "iz_code", "IntZoneName": "iz_name"})
    )
    names["iz_code"] = names["iz_code"].astype(str)

    pred["issue_date"] = pd.to_datetime(pred["issue_date"]).dt.strftime("%Y-%m-%d")
    pred["target_report_date"] = pd.to_datetime(pred["target_report_date"]).dt.strftime("%Y-%m-%d")
    pred["iz_code"] = pred["iz_code"].astype(str)
    lead = (pd.to_datetime(pred["target_report_date"]) - pd.to_datetime(pred["issue_date"])).dt.days
    _assert((lead == 7).all(), "target_report_date must equal issue_date + 7 days.")
    _assert(not pred.duplicated(["target_report_date", "iz_code"]).any(), "Duplicate target_date × IZ rows.")
    _assert(pred["target_report_date"].nunique() == 264, "Retrospective test must have 264 target dates.")
    _assert("2023-03-04" not in set(pred["target_report_date"]), "Future target mixed into retrospective file.")

    thresholds = {}
    checksums = {}
    alpha_rows = []
    for item in manifest["updates"]:
        uid = item["update_id"]
        cal = json.loads(Path(item["calibration_path"]).read_text(encoding="utf-8"))
        thresholds[uid] = float(cal["sigma_p90_threshold"])
        checksums[uid] = item["refit"]["checkpoint_sha256"]
        _assert(cal["checkpoint_sha256"] == checksums[uid], f"{uid} calibration does not match checkpoint.")
        alpha = _alpha_from_checkpoint_path(Path(item["refit"]["checkpoint_path"]))
        s = float(alpha["geo"] + alpha["transport"] + alpha["mobility"])
        _assert(alpha["geo"] > 0 and alpha["transport"] > 0 and alpha["mobility"] > 0, f"{uid} alpha not positive.")
        _assert(abs(s - 1.0) < 1e-6, f"{uid} alpha does not sum to 1 (sum={s}).")
        alpha_rows.append({"update_id": uid, **alpha, "selected_epoch": int(item["selected_epoch"])})
    alpha_map = pd.DataFrame(alpha_rows).set_index("update_id")

    website_ret = pred.merge(alpha_map[["geo", "transport", "mobility"]], left_on="update_id", right_index=True, how="left")
    _assert(website_ret[["geo", "transport", "mobility"]].notna().all().all(), "Missing alpha for some update_id.")
    website_ret["uncertainty_flag"] = [
        "high" if sigma > thresholds[uid] else "normal"
        for uid, sigma in zip(website_ret["update_id"], website_ret["predicted_sigma"])
    ]
    website_ret["checkpoint_id"] = website_ret["update_id"]
    website_ret["model_error"] = website_ret["predicted_mu"] - website_ret["observed_rate"]
    website_ret["model_absolute_error"] = website_ret["model_error"].abs()
    website_ret["persistence_absolute_error"] = (website_ret["persistence_prediction"] - website_ret["observed_rate"]).abs()
    website_ret["forecast_status"] = "retrospective_evaluation"
    website_ret["observed_target_available"] = True
    retrospective = pd.DataFrame(
        {
            "issue_date": website_ret["issue_date"],
            "target_report_date": website_ret["target_report_date"],
            "update_id": website_ret["update_id"],
            "checkpoint_id": website_ret["checkpoint_id"],
            "iz_code": website_ret["iz_code"],
            "node_index": website_ret["node_index"].astype(int),
            "observed_rate": website_ret["observed_rate"],
            "anchor_rate_y_t": website_ret["y_anchor"],
            "predicted_delta": website_ret["predicted_mu_delta"],
            "predicted_rate": website_ret["predicted_mu"],
            "predicted_sigma": website_ret["predicted_sigma"],
            "calibrated80_lower": website_ret["calibrated80_lower"],
            "calibrated80_upper": website_ret["calibrated80_upper"],
            "calibrated95_lower": website_ret["calibrated95_lower"],
            "calibrated95_upper": website_ret["calibrated95_upper"],
            "uncertainty_flag": website_ret["uncertainty_flag"],
            "persistence_prediction": website_ret["persistence_prediction"],
            "model_error": website_ret["model_error"],
            "model_absolute_error": website_ret["model_absolute_error"],
            "persistence_absolute_error": website_ret["persistence_absolute_error"],
            "alpha_geo": website_ret["geo"],
            "alpha_transport": website_ret["transport"],
            "alpha_mobility": website_ret["mobility"],
            "node_order_hash": website_ret["node_order_hash"],
            "forecast_status": website_ret["forecast_status"],
            "observed_target_available": website_ret["observed_target_available"],
        }
    )
    retro_path = out / "website" / "retrospective_predictions.csv"
    retrospective.to_csv(retro_path, index=False)

    dates = (
        retrospective[["issue_date", "target_report_date", "update_id", "checkpoint_id"]]
        .drop_duplicates()
        .sort_values("target_report_date")
    )
    dates_path = out / "website" / "date_selector.csv"
    dates.to_csv(dates_path, index=False)
    _assert(set(dates["issue_date"]) <= set(retrospective["issue_date"]), "Date selector leaked non-test dates.")

    y = retrospective["observed_rate"].to_numpy()
    mu = retrospective["predicted_rate"].to_numpy()
    persist = retrospective["persistence_prediction"].to_numpy()
    sigma = retrospective["predicted_sigma"].to_numpy()
    model_m = _metrics(y, mu)
    persist_m = _metrics(y, persist)
    mae_skill = _skill(model_m["mae"], persist_m["mae"])
    mse_skill = _skill(model_m["mse"], persist_m["mse"])
    _round_check(model_m["mae"], EXPECTED_ROLLING["model_mae"], 2, "rolling MAE")
    _round_check(persist_m["mae"], EXPECTED_ROLLING["persistence_mae"], 2, "persistence MAE")
    _round_check(mae_skill, EXPECTED_ROLLING["mae_skill"], 3, "MAE skill")
    _round_check(model_m["rmse"], EXPECTED_ROLLING["model_rmse"], 2, "rolling RMSE")
    _round_check(persist_m["rmse"], EXPECTED_ROLLING["persistence_rmse"], 2, "persistence RMSE")
    _round_check(model_m["r2"], EXPECTED_ROLLING["model_r2"], 2, "rolling R2")
    _round_check(persist_m["r2"], EXPECTED_ROLLING["persistence_r2"], 2, "persistence R2")

    future = pd.DataFrame(
        {
            "issue_date": iz_ops["issue_date"],
            "target_report_date": iz_ops["target_report_date"],
            "update_id": "U10",
            "checkpoint_id": "U10",
            "iz_code": iz_ops["IntZone"].astype(str),
            "node_index": iz_ops["node_index"].astype(int),
            "anchor_rate_y_t": iz_ops["observed_anchor_rate"],
            "predicted_delta": iz_ops["predicted_delta"],
            "predicted_rate": iz_ops["predicted_mean"],
            "predicted_variance": iz_ops["predicted_variance"],
            "predicted_sigma": iz_ops["predicted_sigma"],
            "calibrated80_lower": iz_ops["calibrated80_lower"],
            "calibrated80_upper": iz_ops["calibrated80_upper"],
            "calibrated95_lower": iz_ops["calibrated95_lower"],
            "calibrated95_upper": iz_ops["calibrated95_upper"],
            "uncertainty_flag": iz_ops["uncertainty_flag"],
            "alpha_geo": iz_ops["alpha_geographic"],
            "alpha_transport": iz_ops["alpha_transport"],
            "alpha_mobility": iz_ops["alpha_mobility"],
            "node_order_hash": iz_ops["node_order_hash"],
            "forecast_status": "unverified_extrapolation",
            "observed_target_available": False,
            "include_in_metrics": False,
            "observed_rate": np.nan,
            "model_error": np.nan,
            "model_absolute_error": np.nan,
            "persistence_prediction": np.nan,
            "persistence_absolute_error": np.nan,
        }
    )
    _assert((future["issue_date"] == "2023-02-25").all(), "Future issue_date must be 2023-02-25.")
    _assert((future["target_report_date"] == "2023-03-04").all(), "Future target must be 2023-03-04.")
    _assert(future["observed_rate"].isna().all(), "Future forecast must not contain observations.")
    _assert(len(future) == 111, "Future forecast must have 111 IZs.")
    future_path = out / "website" / "future_forecast_20230304.csv"
    future.to_csv(future_path, index=False)

    u10_alpha = alpha_map.loc["U10"]
    geo_rows = []
    for iz, part in geo_ops.groupby("iz_code", sort=False):
        first = part.iloc[0]
        geo_rows.append(
            {
                "issue_date": first["issue_date"],
                "target_report_date": first["target_report_date"],
                "update_id": "U10",
                "checkpoint_id": "U10",
                "iz_code": str(iz),
                "component": "baseline",
                "feature_name": "baseline",
                "shapley_value": float(first["phi_0"]),
                "predicted_rate": float(first["reconstructed_prediction"]),
                "reconstructed_prediction": float(first["reconstructed_prediction"]),
                "additivity_error": float(first["additivity_error"]),
                "alpha_geo": float(u10_alpha["geo"]),
                "alpha_transport": float(u10_alpha["transport"]),
                "alpha_mobility": float(u10_alpha["mobility"]),
                "explanation_scope": EXPLANATION_SCOPE,
            }
        )
        for _, row in part.iterrows():
            player = str(row["player_name"])
            if player == LOCATION_PLAYER:
                component = "location"
            elif player.startswith("location_x_"):
                component = "location_interaction"
            else:
                component = "main"
            geo_rows.append(
                {
                    "issue_date": row["issue_date"],
                    "target_report_date": row["target_report_date"],
                    "update_id": "U10",
                    "checkpoint_id": "U10",
                    "iz_code": str(row["iz_code"]),
                    "component": component,
                    "feature_name": player,
                    "shapley_value": float(row["phi"]),
                    "predicted_rate": float(row["reconstructed_prediction"]),
                    "reconstructed_prediction": float(row["reconstructed_prediction"]),
                    "additivity_error": float(row["additivity_error"]),
                    "alpha_geo": float(u10_alpha["geo"]),
                    "alpha_transport": float(u10_alpha["transport"]),
                    "alpha_mobility": float(u10_alpha["mobility"]),
                    "explanation_scope": EXPLANATION_SCOPE,
                }
            )
    geoshapley_web = pd.DataFrame(geo_rows)
    _assert(geoshapley_web["iz_code"].nunique() == 111, "GeoShapley IZ count is not 111.")
    _assert(
        set(geoshapley_web["feature_name"]) <= set(["baseline", *FEATURE_PLAYER_NAMES, LOCATION_PLAYER, *INTERACTION_PLAYER_NAMES]),
        "Unexpected GeoShapley feature names (embedding dimensions are forbidden).",
    )
    geo_path = out / "website" / "geoshapley.csv"
    geoshapley_web.to_csv(geo_path, index=False)

    alpha_web_rows = []
    for item in manifest["updates"]:
        uid = item["update_id"]
        row = alpha_map.loc[uid]
        alpha_web_rows.append(
            {
                "update_id": uid,
                "checkpoint_id": uid,
                "update_date": item["audit"]["update_date"],
                "forecast_start": item["audit"]["predict_issue_start"],
                "forecast_end": item["audit"]["predict_issue_end"],
                "alpha_geo": float(row["geo"]),
                "alpha_transport": float(row["transport"]),
                "alpha_mobility": float(row["mobility"]),
                "selected_epoch": int(item["selected_epoch"]),
                "training_window_days": 730,
                "checkpoint_checksum": checksums[uid],
                "node_order_hash": CANONICAL_HASH,
            }
        )
    alpha_web = pd.DataFrame(alpha_web_rows)
    alpha_path = out / "website" / "rolling_alpha.csv"
    alpha_web.to_csv(alpha_path, index=False)

    import geopandas as gpd

    boundaries = gpd.read_file(shp)
    code_col = next(c for c in (NODE_KEY, "InterZone", "IZ_CODE", "iz_code") if c in boundaries.columns)
    boundaries["iz_code"] = boundaries[code_col].astype(str)
    source_crs = str(boundaries.crs) if boundaries.crs is not None else "unknown"
    iz_index = retrospective[["iz_code", "node_index"]].drop_duplicates()
    geo = boundaries.merge(iz_index, on="iz_code", how="inner").merge(names[["iz_code", "iz_name"]], on="iz_code", how="left")
    _assert(len(geo) == 111, f"Boundary join produced {len(geo)} IZs, expected 111.")
    _assert(geo["iz_code"].is_unique, "Boundary file has duplicate iz_code.")
    geo_4326 = geo.to_crs("EPSG:4326")
    geo_out = geo_4326[["iz_code", "iz_name", "node_index", "geometry"]].copy()
    geo_out["source_crs"] = source_crs
    geo_out["web_crs"] = "EPSG:4326"
    boundary_path = out / "website" / "edinburgh_iz_boundaries.geojson"
    geo_out.to_file(boundary_path, driver="GeoJSON")

    val_sel = _split_stats(split_dir / "validation.npz")
    with np.load(split_dir / "validation.npz", allow_pickle=True) as payload:
        val_targets = pd.to_datetime(payload["target_date"])
        y_val = np.asarray(payload["y_target_raw"], dtype=np.float64)[:, :, 0]
        x_val = np.asarray(payload["X_dynamic_raw"], dtype=np.float64)[:, -1, :, 0]
    n_sel = int(np.floor(len(val_targets) * 0.5))
    order = np.argsort(val_targets, kind="mergesort")
    sel_idx, cal_idx = order[:n_sel], order[n_sel:]

    def _subset_stats(idx: np.ndarray) -> dict[str, Any]:
        y2 = y_val[idx]
        a2 = x_val[idx]
        valid = np.isfinite(y2) & np.isfinite(a2)
        d2 = y2 - a2
        dates = pd.Index(val_targets[idx]).normalize().unique().sort_values()
        return {
            "target_start_date": str(dates.min().date()),
            "target_end_date": str(dates.max().date()),
            "n_unique_target_dates": int(len(dates)),
            "n_valid_iz_date_cells": int(valid.sum()),
            "mean_infection_rate": float(y2[valid].mean()),
            "std_infection_rate": float(y2[valid].std(ddof=1)),
            "mean_target_delta": float(d2[valid].mean()),
            "std_target_delta": float(d2[valid].std(ddof=1)),
        }

    train_s = _split_stats(split_dir / "train.npz")
    test_s = _split_stats(split_dir / "test.npz")
    sel_s = _subset_stats(sel_idx)
    cal_s = _subset_stats(cal_idx)
    n_total = train_s["n_unique_target_dates"] + val_sel["n_unique_target_dates"] + test_s["n_unique_target_dates"]
    split_full = pd.DataFrame(
        [
            {"split_name": "Train", "fraction": train_s["n_unique_target_dates"] / n_total, **train_s},
            {"split_name": "Validation selection", "fraction": sel_s["n_unique_target_dates"] / n_total, **sel_s},
            {"split_name": "Validation calibration", "fraction": cal_s["n_unique_target_dates"] / n_total, **cal_s},
            {"split_name": "Test", "fraction": test_s["n_unique_target_dates"] / n_total, **test_s},
        ]
    )

    fixed = json.loads(fixed_metrics_path.read_text(encoding="utf-8"))
    persist_mse = persist_m["mse"]
    fixed_mse = float(fixed["rmse"] ** 2)
    overall_full = pd.DataFrame(
        [
            {
                "method": "Persistence",
                "evaluation_type": "retrospective_test_65_10_25",
                "MAE": persist_m["mae"],
                "MAE_skill": 0.0,
                "RMSE": persist_m["rmse"],
                "MSE_skill": 0.0,
                "R2": persist_m["r2"],
                "bias": persist_m["bias"],
                "n_unique_target_dates": 264,
                "n_valid_cells": persist_m["n"],
            },
            {
                "method": "Fixed 65/10/25 model",
                "evaluation_type": "retrospective_test_65_10_25",
                "MAE": float(fixed["mae"]),
                "MAE_skill": float(fixed["mae_skill"]),
                "RMSE": float(fixed["rmse"]),
                "MSE_skill": _skill(fixed_mse, persist_mse),
                "R2": float(fixed["r2"]),
                "bias": -float(fixed["bias"]),
                "n_unique_target_dates": 264,
                "n_valid_cells": int(fixed["n_valid_cells"]),
            },
            {
                "method": "Rolling 65/10/25 model",
                "evaluation_type": "retrospective_test_65_10_25",
                "MAE": model_m["mae"],
                "MAE_skill": mae_skill,
                "RMSE": model_m["rmse"],
                "MSE_skill": mse_skill,
                "R2": model_m["r2"],
                "bias": model_m["bias"],
                "n_unique_target_dates": 264,
                "n_valid_cells": model_m["n"],
            },
        ]
    )

    def _period(name: str, start: str | None, end: str | None, mask: pd.Series) -> dict[str, Any]:
        part = retrospective.loc[mask]
        ym = _metrics(part["observed_rate"].to_numpy(), part["predicted_rate"].to_numpy())
        yp = _metrics(part["observed_rate"].to_numpy(), part["persistence_prediction"].to_numpy())
        return {
            "period_name": name,
            "period_start": start or part["target_report_date"].min(),
            "period_end": end or part["target_report_date"].max(),
            "model_MAE": ym["mae"],
            "persistence_MAE": yp["mae"],
            "MAE_skill": _skill(ym["mae"], yp["mae"]),
            "model_RMSE": ym["rmse"],
            "persistence_RMSE": yp["rmse"],
            "MSE_skill": _skill(ym["mse"], yp["mse"]),
            "model_R2": ym["r2"],
            "persistence_R2": yp["r2"],
            "model_bias": ym["bias"],
            "n_unique_target_dates": int(part["target_report_date"].nunique()),
            "n_valid_cells": ym["n"],
        }

    wave_mask = retrospective["target_report_date"] < LATE_STABLE_START
    late_mask = retrospective["target_report_date"] >= LATE_STABLE_START
    period_full = pd.DataFrame(
        [
            _period("declining_or_wave_period", retrospective.loc[wave_mask, "target_report_date"].min(), "2022-09-19", wave_mask),
            _period("late_stable_period", LATE_STABLE_START, retrospective.loc[late_mask, "target_report_date"].max(), late_mask),
            _period("overall_test_period", retrospective["target_report_date"].min(), retrospective["target_report_date"].max(), pd.Series(True, index=retrospective.index)),
        ]
    )

    raw80_lo = mu - RAW80_Z * sigma
    raw80_hi = mu + RAW80_Z * sigma
    raw95_lo = mu - RAW95_Z * sigma
    raw95_hi = mu + RAW95_Z * sigma
    cal80 = _interval_stats(y, retrospective["calibrated80_lower"].to_numpy(), retrospective["calibrated80_upper"].to_numpy())
    cal95 = _interval_stats(y, retrospective["calibrated95_lower"].to_numpy(), retrospective["calibrated95_upper"].to_numpy())
    r80 = _interval_stats(y, raw80_lo, raw80_hi)
    r95 = _interval_stats(y, raw95_lo, raw95_hi)
    abs_err = np.abs(mu - y)
    var = sigma ** 2
    nll = float(np.mean(0.5 * (LOG_TWO_PI + np.log(var) + (y - mu) ** 2 / var)))
    uncertainty_full = pd.DataFrame(
        [
            {"interval_type": "raw_80", "nominal_coverage": 0.80, **r80, "calibration_status": "raw_gaussian"},
            {"interval_type": "calibrated_80", "nominal_coverage": 0.80, **cal80, "calibration_status": "available"},
            {"interval_type": "raw_95", "nominal_coverage": 0.95, **r95, "calibration_status": "raw_gaussian"},
            {"interval_type": "calibrated_95", "nominal_coverage": 0.95, **cal95, "calibration_status": "available"},
        ]
    )
    uncertainty_extra = pd.DataFrame(
        [
            {
                "mean_predicted_sigma": float(np.mean(sigma)),
                "median_predicted_sigma": float(np.median(sigma)),
                "corr_sigma_absolute_error": float(np.corrcoef(sigma, abs_err)[0, 1]),
                "gaussian_nll": nll,
                "n_valid_cells": int(len(y)),
                "evaluation_set": "rolling_retrospective_test_only",
            }
        ]
    )

    alpha_table_a = alpha_web.rename(columns={"forecast_start": "forecast_start", "forecast_end": "forecast_end"})[
        ["update_id", "alpha_geo", "alpha_transport", "alpha_mobility", "selected_epoch", "forecast_start", "forecast_end"]
    ]
    weights = retrospective.groupby("update_id").size()
    pred_mean = {
        "geographic": float(np.average(alpha_web["alpha_geo"], weights=alpha_web["update_id"].map(weights))),
        "transport": float(np.average(alpha_web["alpha_transport"], weights=alpha_web["update_id"].map(weights))),
        "mobility": float(np.average(alpha_web["alpha_mobility"], weights=alpha_web["update_id"].map(weights))),
    }
    alpha_table_b = pd.DataFrame(
        [
            {
                "graph_name": "geographic",
                "prediction_weighted_mean_alpha": pred_mean["geographic"],
                "between_update_standard_deviation": float(alpha_web["alpha_geo"].std(ddof=1)),
                "minimum_alpha": float(alpha_web["alpha_geo"].min()),
                "maximum_alpha": float(alpha_web["alpha_geo"].max()),
            },
            {
                "graph_name": "transport",
                "prediction_weighted_mean_alpha": pred_mean["transport"],
                "between_update_standard_deviation": float(alpha_web["alpha_transport"].std(ddof=1)),
                "minimum_alpha": float(alpha_web["alpha_transport"].min()),
                "maximum_alpha": float(alpha_web["alpha_transport"].max()),
            },
            {
                "graph_name": "mobility",
                "prediction_weighted_mean_alpha": pred_mean["mobility"],
                "between_update_standard_deviation": float(alpha_web["alpha_mobility"].std(ddof=1)),
                "minimum_alpha": float(alpha_web["alpha_mobility"].min()),
                "maximum_alpha": float(alpha_web["alpha_mobility"].max()),
            },
        ]
    )

    geo_main = geoshapley_web[geoshapley_web["component"] == "main"]
    geo_loc = geoshapley_web[geoshapley_web["component"] == "location"]
    geo_int = geoshapley_web[geoshapley_web["component"] == "location_interaction"]
    shapley_rows = []
    for feature in FEATURE_PLAYER_NAMES:
        main = geo_main[geo_main["feature_name"] == feature]["shapley_value"]
        inter = geo_int[geo_int["feature_name"] == f"location_x_{feature}"]["shapley_value"]
        shapley_rows.append(
            {
                "feature_name": feature,
                "mean_absolute_main_effect": float(np.mean(np.abs(main))),
                "mean_signed_main_effect": float(main.mean()),
                "mean_absolute_location_interaction": float(np.mean(np.abs(inter))),
                "positive_effect_iz_fraction": float((main > 0).mean()),
                "negative_effect_iz_fraction": float((main < 0).mean()),
                "checkpoint_id": "U10",
                "target_report_date": "2023-03-04",
            }
        )
    loc = geo_loc["shapley_value"]
    shapley_rows.append(
        {
            "feature_name": "location",
            "mean_absolute_main_effect": float(np.mean(np.abs(loc))),
            "mean_signed_main_effect": float(loc.mean()),
            "mean_absolute_location_interaction": np.nan,
            "positive_effect_iz_fraction": float((loc > 0).mean()),
            "negative_effect_iz_fraction": float((loc < 0).mean()),
            "checkpoint_id": "U10",
            "target_report_date": "2023-03-04",
        }
    )
    shapley_full = pd.DataFrame(shapley_rows)

    write_article_tables(
        out / "article",
        {
            "table01_split_summary": split_full,
            "table02_overall_performance": overall_full,
            "table03_performance_by_period": period_full,
            "table04_uncertainty_intervals": uncertainty_full,
            "table04_uncertainty_sigma": uncertainty_extra,
            "table05a_alpha_by_checkpoint": alpha_table_a,
            "table05b_alpha_descriptive_summary": alpha_table_b,
            "table06_geoshapley_summary": shapley_full,
        },
    )

    metadata = {
        "model_name": MODEL_NAME,
        "prediction_definition": "[Y_{t-6}, ..., Y_t] -> Y_{t+7}; residual mu = Y_t + predicted_delta",
        "rolling_seven_day_rate_definition": "Daily-reported rolling seven-day COVID-19 infection rate per 100,000, not daily new cases.",
        "lookback": 7,
        "lead": 7,
        "output_steps": 1,
        "graph_definitions": {
            "geo": "2011 IZ Queen adjacency, undirected/symmetric allowed",
            "transport": "directed road/public-transport graph, not symmetrised",
            "mobility": "pre-averaged 2019-2023 OD matrix, directed, not real-time flow",
            "fusion": "One global softmax alpha mixes directed diffusion supports. Label: learned relative graph-fusion weights, not COVID risk shares.",
        },
        "context_variable_definitions": {
            "income_rate": "income deprivation",
            "employment_rate": "employment deprivation",
            "university_rate": "higher education / university entry",
            "overcrowded_rate": "overcrowding",
            "crime_rate": "selected neighbourhood crime",
            "pt_gp_min": "public transport time to GP, minutes",
            "location": "joint Easting/Northing player in GeoShapley",
        },
        "uncertainty_interpretation": {
            "sigma": "Predicted standard deviation of the residual-reconstructed rate. Maps should colour by sigma.",
            "intervals": "Calibrated 80/95 are empirical intervals from frozen calibration scores; not an exchangeability coverage guarantee.",
            "uncertainty_flag": "Optional binary overlay: sigma > calibration P90. Not required if the map uses continuous sigma.",
            "exchangeability_limitation": EXCHANGEABILITY_LIMITATION,
            "calibration_method": CALIBRATION_METHOD,
        },
        "checkpoint_information": {
            "rolling_updates": "U01-U10, window W730, 28-day updates",
            "future_forecast_checkpoint": "U10",
            "fixed_65_10_25_not_used_for_website_maps": True,
            "geoshapley_available_for": [
                {
                    "checkpoint_id": "U10",
                    "issue_date": "2023-02-25",
                    "target_report_date": "2023-03-04",
                    "note": "Only GeoShapley computed with the same rolling checkpoint as the displayed forecast.",
                }
            ],
            "retrospective_geoshapley": "Not computed per rolling update. Do not attach the fixed-split GeoShapley file to rolling maps.",
        },
        "calibration_method": CALIBRATION_METHOD,
        "data_date_ranges": {
            "panel": "2020-03-08 to 2023-02-25",
            "retrospective_issue_dates": f"{dates['issue_date'].min()} to {dates['issue_date'].max()}",
            "retrospective_target_dates": f"{dates['target_report_date'].min()} to {dates['target_report_date'].max()}",
            "future_issue_date": "2023-02-25",
            "future_target_date": "2023-03-04",
        },
        "split_design": "Chronological 65/10/25 by target_date on 1054 frozen S1 windows. Validation internally 50/50 selection then calibration.",
        "leakage_check_status": {"passed": bool(leak.get("passed")), "source": str(rolling_dir / "leakage_audit.json")},
        "warning_definitions": {
            "retrospective_evaluation": "Labelled test forecasts; observations exist.",
            "unverified_extrapolation": "No observed target on 2023-03-04; exclude from metrics.",
            "alpha_mean": "Descriptive only; do not substitute for a date-specific checkpoint.",
        },
        "explanation_limitations": {
            "geoshapley_scope": EXPLANATION_SCOPE,
            "does_not_explain_embeddings": True,
            "not_causal": True,
            "alpha_is_not_risk_share": True,
            "no_geoshapley_on_other_test_dates": True,
        },
        "bias_definition": "mean(prediction - observation)",
        "website_date_selector": "Use website/date_selector.csv only (test issue dates). Do not add train or validation dates.",
        "canonical_node_order_hash": CANONICAL_HASH,
    }
    meta_path = out / "website" / "model_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    files = []
    for path in sorted(out.rglob("*")):
        if path.is_file():
            files.append(
                {
                    "path": str(path.relative_to(out)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "n_rows": int(sum(1 for _ in path.open(encoding="utf-8")) - 1) if path.suffix == ".csv" else None,
                    "n_columns": list(pd.read_csv(path, nrows=0).columns) if path.suffix == ".csv" else None,
                }
            )
    export_manifest = {
        "export_dir": project_relative_path(out),
        "sources_not_overwritten": [
            project_relative_path(rolling_dir),
            project_relative_path(root / "data/results/model/split65_10_25/geo_transport_mobility"),
        ],
        "source_checkpoints": checksums,
        "quality_checks": {
            "duplicate_target_iz": False,
            "alpha_positive_sum_to_one": True,
            "target_equals_issue_plus_7": True,
            "rolling_metrics_match_verified_rounded_values": True,
            "future_excluded_from_metrics": True,
            "boundary_join_111": True,
            "geoshapley_checkpoint": "U10",
        },
        "verified_rolling_metrics_raw": {
            **model_m,
            "persistence": persist_m,
            "mae_skill": mae_skill,
            "mse_skill": mse_skill,
        },
        "files": files,
        "column_notes": {
            "website/retrospective_predictions.csv": "One row per test target date × IZ. forecast_status=retrospective_evaluation.",
            "website/future_forecast_20230304.csv": "Unverified U10 extrapolation. Accuracy fields are null.",
            "website/geoshapley.csv": "U10 only, target 2023-03-04. Not for other rolling dates.",
            "website/rolling_alpha.csv": "U01-U10 trajectory.",
            "website/edinburgh_iz_boundaries.geojson": "Join key iz_code, EPSG:4326.",
            "website/date_selector.csv": "Test dates only.",
        },
    }
    manifest_path = out / "EXPORT_MANIFEST.json"
    manifest_path.write_text(json.dumps(export_manifest, indent=2), encoding="utf-8")
    return {"export_dir": str(out), "n_files": len(files), "rolling_mae": model_m["mae"], "mae_skill": mae_skill}


if __name__ == "__main__":
    result = export_website_and_article()
    print(json.dumps(result, indent=2))
