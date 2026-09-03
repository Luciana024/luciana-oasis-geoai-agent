"""Glasgow article tables in the same schema as website_article_v1/article.

Reads existing Glasgow rolling and split artefacts. Does not overwrite
website_article_v1, rolling_v1, or Edinburgh checkpoints.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from agent.region_training import GLASGOW_CA, region_output_dir
from common.errors import ModelError
from common.utils import project_relative_path, project_root
from data.node_order import sha256_file
from model.constants import FEATURE_PLAYER_NAMES, LOCATION_PLAYER
from model.heads import RAW80_Z, RAW95_Z
from presentation.website_export import (
    LATE_STABLE_START,
    _interval_stats,
    _metrics,
    _skill,
    _split_stats,
    write_article_tables,
)

EXPORT_RELATIVE = "data/results/exports/website_article_glasgow_v1"
N_IZ = 136
EDINBURGH_EXPORT = "data/results/exports/website_article_v1"


def _assert(condition: bool, message: str, code: str = "glasgow_export_failed") -> None:
    if not condition:
        raise ModelError(message, code=code)


def _out_dir() -> Path:
    path = project_root() / EXPORT_RELATIVE
    _assert(EDINBURGH_EXPORT not in str(path), "Refused to write into the Edinburgh export.")
    path.mkdir(parents=True, exist_ok=True)
    (path / "article").mkdir(exist_ok=True)
    (path / "article" / "full_precision").mkdir(exist_ok=True)
    return path


def _period(retrospective: pd.DataFrame, name: str, start: str | None, end: str | None, mask: pd.Series) -> dict[str, Any]:
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


def _geoshapley_summary(geo: pd.DataFrame, checkpoint_id: str, target_date: str) -> pd.DataFrame:
    geo = geo.copy()
    geo["component"] = geo["component"].replace({"interaction": "location_interaction"})
    if "feature_name" not in geo.columns:
        geo["feature_name"] = geo["player_name"]
    geo_main = geo[geo["component"] == "main"]
    geo_loc = geo[geo["component"] == "location"]
    geo_int = geo[geo["component"] == "location_interaction"]
    value_col = "shapley_value" if "shapley_value" in geo.columns else "phi"
    rows = []
    for feature in FEATURE_PLAYER_NAMES:
        main = geo_main[geo_main["feature_name"] == feature][value_col]
        inter = geo_int[geo_int["feature_name"] == f"location_x_{feature}"][value_col]
        rows.append(
            {
                "feature_name": feature,
                "mean_absolute_main_effect": float(np.mean(np.abs(main))),
                "mean_signed_main_effect": float(main.mean()),
                "mean_absolute_location_interaction": float(np.mean(np.abs(inter))),
                "positive_effect_iz_fraction": float((main > 0).mean()),
                "negative_effect_iz_fraction": float((main < 0).mean()),
                "checkpoint_id": checkpoint_id,
                "target_report_date": target_date,
            }
        )
    loc = geo_loc[value_col]
    rows.append(
        {
            "feature_name": LOCATION_PLAYER,
            "mean_absolute_main_effect": float(np.mean(np.abs(loc))),
            "mean_signed_main_effect": float(loc.mean()),
            "mean_absolute_location_interaction": np.nan,
            "positive_effect_iz_fraction": float((loc > 0).mean()),
            "negative_effect_iz_fraction": float((loc < 0).mean()),
            "checkpoint_id": checkpoint_id,
            "target_report_date": target_date,
        }
    )
    return pd.DataFrame(rows)


def export_glasgow_article() -> dict[str, Any]:
    region = region_output_dir(GLASGOW_CA)
    out = _out_dir()
    rolling_dir = region / "rolling" / "final_test" / "W730"
    split_dir = region / "forecast_split65_10_25"
    pred_path = rolling_dir / "predictions.csv"
    alpha_path = rolling_dir / "rolling_alpha.csv"
    geo_path = region / "model" / "geo_transport_mobility" / "exports" / "geoshapley.csv"
    _assert(pred_path.is_file(), f"Missing Glasgow rolling predictions: {pred_path}")
    _assert(alpha_path.is_file(), f"Missing Glasgow rolling alpha: {alpha_path}")
    _assert((split_dir / "train.npz").is_file(), f"Missing Glasgow 65/10/25 split: {split_dir}")
    _assert(geo_path.is_file(), f"Missing Glasgow GeoShapley: {geo_path}")
    _assert(EDINBURGH_EXPORT not in str(out), "Refused to write into the Edinburgh export.")

    pred = pd.read_csv(pred_path)
    pred["issue_date"] = pd.to_datetime(pred["issue_date"]).dt.strftime("%Y-%m-%d")
    pred["target_report_date"] = pd.to_datetime(pred["target_report_date"]).dt.strftime("%Y-%m-%d")
    pred["iz_code"] = pred["iz_code"].astype(str)
    lead = (pd.to_datetime(pred["target_report_date"]) - pd.to_datetime(pred["issue_date"])).dt.days
    _assert((lead == 7).all(), "target_report_date must equal issue_date + 7 days.")
    _assert(not pred.duplicated(["target_report_date", "iz_code"]).any(), "Duplicate target_date × IZ rows.")
    _assert("2023-03-04" not in set(pred["target_report_date"]), "Future target mixed into retrospective file.")
    _assert(pred["iz_code"].nunique() == N_IZ, f"Expected {N_IZ} Glasgow IZs, got {pred['iz_code'].nunique()}.")
    n_dates = int(pred["target_report_date"].nunique())
    n_cells = int(len(pred))
    _assert(n_cells == n_dates * N_IZ, f"Cell count {n_cells} is not {n_dates} × {N_IZ}.")

    retrospective = pd.DataFrame(
        {
            "target_report_date": pred["target_report_date"],
            "update_id": pred["update_id"],
            "observed_rate": pred["observed_rate"],
            "predicted_rate": pred["predicted_mu"],
            "persistence_prediction": pred["persistence_prediction"],
            "predicted_sigma": pred["predicted_sigma"],
            "calibrated80_lower": pred["calibrated80_lower"],
            "calibrated80_upper": pred["calibrated80_upper"],
            "calibrated95_lower": pred["calibrated95_lower"],
            "calibrated95_upper": pred["calibrated95_upper"],
        }
    )

    y = retrospective["observed_rate"].to_numpy()
    mu = retrospective["predicted_rate"].to_numpy()
    persist = retrospective["persistence_prediction"].to_numpy()
    sigma = retrospective["predicted_sigma"].to_numpy()
    model_m = _metrics(y, mu)
    persist_m = _metrics(y, persist)
    mae_skill = _skill(model_m["mae"], persist_m["mae"])
    mse_skill = _skill(model_m["mse"], persist_m["mse"])

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
    val_sel = _split_stats(split_dir / "validation.npz")
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
                "n_unique_target_dates": n_dates,
                "n_valid_cells": persist_m["n"],
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
                "n_unique_target_dates": n_dates,
                "n_valid_cells": model_m["n"],
            },
        ]
    )

    wave_mask = retrospective["target_report_date"] < LATE_STABLE_START
    late_mask = retrospective["target_report_date"] >= LATE_STABLE_START
    period_full = pd.DataFrame(
        [
            _period(
                retrospective,
                "declining_or_wave_period",
                retrospective.loc[wave_mask, "target_report_date"].min(),
                "2022-09-19",
                wave_mask,
            ),
            _period(
                retrospective,
                "late_stable_period",
                LATE_STABLE_START,
                retrospective.loc[late_mask, "target_report_date"].max(),
                late_mask,
            ),
            _period(
                retrospective,
                "overall_test_period",
                retrospective["target_report_date"].min(),
                retrospective["target_report_date"].max(),
                pd.Series(True, index=retrospective.index),
            ),
        ]
    )

    raw80_lo = mu - RAW80_Z * sigma
    raw80_hi = mu + RAW80_Z * sigma
    raw95_lo = mu - RAW95_Z * sigma
    raw95_hi = mu + RAW95_Z * sigma
    cal80 = _interval_stats(y, retrospective["calibrated80_lower"].to_numpy(), retrospective["calibrated80_upper"].to_numpy())
    cal95 = _interval_stats(y, retrospective["calibrated95_lower"].to_numpy(), retrospective["calibrated95_upper"].to_numpy())
    uncertainty_full = pd.DataFrame(
        [
            {"interval_type": "raw_80", "nominal_coverage": 0.80, **_interval_stats(y, raw80_lo, raw80_hi), "calibration_status": "raw_gaussian"},
            {"interval_type": "calibrated_80", "nominal_coverage": 0.80, **cal80, "calibration_status": "available"},
            {"interval_type": "raw_95", "nominal_coverage": 0.95, **_interval_stats(y, raw95_lo, raw95_hi), "calibration_status": "raw_gaussian"},
            {"interval_type": "calibrated_95", "nominal_coverage": 0.95, **cal95, "calibration_status": "available"},
        ]
    )
    abs_err = np.abs(mu - y)
    var = sigma ** 2
    nll = float(np.mean(0.5 * (math.log(2.0 * math.pi) + np.log(var) + (y - mu) ** 2 / var)))
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

    alpha_web = pd.read_csv(alpha_path)
    alpha_table_a = alpha_web[
        ["update_id", "alpha_geo", "alpha_transport", "alpha_mobility", "selected_epoch", "forecast_start", "forecast_end"]
    ].copy()
    weights = retrospective.groupby("update_id").size()
    mapped = alpha_web["update_id"].map(weights)
    alpha_table_b = pd.DataFrame(
        [
            {
                "graph_name": "geographic",
                "prediction_weighted_mean_alpha": float(np.average(alpha_web["alpha_geo"], weights=mapped)),
                "between_update_standard_deviation": float(alpha_web["alpha_geo"].std(ddof=1)),
                "minimum_alpha": float(alpha_web["alpha_geo"].min()),
                "maximum_alpha": float(alpha_web["alpha_geo"].max()),
            },
            {
                "graph_name": "transport",
                "prediction_weighted_mean_alpha": float(np.average(alpha_web["alpha_transport"], weights=mapped)),
                "between_update_standard_deviation": float(alpha_web["alpha_transport"].std(ddof=1)),
                "minimum_alpha": float(alpha_web["alpha_transport"].min()),
                "maximum_alpha": float(alpha_web["alpha_transport"].max()),
            },
            {
                "graph_name": "mobility",
                "prediction_weighted_mean_alpha": float(np.average(alpha_web["alpha_mobility"], weights=mapped)),
                "between_update_standard_deviation": float(alpha_web["alpha_mobility"].std(ddof=1)),
                "minimum_alpha": float(alpha_web["alpha_mobility"].min()),
                "maximum_alpha": float(alpha_web["alpha_mobility"].max()),
            },
        ]
    )
    alpha_sums = alpha_web[["alpha_geo", "alpha_transport", "alpha_mobility"]].sum(axis=1)
    _assert(((alpha_web[["alpha_geo", "alpha_transport", "alpha_mobility"]] > 0).all().all()), "Glasgow alpha not positive.")
    _assert(((alpha_sums - 1.0).abs() < 1e-5).all(), "Glasgow alpha does not sum to 1.")

    geo = pd.read_csv(geo_path)
    _assert(geo["iz_code"].nunique() == N_IZ, f"GeoShapley IZ count is {geo['iz_code'].nunique()}, expected {N_IZ}.")
    _assert(set(geo["target_report_date"].astype(str)) == {"2023-03-04"}, "Glasgow GeoShapley is only for 2023-03-04.")
    shapley_full = _geoshapley_summary(geo, checkpoint_id="live", target_date="2023-03-04")

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

    manifest = {
        "area_code": GLASGOW_CA,
        "area_name": "Glasgow City",
        "n_iz": N_IZ,
        "export_dir": project_relative_path(out),
        "did_not_overwrite_edinburgh": True,
        "sources": {
            "predictions": project_relative_path(pred_path),
            "rolling_alpha": project_relative_path(alpha_path),
            "split65": project_relative_path(split_dir),
            "geoshapley": project_relative_path(geo_path),
        },
        "notes": {
            "table02": "Same 65/10/25 rolling test days for Persistence and Rolling. No fixed-65 Glasgow model was scored on these days, so that row is omitted.",
            "table06": "GeoShapley is from the live Glasgow checkpoint on 2023-03-04 (unverified). It is not a siting score and is not U10 rolling.",
            "do_not_pool_with_edinburgh": "111 vs 136 IZs; do not merge MAE.",
        },
        "verified_rolling_metrics_raw": {
            **model_m,
            "persistence": persist_m,
            "mae_skill": mae_skill,
            "mse_skill": mse_skill,
            "n_unique_target_dates": n_dates,
        },
        "files": [
            {
                "path": str(path.relative_to(out)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted((out / "article").rglob("*.csv"))
        ],
    }
    (out / "EXPORT_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "export_dir": str(out),
        "n_dates": n_dates,
        "n_iz": N_IZ,
        "rolling_mae": model_m["mae"],
        "persistence_mae": persist_m["mae"],
        "mae_skill": mae_skill,
        "did_not_overwrite_edinburgh": True,
    }


if __name__ == "__main__":
    print(json.dumps(export_glasgow_article(), indent=2))
