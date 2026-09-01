"""Unlabelled next-report-day inference after the last panel date.

This is not a retrospective test export. It does not rebuild frozen S1 windows
and does not retrain. The live checkpoint is the last rolling update whose
scheduled 28-day issue window contains the request issue date.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn.functional as F

from model.config import config_path, geoshapley_settings, load_model_config, temporal_target
from model.constants import FEATURE_PLAYER_NAMES, LOCATION_PLAYER, THREE_GRAPH_SET
from data.dataset import SplitArrays
from common.errors import LEVEL_ACCEPTED, ModelError, ModelWarning
from model.evaluate import assert_artefact_matches_checkpoint
from presentation.tables import build_forecast_table
from model.heads import calibrated_interval
from data.node_order import sha256_file
from model.residual import apply_residual_scalers_to_split
from model.train import load_raw_checkpoint
from common.utils import NODE_KEY, PANEL_CSV, project_root, results_dir

USER_REQUEST = (
    "Using all COVID-19 observations available through 25 February 2023, "
    "forecast the rolling seven-day COVID-19 infection rate for 4 March 2023 "
    "across Edinburgh IZs, and provide uncertainty, graph-fusion weights and "
    "GeoShapley explanations."
)
REQUESTED_ISSUE_DATE = "2023-02-25"
REQUESTED_TARGET_DATE = "2023-03-04"
REQUESTED_LEAD_DAYS = 7
CANONICAL_NODE_ORDER_HASH = "8f625000ca42af45709b4e887a429c93971443f30f2fbddbe07863342ca16d34"
FIXED_SPLIT_CHECKPOINT_MARKER = "split65_10_25/geo_transport_mobility"

DEFAULT_CHECKPOINT = (
    "data/results/model/rolling_v1_split65_10_25/final_test/W730/U10/checkpoint.pt"
)
DEFAULT_CALIBRATION = (
    "data/results/model/rolling_v1_split65_10_25/final_test/W730/U10/calibration.json"
)
DEFAULT_U10_MANIFEST = (
    "data/results/model/rolling_v1_split65_10_25/final_test/W730/U10/manifest.json"
)
DEFAULT_OUTPUT_DIR = "data/results/model/rolling_v1_split65_10_25/operational_20230225"
IZ_BOUNDARY_SHP = (
    "data/raw/boundaries/SG_IntermediateZoneBdry_2011/SG_IntermediateZone_Bdry_2011.shp"
)
CONFIG_COMPARE_KEYS = (
    "embedding_dim",
    "hidden_dim",
    "context_layers",
    "dcrnn_layers",
    "diffusion_steps",
    "dropout",
    "variance_epsilon",
    "model_variant",
)


def _as_path(path: str | Path) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = project_root() / path
    return path


def _parse_panel_dates(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.replace("-", "", regex=False)
    return pd.to_datetime(text, format="%Y%m%d")


def last_panel_issue_date(panel_path: Path) -> pd.Timestamp:
    dates = _parse_panel_dates(pd.read_csv(panel_path, usecols=["Date"])["Date"])
    return pd.Timestamp(dates.max()).normalize()


def build_operational_split(
    panel_path: Path,
    node_order,
    *,
    issue_date: pd.Timestamp,
    lookback_days: int,
    target_offset_days: int,
) -> SplitArrays:
    """One unlabelled window: [Y_{t-L+1}, ..., Y_t] with t = issue_date."""
    issue_date = pd.Timestamp(issue_date).normalize()
    lookback = pd.date_range(issue_date - pd.Timedelta(days=lookback_days - 1), issue_date, freq="D")
    target_date = issue_date + pd.Timedelta(days=target_offset_days)
    panel = pd.read_csv(panel_path, usecols=["Date", NODE_KEY, "infection_rate"])
    panel["Date"] = _parse_panel_dates(panel["Date"])
    panel[NODE_KEY] = panel[NODE_KEY].astype(str)
    used = panel.loc[panel["Date"].isin(lookback)]
    if (used["Date"] > issue_date).any():
        raise ModelError(
            "Operational window construction would read dates after the issue date.",
            code="post_issue_data_used",
        )
    n_nodes = node_order.n_nodes
    x_raw = np.full((1, lookback_days, n_nodes, 1), np.nan, dtype=np.float64)
    missing_by_day: dict[str, list[str]] = {}
    for step, day in enumerate(lookback):
        sub = panel.loc[panel["Date"] == day]
        if sub.empty:
            raise ModelError(
                f"Panel has no rows for lookback date {day.date()}.",
                code="missing_operational_input",
            )
        rates = sub.drop_duplicates(NODE_KEY).set_index(NODE_KEY)["infection_rate"]
        ordered = rates.reindex(list(node_order.codes))
        if ordered.isna().any():
            missing = [code for code, value in ordered.items() if pd.isna(value)]
            missing_by_day[str(day.date())] = missing
            continue
        x_raw[0, step, :, 0] = ordered.to_numpy(dtype=np.float64)
    if missing_by_day:
        raise ModelError(
            "Incomplete operational lookback: not all 111 IZs are present through the issue date.",
            code="missing_operational_input",
            details={"missing_by_day": {day: codes[:12] for day, codes in missing_by_day.items()}},
        )
    if not np.isfinite(x_raw).all():
        raise ModelError("Operational lookback contains non-finite infection rates.", code="invalid_tensor_shape")
    y_raw = np.full((1, n_nodes, 1), np.nan, dtype=np.float64)
    origin = np.array([np.datetime64(issue_date.date())], dtype="datetime64[D]")
    target = np.array([np.datetime64(target_date.date())], dtype="datetime64[D]")
    dummy = np.zeros_like(x_raw)
    return SplitArrays(
        name="operational",
        x_dynamic_scaled=dummy,
        y_target_scaled=np.zeros_like(y_raw),
        x_dynamic_raw=x_raw,
        y_target_raw=y_raw,
        sample_id=np.array(["operational_last_issue"], dtype=object),
        forecast_origin_date=origin,
        target_date=target,
    )


def _alpha_from_checkpoint(payload: dict[str, Any]) -> dict[str, float]:
    logits = payload["model_state_dict"]["fusion.logits"].float()
    alpha = F.softmax(logits, dim=0).detach().cpu().numpy()
    names = list(payload["graph_set"])
    return {name: float(alpha[i]) for i, name in enumerate(names)}


def _check(name: str, passed: bool, detail: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def _almost_equal_scaler(left: dict[str, Any], right: dict[str, Any]) -> bool:
    for key in ("mean", "scale", "epsilon"):
        if abs(float(left[key]) - float(right[key])) > 1e-12:
            return False
    return str(left.get("transform")) == str(right.get("transform"))


def validate_u10_for_issue(
    *,
    cfg: dict[str, Any],
    payload: dict[str, Any],
    runtime: dict[str, Any],
    artefact: dict[str, Any],
    checkpoint_path: Path,
    calibration_path: Path,
    u10_manifest_path: Path,
    panel_path: Path,
    issue_date: pd.Timestamp,
    target_date: pd.Timestamp,
) -> dict[str, Any]:
    """Compatibility report. Does not run GeoShapley or write predictions."""
    tt = temporal_target(cfg)
    rolling = cfg.get("rolling_evaluation") or {}
    frequency = int(rolling.get("retrain_frequency_days", 28))
    manifest = json.loads(u10_manifest_path.read_text(encoding="utf-8"))
    audit = manifest.get("audit") or {}
    update_date = pd.Timestamp(audit["update_date"]).normalize()
    labelled_predict_start = pd.Timestamp(audit["predict_issue_start"]).normalize()
    labelled_predict_end = pd.Timestamp(audit["predict_issue_end"]).normalize()
    live_start = update_date
    live_end = update_date + pd.Timedelta(days=frequency - 1)
    lookback_start = issue_date - pd.Timedelta(days=tt["lookback_steps"] - 1)
    lookback_dates = pd.date_range(lookback_start, issue_date, freq="D")

    panel = pd.read_csv(panel_path, usecols=["Date", NODE_KEY, "infection_rate"])
    panel["Date"] = _parse_panel_dates(panel["Date"])
    panel[NODE_KEY] = panel[NODE_KEY].astype(str)
    node_order = runtime["dataset"].node_order
    missing_by_day: dict[str, int] = {}
    for day in lookback_dates:
        sub = panel.loc[panel["Date"] == day]
        present = set(sub[NODE_KEY].astype(str))
        missing = [code for code in node_order.codes if code not in present]
        missing_by_day[str(day.date())] = len(missing)
    n_complete_days = sum(1 for count in missing_by_day.values() if count == 0)
    dates_used_max = lookback_dates.max()
    post_issue_used = bool(dates_used_max > issue_date)

    ckpt_hash = sha256_file(checkpoint_path)
    residual_json_path = checkpoint_path.parent / "residual_scalers.json"
    residual_json = json.loads(residual_json_path.read_text(encoding="utf-8")) if residual_json_path.is_file() else None
    payload_scalers = payload.get("residual_scalers")
    scaler_match = False
    if residual_json and payload_scalers:
        scaler_match = all(
            _almost_equal_scaler(payload_scalers[name], residual_json[name])
            for name in ("rate", "first_difference", "delta")
        )

    stored_hashes = payload.get("graph_hashes") or manifest.get("graph_hashes") or {}
    live_hashes = {name: runtime["graphs"][name].file_sha256 for name in runtime["graph_set"]}
    graph_hash_ok = all(stored_hashes.get(name) == live_hashes.get(name) for name in THREE_GRAPH_SET)

    model_cfg = payload.get("model_config") or {}
    config_mismatches = {
        key: {"checkpoint": model_cfg.get(key), "config": cfg.get(key)}
        for key in CONFIG_COMPARE_KEYS
        if key in model_cfg and model_cfg.get(key) != cfg.get(key)
    }
    run_cfg = payload.get("run_config") or {}
    offset_ok = int(run_cfg.get("target_offset_days", tt["target_offset_days"])) == tt["target_offset_days"]
    lookback_ok = int(run_cfg.get("lookback_days", tt["lookback_steps"])) == tt["lookback_steps"]

    alpha = _alpha_from_checkpoint(payload)
    alpha_values = np.array([alpha[name] for name in THREE_GRAPH_SET], dtype=np.float64)
    alpha_ok = bool(np.all(alpha_values > 0) and abs(float(alpha_values.sum()) - 1.0) < 1e-6)

    in_live_window = bool(live_start <= issue_date <= live_end)
    in_labelled_window = bool(labelled_predict_start <= issue_date <= labelled_predict_end)
    is_u10 = str(manifest.get("update_id")) == "U10"
    is_w730 = str(manifest.get("window")) == "W730" and int(audit.get("n_fitting", 0)) == 730
    not_fixed = FIXED_SPLIT_CHECKPOINT_MARKER not in str(checkpoint_path)
    graph_set_ok = tuple(payload.get("graph_set") or ()) == THREE_GRAPH_SET
    node_hash = str(payload.get("node_order", {}).get("canonical_node_order_hash") or manifest.get("canonical_node_order_hash"))
    node_hash_ok = node_hash == CANONICAL_NODE_ORDER_HASH and node_order.canonical_hash == CANONICAL_NODE_ORDER_HASH
    cal_ok = (
        artefact.get("checkpoint_sha256") == ckpt_hash
        and artefact.get("calibration_status") == "available"
        and artefact.get("q80") is not None
        and artefact.get("q95") is not None
        and artefact.get("sigma_p90_threshold") is not None
    )
    covid_scaler_ok = payload.get("residual_scalers") is not None and payload.get("covid_scaler") is not None and scaler_match
    context_ok = payload.get("context_scaler") is not None
    coord_ok = payload.get("coord_scaler") is not None
    dates_ok = str(issue_date.date()) == REQUESTED_ISSUE_DATE and str(target_date.date()) == REQUESTED_TARGET_DATE
    lead_ok = int((target_date - issue_date).days) == REQUESTED_LEAD_DAYS == tt["target_offset_days"]
    inputs_ok = n_complete_days == tt["lookback_steps"] and node_order.n_nodes == 111

    checks = [
        _check("u10_identity", is_u10, manifest.get("update_id")),
        _check("not_fixed_split_checkpoint", not_fixed, str(checkpoint_path)),
        _check("w730_training_window", is_w730, {"window": manifest.get("window"), "n_fitting": audit.get("n_fitting")}),
        _check("graph_set", graph_set_ok, list(payload.get("graph_set") or ())),
        _check("node_order_hash", node_hash_ok, {"checkpoint": node_hash, "runtime": node_order.canonical_hash}),
        _check("model_configuration", not config_mismatches and offset_ok and lookback_ok, config_mismatches),
        _check("covid_residual_scaler", covid_scaler_ok, {"json_present": residual_json is not None, "match": scaler_match}),
        _check("contextual_feature_scaler", context_ok, "context_scaler" in payload),
        _check("coordinate_scaler", coord_ok, "coord_scaler" in payload),
        _check("uncertainty_calibration_artefact", cal_ok, {"status": artefact.get("calibration_status"), "sha256_match": artefact.get("checkpoint_sha256") == ckpt_hash}),
        _check("graph_file_hashes", graph_hash_ok, {"checkpoint": stored_hashes, "loaded": live_hashes}),
        _check("alpha_positive_sum_to_one", alpha_ok, alpha),
        _check(
            "scheduled_live_window_includes_issue_date",
            in_live_window,
            {
                "issue_date": str(issue_date.date()),
                "update_date": str(update_date.date()),
                "scheduled_live_start": str(live_start.date()),
                "scheduled_live_end": str(live_end.date()),
                "retrain_frequency_days": frequency,
                "labelled_predict_issue_start": str(labelled_predict_start.date()),
                "labelled_predict_issue_end": str(labelled_predict_end.date()),
                "in_labelled_retrospective_window": in_labelled_window,
            },
        ),
        _check("requested_dates", dates_ok and lead_ok, {"issue": str(issue_date.date()), "target": str(target_date.date()), "lead_days": int((target_date - issue_date).days)}),
        _check("all_111_iz_inputs_through_issue_date", inputs_ok, {"n_complete_lookback_days": n_complete_days, "missing_by_day": missing_by_day}),
        _check("no_post_issue_date_data_used", (not post_issue_used) and dates_used_max == issue_date, {"max_input_date": str(dates_used_max.date()), "issue_date": str(issue_date.date())}),
    ]
    passed = all(item["passed"] for item in checks)
    warnings = []
    if in_live_window and not in_labelled_window:
        warnings.append(
            {
                "code": "beyond_labelled_predict_window",
                "level": LEVEL_ACCEPTED,
                "message": (
                    "U10's labelled retrospective predict window ends "
                    f"{labelled_predict_end.date()} because S1 targets end. "
                    f"Issue {issue_date.date()} is still inside the scheduled "
                    f"{frequency}-day live window {live_start.date()} to {live_end.date()}, "
                    "so U10 remains the un-updated operational checkpoint. No U11 exists."
                ),
            }
        )
    report = {
        "passed": passed,
        "u10_valid_for_issue_date": in_live_window and is_u10 and passed,
        "issue_date": str(issue_date.date()),
        "target_report_date": str(target_date.date()),
        "information_cutoff": str(issue_date.date()),
        "input_start_date": str(lookback_start.date()),
        "input_end_date": str(issue_date.date()),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": ckpt_hash,
        "calibration_path": str(calibration_path),
        "operational_period_rule": (
            "scheduled_live_window = [update_date, update_date + retrain_frequency_days - 1]; "
            "not the truncated labelled predict_issue_end"
        ),
        "checks": checks,
        "warnings": warnings,
        "alpha_label": "Learned relative graph-fusion weights",
        "forecast_status_if_run": "unverified_extrapolation",
        "observed_target_available": False,
        "include_in_test_metrics": False,
    }
    if not passed:
        report["failed_checks"] = [item["name"] for item in checks if not item["passed"]]
    return report


def _write_maps(table: pd.DataFrame, geoshapley: pd.DataFrame | None, alpha: dict[str, float], output_dir: Path) -> dict[str, str]:
    import geopandas as gpd

    shp = _as_path(IZ_BOUNDARY_SHP)
    boundaries = gpd.read_file(shp)
    code_col = next(name for name in (NODE_KEY, "InterZone", "IZ_CODE", "iz_code") if name in boundaries.columns)
    boundaries[NODE_KEY] = boundaries[code_col].astype(str)
    geo = boundaries.merge(table, left_on=NODE_KEY, right_on="iz_code", how="inner")
    if len(geo) != 111:
        raise ModelError(
            f"Map join produced {len(geo)} IZs, expected 111.",
            code="node_order_mismatch",
        )
    paths: dict[str, str] = {}

    def _choropleth(frame, column: str, title: str, filename: str, cmap: str, categorical: bool = False) -> Path:
        fig, ax = plt.subplots(figsize=(7.2, 8.2), constrained_layout=True)
        frame.plot(
            column=column,
            cmap=None if categorical else cmap,
            categorical=categorical,
            legend=True,
            ax=ax,
            edgecolor="white",
            linewidth=0.2,
            missing_kwds={"color": "#dddddd"},
        )
        ax.set_axis_off()
        ax.set_title(title)
        path = output_dir / filename
        fig.savefig(path, dpi=140)
        plt.close(fig)
        return path

    paths["predicted_risk_map"] = str(
        _choropleth(geo, "predicted_mean", "Predicted rolling 7-day rate · 2023-03-04 · per 100,000", "map_predicted_mean.png", "YlOrRd")
    )
    paths["uncertainty_map"] = str(
        _choropleth(geo, "predicted_sigma", "Predictive sigma · 2023-03-04", "map_sigma.png", "PuBu")
    )
    geo["high_uncertainty"] = (geo["uncertainty_flag"] == "high").astype(int)
    paths["high_uncertainty_map"] = str(
        _choropleth(geo, "high_uncertainty", "High-uncertainty flag · 2023-03-04", "map_high_uncertainty.png", "OrRd")
    )

    fig, ax = plt.subplots(figsize=(6.4, 3.8), constrained_layout=True)
    labels = ["geographic", "transport", "mobility"]
    values = [alpha["geo"], alpha["transport"], alpha["mobility"]]
    ax.bar(labels, values, color=["#4C78A8", "#F58518", "#54A24B"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("alpha")
    ax.set_title("Learned relative graph-fusion weights · U10")
    for tick, value in zip(labels, values):
        ax.text(tick, value + 0.02, f"{value:.2f}", ha="center")
    alpha_path = output_dir / "alpha_u10.png"
    fig.savefig(alpha_path, dpi=140)
    plt.close(fig)
    paths["alpha_chart"] = str(alpha_path)

    if geoshapley is not None and not geoshapley.empty:
        mains = geoshapley.loc[geoshapley["component"].isin(["main", "location"])]
        wide = mains.pivot_table(index="iz_code", columns="player_name", values="phi", aggfunc="first").reset_index()
        wide = wide.rename(columns={"iz_code": NODE_KEY})
        geo_phi = boundaries.merge(wide, on=NODE_KEY, how="inner")
        players = [name for name in list(FEATURE_PLAYER_NAMES) + [LOCATION_PLAYER] if name in geo_phi.columns]
        n = len(players)
        cols = 4
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(14, 3.4 * rows), constrained_layout=True)
        axes = np.atleast_1d(axes).ravel()
        for ax, player in zip(axes, players):
            geo_phi.plot(column=player, cmap="RdBu_r", legend=True, ax=ax, edgecolor="white", linewidth=0.1)
            ax.set_axis_off()
            ax.set_title(player)
        for ax in axes[n:]:
            ax.set_axis_off()
        fig.suptitle("GeoShapley main effects · 2023-03-04 prediction")
        geo_path = output_dir / "map_geoshapley_main.png"
        fig.savefig(geo_path, dpi=120)
        plt.close(fig)
        paths["geoshapley_main_maps"] = str(geo_path)

        interactions = geoshapley.loc[geoshapley["component"] == "interaction"]
        if not interactions.empty:
            wide_i = interactions.pivot_table(index="iz_code", columns="player_name", values="phi", aggfunc="first").reset_index()
            wide_i = wide_i.rename(columns={"iz_code": NODE_KEY})
            geo_i = boundaries.merge(wide_i, on=NODE_KEY, how="inner")
            i_players = [col for col in wide_i.columns if col != NODE_KEY]
            n = len(i_players)
            cols = 3
            rows = int(np.ceil(n / cols))
            fig, axes = plt.subplots(rows, cols, figsize=(14, 3.4 * rows), constrained_layout=True)
            axes = np.atleast_1d(axes).ravel()
            for ax, player in zip(axes, i_players):
                geo_i.plot(column=player, cmap="RdBu_r", legend=True, ax=ax, edgecolor="white", linewidth=0.1)
                ax.set_axis_off()
                ax.set_title(player)
            for ax in axes[n:]:
                ax.set_axis_off()
            fig.suptitle("GeoShapley location–feature interactions · 2023-03-04 prediction")
            i_path = output_dir / "map_geoshapley_interactions.png"
            fig.savefig(i_path, dpi=120)
            plt.close(fig)
            paths["geoshapley_interaction_maps"] = str(i_path)
    return paths


GEOSHAPLEY_PLAYER_ORDER = (
    "income_deprivation",
    "employment_deprivation",
    "higher_education",
    "overcrowding",
    "crime",
    "public_transport_time_to_gp",
    "location",
    "location_x_income_deprivation",
    "location_x_employment_deprivation",
    "location_x_higher_education",
    "location_x_overcrowding",
    "location_x_crime",
    "location_x_public_transport_time_to_gp",
)


def _write_iz_outputs(
    *,
    forecast_table: pd.DataFrame,
    map_table: pd.DataFrame,
    geoshapley: pd.DataFrame,
    panel_path: Path,
    output_path: Path,
) -> Path:
    """One row per IZ with forecast, uncertainty, alpha and GeoShapley columns."""
    names = (
        pd.read_csv(panel_path, usecols=[NODE_KEY, "IntZoneName", "node_index"])
        .drop_duplicates(NODE_KEY)
        .rename(columns={NODE_KEY: "IntZone"})
    )
    base = forecast_table.rename(
        columns={
            "iz_code": "IntZone",
            "y_anchor": "observed_anchor_rate",
            "predicted_mu_original": "predicted_mean",
            "predicted_mu_delta": "predicted_delta",
            "predicted_variance_original": "predicted_variance",
            "predicted_sigma_original": "predicted_sigma",
            "model_checkpoint": "u10_checkpoint_id",
            "calibration_artefact": "calibration_artefact_id",
            "target_offset_days": "lead_days",
        }
    )
    extra = map_table.rename(columns={"iz_code": "IntZone"})[
        [
            "IntZone",
            "calibrated80_lower",
            "calibrated80_upper",
            "calibrated95_lower",
            "calibrated95_upper",
            "alpha_geographic",
            "alpha_transport",
            "alpha_mobility",
            "alpha_label",
            "forecast_status",
            "observed_target_available",
            "include_in_test_metrics",
        ]
    ]
    out = base.merge(names, on=["IntZone", "node_index"], how="left").merge(extra, on="IntZone", how="left")
    if not geoshapley.empty:
        phi_wide = geoshapley.pivot_table(index="iz_code", columns="player_name", values="phi", aggfunc="first")
        phi_wide = phi_wide.reindex(columns=list(GEOSHAPLEY_PLAYER_ORDER))
        phi_wide.columns = [f"geoshapley_{name}" for name in phi_wide.columns]
        meta = geoshapley.groupby("iz_code").agg(
            geoshapley_phi0=("phi_0", "first"),
            geoshapley_reconstructed_prediction=("reconstructed_prediction", "first"),
            geoshapley_additivity_error=("additivity_error", "max"),
            geoshapley_n_coalitions=("n_coalitions", "first"),
            geoshapley_scope=("explanation_scope", "first"),
        )
        out = out.merge(phi_wide.reset_index().rename(columns={"iz_code": "IntZone"}), on="IntZone", how="left")
        out = out.merge(meta.reset_index().rename(columns={"iz_code": "IntZone"}), on="IntZone", how="left")
    cols = [
        "IntZone",
        "IntZoneName",
        "node_index",
        "input_start_date",
        "issue_date",
        "target_report_date",
        "lead_days",
        "observed_anchor_rate",
        "predicted_delta",
        "predicted_mean",
        "predicted_variance",
        "predicted_sigma",
        "calibrated80_lower",
        "calibrated80_upper",
        "calibrated95_lower",
        "calibrated95_upper",
        "display_mean",
        "display_lower",
        "display_upper",
        "display_interval_type",
        "uncertainty_flag",
        "uncertainty_threshold",
        "alpha_geographic",
        "alpha_transport",
        "alpha_mobility",
        "alpha_label",
        "geoshapley_phi0",
        *[f"geoshapley_{name}" for name in GEOSHAPLEY_PLAYER_ORDER],
        "geoshapley_reconstructed_prediction",
        "geoshapley_additivity_error",
        "geoshapley_n_coalitions",
        "geoshapley_scope",
        "forecast_status",
        "observed_target_available",
        "include_in_test_metrics",
        "u10_checkpoint_id",
        "calibration_artefact_id",
        "node_order_hash",
    ]
    present = [col for col in cols if col in out.columns]
    table = out[present].sort_values("node_index")
    table.to_csv(output_path, index=False)
    return output_path


def _leading_geoshapley_factors(geoshapley: pd.DataFrame) -> list[dict[str, Any]]:
    if geoshapley.empty:
        return []
    grouped = (
        geoshapley.groupby(["player_name", "component"], as_index=False)
        .agg(mean_phi=("phi", "mean"), mean_abs_phi=("phi", lambda s: float(np.mean(np.abs(s)))))
        .sort_values("mean_abs_phi", ascending=False)
    )
    return grouped.head(8).to_dict(orient="records")


def run_operational_forecast(
    *,
    restore_runtime: Callable,
    predict_residual: Callable,
    explain_all_iz: Callable | None,
    checkpoint_path: Path,
    calibration_path: Path,
    output_dir: Path,
    u10_manifest_path: Path | None = None,
    config_path_override: str | Path | None = None,
    issue_date: str | None = None,
    run_geoshapley: bool = True,
) -> dict[str, Any]:
    """Forecast the unlabelled target after validating U10 for the issue date."""
    cfg = load_model_config(config_path_override)
    tt = temporal_target(cfg)
    checkpoint_path = _as_path(checkpoint_path)
    calibration_path = _as_path(calibration_path)
    output_dir = _as_path(output_dir)
    u10_manifest_path = _as_path(u10_manifest_path or DEFAULT_U10_MANIFEST)
    panel_path = results_dir() / PANEL_CSV
    if not panel_path.is_file():
        raise ModelError(f"Missing panel.csv at {panel_path}", code="missing_dataset")
    issue = pd.Timestamp(issue_date or REQUESTED_ISSUE_DATE).normalize()
    target = issue + pd.Timedelta(days=tt["target_offset_days"])
    output_dir.mkdir(parents=True, exist_ok=True)

    artefact = json.loads(calibration_path.read_text(encoding="utf-8"))
    assert_artefact_matches_checkpoint(artefact, checkpoint_path)
    artefact = dict(artefact)
    artefact.setdefault("uncertainty_flag_enabled", True)
    artefact.setdefault("require_calibration_available", True)
    payload = load_raw_checkpoint(checkpoint_path)
    runtime = restore_runtime(cfg, checkpoint_path)

    validation = validate_u10_for_issue(
        cfg=cfg,
        payload=payload,
        runtime=runtime,
        artefact=artefact,
        checkpoint_path=checkpoint_path,
        calibration_path=calibration_path,
        u10_manifest_path=u10_manifest_path,
        panel_path=panel_path,
        issue_date=issue,
        target_date=target,
    )
    validation_path = output_dir / "validation_report.json"
    validation_path.write_text(json.dumps(validation, indent=2), encoding="utf-8")
    if not validation["passed"]:
        raise ModelError(
            "U10 is incompatible with this operational request. Inference stopped before GeoShapley.",
            code="u10_incompatible_issue_date",
            details={"failed_checks": validation.get("failed_checks"), "validation_report": str(validation_path)},
        )

    dataset = runtime["dataset"]
    split = build_operational_split(
        panel_path,
        dataset.node_order,
        issue_date=issue,
        lookback_days=tt["lookback_steps"],
        target_offset_days=tt["target_offset_days"],
    )
    split = apply_residual_scalers_to_split(split, dataset.residual_scalers)
    preds = predict_residual(runtime, split)
    issue_str = str(issue.date())
    target_str = str(target.date())
    input_start = str((issue - pd.Timedelta(days=tt["lookback_steps"] - 1)).date())
    ckpt_id = sha256_file(checkpoint_path)
    table = build_forecast_table(
        node_order=dataset.node_order,
        issue_dates=[issue_str],
        input_start_dates=[input_start],
        target_dates=[target_str],
        target_offset_days=tt["target_offset_days"],
        mu_z=preds["mu_z"],
        variance_z=preds["variance_z"],
        sigma_z=preds["sigma_z"],
        mu=preds["mu"],
        variance=preds["variance"],
        sigma=preds["sigma"],
        artefact=artefact,
        checkpoint_id=ckpt_id,
        calibration_artefact_id=str(calibration_path),
        y_anchor=preds["y_anchor"],
        mu_delta=preds["mu_delta"],
    )
    q80 = artefact.get("q80")
    q95 = artefact.get("q95")
    if q80 is None or q95 is None:
        raise ModelError("U10 calibration artefact is missing q80 or q95.", code="missing_calibration")
    lo80, hi80 = calibrated_interval(preds["mu"], preds["sigma"], float(q80))
    lo95, hi95 = calibrated_interval(preds["mu"], preds["sigma"], float(q95))
    alpha = _alpha_from_checkpoint(payload)
    map_table = pd.DataFrame(
        {
            "issue_date": issue_str,
            "target_report_date": target_str,
            "iz_code": table["iz_code"],
            "node_index": table["node_index"],
            "observed_anchor_rate_20230225": table["y_anchor"],
            "predicted_delta": table["predicted_mu_delta"],
            "predicted_mean": table["predicted_mu_original"],
            "predicted_variance": table["predicted_variance_original"],
            "predicted_sigma": table["predicted_sigma_original"],
            "calibrated80_lower": lo80.reshape(-1),
            "calibrated80_upper": hi80.reshape(-1),
            "calibrated95_lower": lo95.reshape(-1),
            "calibrated95_upper": hi95.reshape(-1),
            "uncertainty_flag": table["uncertainty_flag"],
            "u10_checkpoint_id": ckpt_id,
            "calibration_artefact_id": str(calibration_path),
            "alpha_geographic": alpha["geo"],
            "alpha_transport": alpha["transport"],
            "alpha_mobility": alpha["mobility"],
            "alpha_label": "Learned relative graph-fusion weights",
            "node_order_hash": dataset.node_order.canonical_hash,
            "forecast_status": "unverified_extrapolation",
            "observed_target_available": False,
            "include_in_test_metrics": False,
        }
    )
    geo_cfg = geoshapley_settings(cfg)
    geo_table = pd.DataFrame()
    if run_geoshapley:
        if explain_all_iz is None:
            raise ModelError("GeoShapley callback is missing.", code="invalid_config")
        geo_table = explain_all_iz(
            runtime,
            split,
            np.asarray([0], dtype=int),
            additivity_tolerance=geo_cfg["additivity_tolerance"],
        )
        geo_table["forecast_status"] = "unverified_extrapolation"
        geo_table["observed_target_available"] = False
        geo_table["include_in_test_metrics"] = False
        geo_table["u10_checkpoint_id"] = ckpt_id
        if "iz_code" in geo_table.columns:
            phi_wide = geo_table.pivot_table(index="iz_code", columns="player_name", values="phi", aggfunc="first")
            phi_wide.columns = [f"geoshapley_{col}" for col in phi_wide.columns]
            map_table = map_table.merge(phi_wide, left_on="iz_code", right_index=True, how="left")
            add_err = geo_table.groupby("iz_code")["additivity_error"].max()
            map_table = map_table.merge(add_err.rename("geoshapley_additivity_error"), left_on="iz_code", right_index=True, how="left")

    map_paths = _write_maps(map_table, geo_table if not geo_table.empty else None, alpha, output_dir)
    iz_path = _write_iz_outputs(
        forecast_table=table,
        map_table=map_table,
        geoshapley=geo_table,
        panel_path=panel_path,
        output_path=output_dir / "iz_outputs.csv",
    )
    geo_path = None
    if not geo_table.empty:
        geo_path = output_dir / "geoshapley.csv"
        geo_table.to_csv(geo_path, index=False)

    high = map_table.loc[map_table["uncertainty_flag"] == "high"].sort_values("predicted_sigma", ascending=False)
    leading = _leading_geoshapley_factors(geo_table)
    graph_files = {
        name: {
            "path": str(runtime["graphs"][name].path),
            "sha256": runtime["graphs"][name].file_sha256,
        }
        for name in runtime["graph_set"]
    }
    provenance = {
        "user_request": USER_REQUEST,
        "information_cutoff": issue_str,
        "input_date_range": [input_start, issue_str],
        "target_date": target_str,
        "lead_days": tt["target_offset_days"],
        "data_source": {
            "panel": str(panel_path),
            "panel_sha256": sha256_file(panel_path),
            "study_area": "City of Edinburgh",
            "geography": "2011 Intermediate Zones",
            "n_iz": dataset.node_order.n_nodes,
        },
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": ckpt_id,
        "selected_epoch": payload.get("selected_epoch"),
        "update_id": "U10",
        "window": "W730",
        "scaler_identifiers": {
            "residual_scalers": payload.get("residual_scalers"),
            "context_scaler_keys": sorted((payload.get("context_scaler") or {}).keys()),
            "coord_scaler_keys": sorted((payload.get("coord_scaler") or {}).keys()),
        },
        "calibration_artefact": {
            "path": str(calibration_path),
            "checkpoint_sha256": artefact.get("checkpoint_sha256"),
            "q80": artefact.get("q80"),
            "q95": artefact.get("q95"),
            "sigma_p90_threshold": artefact.get("sigma_p90_threshold"),
        },
        "graph_files": graph_files,
        "alpha": {
            "label": "Learned relative graph-fusion weights",
            "alpha_geographic": alpha["geo"],
            "alpha_transport": alpha["transport"],
            "alpha_mobility": alpha["mobility"],
        },
        "geoshapley_configuration": geo_cfg,
        "forecast_status": "unverified_extrapolation",
        "observed_target_available": False,
        "include_in_test_metrics": False,
        "no_test_metrics_computed": True,
        "canonical_node_order_hash": dataset.node_order.canonical_hash,
    }
    provenance_path = output_dir / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    summary = {
        "n_iz": int(len(map_table)),
        "predicted_mean_min": float(map_table["predicted_mean"].min()),
        "predicted_mean_max": float(map_table["predicted_mean"].max()),
        "predicted_mean_avg": float(map_table["predicted_mean"].mean()),
        "n_high_uncertainty": int((map_table["uncertainty_flag"] == "high").sum()),
        "highest_uncertainty_iz": high["iz_code"].head(10).tolist(),
        "leading_geoshapley_factors": leading,
        "alpha": provenance["alpha"],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    warning = ModelWarning(
        code="unverified_extrapolation",
        level=LEVEL_ACCEPTED,
        message=(
            f"No observed rolling seven-day rate exists for {target_str}. "
            "forecast_status=unverified_extrapolation; include_in_test_metrics=false. "
            "MAE/RMSE/R²/coverage were not calculated."
        ),
        details={"issue_date": issue_str, "target_report_date": target_str},
    )
    extra_warnings = [
        ModelWarning(code=item["code"], level=item["level"], message=item["message"])
        for item in validation.get("warnings", [])
    ]
    paths = {
        "validation_report": str(validation_path),
        "iz_outputs": str(iz_path),
        "provenance": str(provenance_path),
        "summary": str(output_dir / "summary.json"),
        **map_paths,
    }
    if geo_path is not None:
        paths["geoshapley"] = str(geo_path)
    return {
        "warnings": list(runtime["warnings"]) + extra_warnings + [warning],
        "paths": paths,
        "validation": validation,
        "summary": summary,
        "provenance": provenance,
        "n_iz": int(len(map_table)),
        "device": str(runtime.get("device")),
    }
