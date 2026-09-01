"""Map-ready and diagnostic tables.

See docs/model.md section 16. One row per issue date x IZ, never seven rows
per IZ. Display intervals are either calibrated_95 or raw_gaussian_95.
Coverage diagnostics must keep using unclipped intervals from heads.py.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from model.constants import DISPLAY_CALIBRATED, DISPLAY_RAW
from model.heads import (
    RAW80_Z,
    RAW95_Z,
    calibrated_interval,
    display_interval,
    raw_interval,
)
from data.node_order import NodeOrder


FORECAST_COLUMNS = [
    "input_start_date",
    "issue_date",
    "target_report_date",
    "target_offset_days",
    "iz_code",
    "node_index",
    "predicted_mu_z",
    "predicted_variance_z",
    "predicted_sigma_z",
    "predicted_mu_original",
    "predicted_mu_delta",
    "y_anchor",
    "predicted_variance_original",
    "predicted_sigma_original",
    "raw80_lower",
    "raw80_upper",
    "raw95_lower",
    "raw95_upper",
    "calibrated_lower",
    "calibrated_upper",
    "calibration_status",
    "display_mean",
    "display_lower",
    "display_upper",
    "display_interval_type",
    "mean_clipped",
    "lower_bound_clipped",
    "uncertainty_flag",
    "uncertainty_threshold",
    "model_checkpoint",
    "calibration_artefact",
    "node_order_hash",
]


def build_forecast_table(
    *,
    node_order: NodeOrder,
    issue_dates: Sequence[str],
    input_start_dates: Sequence[str],
    target_dates: Sequence[str],
    target_offset_days: int,
    mu_z: np.ndarray,
    variance_z: np.ndarray,
    sigma_z: np.ndarray,
    mu: np.ndarray,
    variance: np.ndarray,
    sigma: np.ndarray,
    artefact: dict[str, Any],
    checkpoint_id: str,
    calibration_artefact_id: str,
    y_anchor: np.ndarray | None = None,
    mu_delta: np.ndarray | None = None,
) -> pd.DataFrame:
    n_samples, n_nodes, _ = mu.shape
    if n_nodes != node_order.n_nodes:
        raise ValueError("Prediction N does not match node_order.")
    status = artefact.get("calibration_status", "unavailable")
    q95 = artefact.get("q95")
    tau = artefact.get("sigma_p90_threshold")
    raw80_lo, raw80_hi = raw_interval(mu, sigma, RAW80_Z)
    raw95_lo, raw95_hi = raw_interval(mu, sigma, RAW95_Z)
    if status == "available" and q95 is not None:
        cal_lo, cal_hi = calibrated_interval(mu, sigma, float(q95))
    else:
        cal_lo, cal_hi = None, None
    display = display_interval(mu, raw95_lo, raw95_hi, cal_lo, cal_hi, status)
    rows: list[dict[str, Any]] = []
    for sample_index in range(n_samples):
        for node_index, iz_code in enumerate(node_order.codes):
            calibrated_lower = None if cal_lo is None else float(cal_lo[sample_index, node_index, 0])
            calibrated_upper = None if cal_hi is None else float(cal_hi[sample_index, node_index, 0])
            sigma_i = float(sigma[sample_index, node_index, 0])
            flag_enabled = bool(artefact.get("uncertainty_flag_enabled", True))
            require_cal = bool(artefact.get("require_calibration_available", True))
            can_flag = flag_enabled and tau is not None
            if require_cal:
                can_flag = can_flag and status == "available"
            if can_flag:
                flag = "high" if sigma_i > float(tau) else "normal"
                threshold: float | None = float(tau)
            else:
                flag = None
                threshold = None
            rows.append(
                {
                    "input_start_date": str(input_start_dates[sample_index]),
                    "issue_date": str(issue_dates[sample_index]),
                    "target_report_date": str(target_dates[sample_index]),
                    "target_offset_days": int(target_offset_days),
                    "iz_code": iz_code,
                    "node_index": int(node_index),
                    "predicted_mu_z": float(mu_z[sample_index, node_index, 0]),
                    "predicted_variance_z": float(variance_z[sample_index, node_index, 0]),
                    "predicted_sigma_z": float(sigma_z[sample_index, node_index, 0]),
                    "predicted_mu_original": float(mu[sample_index, node_index, 0]),
                    "predicted_mu_delta": None
                    if mu_delta is None
                    else float(mu_delta[sample_index, node_index, 0]),
                    "y_anchor": None
                    if y_anchor is None
                    else float(y_anchor[sample_index, node_index, 0]),
                    "predicted_variance_original": float(variance[sample_index, node_index, 0]),
                    "predicted_sigma_original": sigma_i,
                    "raw80_lower": float(raw80_lo[sample_index, node_index, 0]),
                    "raw80_upper": float(raw80_hi[sample_index, node_index, 0]),
                    "raw95_lower": float(raw95_lo[sample_index, node_index, 0]),
                    "raw95_upper": float(raw95_hi[sample_index, node_index, 0]),
                    "calibrated_lower": calibrated_lower,
                    "calibrated_upper": calibrated_upper,
                    "calibration_status": status,
                    "display_mean": float(display["display_mean"][sample_index, node_index, 0]),
                    "display_lower": float(display["display_lower"][sample_index, node_index, 0]),
                    "display_upper": float(display["display_upper"][sample_index, node_index, 0]),
                    "display_interval_type": display["display_interval_type"],
                    "mean_clipped": bool(display["mean_clipped"][sample_index, node_index, 0]),
                    "lower_bound_clipped": bool(display["lower_bound_clipped"][sample_index, node_index, 0]),
                    "uncertainty_flag": flag,
                    "uncertainty_threshold": threshold,
                    "model_checkpoint": checkpoint_id,
                    "calibration_artefact": calibration_artefact_id,
                    "node_order_hash": node_order.canonical_hash,
                }
            )
    table = pd.DataFrame(rows, columns=FORECAST_COLUMNS)
    allowed = {DISPLAY_CALIBRATED, DISPLAY_RAW}
    if not set(table["display_interval_type"].unique()).issubset(allowed):
        raise ValueError("display_interval_type must be calibrated_95 or raw_gaussian_95.")
    return table


def build_geoshapley_table(
    *,
    iz_code: str,
    node_index: int,
    explanation: dict[str, Any],
    node_order_hash: str,
    issue_date: str | None = None,
    target_report_date: str | None = None,
    input_start_date: str | None = None,
) -> pd.DataFrame:
    rows = []
    for component in explanation["components"]:
        rows.append(
            {
                "input_start_date": input_start_date,
                "issue_date": issue_date,
                "target_report_date": target_report_date,
                "iz_code": iz_code,
                "node_index": node_index,
                "player_name": component["player_name"],
                "component": component["component"],
                "phi": component["phi"],
                "phi_0": explanation["phi_0"],
                "reconstructed_prediction": explanation["reconstructed_prediction"],
                "additivity_error": explanation["additivity_error"],
                "explanation_scope": explanation["explanation_scope"],
                "n_coalitions": explanation.get("n_coalitions"),
                "node_order_hash": node_order_hash,
                "label": component.get("label"),
            }
        )
    return pd.DataFrame(rows)


def attach_observed_columns(
    table: pd.DataFrame,
    *,
    node_order: NodeOrder,
    issue_dates: Sequence[str],
    y_raw: np.ndarray,
    artefact: dict[str, Any],
) -> pd.DataFrame:
    """Add observed rate, residual, and unclipped coverage flags when labels exist."""
    status = artefact.get("calibration_status", "unavailable")
    observed = []
    residual = []
    cov80 = []
    cov95 = []
    cov_cal = []
    for sample_index, _issue in enumerate(issue_dates):
        for node_index, _iz in enumerate(node_order.codes):
            y = float(y_raw[sample_index, node_index, 0])
            mu = float(table.iloc[sample_index * node_order.n_nodes + node_index]["predicted_mu_original"])
            lo80 = float(table.iloc[sample_index * node_order.n_nodes + node_index]["raw80_lower"])
            hi80 = float(table.iloc[sample_index * node_order.n_nodes + node_index]["raw80_upper"])
            lo95 = float(table.iloc[sample_index * node_order.n_nodes + node_index]["raw95_lower"])
            hi95 = float(table.iloc[sample_index * node_order.n_nodes + node_index]["raw95_upper"])
            observed.append(y)
            residual.append(y - mu)
            cov80.append(bool(lo80 <= y <= hi80))
            cov95.append(bool(lo95 <= y <= hi95))
            if status == "available":
                lo_c = table.iloc[sample_index * node_order.n_nodes + node_index]["calibrated_lower"]
                hi_c = table.iloc[sample_index * node_order.n_nodes + node_index]["calibrated_upper"]
                cov_cal.append(bool(float(lo_c) <= y <= float(hi_c)) if pd.notna(lo_c) and pd.notna(hi_c) else None)
            else:
                cov_cal.append(None)
    table = table.copy()
    table["observed_rate"] = observed
    table["residual"] = residual
    table["raw80_covered"] = cov80
    table["raw95_covered"] = cov95
    table["calibrated95_covered"] = cov_cal
    return table


def build_embedding_table(embedding: np.ndarray, node_order: NodeOrder, diagnostics: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for node_index, iz_code in enumerate(node_order.codes):
        row = {
            "iz_code": iz_code,
            "node_index": node_index,
            "node_order_hash": node_order.canonical_hash,
        }
        for dim in range(embedding.shape[-1]):
            row[f"z_{dim}"] = float(embedding[node_index, dim])
        rows.append(row)
    table = pd.DataFrame(rows)
    table.attrs["diagnostics"] = diagnostics
    return table
