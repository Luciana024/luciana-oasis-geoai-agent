"""Calibrated and raw Gaussian intervals. Not a coverage guarantee."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from model.constants import CALIBRATION_METHOD, DISPLAY_CALIBRATED, DISPLAY_RAW, EXCHANGEABILITY_LIMITATION, HEAD_NAME

RAW80_Z = 1.2815515655
RAW95_Z = 1.9599639845


def inverse_transform_moments(
    mu_z: np.ndarray,
    variance_z: np.ndarray,
    sigma_z: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Original units. Mean is added only to mu. v *= s^2; sigma *= |s|."""
    mean = np.asarray(mean, dtype=np.float64).reshape(1, -1, 1)
    scale = np.asarray(scale, dtype=np.float64).reshape(1, -1, 1)
    mu = mu_z * scale + mean
    variance = variance_z * (scale ** 2)
    sigma = sigma_z * np.abs(scale)
    return mu, variance, sigma


def raw_interval(mu: np.ndarray, sigma: np.ndarray, z: float) -> tuple[np.ndarray, np.ndarray]:
    return mu - z * sigma, mu + z * sigma


def calibrated_interval(mu: np.ndarray, sigma: np.ndarray, q95: float) -> tuple[np.ndarray, np.ndarray]:
    return mu - q95 * sigma, mu + q95 * sigma


def display_interval(
    mu: np.ndarray,
    raw95_lower: np.ndarray,
    raw95_upper: np.ndarray,
    calibrated_lower: np.ndarray | None,
    calibrated_upper: np.ndarray | None,
    calibration_status: str,
) -> dict[str, Any]:
    """Map display uses calibrated 95% if available, otherwise raw Gaussian 95%. Then clip at 0."""
    if calibration_status == "available" and calibrated_lower is not None and calibrated_upper is not None:
        source_lower, source_upper = calibrated_lower, calibrated_upper
        interval_type = DISPLAY_CALIBRATED
    else:
        source_lower, source_upper = raw95_lower, raw95_upper
        interval_type = DISPLAY_RAW
    display_mean = np.maximum(0.0, mu)
    display_lower = np.maximum(0.0, source_lower)
    display_upper = np.maximum(0.0, source_upper)
    return {
        "display_mean": display_mean,
        "display_lower": display_lower,
        "display_upper": display_upper,
        "display_interval_type": interval_type,
        "mean_clipped": display_mean != mu,
        "lower_bound_clipped": display_lower != source_lower,
    }


def finite_sample_empirical_calibration(
    scores: np.ndarray,
    *,
    gamma: float = 0.05,
    n_min: int = 20,
) -> dict[str, Any]:
    """Finite-sample corrected empirical calibration. Not a conformal coverage guarantee."""
    scores = np.asarray(scores, dtype=np.float64)
    scores = scores[np.isfinite(scores)]
    n_calibration = int(scores.size)
    result = {
        "method": CALIBRATION_METHOD,
        "exchangeability_limitation": EXCHANGEABILITY_LIMITATION,
        "gamma": float(gamma),
        "n_min": int(n_min),
        "n_calibration": n_calibration,
        "k": None,
        "q95": None,
        "calibration_status": "unavailable",
        "reason": None,
        "head_name": HEAD_NAME,
    }
    if n_calibration < n_min:
        result["reason"] = f"n_calibration={n_calibration} < n_min={n_min}"
        return result
    k = int(math.ceil((n_calibration + 1) * (1.0 - gamma)))
    result["k"] = k
    if not 1 <= k <= n_calibration:
        result["reason"] = f"k={k} is outside [1, n={n_calibration}]; k was not truncated"
        return result
    ordered = np.sort(scores)
    result["q95"] = float(ordered[k - 1])
    result["calibration_status"] = "available"
    return result
