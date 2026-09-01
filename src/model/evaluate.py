"""Calibration artefact and test-period evaluation.

See docs/model.md sections 11, 12 and 15.

q95 and the P90 sigma threshold are computed only on validation_calibration
after the checkpoint is frozen. Test is never used for selection, calibration,
or thresholds. Coverage uses unclipped intervals.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from model.constants import CALIBRATION_METHOD, EXCHANGEABILITY_LIMITATION
from data.dataset import SplitArrays, TemporalDataset, subset_split
from common.errors import LEVEL_REVIEW, ModelError, ModelWarning
from model.heads import (
    RAW80_Z,
    RAW95_Z,
    calibrated_interval,
    finite_sample_empirical_calibration,
    gaussian_nll,
    raw_interval,
)
from model.network import ForecastModel
from model.residual import ResidualScalers, reconstruct_rate_from_delta
from data.node_order import sha256_file


def predict_split(
    model: ForecastModel,
    split: SplitArrays,
    simd_scaled: np.ndarray,
    coords_scaled: np.ndarray | None,
    supports_fwd: np.ndarray,
    supports_bwd: np.ndarray,
    residual_scalers: ResidualScalers,
    *,
    device: torch.device | None = None,
    batch_size: int = 32,
) -> dict[str, np.ndarray]:
    device = device or torch.device("cpu")
    model.eval()
    if split.x_dynamic_model is None or split.y_anchor_raw is None:
        raise ModelError("Residual features are missing on this split.", code="invalid_tensor_shape")
    if residual_scalers is None:
        raise ModelError("Residual scalers are missing.", code="invalid_scaler")
    simd = torch.tensor(simd_scaled, dtype=torch.float32, device=device)
    coords = None if coords_scaled is None else torch.tensor(coords_scaled, dtype=torch.float32, device=device)
    s_fwd = torch.tensor(supports_fwd, dtype=torch.float32, device=device)
    s_bwd = torch.tensor(supports_bwd, dtype=torch.float32, device=device)
    mus, vars_, sigmas = [], [], []
    with torch.no_grad():
        x_all = split.x_dynamic_model
        for start in range(0, x_all.shape[0], batch_size):
            x = torch.tensor(x_all[start : start + batch_size], dtype=torch.float32, device=device)
            outputs = model(x, simd, coords, s_fwd, s_bwd)
            mus.append(outputs["mu"].cpu().numpy())
            vars_.append(outputs["variance"].cpu().numpy())
            sigmas.append(outputs["sigma"].cpu().numpy())
    mu_z = np.concatenate(mus, axis=0)
    var_z = np.concatenate(vars_, axis=0)
    sigma_z = np.concatenate(sigmas, axis=0)
    reconstructed = reconstruct_rate_from_delta(
        mu_z,
        var_z,
        sigma_z,
        delta_scaler=residual_scalers.delta,
        y_anchor=split.y_anchor_raw,
    )
    return reconstructed


def build_calibration_artefact(
    model: ForecastModel,
    dataset: TemporalDataset,
    simd_scaled: np.ndarray,
    coords_scaled: np.ndarray | None,
    supports_fwd: np.ndarray,
    supports_bwd: np.ndarray,
    *,
    checkpoint_path: Path,
    gamma: float = 0.05,
    n_min: int = 20,
    sigma_quantile: float = 0.90,
    sigma_source_split: str = "validation_calibration",
    uncertainty_flag_enabled: bool = True,
    require_calibration_available: bool = True,
    output_path: Path | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    calibration_split = subset_split(dataset.splits["validation"], dataset.validation_calibration_index)
    preds = predict_split(
        model,
        calibration_split,
        simd_scaled,
        coords_scaled,
        supports_fwd,
        supports_bwd,
        dataset.residual_scalers,
        device=device,
    )
    y = calibration_split.y_target_raw
    if sigma_source_split != "validation_calibration":
        raise ModelError(
            "σ quantile must be computed on validation_calibration, not test.",
            code="invalid_config",
            details={"source_split": sigma_source_split},
        )
    valid = np.isfinite(y) & np.isfinite(preds["mu"]) & (preds["sigma"] > 0)
    scores = np.abs(y[valid] - preds["mu"][valid]) / preds["sigma"][valid]
    calibrated = finite_sample_empirical_calibration(scores, gamma=gamma, n_min=n_min)
    warnings: list[dict[str, Any]] = []
    if calibrated["calibration_status"] != "available":
        warnings.append(
            ModelWarning(
                code="calibration_sample_insufficient",
                level=LEVEL_REVIEW,
                message="Calibration sample is below n_min; k was not truncated.",
                details={"n_calibration": calibrated["n_calibration"], "n_min": n_min},
            ).to_dict()
        )
        p90 = None
    else:
        p90 = float(np.quantile(preds["sigma"][valid], sigma_quantile))
    artefact = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "method": CALIBRATION_METHOD,
        "exchangeability_limitation": EXCHANGEABILITY_LIMITATION,
        "calibration_status": calibrated["calibration_status"],
        "n_calibration": calibrated["n_calibration"],
        "n_min": n_min,
        "gamma": gamma,
        "k": calibrated["k"],
        "q95": calibrated["q95"],
        "sigma_p90_threshold": p90,
        "sigma_flag_quantile": float(sigma_quantile),
        "sigma_flag_source_split": sigma_source_split,
        "uncertainty_flag_enabled": bool(uncertainty_flag_enabled),
        "require_calibration_available": bool(require_calibration_available),
        "validation_calibration": dataset.internal_split_provenance,
        "warnings": warnings,
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(artefact, indent=2), encoding="utf-8")
        artefact["path"] = str(output_path)
    return artefact


def build_calibration_from_split(
    model: ForecastModel,
    calibration_split: SplitArrays,
    simd_scaled: np.ndarray,
    coords_scaled: np.ndarray | None,
    supports_fwd: np.ndarray,
    supports_bwd: np.ndarray,
    residual_scalers: ResidualScalers,
    *,
    checkpoint_path: Path,
    gamma_95: float = 0.05,
    gamma_80: float = 0.20,
    n_min: int = 20,
    sigma_quantile: float = 0.90,
    output_path: Path | None = None,
    device: torch.device | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Empirical q80/q95 from a historical calibration block only."""
    preds = predict_split(
        model,
        calibration_split,
        simd_scaled,
        coords_scaled,
        supports_fwd,
        supports_bwd,
        residual_scalers,
        device=device,
    )
    y = calibration_split.y_target_raw
    valid = np.isfinite(y) & np.isfinite(preds["mu"]) & (preds["sigma"] > 0)
    scores = np.abs(y[valid] - preds["mu"][valid]) / preds["sigma"][valid]
    cal95 = finite_sample_empirical_calibration(scores, gamma=gamma_95, n_min=n_min)
    cal80 = finite_sample_empirical_calibration(scores, gamma=gamma_80, n_min=n_min)
    warnings: list[dict[str, Any]] = []
    if cal95["calibration_status"] != "available":
        warnings.append(
            ModelWarning(
                code="calibration_sample_insufficient",
                level=LEVEL_REVIEW,
                message="Rolling calibration sample is below n_min; k was not truncated.",
                details={"n_calibration": cal95["n_calibration"], "n_min": n_min},
            ).to_dict()
        )
        p90 = None
    else:
        p90 = float(np.quantile(preds["sigma"][valid], sigma_quantile))
    artefact = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "method": CALIBRATION_METHOD,
        "exchangeability_limitation": EXCHANGEABILITY_LIMITATION,
        "calibration_status": cal95["calibration_status"],
        "n_calibration": cal95["n_calibration"],
        "n_min": n_min,
        "gamma_95": gamma_95,
        "gamma_80": gamma_80,
        "k95": cal95["k"],
        "q95": cal95["q95"],
        "k80": cal80["k"],
        "q80": cal80["q80"] if "q80" in cal80 else cal80["q95"],
        "sigma_p90_threshold": p90,
        "sigma_flag_quantile": float(sigma_quantile),
        "sigma_flag_source_split": "rolling_calibration",
        "warnings": warnings,
    }
    if extra:
        artefact.update(extra)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(artefact, indent=2), encoding="utf-8")
        artefact["path"] = str(output_path)
    return artefact


def assert_artefact_matches_checkpoint(artefact: dict[str, Any], checkpoint_path: Path) -> None:
    expected = sha256_file(checkpoint_path)
    stored = artefact.get("checkpoint_sha256")
    if stored != expected:
        raise ModelError(
            "Calibration artefact does not belong to the loaded checkpoint.",
            code="calibration_checkpoint_mismatch",
            details={"artefact_sha256": stored, "checkpoint_sha256": expected},
        )


def evaluate_split(
    preds: dict[str, np.ndarray],
    y_raw: np.ndarray,
    artefact: dict[str, Any],
) -> dict[str, Any]:
    """Point and probability metrics on unclipped intervals. Not a leaderboard."""
    valid = np.isfinite(y_raw) & np.isfinite(preds["mu"])
    y = y_raw[valid]
    mu = preds["mu"][valid]
    sigma = preds["sigma"][valid]
    residual = y - mu
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    bias = float(np.mean(residual))
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) if y.size else 0.0
    r2 = None if ss_tot == 0 else float(1.0 - ss_res / ss_tot)
    nll = float(
        gaussian_nll(
            torch.tensor(y_raw, dtype=torch.float32),
            torch.tensor(preds["mu"], dtype=torch.float32),
            torch.tensor(preds["variance"], dtype=torch.float32),
            mask=torch.tensor(valid, dtype=torch.bool),
        ).item()
    )
    raw80_lo, raw80_hi = raw_interval(preds["mu"], preds["sigma"], RAW80_Z)
    raw95_lo, raw95_hi = raw_interval(preds["mu"], preds["sigma"], RAW95_Z)
    metrics = {
        "n_valid_cells": int(valid.sum()),
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "r2": r2,
        "gaussian_nll_original": nll,
        "raw80_coverage": float(np.mean((y_raw[valid] >= raw80_lo[valid]) & (y_raw[valid] <= raw80_hi[valid]))),
        "raw80_mean_width": float(np.mean(raw80_hi[valid] - raw80_lo[valid])),
        "raw95_coverage": float(np.mean((y_raw[valid] >= raw95_lo[valid]) & (y_raw[valid] <= raw95_hi[valid]))),
        "raw95_mean_width": float(np.mean(raw95_hi[valid] - raw95_lo[valid])),
        "calibration_status": artefact.get("calibration_status"),
        "coverage_uses_unclipped_intervals": True,
        "not_a_coverage_guarantee": True,
    }
    if preds.get("y_anchor") is not None:
        persist = float(np.mean(np.abs(y_raw[valid] - preds["y_anchor"][valid])))
        metrics["persistence_mae"] = persist
        metrics["mae_skill"] = None if persist == 0 else float(1.0 - mae / persist)
    if artefact.get("calibration_status") == "available":
        cal_lo, cal_hi = calibrated_interval(preds["mu"], preds["sigma"], float(artefact["q95"]))
        metrics["calibrated95_coverage"] = float(
            np.mean((y_raw[valid] >= cal_lo[valid]) & (y_raw[valid] <= cal_hi[valid]))
        )
        metrics["calibrated95_mean_width"] = float(np.mean(cal_hi[valid] - cal_lo[valid]))
    return metrics
