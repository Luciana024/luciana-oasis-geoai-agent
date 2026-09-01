"""Persistence-anchored residual features from frozen S1 raw windows.

Do not rebuild forecast windows. Y_t is the last lookback report.
Delta target is Y_{t+7} - Y_t. First differences are in-window; step 0 is 0.
COVID input uses a train-only global scaler (optional log1p). Delta has its
own train-only scaler. Frozen per-IZ scaler.csv is provenance only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from data.dataset import SplitArrays, TemporalDataset, subset_split
from common.errors import ModelError


@dataclass
class GlobalScaler:
    mean: float
    scale: float
    epsilon: float
    transform: str = "identity"

    def _prepare(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if self.transform == "log1p":
            return np.log1p(np.clip(values, 0.0, None))
        if self.transform == "identity":
            return values
        raise ModelError(f"Unknown scaler transform {self.transform}", code="invalid_config")

    def transform_values(self, values: np.ndarray) -> np.ndarray:
        prepared = self._prepare(values)
        scale = self.scale if self.scale >= self.epsilon else 1.0
        return (prepared - self.mean) / scale

    def inverse_mean_scale(self) -> tuple[float, float]:
        return self.mean, self.scale if self.scale >= self.epsilon else 1.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "mean": float(self.mean),
            "scale": float(self.scale),
            "epsilon": float(self.epsilon),
            "transform": self.transform,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GlobalScaler":
        return cls(
            mean=float(payload["mean"]),
            scale=float(payload["scale"]),
            epsilon=float(payload.get("epsilon", 1e-8)),
            transform=str(payload.get("transform", "identity")),
        )


@dataclass
class ResidualScalers:
    rate: GlobalScaler
    first_difference: GlobalScaler
    delta: GlobalScaler

    def as_dict(self) -> dict[str, Any]:
        return {
            "rate": self.rate.as_dict(),
            "first_difference": self.first_difference.as_dict(),
            "delta": self.delta.as_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ResidualScalers":
        return cls(
            rate=GlobalScaler.from_dict(payload["rate"]),
            first_difference=GlobalScaler.from_dict(payload["first_difference"]),
            delta=GlobalScaler.from_dict(payload["delta"]),
        )


def first_difference_padded(rates: np.ndarray) -> np.ndarray:
    """rates: [B, L, N, 1]. Diffs along lookback; first step is 0."""
    diffs = np.zeros_like(rates, dtype=np.float64)
    diffs[:, 1:, :, :] = rates[:, 1:, :, :] - rates[:, :-1, :, :]
    return diffs


def _fit_global(values: np.ndarray, *, transform: str, epsilon: float, ddof: int) -> GlobalScaler:
    scaler = GlobalScaler(mean=0.0, scale=1.0, epsilon=epsilon, transform=transform)
    prepared = scaler._prepare(values)
    finite = prepared[np.isfinite(prepared)]
    if finite.size == 0:
        raise ModelError("Cannot fit a scaler on an empty finite sample.", code="invalid_scaler")
    mean = float(finite.mean())
    std = float(finite.std(ddof=ddof))
    scale = std if std >= epsilon else 1.0
    return GlobalScaler(mean=mean, scale=scale, epsilon=epsilon, transform=transform)


def _attach_residual_fields(
    split: SplitArrays,
    scalers: ResidualScalers,
) -> SplitArrays:
    rates = split.x_dynamic_raw
    diffs = first_difference_padded(rates)
    rate_z = scalers.rate.transform_values(rates)
    diff_z = scalers.first_difference.transform_values(diffs)
    diff_z[:, 0, :, :] = 0.0
    x_model = np.concatenate([rate_z, diff_z], axis=-1)
    anchor = rates[:, -1, :, :].copy()
    delta = split.y_target_raw - anchor
    delta_z = scalers.delta.transform_values(delta)
    split.x_dynamic_model = x_model
    split.y_anchor_raw = anchor
    split.y_delta_raw = delta
    split.y_delta_scaled = delta_z
    return split


def fit_residual_scalers(split: SplitArrays, cfg: dict[str, Any]) -> ResidualScalers:
    """Fit COVID/delta scalers on one permitted split only. Never use future labels."""
    covid_cfg = cfg.get("covid_scaler") or {}
    delta_cfg = cfg.get("delta_scaler") or {}
    epsilon = float(covid_cfg.get("epsilon", 1e-8))
    ddof = int(covid_cfg.get("ddof", 0))
    rate_transform = str(covid_cfg.get("transform", "log1p"))
    rates = split.x_dynamic_raw
    diffs = first_difference_padded(rates)
    rate_scaler = _fit_global(rates, transform=rate_transform, epsilon=epsilon, ddof=ddof)
    diff_scaler = _fit_global(diffs[:, 1:, :, :], transform="identity", epsilon=epsilon, ddof=ddof)
    anchors = rates[:, -1, :, :]
    deltas = split.y_target_raw - anchors
    delta_scaler = _fit_global(
        deltas,
        transform=str(delta_cfg.get("transform", "identity")),
        epsilon=float(delta_cfg.get("epsilon", epsilon)),
        ddof=int(delta_cfg.get("ddof", ddof)),
    )
    return ResidualScalers(rate=rate_scaler, first_difference=diff_scaler, delta=delta_scaler)


def prepare_residual_dataset(dataset: TemporalDataset, cfg: dict[str, Any]) -> ResidualScalers:
    """Fit global rate/diff/delta scalers on train raw windows, then transform all splits."""
    scalers = fit_residual_scalers(dataset.splits["train"], cfg)
    apply_residual_scalers(dataset, scalers)
    return scalers


def apply_residual_scalers_to_split(split: SplitArrays, scalers: ResidualScalers) -> SplitArrays:
    return _attach_residual_fields(split, scalers)


def apply_residual_scalers(dataset: TemporalDataset, scalers: ResidualScalers) -> None:
    for split in dataset.splits.values():
        _attach_residual_fields(split, scalers)
    dataset.residual_scalers = scalers


def persistence_mae(split: SplitArrays) -> float:
    valid = np.isfinite(split.y_target_raw) & np.isfinite(split.y_anchor_raw)
    return float(np.mean(np.abs(split.y_target_raw[valid] - split.y_anchor_raw[valid])))


def reconstruct_rate_from_delta(
    mu_delta_z: np.ndarray,
    variance_z: np.ndarray,
    sigma_z: np.ndarray,
    *,
    delta_scaler: GlobalScaler,
    y_anchor: np.ndarray,
) -> dict[str, np.ndarray]:
    mean, scale = delta_scaler.inverse_mean_scale()
    mu_delta = mu_delta_z * scale + mean
    variance_delta = variance_z * (scale ** 2)
    sigma_delta = sigma_z * abs(scale)
    mu_rate = y_anchor + mu_delta
    return {
        "mu_delta_z": mu_delta_z,
        "variance_delta_z": variance_z,
        "sigma_delta_z": sigma_z,
        "mu_z": mu_delta_z,
        "variance_z": variance_z,
        "sigma_z": sigma_z,
        "mu_delta": mu_delta,
        "variance": variance_delta,
        "sigma": sigma_delta,
        "mu": mu_rate,
        "y_anchor": y_anchor,
    }


def subset_residual_split(split: SplitArrays, index: np.ndarray) -> SplitArrays:
    out = subset_split(split, index)
    if getattr(split, "x_dynamic_model", None) is not None:
        out.x_dynamic_model = split.x_dynamic_model[index]
        out.y_anchor_raw = split.y_anchor_raw[index]
        out.y_delta_raw = split.y_delta_raw[index]
        out.y_delta_scaled = split.y_delta_scaled[index]
    return out
