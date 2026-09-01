"""UQGNN-inspired univariate probabilistic prediction head.

See docs/model.md sections 10 and 11.

This is a 1x1 Gaussian head, not a full multivariate UQGNN. The name must
remain 'UQGNN-inspired univariate probabilistic prediction head'.
Primary uncertainty is this Gaussian, not MC Dropout.
"""

from __future__ import annotations

import torch
from torch import nn

from explain.uncertainty import (
    RAW80_Z,
    RAW95_Z,
    calibrated_interval,
    display_interval,
    finite_sample_empirical_calibration,
    inverse_transform_moments,
    raw_interval,
)
from model.losses import LOG_TWO_PI, combined_forecast_loss, gaussian_nll, huber_loss

__all__ = [
    "RAW80_Z",
    "RAW95_Z",
    "LOG_TWO_PI",
    "UnivariateProbabilisticHead",
    "calibrated_interval",
    "combined_forecast_loss",
    "display_interval",
    "finite_sample_empirical_calibration",
    "gaussian_nll",
    "huber_loss",
    "inverse_transform_moments",
    "raw_interval",
]


class UnivariateProbabilisticHead(nn.Module):
    """Maps H_final [B, N, R] to mu, variance, sigma in scaled delta space, each [B, N, 1]."""

    def __init__(self, hidden_dim: int, variance_epsilon: float = 1e-6):
        super().__init__()
        self.variance_epsilon = variance_epsilon
        self.mean_head = nn.Linear(hidden_dim, 1)
        self.variance_head = nn.Linear(hidden_dim, 1)

    def forward(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        mu = self.mean_head(hidden)
        variance = torch.nn.functional.softplus(self.variance_head(hidden)) + self.variance_epsilon
        sigma = torch.sqrt(variance)
        return {"mu": mu, "variance": variance, "sigma": sigma}
