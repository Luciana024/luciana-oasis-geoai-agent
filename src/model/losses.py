"""Training losses for the residual probabilistic head."""

from __future__ import annotations

import math

import torch

LOG_TWO_PI = math.log(2.0 * math.pi)


def huber_loss(
    target: torch.Tensor,
    mu: torch.Tensor,
    *,
    delta: float = 1.0,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Masked Smooth-L1 / Huber on the same cells as the NLL."""
    error = target - mu
    abs_error = torch.abs(error)
    quadratic = torch.clamp(abs_error, max=delta)
    linear = abs_error - quadratic
    loss = 0.5 * quadratic ** 2 + delta * linear
    if mask is None:
        return loss.mean()
    valid = mask.to(dtype=loss.dtype)
    if float(valid.sum()) == 0:
        return loss.new_zeros(())
    return (loss * valid).sum() / valid.sum()


def gaussian_nll(
    target: torch.Tensor,
    mu: torch.Tensor,
    variance: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean NLL over valid cells: 0.5 [log(2 pi) + log v + (y-mu)^2 / v]."""
    nll = 0.5 * (LOG_TWO_PI + torch.log(variance) + (target - mu) ** 2 / variance)
    if mask is None:
        return nll.mean()
    valid = mask.to(dtype=nll.dtype)
    if float(valid.sum()) == 0:
        return nll.new_zeros(())
    return (nll * valid).sum() / valid.sum()


def combined_forecast_loss(
    target: torch.Tensor,
    mu: torch.Tensor,
    variance: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    mean_loss_weight: float = 0.5,
    huber_delta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    nll = gaussian_nll(target, mu, variance, mask=mask)
    mean_term = huber_loss(target, mu, delta=huber_delta, mask=mask)
    total = nll + float(mean_loss_weight) * mean_term
    return total, nll, mean_term
