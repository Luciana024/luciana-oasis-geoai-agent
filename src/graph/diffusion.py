"""Directed diffusion supports and one global alpha.

See docs/model.md sections 6 and 7.

T is row-stochastic: T = D_out^{-1} A for rows with out-degree > 0.
S = T^T is used as a left multiplier: propagated = S X.
Check T.sum(axis=1) and S.sum(axis=0). Do not require S rows to sum to 1.
Zero out-degree keeps a zero T row / zero S column. Do not fill with a uniform row.

One softmax alpha is shared by the context encoder and the DCRNN encoder.
Do not average forward and backward supports. Do not renormalise after fusion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import nn

from common.errors import LEVEL_ACCEPTED, ModelWarning


@dataclass
class GraphSupports:
    name: str
    adjacency: np.ndarray
    t_fwd: np.ndarray
    t_bwd: np.ndarray
    s_fwd: np.ndarray
    s_bwd: np.ndarray


def _row_normalise(adjacency: np.ndarray, degree: np.ndarray) -> np.ndarray:
    transition = np.zeros_like(adjacency, dtype=np.float64)
    positive = degree > 0
    transition[positive] = adjacency[positive] / degree[positive, None]
    return transition


def directed_supports(adjacency: np.ndarray, *, name: str = "graph") -> GraphSupports:
    """Build T_fwd/T_bwd and S_fwd/S_bwd from a raw directed adjacency."""
    adjacency = np.asarray(adjacency, dtype=np.float64)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(f"{name} adjacency must be square.")
    out_degree = adjacency.sum(axis=1)
    in_degree = adjacency.sum(axis=0)
    t_fwd = _row_normalise(adjacency, out_degree)
    t_bwd = _row_normalise(adjacency.T, in_degree)
    s_fwd = t_fwd.T
    s_bwd = t_bwd.T
    return GraphSupports(
        name=name,
        adjacency=adjacency,
        t_fwd=t_fwd,
        t_bwd=t_bwd,
        s_fwd=s_fwd,
        s_bwd=s_bwd,
    )


def support_checksums(supports: GraphSupports, *, atol: float = 1e-8) -> dict[str, np.ndarray]:
    """Return T row sums and S column sums for tests and reports."""
    return {
        "t_fwd_row_sum": supports.t_fwd.sum(axis=1),
        "t_bwd_row_sum": supports.t_bwd.sum(axis=1),
        "s_fwd_col_sum": supports.s_fwd.sum(axis=0),
        "s_bwd_col_sum": supports.s_bwd.sum(axis=0),
    }


def fuse_supports(supports: Sequence[np.ndarray], alpha: np.ndarray) -> np.ndarray:
    """Weighted sum of supports. Do not row- or column-renormalise the result."""
    if len(supports) != len(alpha):
        raise ValueError("alpha length must match the number of graphs.")
    fused = np.zeros_like(supports[0], dtype=np.float64)
    for weight, matrix in zip(alpha, supports, strict=True):
        fused = fused + float(weight) * matrix
    return fused


def fused_column_sum_warning(
    fused: np.ndarray,
    *,
    name: str,
    atol: float = 1e-6,
) -> ModelWarning | None:
    """Fusion column sums may be < 1 when a graph has a zero out-degree column."""
    col_sum = fused.sum(axis=0)
    short = col_sum < 1.0 - atol
    if not np.any(short):
        return None
    return ModelWarning(
        code="fused_support_column_sum_lt_1",
        level=LEVEL_ACCEPTED,
        message=f"{name} fused support has column sums < 1 because of zero out-degree columns.",
        details={
            "n_short_columns": int(short.sum()),
            "min_column_sum": float(col_sum.min()),
        },
    )


class GraphFusion(nn.Module):
    """One global softmax over graph logits. Same alpha for embedding and DCRNN."""

    def __init__(self, n_graphs: int):
        super().__init__()
        if n_graphs < 1:
            raise ValueError("n_graphs must be >= 1.")
        self.logits = nn.Parameter(torch.zeros(n_graphs))

    def alpha(self) -> torch.Tensor:
        return torch.softmax(self.logits, dim=0)

    def forward(
        self,
        supports_fwd: torch.Tensor,
        supports_bwd: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """supports_*: [G, N, N] -> fused [N, N]. Forward and backward stay separate."""
        weights = self.alpha()
        fused_fwd = torch.einsum("g,gij->ij", weights, supports_fwd)
        fused_bwd = torch.einsum("g,gij->ij", weights, supports_bwd)
        return fused_fwd, fused_bwd
