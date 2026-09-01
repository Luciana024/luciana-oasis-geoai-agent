"""Contextual node embedding from six SIMD variables and projected centroids.

See docs/model.md sections 2, 3 and 8.

SIMD is z-scored across IZs (not across dates) and never by the COVID scaler.
Coordinates are a separate EPSG:27700 z-score. GraphConv consumes only the six
SIMD columns: Z = phi(S (X W) + b). Coordinates enter the local MLP only.

The embedding is static in time. Repeating it across lookback steps aligns
tensors; it does not make embedding dimensions dynamic variables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

from common.errors import LEVEL_ACCEPTED, ModelWarning


@dataclass
class FrozenScaler:
    """Fit once on the training configuration, then freeze for val/test/GeoShapley."""

    names: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray
    epsilon: float
    zero_variance_columns: tuple[str, ...]
    ddof: int = 0

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        scale = np.where(self.std < self.epsilon, 1.0, self.std)
        scaled = (values - self.mean) / scale
        if scaled.ndim == 1:
            for index, name in enumerate(self.names):
                if name in self.zero_variance_columns:
                    scaled[index] = 0.0
            return scaled
        for index, name in enumerate(self.names):
            if name in self.zero_variance_columns:
                scaled[..., index] = 0.0
        return scaled

    def as_dict(self) -> dict[str, Any]:
        return {
            "names": list(self.names),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "epsilon": self.epsilon,
            "zero_variance_columns": list(self.zero_variance_columns),
            "ddof": self.ddof,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FrozenScaler":
        return cls(
            names=tuple(payload["names"]),
            mean=np.asarray(payload["mean"], dtype=np.float64),
            std=np.asarray(payload["std"], dtype=np.float64),
            epsilon=float(payload["epsilon"]),
            zero_variance_columns=tuple(payload.get("zero_variance_columns", ())),
            ddof=int(payload.get("ddof", 0)),
        )


def fit_cross_section_scaler(
    values: np.ndarray,
    names: Sequence[str],
    *,
    epsilon: float = 1e-8,
    ddof: int = 0,
) -> tuple[FrozenScaler, list[ModelWarning]]:
    """Z-score across IZs. Near-zero std maps that column to 0 and keeps the column."""
    values = np.asarray(values, dtype=np.float64)
    names = tuple(names)
    if values.ndim != 2 or values.shape[1] != len(names):
        raise ValueError(f"Expected values shape [N, {len(names)}], got {values.shape}.")
    mean = values.mean(axis=0)
    std = values.std(axis=0, ddof=ddof)
    zero_cols = tuple(names[i] for i, scale in enumerate(std) if scale < epsilon)
    warnings: list[ModelWarning] = []
    if zero_cols:
        warnings.append(
            ModelWarning(
                code="zero_variance_context_column",
                level=LEVEL_ACCEPTED,
                message="One or more context columns have near-zero variance and are mapped to 0.",
                details={"columns": list(zero_cols)},
            )
        )
    scaler = FrozenScaler(
        names=names,
        mean=mean,
        std=std,
        epsilon=epsilon,
        zero_variance_columns=zero_cols,
        ddof=ddof,
    )
    return scaler, warnings


class GraphConv(nn.Module):
    """One graph convolution: Z = phi(S @ (X @ W) + b).

    Never compute S @ W @ X. Bias is added after S multiplies the transformed features.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        transformed = features @ self.weight
        if features.dim() == 2:
            propagated = support @ transformed
        elif features.dim() == 3:
            propagated = torch.einsum("ij,bjf->bif", support, transformed)
        else:
            raise ValueError("GraphConv expects [N, F] or [B, N, F].")
        return self.dropout(torch.relu(propagated + self.bias))


class ContextualEncoder(nn.Module):
    """Local MLP on [SIMD || coords] plus forward/backward GraphConv on SIMD only."""

    def __init__(
        self,
        n_features: int = 6,
        embedding_dim: int = 8,
        n_layers: int = 2,
        dropout: float = 0.1,
        has_location: bool = True,
    ):
        super().__init__()
        self.n_features = n_features
        self.embedding_dim = embedding_dim
        self.has_location = has_location
        local_in = n_features + (2 if has_location else 0)
        layers: list[nn.Module] = []
        in_dim = local_in
        for layer_index in range(n_layers):
            layers.append(nn.Linear(in_dim, embedding_dim))
            if layer_index < n_layers - 1:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
            in_dim = embedding_dim
        self.local_mlp = nn.Sequential(*layers)

        graph_layers_fwd = []
        graph_layers_bwd = []
        graph_in = n_features
        for _ in range(n_layers):
            graph_layers_fwd.append(GraphConv(graph_in, embedding_dim, dropout=dropout))
            graph_layers_bwd.append(GraphConv(graph_in, embedding_dim, dropout=dropout))
            graph_in = embedding_dim
        self.graph_fwd = nn.ModuleList(graph_layers_fwd)
        self.graph_bwd = nn.ModuleList(graph_layers_bwd)
        self.graph_proj = nn.Linear(2 * embedding_dim, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)

    def _stack_graph(self, features: torch.Tensor, support: torch.Tensor, layers: nn.ModuleList) -> torch.Tensor:
        hidden = features
        for layer in layers:
            hidden = layer(hidden, support)
        return hidden

    def forward(
        self,
        simd_scaled: torch.Tensor,
        coords_scaled: torch.Tensor | None,
        support_fwd: torch.Tensor,
        support_bwd: torch.Tensor,
    ) -> torch.Tensor:
        if self.has_location:
            if coords_scaled is None:
                raise ValueError("Location is enabled but coordinates were not provided.")
            local_in = torch.cat([simd_scaled, coords_scaled], dim=-1)
        else:
            local_in = simd_scaled
        z_local = self.local_mlp(local_in)
        z_fwd = self._stack_graph(simd_scaled, support_fwd, self.graph_fwd)
        z_bwd = self._stack_graph(simd_scaled, support_bwd, self.graph_bwd)
        z_graph = self.graph_proj(torch.cat([z_fwd, z_bwd], dim=-1))
        return self.norm(z_local + z_graph)


def diagnose_embedding(embedding: np.ndarray, *, near_constant_std: float = 1e-6) -> dict[str, Any]:
    """Finite-ness, per-dimension mean/std, and near-constant dimension count."""
    values = np.asarray(embedding, dtype=np.float64)
    std = values.std(axis=0)
    return {
        "shape": list(values.shape),
        "finite": bool(np.isfinite(values).all()),
        "dim_mean": values.mean(axis=0).tolist(),
        "dim_std": std.tolist(),
        "n_near_constant_dims": int((std < near_constant_std).sum()),
        "node_std_mean": float(values.std(axis=1).mean()) if values.size else 0.0,
    }
