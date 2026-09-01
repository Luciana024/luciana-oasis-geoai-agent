"""Wire approved blocks into one encoder-only residual forecast model.

DCRNN reads COVID history only (rate + first difference). Contextual embedding
is concatenated at the prediction head, not copied across lookback steps.
The head predicts μ_Δ, σ_Δ in scaled delta space. Reconstruct
Y_{t+7} = Y_t + μ_Δ outside the network.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from model.context import ContextualEncoder
from model.dcrnn_encoder import DCRNNEncoder
from graph.diffusion import GraphFusion
from model.heads import UnivariateProbabilisticHead

DYNAMIC_INPUT_DIM = 2


class ForecastModel(nn.Module):
    """Probabilistic Adaptive Multi-Graph DCRNN Encoder with Contextual Node Embedding."""

    def __init__(
        self,
        n_graphs: int,
        n_features: int = 6,
        embedding_dim: int = 8,
        hidden_dim: int = 64,
        context_layers: int = 2,
        dcrnn_layers: int = 2,
        diffusion_steps: int = 2,
        dropout: float = 0.1,
        variance_epsilon: float = 1e-6,
        has_location: bool = True,
        use_context: bool = True,
        dynamic_input_dim: int = DYNAMIC_INPUT_DIM,
    ):
        super().__init__()
        self.n_graphs = n_graphs
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.has_location = has_location
        self.use_context = use_context
        self.dynamic_input_dim = dynamic_input_dim
        self.fusion = GraphFusion(n_graphs)
        self.context = ContextualEncoder(
            n_features=n_features,
            embedding_dim=embedding_dim,
            n_layers=context_layers,
            dropout=dropout,
            has_location=has_location,
        )
        self.encoder = DCRNNEncoder(
            input_dim=dynamic_input_dim,
            hidden_dim=hidden_dim,
            n_layers=dcrnn_layers,
            diffusion_steps=diffusion_steps,
        )
        head_in = hidden_dim + (embedding_dim if use_context else 0)
        self.head = UnivariateProbabilisticHead(head_in, variance_epsilon=variance_epsilon)

    def alpha(self) -> torch.Tensor:
        return self.fusion.alpha()

    def embed(
        self,
        simd_scaled: torch.Tensor,
        coords_scaled: torch.Tensor | None,
        supports_fwd: torch.Tensor,
        supports_bwd: torch.Tensor,
        *,
        zero_embedding: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        support_fwd, support_bwd = self.fusion(supports_fwd, supports_bwd)
        if (not self.use_context) or zero_embedding:
            n_nodes = simd_scaled.size(-2)
            embedding = simd_scaled.new_zeros(n_nodes, self.embedding_dim)
            if simd_scaled.dim() == 3:
                embedding = simd_scaled.new_zeros(simd_scaled.size(0), n_nodes, self.embedding_dim)
            return embedding, support_fwd, support_bwd
        embedding = self.context(simd_scaled, coords_scaled, support_fwd, support_bwd)
        return embedding, support_fwd, support_bwd

    def forward(
        self,
        x_covid: torch.Tensor,
        simd_scaled: torch.Tensor,
        coords_scaled: torch.Tensor | None,
        supports_fwd: torch.Tensor,
        supports_bwd: torch.Tensor,
        *,
        zero_embedding: bool = False,
    ) -> dict[str, torch.Tensor]:
        """x_covid: [B, L, N, 2]. Head output is scaled Δ, not absolute rate."""
        embedding, support_fwd, support_bwd = self.embed(
            simd_scaled,
            coords_scaled,
            supports_fwd,
            supports_bwd,
            zero_embedding=zero_embedding,
        )
        hidden = self.encoder(x_covid, support_fwd, support_bwd)
        if self.use_context and not zero_embedding:
            if embedding.dim() == 2:
                context = embedding.unsqueeze(0).expand(hidden.size(0), -1, -1)
            else:
                context = embedding
            hidden = torch.cat([hidden, context], dim=-1)
        outputs = self.head(hidden)
        outputs["embedding"] = embedding
        outputs["alpha"] = self.alpha()
        return outputs

    def config_dict(self) -> dict[str, Any]:
        return {
            "n_graphs": self.n_graphs,
            "n_features": self.context.n_features,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "context_layers": len(self.context.graph_fwd),
            "dcrnn_layers": self.encoder.n_layers,
            "diffusion_steps": self.encoder.cells[0].diffusion_steps,
            "has_location": self.has_location,
            "use_context": self.use_context,
            "dynamic_input_dim": self.dynamic_input_dim,
            "predicts": "delta_from_latest_report",
        }
