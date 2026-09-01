"""Encoder-only DCRNN (DCGRU) time encoder.

See docs/model.md section 9.

Adapted from DCRNN (Li et al., ICLR 2018):
https://github.com/liyaguang/DCRNN
commit 602afd9d767d3aa1c9b3eac51710d6aeee12c227
MIT License, Copyright (c) 2017 Yaguang Li

This file keeps dual_random_walk diffusion and DCGRU gates. It does not
include the seq2seq decoder or teacher forcing. N is taken from the input
tensor, not hard-coded.

For each support the Chebyshev recurrence starts from the original X^{(0)}:
    X^{(1)} = S X^{(0)}
    X^{(k)} = 2 S X^{(k-1)} - X^{(k-2)}   (k >= 2)
Forward and backward supports are passed separately and never averaged.
"""

from __future__ import annotations

import torch
from torch import nn


class DCGRUCell(nn.Module):
    """One DCGRU step with K-step diffusion on two directed supports."""

    def __init__(self, input_dim: int, hidden_dim: int, diffusion_steps: int = 2):
        super().__init__()
        if diffusion_steps < 1:
            raise ValueError("diffusion_steps must be >= 1.")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.diffusion_steps = diffusion_steps
        # Identity plus K steps for each of two supports.
        self.n_matrices = 2 * diffusion_steps + 1
        concat_dim = (input_dim + hidden_dim) * self.n_matrices
        self.gate_weight = nn.Parameter(torch.empty(concat_dim, 2 * hidden_dim))
        # Official DCGRU initialises gate bias to 1.
        self.gate_bias = nn.Parameter(torch.ones(2 * hidden_dim))
        self.candidate_weight = nn.Parameter(torch.empty(concat_dim, hidden_dim))
        self.candidate_bias = nn.Parameter(torch.zeros(hidden_dim))
        nn.init.xavier_uniform_(self.gate_weight)
        nn.init.xavier_uniform_(self.candidate_weight)

    def _diffuse(self, features: torch.Tensor, supports: list[torch.Tensor]) -> torch.Tensor:
        """Chebyshev diffusion. features: [B, N, F] -> [B, N, F * n_matrices]."""
        batch, n_nodes, feat_dim = features.shape
        # x0: [N, F*B], matching the official dense layout.
        x0 = features.permute(1, 2, 0).reshape(n_nodes, feat_dim * batch)
        terms = [x0]
        for support in supports:
            # Reset to the original features for every support.
            current = support @ x0
            terms.append(current)
            prev = x0
            for _ in range(2, self.diffusion_steps + 1):
                nxt = 2.0 * (support @ current) - prev
                terms.append(nxt)
                prev, current = current, nxt
        stacked = torch.stack(terms, dim=0)
        stacked = stacked.reshape(self.n_matrices, n_nodes, feat_dim, batch)
        stacked = stacked.permute(3, 1, 2, 0).reshape(batch * n_nodes, feat_dim * self.n_matrices)
        return stacked

    def _gconv(
        self,
        inputs: torch.Tensor,
        state: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        supports: list[torch.Tensor],
    ) -> torch.Tensor:
        combined = torch.cat([inputs, state], dim=-1)
        diffused = self._diffuse(combined, supports)
        out = diffused @ weight + bias
        batch, n_nodes, _ = inputs.shape
        return out.reshape(batch, n_nodes, -1)

    def forward(
        self,
        inputs: torch.Tensor,
        state: torch.Tensor,
        support_fwd: torch.Tensor,
        support_bwd: torch.Tensor,
    ) -> torch.Tensor:
        supports = [support_fwd, support_bwd]
        gates = torch.sigmoid(self._gconv(inputs, state, self.gate_weight, self.gate_bias, supports))
        reset, update = torch.chunk(gates, 2, dim=-1)
        candidate = torch.tanh(
            self._gconv(inputs, reset * state, self.candidate_weight, self.candidate_bias, supports)
        )
        return update * state + (1.0 - update) * candidate


class DCRNNEncoder(nn.Module):
    """Stack of DCGRU cells. Returns the last hidden state [B, N, R], not [B, H, N, R]."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        n_layers: int = 2,
        diffusion_steps: int = 2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        cells = []
        for layer_index in range(n_layers):
            layer_in = input_dim if layer_index == 0 else hidden_dim
            cells.append(DCGRUCell(layer_in, hidden_dim, diffusion_steps=diffusion_steps))
        self.cells = nn.ModuleList(cells)

    def forward(
        self,
        inputs: torch.Tensor,
        support_fwd: torch.Tensor,
        support_bwd: torch.Tensor,
    ) -> torch.Tensor:
        """inputs: [B, L, N, F]. support_*: [N, N]."""
        batch, _lookback, n_nodes, _feat = inputs.shape
        states = [
            inputs.new_zeros(batch, n_nodes, self.hidden_dim) for _ in range(self.n_layers)
        ]
        last = None
        for time_index in range(inputs.size(1)):
            current = inputs[:, time_index]
            for layer_index, cell in enumerate(self.cells):
                current = cell(current, states[layer_index], support_fwd, support_bwd)
                states[layer_index] = current
            last = current
        assert last is not None
        return last
