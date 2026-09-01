"""Graph-fusion alpha: learned mixing weights over geo / transport / mobility.

Alpha is positive and sums to 1. It is not a risk share and not an allocation weight.
"""

from __future__ import annotations

from typing import Any

import torch.nn.functional as F


def alpha_from_checkpoint(payload: dict[str, Any]) -> dict[str, float]:
    """Read the three-graph softmax weights stored on a checkpoint."""
    logits = payload["model_state_dict"]["fusion.logits"].float()
    alpha = F.softmax(logits, dim=0).detach().cpu().numpy()
    names = list(payload["graph_set"])
    return {name: float(alpha[i]) for i, name in enumerate(names)}


def alpha_from_checkpoint_path(path: str) -> dict[str, float]:
    import torch
    from pathlib import Path

    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    return alpha_from_checkpoint(payload)
