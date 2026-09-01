"""Single-target residual forecast: one Y_{t+7} distribution per IZ.

Agent-callable wrappers live in agent.tools. This module re-exports them
with a lazy import so model.forecast does not load the tool registry at import.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_checkpoint(*args: Any, **kwargs: Any) -> dict[str, Any]:
    from agent.tools import load_checkpoint as _load_checkpoint

    return _load_checkpoint(*args, **kwargs)


def forecast_single_target(*args: Any, **kwargs: Any) -> dict[str, Any]:
    from agent.tools import forecast_single_target as _forecast_single_target

    return _forecast_single_target(*args, **kwargs)


def restore_runtime(cfg: dict[str, Any], checkpoint_path: Path) -> dict[str, Any]:
    from agent.tools import _restore_runtime

    return _restore_runtime(cfg, checkpoint_path)
