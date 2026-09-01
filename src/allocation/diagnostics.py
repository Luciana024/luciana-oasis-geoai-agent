"""Human-readable allocation diagnostics. Not a siting score."""

from __future__ import annotations

from typing import Any


def selection_diagnostics(
    scenario: str,
    selected: list[str],
    gains: list[float],
    metrics: dict[str, Any],
    weight_note: str,
    warnings: list[str],
    *,
    method: str,
) -> dict[str, Any]:
    return {
        "method": method,
        "n_steps": len(selected),
        "step_gains": [float(value) for value in gains],
        "weight_note": weight_note,
        "scenario": scenario,
        "message": (
            f"Selected {len(selected)} sites by {method} under {scenario} weights. "
            f"{metrics.get('iz_covered')} IZs and {metrics.get('population_covered')} people "
            "are within the travel-time threshold."
        ),
        "warnings": warnings,
    }
