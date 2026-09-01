"""Intervention allocation. Solver is pluggable; default does not invent sites."""

from __future__ import annotations

from common.errors import ModelError

from allocation.contracts import N_SITES, SCENARIOS, SCENARIO_LABELS
from allocation.engine import get_solver, greedy_scenario_solver, run_allocation, set_solver
from allocation.validate import validate_allocation_result

_MESSAGE = (
    "This allocation submodule is not wired. Use allocation.run_allocation "
    "or allocation.set_solver. Do not invent sites."
)


def _stop(name: str) -> None:
    raise ModelError(_MESSAGE, code="not_implemented", details={"module": name})


__all__ = [
    "N_SITES",
    "SCENARIOS",
    "SCENARIO_LABELS",
    "greedy_scenario_solver",
    "get_solver",
    "run_allocation",
    "set_solver",
    "validate_allocation_result",
]
