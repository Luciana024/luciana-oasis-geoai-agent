"""Structured model errors and warning records.

See docs/model.md section 14. Agents must read warning codes, not only status.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# accepted_limitation: continue and disclose
# review_required: human review before treating the result as unqualified
# critical_failure: stop; do not continue silently
LEVEL_ACCEPTED = "accepted_limitation"
LEVEL_REVIEW = "review_required"
LEVEL_CRITICAL = "critical_failure"
VALID_LEVELS = (LEVEL_ACCEPTED, LEVEL_REVIEW, LEVEL_CRITICAL)


class ModelError(Exception):
    """Fatal modelling failure. Stop; do not silently change graphs, IZ order, or alpha."""

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.code = code or "model_error"
        self.details = details or {}


@dataclass
class ModelWarning:
    code: str
    level: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.level not in VALID_LEVELS:
            raise ValueError(f"Unknown warning level: {self.level}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def highest_level(warnings: list[ModelWarning]) -> str | None:
    if any(item.level == LEVEL_CRITICAL for item in warnings):
        return LEVEL_CRITICAL
    if any(item.level == LEVEL_REVIEW for item in warnings):
        return LEVEL_REVIEW
    if warnings:
        return LEVEL_ACCEPTED
    return None


def status_from_warnings(warnings: list[ModelWarning]) -> str:
    """Map warning severity to tool status: failed / ok_with_warnings / ok."""
    if highest_level(warnings) == LEVEL_CRITICAL:
        return "failed"
    if warnings:
        return "ok_with_warnings"
    return "ok"
