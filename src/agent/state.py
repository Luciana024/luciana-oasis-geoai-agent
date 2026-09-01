"""Mutable run state. Required user choices are never invented."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentState:
    """Run state. years and data_source must not be invented for covid_prepare.

    forecast_prepare lookback/horizon default to 7 if omitted.
    """

    request: str
    task: str = "covid_prepare"
    years: list[int] = field(default_factory=list)
    data_source: str | None = None  # "api" or "local"; None means ask the user
    lookback_days: int | None = None
    forecast_horizon_days: int | None = None
    area_code: str = "S12000036"
    area_name: str = "City of Edinburgh"
    osm_place: str | None = None
    sites_path: str | None = None
    missing_parameters: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)
    status: str = "initialized"  # initialized | awaiting_user | planned | ok

    def as_dict(self) -> dict[str, Any]:
        return {
            "request": self.request,
            "task": self.task,
            "years": self.years,
            "data_source": self.data_source,
            "lookback_days": self.lookback_days,
            "forecast_horizon_days": self.forecast_horizon_days,
            "area_code": self.area_code,
            "area_name": self.area_name,
            "osm_place": self.osm_place,
            "sites_path": self.sites_path,
            "missing_parameters": self.missing_parameters,
            "warnings": self.warnings,
            "artifacts": self.artifacts,
            "status": self.status,
        }
