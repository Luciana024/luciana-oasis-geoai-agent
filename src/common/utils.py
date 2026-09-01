"""Shared constants, path helpers and logging for the COVID prepare tools."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

NODE_KEY = "IntZone"
LOCAL_AUTHORITY_CODE = "S12000036"
LOCAL_AUTHORITY_NAME = "City of Edinburgh"
GEOGRAPHY_VINTAGE = "2011"
EXPECTED_IZ_COUNT = 111
SUPPRESSION_FLAG = "c"
SUPPRESSION_FILL_VALUES = (0, 1, 2)
PRIMARY_SCENARIO_FILL = 1
PANEL_CSV = "panel.csv"
KNOWN_QUALITY_FLAGS = frozenset({"", "c"})
RATE_DISCREPANCY_TOLERANCE = 0.5
ALLOWED_YEARS = (2020, 2021, 2022, 2023)
ALLOWED_DATA_SOURCES = ("api", "local")
COVID_COLUMNS = [
    "Date",
    "IntZone",
    "IntZoneName",
    "CA",
    "CAName",
    "Positive7Day",
    "Positive7DayQF",
    "Population",
    "CrudeRate7DayPositive",
    "CrudeRate7DayPositiveQF",
]


def project_root() -> Path:
    """Walk up from this file until pyproject.toml and configs/ are found."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists() and (parent / "configs").exists():
            return parent
    raise FileNotFoundError("Could not locate oasis_geoai_agent project root")


def results_dir() -> Path:
    """Writable output folder for COVID extracts, fill scenarios and reports."""
    return project_root() / "data" / "results"


def load_yaml(relative_path: str) -> dict[str, Any]:
    path = project_root() / relative_path
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def read_table(path: Path) -> pd.DataFrame:
    """Read CSV as strings so suppression flags and empty cells are preserved."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table format: {path}")


def write_table(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=False)
    elif suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        raise ValueError(f"Unsupported table format: {path}")
    return path


def write_json(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def get_logger(name: str = "oasis_geoai_agent") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
    return logger


def write_run_log(event: dict, filename: str = "covid_prepare.jsonl") -> Path:
    """Append one JSON line to logs/ for the audit trail."""
    log_dir = project_root() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / filename
    record = {"timestamp": datetime.now(timezone.utc).isoformat(), **event}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")
    return path
