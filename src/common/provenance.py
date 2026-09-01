"""Write audit JSON next to tool outputs."""

from pathlib import Path
from typing import Any

from common.utils import write_json, write_run_log


def write_provenance(payload: dict[str, Any], path: str | Path) -> Path:
    """Write a provenance JSON file. Does not modify source data."""
    return write_json(payload, Path(path))


def append_run_event(event: dict[str, Any], filename: str = "agent.jsonl") -> Path:
    return write_run_log(event, filename=filename)
