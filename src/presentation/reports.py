"""Human-readable reports next to website and article tables."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from common.utils import write_json


def write_json_report(payload: dict[str, Any], path: str | Path) -> Path:
    return write_json(payload, Path(path))
