"""Write allocation tables. Does not invent sites."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from common.utils import project_root, write_json, write_table


def export_allocation(result: dict[str, Any], output_dir: str | Path | None = None) -> dict[str, str]:
    scenario = str(result.get("scenario") or "balanced")
    folder = Path(output_dir) if output_dir is not None else project_root() / "data" / "results" / "allocation" / scenario
    folder.mkdir(parents=True, exist_ok=True)
    sites = pd.DataFrame(result.get("selected_sites") or [])
    assigns = pd.DataFrame(result.get("assignments") or [])
    paths = {
        "selected_sites": str(write_table(sites, folder / "selected_sites.csv") if len(sites) else folder / "selected_sites.csv"),
        "assignments": str(write_table(assigns, folder / "assignments.csv") if len(assigns) else folder / "assignments.csv"),
        "result": str(write_json(result, folder / "result.json")),
    }
    if not len(sites):
        (folder / "selected_sites.csv").write_text("site_id,site_name,site_type\n")
    if not len(assigns):
        (folder / "assignments.csv").write_text("iz_code,site_id,travel_time_min,served\n")
    return paths
