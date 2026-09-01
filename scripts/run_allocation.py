"""Run four-scenario allocation from existing forecast, sites and travel time."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from allocation.contracts import N_SITES, SCENARIOS
from allocation.engine import run_allocation
from allocation.export import export_allocation


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Deterministic six-site location allocation.")
    parser.add_argument("--area-code", default="S12000036")
    parser.add_argument("--scenario", choices=list(SCENARIOS), default="balanced")
    parser.add_argument("--travel-mode", choices=["drive", "walk"], default="drive")
    parser.add_argument("--threshold", type=float, default=20.0)
    parser.add_argument("--site-types", default="gp,pharmacy,mobile_stop")
    parser.add_argument("--all-scenarios", action="store_true")
    args = parser.parse_args(argv)
    scenarios = list(SCENARIOS) if args.all_scenarios else [args.scenario]
    payload = {
        "area_code": args.area_code,
        "travel_mode": args.travel_mode,
        "travel_time_threshold_min": args.threshold,
        "eligible_site_types": [part.strip() for part in args.site_types.split(",") if part.strip()],
        "n_sites": N_SITES,
    }
    outputs = []
    for scenario in scenarios:
        result = run_allocation({**payload, "scenario": scenario})
        paths = export_allocation(result)
        outputs.append({"scenario": scenario, "n_sites": result["n_sites_selected"], "metrics": result["metrics"], "paths": paths})
    print(json.dumps(outputs if args.all_scenarios else outputs[0], indent=2, default=str))


if __name__ == "__main__":
    main()
