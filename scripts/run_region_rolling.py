"""Rolling-origin evaluation for one new city. Does not overwrite rolling_v1 or Edinburgh U10."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent.region_training import GLASGOW_CA, prepare_region_rolling
from agent.tools import run_rolling_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--area", default=GLASGOW_CA, help="Council area code, e.g. S12000049")
    parser.add_argument("--stage", choices=("plan", "final_test"), default="final_test")
    args = parser.parse_args()
    prepared = prepare_region_rolling(args.area, stage=args.stage)
    result = run_rolling_evaluation(prepared["config_path"])
    print(json.dumps({"prepared": prepared, "rolling": result}, indent=2, default=str))


if __name__ == "__main__":
    main()
