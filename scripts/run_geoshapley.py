"""GeoShapley for one IZ. Default is the last retrospective test issue, not 4 March."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent.tools import explain_target_iz_with_geoshapley
from model.operational import DEFAULT_CALIBRATION, DEFAULT_CHECKPOINT


def main() -> None:
    parser = argparse.ArgumentParser(description="GeoShapley for one Edinburgh IZ.")
    parser.add_argument("iz_code", help="Intermediate Zone code, e.g. S02001576")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--calibration", default=DEFAULT_CALIBRATION)
    args = parser.parse_args()
    result = explain_target_iz_with_geoshapley(args.checkpoint, args.calibration, args.iz_code)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
