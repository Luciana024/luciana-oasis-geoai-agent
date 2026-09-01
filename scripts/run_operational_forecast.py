"""U10 operational forecast for the unlabelled t+7 target. Does not retrain."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent.tools import export_operational_forecast


def main() -> None:
    result = export_operational_forecast()
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
