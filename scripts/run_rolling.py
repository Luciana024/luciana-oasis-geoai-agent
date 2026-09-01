"""Leakage-safe rolling evaluation. Never overwrites the fixed S1 checkpoint."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent.tools import run_rolling_evaluation


def main() -> None:
    result = run_rolling_evaluation()
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
