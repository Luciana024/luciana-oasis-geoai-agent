"""Train the fixed-split residual model. Does not overwrite rolling_v1."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent.tools import train_model


def main() -> None:
    result = train_model()
    print(json.dumps({k: v for k, v in result.items() if k != "outputs"}, indent=2, default=str))
    if result.get("outputs"):
        print(json.dumps(result["outputs"], indent=2, default=str))


if __name__ == "__main__":
    main()
