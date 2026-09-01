"""Start the OASIS GeoAI agent CLI."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent.agent import main

if __name__ == "__main__":
    main()
