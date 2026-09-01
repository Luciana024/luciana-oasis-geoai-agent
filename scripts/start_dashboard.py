"""Start the local Streamlit planning dashboard."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "web" / "app.py"


def main() -> None:
    env = os.environ.copy()
    src = str(ROOT / "src")
    current = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src if not current else f"{src}{os.pathsep}{current}"
    raise SystemExit(
        subprocess.call(
            [sys.executable, "-m", "streamlit", "run", str(APP), "--server.headless", "true"],
            cwd=str(ROOT),
            env=env,
        )
    )


if __name__ == "__main__":
    main()
