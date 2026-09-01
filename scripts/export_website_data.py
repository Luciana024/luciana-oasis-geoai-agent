"""Assemble website and article tables from existing rolling/operational artefacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from presentation.website_export import export_website_and_article


def main() -> None:
    result = export_website_and_article()
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
