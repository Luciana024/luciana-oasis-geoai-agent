"""Dispatch: python -m graph geo | python -m graph road | python -m graph mobility."""

from __future__ import annotations

import sys

HELP = """Spatial graphs (same node_index; do not merge the three graphs).

    PYTHONPATH=src python -m graph geo
    PYTHONPATH=src python -m graph road
    PYTHONPATH=src python -m graph mobility
    PYTHONPATH=src python -m graph.geo
    PYTHONPATH=src python -m graph.road
    PYTHONPATH=src python -m graph.mobility
"""


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print(HELP.strip())
        return
    command, rest = args[0], args[1:]
    if command == "geo":
        from graph.geo import main as geo_main

        geo_main(rest)
        return
    if command in {"road", "transport"}:
        from graph.road import main as road_main

        road_main(rest)
        return
    if command in {"mobility", "od"}:
        from graph.mobility import main as mobility_main

        mobility_main(rest)
        return
    raise SystemExit(f"Unknown graph {command!r}. Use geo, road or mobility.")


if __name__ == "__main__":
    main()
