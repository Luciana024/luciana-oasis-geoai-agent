"""Choropleth helpers for operational maps.

Map drawing currently lives in model.operational._write_maps.
This module is the presentation entry for future website layers.
"""

from __future__ import annotations

from typing import Any


def available_map_layers() -> list[str]:
    return [
        "predicted_rate",
        "observed_rate",
        "error",
        "predicted_sigma",
        "geoshapley_component",
    ]


def layer_join_key() -> str:
    return "iz_code"
