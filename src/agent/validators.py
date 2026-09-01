"""Parse user text. Do not invent years, source, or windows."""

from __future__ import annotations

import re
from typing import Any

from common.utils import ALLOWED_DATA_SOURCES, ALLOWED_YEARS

YEAR_PATTERN = re.compile(r"(?<!\d)(20[2-3]\d)(?!\d)")
RANGE_PATTERN = re.compile(
    r"(?<!\d)(20[2-3]\d)\s*(?:-|to|至|到)\s*(20[2-3]\d)(?!\d)",
    re.I,
)
PAIR_WINDOW_PATTERN = re.compile(
    r"(?<!\d)(\d+)\s*天?\s*(?:预测|预报|forecast(?:ing)?|predict(?:ing)?|→|->|to|到)\s*(\d+)\s*天?",
    re.I,
)
SINGLE_WINDOW_PATTERN = re.compile(
    r"(?<!\d)(\d+)\s*(?:天(?:窗口|滑窗|预测)?|[- ]?day(?:s)?(?:\s+window)?)",
    re.I,
)


def extract_years(text: str) -> list[int]:
    years: set[int] = set()
    for start, end in RANGE_PATTERN.findall(text or ""):
        low, high = int(start), int(end)
        if low > high:
            low, high = high, low
        years.update(range(low, high + 1))
    years.update(int(match) for match in YEAR_PATTERN.findall(text or ""))
    return sorted(year for year in years if year in ALLOWED_YEARS)


def extract_source(text: str) -> str | None:
    """Return 'api' or 'local' from wording. Ambiguous text returns None."""
    lowered = (text or "").lower()
    api_hits = any(token in lowered for token in ("api", "ckan", "接口", "在线", "网络"))
    local_hits = any(token in lowered for token in ("local", "本地", "离线", "raw"))
    osm_hits = any(token in lowered for token in ("osmnx", "openstreetmap")) or bool(
        re.search(r"(?<![a-z])osm(?![a-z])", lowered)
    )
    if sum(int(flag) for flag in (api_hits, local_hits, osm_hits)) > 1:
        return None
    if osm_hits:
        return "osm"
    if api_hits:
        return "api"
    if local_hits:
        return "local"
    return None


def extract_window(text: str) -> tuple[int, int] | None:
    """Parse lookback and horizon if the user stated them. Does not invent 7/7."""
    pair = PAIR_WINDOW_PATTERN.search(text or "")
    if pair:
        lookback, horizon = int(pair.group(1)), int(pair.group(2))
        if lookback > 0 and horizon > 0:
            return lookback, horizon
    single = SINGLE_WINDOW_PATTERN.search(text or "")
    if single:
        days = int(single.group(1))
        if days > 0:
            return days, days
    return None


def extract_task(text: str) -> str | None:
    lowered = (text or "").lower()
    mapping = (
        ("candidate_sites_prepare", ("candidate_sites_prepare", "candidate_sites", "候选点", "选点")),
        ("travel_time_prepare", ("travel_time_prepare", "travel_time", "出行时间", "时间矩阵")),
        ("inventory", ("inventory", "盘点")),
        ("forecast_prepare", ("forecast_prepare", "forecast", "滑窗", "预测样本")),
        ("covid_prepare", ("covid_prepare", "covid")),
    )
    hits = [task for task, tokens in mapping if any(token in lowered for token in tokens)]
    if len(hits) == 1:
        return hits[0]
    return None


def coerce_years(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (int, str)) and str(value).isdigit():
        value = [int(value)]
    if not isinstance(value, (list, tuple)):
        return []
    years = []
    for item in value:
        try:
            years.append(int(item))
        except (TypeError, ValueError):
            continue
    return sorted({year for year in years if year in ALLOWED_YEARS})


def coerce_source(value: Any) -> str | None:
    if value is None:
        return None
    source = str(value).strip().lower()
    aliases = {
        "api": "api",
        "ckan": "api",
        "online": "api",
        "local": "local",
        "file": "local",
        "files": "local",
        "raw": "local",
        "osm": "osm",
        "osmnx": "osm",
        "openstreetmap": "osm",
    }
    return aliases.get(source)


def coerce_task(value: Any) -> str | None:
    if value is None:
        return None
    task = str(value).strip().lower()
    aliases = {
        "covid": "covid_prepare",
        "covid_prepare": "covid_prepare",
        "inventory": "inventory",
        "forecast": "forecast_prepare",
        "forecast_prepare": "forecast_prepare",
        "travel_time": "travel_time_prepare",
        "travel_time_prepare": "travel_time_prepare",
        "candidate_sites": "candidate_sites_prepare",
        "candidate_sites_prepare": "candidate_sites_prepare",
    }
    return aliases.get(task)


def coerce_positive_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def window_from_payload(payload: dict[str, Any]) -> tuple[int | None, int | None]:
    window = coerce_positive_int(payload.get("window") or payload.get("window_days"))
    lookback = coerce_positive_int(
        payload.get("lookback_days") if payload.get("lookback_days") is not None else payload.get("lookback")
    )
    horizon = coerce_positive_int(
        payload.get("forecast_horizon_days")
        if payload.get("forecast_horizon_days") is not None
        else payload.get("horizon")
    )
    if window is not None:
        lookback = lookback if lookback is not None else window
        horizon = horizon if horizon is not None else window
    return lookback, horizon
