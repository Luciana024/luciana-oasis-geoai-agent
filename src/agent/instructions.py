"""Prompts shown when the agent must stop and ask the user."""

YEAR_PROMPT = (
    "Please specify the COVID report year or year range to retrieve. "
    "Approved neighbourhood extracts exist for 2020, 2021, 2022 and 2023. "
    "The agent will not choose a year for you."
)
SOURCE_PROMPT = (
    "Please choose a data source: 'api' (PHS CKAN) or 'local' (files in data/raw/covid). "
    "The agent will not choose the source for you, and it will not silently switch."
)
TRAVEL_TIME_SOURCE_PROMPT = (
    "Please choose a travel-time source: 'local' (existing travel_time_matrix.csv) "
    "or 'osm' (compute once from OpenStreetMap for the requested city). "
    "The agent will not choose the source for you. Road-graph kilometres are not travel time."
)
CANDIDATE_SITES_SOURCE_PROMPT = (
    "Please choose a candidate-site source: 'local' (existing merged table), "
    "'api' (PHS CKAN GP/pharmacy lists + Geofabrik Scotland OSM), "
    "or 'osm' (local lists + Geofabrik OSM). "
    "The agent will not invent vaccination sites."
)
WINDOW_PROMPT = (
    "Optional: set lookback and horizon (defaults 7 and 7). Horizon is the "
    "lead time to one rolling-seven-day rate, not an H-day cumulative total. "
    "They need not be equal, e.g. --lookback 14 --horizon 7."
)
