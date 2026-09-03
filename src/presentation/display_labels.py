"""Human-readable labels for the planning dashboard. Internal keys stay unchanged."""

from __future__ import annotations

from typing import Any

import pandas as pd

from model.constants import FEATURE_PLAYER_NAMES, LOCATION_PLAYER

SITE_TYPE_LABELS = {
    "gp": "GP",
    "pharmacy": "Pharmacy",
    "mobile_stop": "Mobile stop (car park)",
}

SCENARIO_LABELS = {
    "balanced": "Balanced",
    "coverage priority": "Coverage priority",
    "equity priority": "Equity priority",
    "preventive priority": "Preventive priority",
}

TRAVEL_MODE_LABELS = {
    "drive": "Drive",
    "walk": "Walk",
}

MAP_COLOUR_LABELS = {
    "predicted_rate": "Predicted rate",
    "predicted_sigma": "Uncertainty (σ)",
}

FEATURE_LABELS = {
    "baseline": "Baseline",
    "location": "Location",
    "income_deprivation": "Income deprivation",
    "employment_deprivation": "Employment deprivation",
    "higher_education": "Higher education",
    "overcrowding": "Overcrowding",
    "crime": "Crime",
    "public_transport_time_to_gp": "Public transport time to GP",
    "location_x_income_deprivation": "Location × income deprivation",
    "location_x_employment_deprivation": "Location × employment deprivation",
    "location_x_higher_education": "Location × higher education",
    "location_x_overcrowding": "Location × overcrowding",
    "location_x_crime": "Location × crime",
    "location_x_public_transport_time_to_gp": "Location × public transport time to GP",
}

ALPHA_LABELS = {
    "update_id": "Update",
    "alpha_geo": "Geographic graph",
    "alpha_transport": "Transport graph",
    "alpha_mobility": "Mobility graph",
}

MAP_FIELD_LABELS = {
    "predicted_rate": "Predicted rate",
    "predicted_sigma": "Uncertainty (σ)",
    "uncertainty_flag": "Uncertainty flag",
    "status": "Coverage status",
    "assigned_site": "Assigned site",
    "travel_time": "Travel time (min)",
    "iz_code": "Zone",
}

EXPLANATION_SCOPE_LABEL = "Local explanation for the selected zone"


def label_of(mapping: dict[str, str], key: str) -> str:
    if key in mapping:
        return mapping[key]
    return str(key).replace("_", " ").replace(" x ", " × ")


def invert(mapping: dict[str, str]) -> dict[str, str]:
    return {label: key for key, label in mapping.items()}


def normalize_geoshapley_table(df: pd.DataFrame) -> pd.DataFrame:
    """Accept train-export columns (player_name, phi, phi_0) or website columns."""
    if df is None or df.empty:
        return df
    out = df.copy()
    if "feature_name" not in out.columns and "player_name" in out.columns:
        out["feature_name"] = out["player_name"].astype(str)
    if "shapley_value" not in out.columns and "phi" in out.columns:
        out["shapley_value"] = out["phi"]
    if "predicted_rate" not in out.columns and "reconstructed_prediction" in out.columns:
        out["predicted_rate"] = out["reconstructed_prediction"]
    if "feature_name" not in out.columns or "phi_0" not in out.columns:
        return out
    frames = []
    grouped = out.groupby(out["iz_code"].astype(str), sort=False) if "iz_code" in out.columns else [(None, out)]
    for _, group in grouped:
        if group["feature_name"].astype(str).eq("baseline").any():
            frames.append(group)
            continue
        row = group.iloc[[0]].copy()
        phi0 = float(group["phi_0"].iloc[0])
        row["feature_name"] = "baseline"
        if "player_name" in row.columns:
            row["player_name"] = "baseline"
        row["component"] = "baseline"
        row["shapley_value"] = phi0
        if "phi" in row.columns:
            row["phi"] = phi0
        frames.append(pd.concat([row, group], ignore_index=True))
    return pd.concat(frames, ignore_index=True)


def simple_geoshapley_contributions(df_iz: pd.DataFrame) -> dict[str, Any]:
    """One net contribution per indicator: main effect plus location interaction."""
    df_iz = normalize_geoshapley_table(df_iz)
    values = {
        str(row["feature_name"]): float(row["shapley_value"])
        for _, row in df_iz.iterrows()
        if "feature_name" in df_iz.columns and pd.notna(row.get("shapley_value"))
    }
    rows = []
    for name in FEATURE_PLAYER_NAMES:
        total = values.get(name, 0.0) + values.get(f"location_x_{name}", 0.0)
        rows.append(
            {
                "indicator": name,
                "label": FEATURE_LABELS[name],
                "contribution": total,
                "effect": "Raises forecast" if total >= 0 else "Lowers forecast",
            }
        )
    loc = values.get(LOCATION_PLAYER, 0.0)
    rows.append(
        {
            "indicator": LOCATION_PLAYER,
            "label": FEATURE_LABELS[LOCATION_PLAYER],
            "contribution": loc,
            "effect": "Raises forecast" if loc >= 0 else "Lowers forecast",
        }
    )
    table = pd.DataFrame(rows).sort_values("contribution", ascending=True)
    baseline = values.get("baseline")
    if baseline is None and "phi_0" in df_iz.columns and df_iz["phi_0"].notna().any():
        baseline = float(df_iz["phi_0"].iloc[0])
    predicted = None
    if "predicted_rate" in df_iz.columns and df_iz["predicted_rate"].notna().any():
        predicted = float(df_iz["predicted_rate"].iloc[0])
    return {
        "table": table,
        "baseline": baseline,
        "predicted_rate": predicted,
        "net": float(table["contribution"].sum()) if len(table) else 0.0,
    }


def iz_code_from_map_event(event, iz_codes: list[str]) -> str | None:
    """Read a clicked Intermediate Zone from a Plotly choropleth selection."""
    selection = getattr(event, "selection", None)
    points = getattr(selection, "points", None) if selection is not None else None
    if points is None and isinstance(event, dict):
        points = (event.get("selection") or {}).get("points")
    if not points:
        return None
    known = {str(code) for code in iz_codes}
    for point in points:
        payload = point if isinstance(point, dict) else {}
        if payload.get("curve_number") not in (None, 0):
            continue
        custom = payload.get("customdata")
        if isinstance(custom, (list, tuple)) and custom:
            custom = custom[0]
        loc = payload.get("location")
        for candidate in (custom, payload.get("hovertext"), payload.get("text"), loc):
            if candidate is not None and str(candidate) in known:
                return str(candidate)
        if loc is not None:
            try:
                idx = int(loc)
            except (TypeError, ValueError):
                idx = None
            if idx is not None and 0 <= idx < len(iz_codes):
                return str(iz_codes[idx])
    return None


def site_id_from_map_event(event, plottable: list[dict]) -> str | None:
    """Read a clicked allocated site from Streamlit's Plotly selection."""
    selection = getattr(event, "selection", None)
    points = getattr(selection, "points", None) if selection is not None else None
    if points is None and isinstance(event, dict):
        points = (event.get("selection") or {}).get("points")
    if not points:
        return None
    known = [str(site["site_id"]) for site in plottable]
    known_set = set(known)
    for point in points:
        payload = point if isinstance(point, dict) else {}
        if not payload and point is not None:
            payload = {
                "curve_number": getattr(point, "curve_number", None),
                "customdata": getattr(point, "customdata", None),
                "text": getattr(point, "text", None),
                "hovertext": getattr(point, "hovertext", None),
                "point_index": getattr(point, "point_index", None),
                "point_number": getattr(point, "point_number", None),
            }
        custom = payload.get("customdata")
        while isinstance(custom, (list, tuple)) and custom:
            custom = custom[0]
        idx = payload.get("point_index", payload.get("point_number"))
        curve = payload.get("curve_number")
        for candidate in (custom, payload.get("text")):
            if candidate is not None and str(candidate) in known_set:
                return str(candidate)
        if curve in (None, 0):
            continue
        hover = payload.get("hovertext")
        if hover is not None:
            hover_text = str(hover)
            for site_id in known:
                if site_id and site_id in hover_text:
                    return site_id
        if idx is not None and 0 <= int(idx) < len(plottable):
            return str(plottable[int(idx)]["site_id"])
    return None


GLOSSARY_ROWS = [
    ("Predicted rate", "Forecast rolling 7-day COVID-19 infection rate per 100,000 people. This is not daily new cases."),
    ("Uncertainty (σ)", "Predicted standard deviation of that rate. Larger σ means a less certain forecast."),
    ("Uncertainty flag", "High if σ is above the calibration 90th percentile; otherwise Normal. A map overlay, not a siting score."),
    ("Zone / IZ", "2011 Intermediate Zone. Edinburgh has 111 neighbourhoods. Maps join on the zone code."),
    ("GP", "NHS GP practice used as a candidate vaccination site."),
    ("Pharmacy", "Community pharmacy used as a candidate vaccination site."),
    ("Mobile stop (car park)", "OSM car park used as a possible mobile / pop-up vaccination site."),
    ("Drive / Walk", "Travel mode on the OSM network. Drive assumes 30 km/h; walk assumes 4.5 km/h."),
    ("Travel-time threshold", "An Intermediate Zone is covered if the nearest selected site is within this many minutes."),
    ("Coverage priority", "Choose 6 vaccination sites to cover as many people as possible within the threshold."),
    ("Equity priority", "Prefer income-deprived and currently underserved zones (Scottish Index of Multiple Deprivation income rate and public-transport time to a GP)."),
    ("Preventive priority", "Prefer zones with high predicted rate and/or high uncertainty."),
    ("Balanced", "Combines coverage, equity, and preventive hotspot weights."),
    ("Baseline", "The model's starting point: a reference prediction before this area's own socioeconomic and location details adjust the result up or down."),
    ("Income deprivation", "Scottish Index of Multiple Deprivation share of people who are income-deprived. Explains the forecast; not used as a siting score."),
    ("Employment deprivation", "Scottish Index of Multiple Deprivation employment-deprivation indicator for the zone."),
    ("Higher education", "Scottish Index of Multiple Deprivation higher-education / university-entry indicator."),
    ("Overcrowding", "Scottish Index of Multiple Deprivation overcrowding indicator."),
    ("Crime", "Scottish Index of Multiple Deprivation selected neighbourhood-crime indicator."),
    ("Public transport time to GP", "Scottish Index of Multiple Deprivation public-transport minutes to a GP. Current access, not our OSM travel matrix."),
    ("Location", "Residual spatial contribution of the zone (easting/northing) after the named features."),
    ("Location × …", "How location interacts with that feature. Conditional on the fixed graph structure."),
    ("Geographic / transport / mobility graph", "Learned fusion weights (α) for the three graphs. Relative importance of graph sources, not COVID risk shares."),
    ("Local explanation for the selected zone", "GeoShapley here is target-zone local. It is not causal."),
]
