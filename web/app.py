"""Edinburgh planning dashboard, adapted from ayushdabra/GeoMind.

Numbers come from the local OASIS planning agent, not the placeholder solver.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import importlib

import geopandas as gpd
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import agent.dashboard_bridge as _dashboard_bridge

_dashboard_bridge = importlib.reload(_dashboard_bridge)

from agent.dashboard_bridge import (
    AGENT_GREETING,
    AGENT_HELP,
    EDINBURGH_CA,
    GLASGOW_CA,
    GLASGOW_NAME,
    PlanningAgent,
    STUDY_AREA_OPTIONS,
    brief_fingerprint,
    hydrate_region_ui,
    map_study_area,
    narrate_planning,
    parse_agent_intent,
    extract_forecast_date,
    looks_like_allocation,
    looks_like_compare,
    looks_like_forecast,
    looks_like_plan,
    match_zone_names,
    text_wants_iz_view,
    text_wants_region_view,
    plan_from_brief_text,
    map_scenario,
)
from agent.region_training import region_artefacts_ready, write_region_iz_boundaries
from allocation.contracts import (
    BOUNDARIES_GEOJSON,
    DATE_SELECTOR,
    DEMO_FORECAST_DATE,
    GEOSHAPLEY_CACHE,
    MODEL_METADATA,
    ROLLING_ALPHA,
)
from common.utils import project_root
from presentation.display_labels import (
    ALPHA_LABELS,
    EXPLANATION_SCOPE_LABEL,
    FEATURE_LABELS,
    GLOSSARY_ROWS,
    MAP_COLOUR_LABELS,
    MAP_FIELD_LABELS,
    SCENARIO_LABELS,
    SITE_TYPE_LABELS,
    TRAVEL_MODE_LABELS,
    invert,
    label_of,
    simple_geoshapley_contributions,
    site_id_from_map_event,
    iz_code_from_map_event,
)
import presentation.ui_i18n as _ui_i18n

_ui_i18n = importlib.reload(_ui_i18n)
from presentation.ui_i18n import (
    city_display,
    feature_display,
    glossary_rows,
    scenario_display,
    site_type_display,
    t,
    travel_display,
)

st.set_page_config(
    page_title="Vaccination Site Planning Agent",
    layout="wide",
    initial_sidebar_state="collapsed",
)

AGENT_AVATAR = Path(__file__).resolve().parent / "assets" / "agent_robot.png"
USER_AVATAR = Path(__file__).resolve().parent / "assets" / "user_avatar.png"
UOFG_CREST = Path(__file__).resolve().parent / "assets" / "uofg_crest.png"


def plain_agent_text(text: str) -> str:
    """Strip internal names from chat copy. Local so Streamlit does not depend on a stale dashboard_bridge reload."""
    try:
        from agent.dashboard_bridge import plain_agent_text as _plain

        if _plain is not plain_agent_text:
            return _plain(text)
    except ImportError:
        pass
    out = str(text or "")
    for old, new in (
        ("Edinburgh U10 checkpoint", "Edinburgh's saved model"),
        ("frozen Edinburgh U10 checkpoint", "Edinburgh's saved model"),
        ("Edinburgh U10", "Edinburgh's saved model"),
        ("**U10** checkpoint", "saved model"),
        ("**U10**", "saved model"),
        ("U10 checkpoint", "saved model"),
        ("U10", "saved model"),
        ("rolling_v1", "Edinburgh files"),
        ("checkpoint_used", "model"),
        ("checkpoint_id", "model"),
        ("checkpoint", "model"),
        ("Checkpoint", "Model"),
        ("existing trained model", "saved model"),
        ("trained model", "saved model"),
        ("Confirm training", "Allow a new city model"),
        ("confirm training", "allow a new city model"),
        ("I did not retrain", "I did not rebuild it"),
        ("I do not retrain", "I do not rebuild the model"),
        ("will not train silently", "will not build a model unless you ask"),
        ("before I train", "before I build one"),
        ("I trained a model", "I built a model"),
        ("Trained a new model", "Built a model"),
        ("No new training", "The model was not rebuilt"),
        ("training-set", "reference"),
        ("training set", "reference"),
    ):
        out = out.replace(old, new)
    return out


def format_agent_markdown(text: str) -> str:
    """Keep 1./2./3./4. as separate markdown paragraphs so they do not wrap into one line."""
    out = plain_agent_text(text)
    out = out.replace("干预点", "疫苗接种点").replace("intervention sites", "vaccination sites")
    out = out.replace("Intervention sites", "Vaccination sites")
    return re.sub(r"\s*(?=\*\*[1-4]\. )", "\n\n", out).strip()


def _lang() -> str:
    return "zh" if st.session_state.get("ui_lang") == "zh" else "en"


_TX_FALLBACK = {
    "btn_scope_region": {
        "en": "Whole-region view (all Intermediate Zones)",
        "zh": "全区域展示（整座城市的全部中间区）",
    },
    "btn_scope_iz": {
        "en": "Selected Intermediate Zone view",
        "zh": "按所选中间区（Intermediate Zone）展示",
    },
    "btn_switch_region": {
        "en": "Back to whole-region view (all Intermediate Zones)",
        "zh": "改回全区域展示（全部中间区）",
    },
    "btn_another_zone": {
        "en": "Choose another Intermediate Zone",
        "zh": "重新选择中间区（Intermediate Zone）",
    },
    "scope_offer": {
        "en": "Choose a view first. The matching maps appear after you select one:",
        "zh": "请先选择展示方式，选完后才会出现对应结果：",
    },
    "btn_home": {
        "en": "Back to home",
        "zh": "回到初始主页",
    },
    "btn_input_type": {
        "en": "I will type my answer",
        "zh": "自己输入诉求",
    },
    "btn_input_buttons": {
        "en": "Show me the agent's options",
        "zh": "点选 Luciana 给出的范围",
    },
    "input_mode_offer": {
        "en": "How do you want to answer?",
        "zh": "请先选一种作答方式：",
    },
    "input_mode_type_scenario": {
        "en": "Type a policy: Coverage priority, Equity priority, Preventive priority, or Balanced.",
        "zh": "请输入政策：覆盖优先、公平优先、预防优先或均衡。",
    },
    "input_mode_type_travel": {
        "en": "Type a travel mode and a time limit, for example \"walk 20 minutes\" or \"drive 15\".",
        "zh": "请输入出行方式和时限，例如「步行 20 分钟」或「驾车 15」。",
    },
    "input_mode_type_scope": {
        "en": "Type whole-region view, or name one Intermediate Zone.",
        "zh": "请输入「全区域展示」，或写出一个中间区名称。",
    },
    "type_box_caption": {
        "en": "Type your answer in the box:",
        "zh": "请在输入框里填写：",
    },
    "placeholder_type_scenario": {
        "en": "e.g. Preventive priority…",
        "zh": "例如：预防优先…",
    },
    "placeholder_type_travel": {
        "en": "e.g. walk 20 minutes…",
        "zh": "例如：步行 20 分钟…",
    },
    "placeholder_type_scope": {
        "en": "e.g. whole region, or an Intermediate Zone name…",
        "zh": "例如：全区域，或一个中间区名称…",
    },
    "scenario_question": {
        "en": (
            "I can place exactly six vaccination sites. Which policy should I use? "
            "Type it below, or press Enter on an empty box to see the four policies."
        ),
        "zh": (
            "我将放置 6 个疫苗接种点。要用哪种政策？"
            "请在下方输入；若想看四种政策，把输入框留空后回车。"
        ),
    },
    "show_policy_options": {
        "en": "Here are the four policies. Click one:",
        "zh": "这是四种政策，请点选一项：",
    },
    "show_travel_options": {
        "en": "Here are the travel options. Click Drive or Walk, then a time limit:",
        "zh": "这是出行选项。请先点选驾车或步行，再点选时限：",
    },
    "show_scope_options": {
        "en": "Here are the two views. Click one:",
        "zh": "这是两种展示方式，请点选一项：",
    },
    "show_zone_options": {
        "en": "Here is the full Intermediate Zone list. Pull one neighbourhood:",
        "zh": "这是完整中间区名单，请下拉选择一个街区：",
    },
    "zone_question": {
        "en": (
            "Type an Intermediate Zone name in the box below, or press Enter on an empty box "
            "to open the full list."
        ),
        "zh": "请在下方输入中间区名称；若想打开完整名单，把输入框留空后回车。",
    },
    "title_agent": {"en": "Vaccination Site Planning Agent", "zh": "疫苗接种点规划助手"},
    "label_agent": {"en": "Luciana", "zh": "Luciana"},
    "continue_agent": {"en": "Continue with Luciana", "zh": "继续与 Luciana 对话"},
    "title_result": {"en": "Vaccination planning result", "zh": "接种点规划结果"},
    "home_coverage": {
        "en": (
            "**Latest forecast date:** {latest}  \n"
            "That day is a 7-day extrapolation from the last observed panel; there is no ground truth on that date.\n\n"
            "**Cities supported now:** City of Edinburgh · Glasgow City"
        ),
        "zh": (
            "**最新预测日期：** {latest}  \n"
            "该日为最后观测日后的 7 日外推，当天没有观测真值。\n\n"
            "**目前支持的城市：** 爱丁堡（City of Edinburgh）· 格拉斯哥（Glasgow City）"
        ),
    },
}


def _tx(key: str, **kwargs) -> str:
    out = t(_lang(), key, **kwargs)
    if out == key:
        row = _TX_FALLBACK.get(key) or {}
        out = row.get(_lang()) or row.get("en") or out
        if kwargs:
            try:
                return out.format(**kwargs)
            except (KeyError, IndexError):
                return out
    return out


def _note_user_language(text: str) -> None:
    try:
        from presentation.ui_i18n import detect_ui_lang
    except ImportError:
        return
    detected = detect_ui_lang(text)
    if not detected:
        return
    st.session_state.ui_lang = detected
    if not st.session_state.get("messages"):
        return
    first = st.session_state.messages[0]
    if first.get("role") != "assistant":
        return
    raw = str(first.get("text") or "")
    markers = (
        "Luciana",
        "Hexa",
        "I am the health planning agent.",
        "我是卫生规划助手",
        "I plan **six vaccination",
        "我根据感染预测",
    )
    if any(mark in raw for mark in markers):
        first["text"] = t(detected, "greeting")


def _rewrite_stale_site_copy() -> None:
    """Replace old '干预点' bubbles already stored in this browser session."""
    swaps = (
        ("干预点", "疫苗接种点"),
        ("Intervention sites", "Vaccination sites"),
        ("intervention sites", "vaccination sites"),
        ("已分配干预点", "疫苗接种点"),
        ("这 6 个疫苗接种点不是我选的。\n\n", ""),
        ("这 6 个疫苗接种点不是我选的。", ""),
        ("I did not choose the six vaccination sites.\n\n", ""),
        ("I did not choose the six vaccination sites.", ""),
        ("I did not choose the sites.", ""),
        ("I did not pick any of these sites.", ""),
        ("I do not choose sites.\n\n", ""),
        ("I do not choose the IDs.\n\n", ""),
        ("I do not choose the site IDs.\n\n", ""),
        (" I do not choose the site IDs.", ""),
        (" I still do not pick the sites.", ""),
        (" The solver chooses the IDs; I do not.", ""),
        (" I do not pick sites.", ""),
        (" I do not pick the zone.", ""),
        ("选址编号不由我决定。\n\n", ""),
        ("编号不由我选。", ""),
        ("编号由求解器决定，不由我选。", ""),
        ("我仍然不选点。", ""),
        ("街区不由我选。", ""),
        ("点位不是我选的。", ""),
        ("这些接种点都不是我选的。", ""),
        ("我不会选点。", ""),
        ("但政策不由我选。", ""),
    )
    for message in st.session_state.get("messages") or []:
        text = str(message.get("text") or "")
        updated = text
        for old, new in swaps:
            updated = updated.replace(old, new)
        if updated != text:
            message["text"] = updated
    st.session_state.site_label_version = 3


def _website(rel: str) -> Path:
    return project_root() / rel


def _attach_iz_names(gdf):
    """Keep official Intermediate Zone names for display. Codes stay for joins."""
    out = gdf.copy()
    if "iz_code" not in out.columns:
        for col in ("IntZone", "InterZone", "IZ_CODE"):
            if col in out.columns:
                out["iz_code"] = out[col].astype(str)
                break
    name_col = next(
        (col for col in ("iz_name", "IntZoneName", "Name", "IZ_NAME") if col in out.columns),
        None,
    )
    if name_col:
        out["iz_name"] = out[name_col].astype(str).str.strip()
    elif "iz_name" not in out.columns:
        out["iz_name"] = out["iz_code"].astype(str)
    blank = out["iz_name"].isna() | out["iz_name"].isin(["", "nan", "None"])
    out.loc[blank, "iz_name"] = out.loc[blank, "iz_code"].astype(str)
    return out


def _iz_label(code: str) -> str:
    frame = globals().get("gdf_plot")
    if frame is None or "iz_name" not in getattr(frame, "columns", []):
        return str(code)
    hit = frame.loc[frame["iz_code"].astype(str) == str(code), "iz_name"]
    if hit.empty:
        return str(code)
    name = str(hit.iloc[0]).strip()
    return name or str(code)


@st.cache_data
def load_base_data():
    gdf_zones = gpd.read_file(_website(BOUNDARIES_GEOJSON))
    if "iz_code" not in gdf_zones.columns and "IntZone" in gdf_zones.columns:
        gdf_zones = gdf_zones.rename(columns={"IntZone": "iz_code"})
    gdf_zones = _attach_iz_names(gdf_zones)
    if gdf_zones.crs is not None and gdf_zones.crs.to_epsg() != 4326:
        gdf_zones = gdf_zones.to_crs(epsg=4326)

    dates_path = _website(DATE_SELECTOR)
    df_dates = pd.read_csv(dates_path) if dates_path.exists() else pd.DataFrame()

    shapley_path = _website(GEOSHAPLEY_CACHE)
    df_shapley = pd.read_csv(shapley_path) if shapley_path.exists() else pd.DataFrame()

    alpha_path = _website(ROLLING_ALPHA)
    df_alpha = pd.read_csv(alpha_path) if alpha_path.exists() else pd.DataFrame()

    meta_path = _website(MODEL_METADATA)
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return gdf_zones, df_dates, df_shapley, df_alpha, metadata


@st.cache_resource
def get_agent():
    return PlanningAgent()


def run_agent_pipeline(
    study_area: str,
    forecast_date: str,
    scenario: str,
    travel_mode: str,
    travel_threshold: float,
    eligible_types: tuple[str, ...],
    user_confirmed_retraining: bool = False,
):
    return PlanningAgent().run_planning_pipeline(
        study_area=study_area,
        forecast_date=forecast_date,
        scenario=scenario,
        travel_mode=travel_mode,
        travel_time_threshold=travel_threshold,
        eligible_site_types=list(eligible_types),
        priority_population="Total Population",
        user_confirmed_retraining=user_confirmed_retraining,
    )


gdf_zones, df_dates, df_shapley, df_alpha, _metadata = load_base_data()
agent = get_agent()
_scenario_keys = invert(SCENARIO_LABELS)
_travel_keys = invert(TRAVEL_MODE_LABELS)
_site_keys = invert(SITE_TYPE_LABELS)
_colour_keys = invert(MAP_COLOUR_LABELS)


MAP_CENTERS = {
    EDINBURGH_CA: {"lat": 55.935, "lon": -3.220},
    GLASGOW_CA: {"lat": 55.864, "lon": -4.252},
}
MAP_CENTER = MAP_CENTERS[EDINBURGH_CA]


@st.cache_data
def load_zone_boundaries(area_code: str):
    if area_code == EDINBURGH_CA:
        gdf_zones, *_rest = load_base_data()
        return gdf_zones
    from agent.region_training import region_boundaries_path

    cache = region_boundaries_path(area_code)
    if not cache.is_file() or cache.stat().st_size == 0:
        cache = write_region_iz_boundaries(area_code)
    gdf = gpd.read_file(cache)
    gdf = _attach_iz_names(gdf)
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    return gdf.reset_index(drop=True)


def _map_labels() -> dict[str, str]:
    return {
        "Predicted rate": _tx("predicted_rate"),
        "Uncertainty (σ)": _tx("uncertainty"),
        "Uncertainty flag": _tx("uncertainty_flag"),
        "Coverage": _tx("coverage"),
        "Assigned site": _tx("assigned_site"),
        "Travel time (min)": _tx("travel_time"),
        "Covered": _tx("covered"),
        "Unserved": _tx("unserved_label"),
        "iz_name": _tx("neighbourhood"),
        "iz_code": _tx("iz_code"),
    }


def _choropleth(gdf, color, hover_data, *, discrete=False, scale="Viridis"):
    if gdf is None or getattr(gdf, "empty", True) or color not in getattr(gdf, "columns", []):
        return go.Figure()
    plot = gdf.reset_index(drop=True)
    hover = dict(hover_data or {})
    hover.setdefault("iz_code", False)
    hover.setdefault("iz_name", False)
    hover_name = "iz_name" if "iz_name" in plot.columns else "iz_code"
    use_code = "iz_code" in plot.columns
    kwargs = dict(
        geojson=plot.__geo_interface__,
        locations=plot["iz_code"] if use_code else plot.index,
        color=color,
        zoom=10.2,
        center=MAP_CENTER,
        opacity=0.65,
        hover_name=hover_name,
        hover_data=hover,
        labels=_map_labels(),
    )
    if use_code:
        kwargs["featureidkey"] = "properties.iz_code"
        kwargs["custom_data"] = ["iz_code"]
    if discrete:
        covered_map = {"Covered": "#2A9D8F", "Unserved": "#9B2226"}
        if hasattr(px, "choropleth_map"):
            fig = px.choropleth_map(
                plot,
                map_style="carto-positron",
                color_discrete_map=covered_map,
                **kwargs,
            )
        else:
            fig = px.choropleth_mapbox(
                plot,
                mapbox_style="carto-positron",
                color_discrete_map=covered_map,
                **kwargs,
            )
    elif hasattr(px, "choropleth_map"):
        fig = px.choropleth_map(plot, map_style="carto-positron", color_continuous_scale=scale, **kwargs)
    else:
        fig = px.choropleth_mapbox(plot, mapbox_style="carto-positron", color_continuous_scale=scale, **kwargs)
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, height=470, clickmode="event+select")
    if not discrete:
        fig.update_layout(coloraxis_colorbar={"title": _tx("predicted_rate") if color == "Predicted rate" else _tx("uncertainty")})
    return fig


def _add_allocated_sites(fig, sites_xy: list[dict], highlight_id: str | None = None) -> None:
    """Overlay the six solver sites. Does not invent locations."""
    if not sites_xy:
        return
    gdf_selected = gpd.GeoDataFrame(
        sites_xy,
        geometry=gpd.points_from_xy(
            [site["easting"] for site in sites_xy],
            [site["northing"] for site in sites_xy],
        ),
        crs="EPSG:27700",
    ).to_crs(epsg=4326)
    base_type = str(fig.data[0].type) if fig.data else ""
    if "mapbox" in base_type or not hasattr(go, "Scattermap"):
        scatter_cls = go.Scattermapbox
    else:
        scatter_cls = go.Scattermap
    hover = [
        f"{site['site_id']} — {site['site_name']} ({site_type_display(_lang(), label_of(SITE_TYPE_LABELS, site['site_type']))})"
        for site in sites_xy
    ]
    fig.add_trace(
        scatter_cls(
            lat=gdf_selected.geometry.y.tolist(),
            lon=gdf_selected.geometry.x.tolist(),
            mode="markers+text",
            marker=dict(size=16, color="#D90429"),
            text=[site["site_id"] for site in sites_xy],
            textfont=dict(size=11, color="#111111"),
            customdata=[[site["site_id"]] for site in sites_xy],
            textposition="top right",
            name=_tx("allocated_sites"),
            hovertext=hover,
            hovertemplate="%{hovertext}<extra></extra>",
        )
    )
    if not highlight_id:
        return
    idx = next((i for i, site in enumerate(sites_xy) if site["site_id"] == highlight_id), None)
    if idx is None:
        return
    hi = gdf_selected.iloc[[idx]]
    hi_site = sites_xy[idx]
    fig.add_trace(
        scatter_cls(
            lat=hi.geometry.y.tolist(),
            lon=hi.geometry.x.tolist(),
            mode="markers",
            marker=dict(size=22, color="#F4A261"),
            customdata=[[hi_site["site_id"]]],
            name="Selected site",
            showlegend=False,
            hovertext=[
                f"{hi_site['site_id']} — {hi_site['site_name']} ({site_type_display(_lang(), label_of(SITE_TYPE_LABELS, hi_site['site_type']))})"
            ],
            hovertemplate="%{hovertext}<extra></extra>",
        )
    )


def _iz_code_under_site(site_id: str, sites_xy: list[dict], gdf) -> str | None:
    """IZ polygon that contains this allocated site. Used so a site click matches a zone click."""
    site = next((row for row in sites_xy if str(row.get("site_id")) == str(site_id)), None)
    if site is None or gdf is None or gdf.empty:
        return None
    point = (
        gpd.GeoSeries(
            gpd.points_from_xy([site["easting"]], [site["northing"]]),
            crs="EPSG:27700",
        )
        .to_crs(gdf.crs)
        .iloc[0]
    )
    inside = gdf.geometry.contains(point)
    if bool(inside.any()):
        return str(gdf.loc[inside, "iz_code"].iloc[0])
    nearest = gdf.geometry.centroid.distance(point).idxmin()
    return str(gdf.loc[nearest, "iz_code"])


def render_glossary() -> None:
    st.header(_tx("glossary"))
    st.caption(_tx("glossary_caption"))
    rows = glossary_rows(_lang(), GLOSSARY_ROWS)
    st.dataframe(
        {_tx("glossary_name"): [row[0] for row in rows], _tx("glossary_meaning"): [row[1] for row in rows]},
        width="stretch",
        hide_index=True,
    )


def _pick_alpha_row(frame: pd.DataFrame) -> dict[str, float]:
    if frame is None or frame.empty:
        return {}
    pick = frame.iloc[-1]
    if "forecast_start" in frame.columns and "forecast_end" in frame.columns:
        target = pd.Timestamp(str(selected_date))
        start = pd.to_datetime(frame["forecast_start"], errors="coerce")
        end = pd.to_datetime(frame["forecast_end"], errors="coerce")
        inside = frame.loc[(start <= target) & (target <= end)]
        if inside.empty:
            inside = frame.loc[end <= target]
        if not inside.empty:
            pick = inside.iloc[-1]
    return {
        key: float(pick[key])
        for key in ("alpha_geo", "alpha_transport", "alpha_mobility")
        if key in pick.index and pd.notna(pick[key])
    }


def _current_graph_weights() -> dict[str, float]:
    if zones:
        row = next(iter(zones.values()))
        values = {}
        for key in ("alpha_geo", "alpha_transport", "alpha_mobility"):
            val = row.get(key)
            if val is not None and pd.notna(val):
                values[key] = float(val)
        if len(values) == 3:
            return values
    area = str((res.get("bundle") or {}).get("request", {}).get("area_code") or "") if res else ""
    region = _region_rolling_alpha(area, _region_alpha_mtime(area))
    picked = _pick_alpha_row(region)
    if len(picked) == 3:
        return picked
    from_ckpt = _checkpoint_alpha(area, _region_ckpt_mtime(area))
    if len(from_ckpt) == 3:
        return from_ckpt
    return _pick_alpha_row(df_alpha)


@st.cache_data
def _checkpoint_alpha(area_code: str, mtime: float = 0.0) -> dict[str, float]:
    """Learned fusion weights from a city's saved model. Does not invent a rolling path."""
    code = str(area_code or "").strip()
    if code in {"", EDINBURGH_CA, "UNKNOWN"}:
        return {}
    ckpt = project_root() / "data" / "results" / "regions" / code / "model" / "geo_transport_mobility" / "checkpoint.pt"
    if not ckpt.exists():
        return {}
    try:
        from model.operational import _alpha_from_checkpoint
        from model.train import load_raw_checkpoint

        raw = _alpha_from_checkpoint(load_raw_checkpoint(ckpt))
        geo = raw.get("geo")
        transport = raw.get("transport")
        mobility = raw.get("mobility")
        if geo is None or transport is None or mobility is None:
            return {}
        return {
            "alpha_geo": float(geo),
            "alpha_transport": float(transport),
            "alpha_mobility": float(mobility),
        }
    except Exception:
        return {}


@st.cache_data
def _region_rolling_alpha(area_code: str, mtime: float = 0.0) -> pd.DataFrame:
    code = str(area_code or "").strip()
    if code in {"", EDINBURGH_CA, "UNKNOWN"}:
        return pd.DataFrame()
    path = project_root() / "data" / "results" / "regions" / code / "rolling" / "final_test" / "W730" / "rolling_alpha.csv"
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _region_alpha_mtime(area_code: str) -> float:
    path = project_root() / "data" / "results" / "regions" / str(area_code) / "rolling" / "final_test" / "W730" / "rolling_alpha.csv"
    try:
        return float(path.stat().st_mtime)
    except OSError:
        return 0.0


def _alpha_trajectory(area: str) -> tuple[pd.DataFrame, bool]:
    """Return (frame, from_rolling). from_rolling is False for a one-point checkpoint fallback."""
    if area in {"", EDINBURGH_CA} and df_alpha is not None and not df_alpha.empty:
        return df_alpha, True
    region = _region_rolling_alpha(area, _region_alpha_mtime(area))
    if region is not None and not region.empty:
        return region, True
    weights = _checkpoint_alpha(area, _region_ckpt_mtime(area))
    if len(weights) != 3:
        return pd.DataFrame(), False
    return (
        pd.DataFrame(
            [
                {
                    "update_date": selected_date,
                    "alpha_geo": weights["alpha_geo"],
                    "alpha_transport": weights["alpha_transport"],
                    "alpha_mobility": weights["alpha_mobility"],
                }
            ]
        ),
        False,
    )


def _region_ckpt_mtime(area_code: str) -> float:
    ckpt = project_root() / "data" / "results" / "regions" / str(area_code) / "model" / "geo_transport_mobility" / "checkpoint.pt"
    try:
        return float(ckpt.stat().st_mtime)
    except OSError:
        return 0.0


def render_graph_fusion() -> None:
    """City-level graph mix and how it changed. Not a siting score."""
    area = str((res.get("bundle") or {}).get("request", {}).get("area_code") or "") if res else ""
    weights = _current_graph_weights()
    if len(weights) != 3:
        weights = _checkpoint_alpha(area, _region_ckpt_mtime(area))
    trajectory, from_rolling = _alpha_trajectory(area)
    if not weights and (trajectory is None or trajectory.empty):
        return
    st.subheader(_tx("graph_mix"))
    st.caption(_tx("graph_mix_caption"))
    alpha_shown = {
        "alpha_geo": _tx("graph_geo"),
        "alpha_transport": _tx("graph_transport"),
        "alpha_mobility": _tx("graph_mobility"),
    }
    if weights:
        cols = st.columns(3)
        for col, key in zip(cols, ("alpha_geo", "alpha_transport", "alpha_mobility")):
            if key in weights:
                col.metric(alpha_shown[key], f"{weights[key]:.3f}")
    if trajectory is None or trajectory.empty:
        return
    frame = trajectory.copy()
    if "update_date" in frame.columns:
        frame["Date"] = pd.to_datetime(frame["update_date"], errors="coerce")
    elif "forecast_start" in frame.columns:
        frame["Date"] = pd.to_datetime(frame["forecast_start"], errors="coerce")
    else:
        frame["Date"] = pd.to_datetime(range(len(frame)), unit="D", origin="2022-05-31")
    frame = frame.dropna(subset=["Date"]).sort_values("Date")
    value_cols = [col for col in ("alpha_geo", "alpha_transport", "alpha_mobility") if col in frame.columns]
    if frame.empty or not value_cols:
        return
    if (not from_rolling) and len(frame) < 2:
        st.caption(_tx("graph_mix_saved"))
    # Keep English series ids in the data; Chinese names only go on traces.
    colours = {"alpha_geo": "#1f77b4", "alpha_transport": "#ff7f0e", "alpha_mobility": "#2ca02c"}
    fig = go.Figure()
    ymax = 0.0
    for key in value_cols:
        ys = pd.to_numeric(frame[key], errors="coerce")
        ymax = max(ymax, float(ys.max()) if ys.notna().any() else 0.0)
        fig.add_trace(
            go.Scatter(
                x=frame["Date"],
                y=ys,
                mode="lines+markers",
                name=str(key),
                line={"color": colours.get(key, "#333333"), "width": 2},
                marker={"size": 8},
            )
        )
    for trace in fig.data:
        trace.name = alpha_shown.get(str(trace.name), trace.name)
    fig.update_layout(
        margin={"r": 8, "t": 12, "l": 8, "b": 8},
        height=300,
        yaxis_title=_tx("fusion_weight"),
        xaxis_title=_tx("refresh_date"),
        legend_title="",
        yaxis={"range": [0, max(0.5, ymax * 1.15)]},
        hovermode="x unified",
    )
    page = str(st.session_state.get("result_artifact") or "forecast")
    st.plotly_chart(fig, width="stretch", key=f"alpha_trajectory_{page}")


def _normalize_geoshapley_table(df: pd.DataFrame) -> pd.DataFrame:
    """Train exports store baseline in phi_0; the website table has a baseline row."""
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


def render_geoshapley(forecast, zones, df_shapley) -> None:
    path = forecast.get("geoshapley_path")
    if path:
        shapley_file = Path(path)
        if shapley_file.exists():
            df_shapley = pd.read_csv(shapley_file)
    df_shapley = _normalize_geoshapley_table(df_shapley)
    if df_shapley is None or df_shapley.empty:
        st.info(_tx("no_geoshapley"))
        return
    st.subheader(_tx("geoshapley_title"))
    st.caption(_tx("geoshapley_caption"))
    iz_choice = st.session_state.get("iz_inspect_box") or st.session_state.get("inspected_iz_code")
    if iz_choice not in zones:
        iz_choice = sorted(zones.keys())[0]
    date_note = ""
    if "target_report_date" in df_shapley.columns:
        dates = sorted(df_shapley["target_report_date"].astype(str).unique())
        date_note = f" · target {dates[-1]}"
    st.caption(_tx("zone_label", name=_iz_label(iz_choice)) + date_note)
    df_iz = df_shapley.loc[df_shapley["iz_code"].astype(str) == iz_choice].copy()
    if df_iz.empty:
        st.info(_tx("no_geoshapley_iz"))
        return
    df_iz = _normalize_geoshapley_table(df_iz)
    simple = simple_geoshapley_contributions(df_iz)
    table = simple["table"].copy()
    table["label"] = table["indicator"].map(
        lambda key: feature_display(_lang(), str(key), FEATURE_LABELS.get(str(key), str(key)))
    )
    table["effect"] = table["effect"].map(
        lambda effect: _tx("raises") if str(effect).startswith("Raises") else _tx("lowers")
    )
    c1, c2, c3 = st.columns(3)
    c1.metric(
        _tx("baseline"),
        f"{simple['baseline']:.1f}" if simple["baseline"] is not None else "—",
        help=_tx("geo_baseline_help"),
    )
    c2.metric(
        _tx("predicted_rate"),
        f"{simple['predicted_rate']:.1f}" if simple["predicted_rate"] is not None else "—",
        help=_tx("geo_rate_help"),
    )
    c3.metric(
        _tx("contributions_sum"),
        f"{simple['net']:+.1f}",
        help=_tx("geo_sum_help"),
    )
    st.info(_tx("geo_baseline_info"))
    chart = table.copy()
    chart["display"] = chart["contribution"].map(lambda v: f"{v:+.1f}")
    fig_bar = px.bar(
        chart,
        x="contribution",
        y="label",
        orientation="h",
        color="effect",
        color_discrete_map={_tx("raises"): "#2A9D8F", _tx("lowers"): "#E76F51"},
        text="display",
        labels={"contribution": _tx("contribution"), "label": _tx("indicator")},
    )
    fig_bar.update_traces(textposition="outside", cliponaxis=False, textfont_size=12)
    fig_bar.update_layout(
        height=420,
        margin=dict(t=70, b=50, l=240, r=110),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.08, x=0, title_text=""),
        yaxis=dict(automargin=True, title="", tickfont=dict(size=13)),
        xaxis=dict(automargin=True, zeroline=True, title=_tx("contribution")),
    )
    st.plotly_chart(fig_bar, width="stretch")
    st.subheader(_tx("numbers"))
    st.dataframe(
        table.assign(**{_tx("contribution"): table["contribution"].map(lambda v: f"{v:+.2f}")})[
            ["label", _tx("contribution"), "effect"]
        ].rename(columns={"label": _tx("indicator"), "effect": _tx("direction")}),
        width="stretch",
        hide_index=True,
    )
    df_iz = df_iz.sort_values(["component", "feature_name"])
    df_iz["display_name"] = df_iz["feature_name"].map(
        lambda name: feature_display(_lang(), str(name), label_of(FEATURE_LABELS, str(name)))
    )
    fig_waterfall = go.Figure(
        go.Waterfall(
            name="Contribution",
            orientation="v",
            measure=["relative"] * len(df_iz),
            x=df_iz["display_name"],
            y=df_iz["shapley_value"],
            textposition="outside",
        )
    )
    fig_waterfall.update_layout(
        height=440,
        margin=dict(t=30, b=160, l=50, r=30),
        xaxis=dict(tickangle=-40, automargin=True, tickfont=dict(size=11)),
        yaxis=dict(automargin=True),
    )
    st.plotly_chart(fig_waterfall, width="stretch")
    st.caption(_tx("geo_scope"))


date_options = [f"{DEMO_FORECAST_DATE} (extrapolation)"]
if not df_dates.empty and "target_report_date" in df_dates.columns:
    date_options += [str(value) for value in df_dates["target_report_date"].tolist()[::-1]]
if "ui_lang" not in st.session_state:
    st.session_state.ui_lang = "en"
if "brief_study_area" not in st.session_state:
    st.session_state.brief_study_area = list(STUDY_AREA_OPTIONS)[0]
if "brief_date" not in st.session_state:
    st.session_state.brief_date = date_options[0]
if "brief_scenario_label" not in st.session_state:
    st.session_state.brief_scenario_label = list(SCENARIO_LABELS.values())[0]
if "brief_travel_label" not in st.session_state:
    st.session_state.brief_travel_label = list(TRAVEL_MODE_LABELS.values())[0]
if "brief_threshold" not in st.session_state:
    st.session_state.brief_threshold = 20.0
if "brief_confirm_new_region" not in st.session_state:
    st.session_state.brief_confirm_new_region = False
if "brief_city_ready" not in st.session_state:
    st.session_state.brief_city_ready = False
if "brief_date_ready" not in st.session_state:
    st.session_state.brief_date_ready = False
if "brief_qa_step" not in st.session_state:
    st.session_state.brief_qa_step = "city"
if "brief_path" not in st.session_state:
    st.session_state.brief_path = None
if "pending_brief_task" not in st.session_state:
    st.session_state.pending_brief_task = None
if "iz_qa_step" not in st.session_state:
    st.session_state.iz_qa_step = None
if "iz_qa_matches" not in st.session_state:
    st.session_state.iz_qa_matches = []
if "iz_picked_by_user" not in st.session_state:
    st.session_state.iz_picked_by_user = False
if "forecast_scope" not in st.session_state:
    st.session_state.forecast_scope = None
if "alloc_qa_step" not in st.session_state:
    st.session_state.alloc_qa_step = None
if "alloc_qa_travel" not in st.session_state:
    st.session_state.alloc_qa_travel = None
if "alloc_qa_threshold" not in st.session_state:
    st.session_state.alloc_qa_threshold = None
if "alloc_input_mode" not in st.session_state:
    st.session_state.alloc_input_mode = None
if "forecast_input_mode" not in st.session_state:
    st.session_state.forecast_input_mode = None

study_area = st.session_state.brief_study_area
area_code, _area_name = map_study_area(study_area)
glasgow_ready = region_artefacts_ready(area_code)
needs_first_train_confirm = area_code not in {EDINBURGH_CA, "", "UNKNOWN"} and not glasgow_ready
confirm_new_region = bool(st.session_state.brief_confirm_new_region) if needs_first_train_confirm else False
chosen_date_label = str(st.session_state.brief_date)
if chosen_date_label not in date_options:
    chosen_date_label = date_options[0]
    st.session_state.brief_date = chosen_date_label
selected_date = chosen_date_label.split(" ")[0]
scenario_label = st.session_state.brief_scenario_label
if scenario_label not in SCENARIO_LABELS.values():
    scenario_label = list(SCENARIO_LABELS.values())[0]
    st.session_state.brief_scenario_label = scenario_label
scenario = _scenario_keys[scenario_label]
travel_label = st.session_state.brief_travel_label
travel_mode = _travel_keys[travel_label]
travel_threshold = float(st.session_state.brief_threshold)
eligible_labels = list(SITE_TYPE_LABELS.values())
eligible_types = list(_site_keys[label] for label in eligible_labels if label in _site_keys)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "text": _tx("greeting"), "artifact": None}]
if st.session_state.get("ask_first_version") != 3:
    st.session_state.ask_first_version = 3
    st.session_state.explicit_task = False
    st.session_state.messages = [{"role": "assistant", "text": _tx("greeting"), "artifact": None}]
    st.session_state.pipeline_result = None
    st.session_state.applied_brief = None
    st.session_state.view = "agent"
    st.session_state.brief_city_ready = False
    st.session_state.brief_date_ready = False
    st.session_state.brief_qa_step = "city"
    st.session_state.brief_path = None
    st.session_state.pending_brief_task = None
    st.session_state.brief_confirm_new_region = False
    st.session_state.alloc_qa_step = None
elif (
    st.session_state.get("brief_qa_step") == "city"
    and st.session_state.messages
    and st.session_state.messages[0].get("role") == "assistant"
    and len(st.session_state.messages) == 1
):
    st.session_state.messages[0]["text"] = _tx("greeting")
_rewrite_stale_site_copy()

current_brief = brief_fingerprint(
    study_area=study_area,
    forecast_date=selected_date,
    scenario=scenario,
    travel_mode=travel_mode,
    travel_threshold=float(travel_threshold),
    eligible_types=eligible_types,
    confirm_new_region=confirm_new_region,
)
if st.session_state.get("brief_scenario_committed") not in (None, scenario):
    st.session_state.override_scenario = None
st.session_state.brief_scenario_committed = scenario


def _display_scenario_label() -> str:
    override = st.session_state.get("override_scenario")
    if override:
        try:
            english = SCENARIO_LABELS[map_scenario(override)]
        except Exception:
            english = str(override)
    else:
        english = scenario_label
    return scenario_display(_lang(), english)


def _brief_plan_text() -> str:
    return plan_from_brief_text(
        study_area=study_area,
        forecast_date=selected_date,
        scenario_label=_display_scenario_label(),
        travel_threshold=float(travel_threshold),
        travel_label=travel_label,
        eligible_labels=eligible_labels,
    )


def _active_scenario() -> str:
    return st.session_state.get("override_scenario") or scenario


def _known_ids(result) -> tuple[list[str], list[str]]:
    if not result or not result.get("success"):
        return [], []
    selected = [row["site_id"] for row in result["allocation"]["selected_sites"]]
    return selected, sorted(result["forecast"]["zone_forecasts"].keys())


def _open_result_page(artifact: str) -> None:
    st.session_state.view = "results"
    st.session_state.result_artifact = artifact


def _is_hidden_chat_line(message: dict) -> bool:
    text = str(message.get("text") or "")
    if text.startswith("I am the health planning agent.") or text.startswith("我是卫生规划助手"):
        return True
    if text.startswith("Plan 6 intervention sites using the task brief:") or text.startswith(
        "Plan 6 vaccination sites using the task brief:"
    ):
        return True
    if message.get("source") == "brief":
        return True
    return False


def _is_home_page() -> bool:
    return (
        st.session_state.get("view", "agent") == "agent"
        and not st.session_state.get("brief_city_ready")
        and st.session_state.get("brief_qa_step") == "city"
        and not st.session_state.get("explicit_task")
    )


def _go_home() -> None:
    """Return to the first agent screen: greeting and city question."""
    st.session_state.view = "agent"
    st.session_state.explicit_task = False
    st.session_state.messages = [{"role": "assistant", "text": _tx("greeting"), "artifact": None}]
    st.session_state.brief_city_ready = False
    st.session_state.brief_date_ready = False
    st.session_state.brief_qa_step = "city"
    st.session_state.pending_brief_task = None
    st.session_state.pending_user = None
    st.session_state.pending_from_brief = False
    st.session_state.brief_confirm_new_region = False
    st.session_state.alloc_qa_step = None
    st.session_state.alloc_qa_travel = None
    st.session_state.alloc_qa_threshold = None
    st.session_state.alloc_input_mode = None
    st.session_state.forecast_input_mode = None
    st.session_state.iz_qa_step = None
    st.session_state.iz_qa_matches = []
    st.session_state.iz_picked_by_user = False
    st.session_state.forecast_scope = None
    st.session_state.result_artifact = None
    st.session_state.force_realloc = False
    st.session_state.inspected_iz_code = None
    st.session_state.iz_inspect_box = None


def _render_brand_mark() -> None:
    import html as html_lib

    mark, copy = st.columns([0.55, 8], gap="small", vertical_alignment="center")
    with mark:
        if UOFG_CREST.exists():
            st.image(str(UOFG_CREST), width=72)
    with copy:
        uni = html_lib.escape(_tx("brand_uni"))
        sub = html_lib.escape(_tx("brand_sub"))
        st.html(
            f'<div style="color:#003865;line-height:1.15;margin:0;">'
            f'<div style="font-weight:800;font-size:1.85rem;">{uni}</div>'
            f'<div style="font-size:1.05rem;margin-top:2px;opacity:0.92;">{sub}</div>'
            f"</div>"
        )


def _render_brand_bar() -> None:
    """University of Glasgow crest and name. Red home button on every page except the start screen."""
    show_home = not _is_home_page()
    if show_home:
        brand, home = st.columns([4, 1.7])
        with brand:
            _render_brand_mark()
        with home:
            if st.button(_tx("btn_home"), key="home_to_start", type="primary", width="stretch"):
                _go_home()
                st.rerun()
        return
    _render_brand_mark()


def _agent_heading(text: str) -> None:
    """Section label with the same navy-and-white robot used in chat."""
    icon, title = st.columns([1, 18], vertical_alignment="center")
    with icon:
        if AGENT_AVATAR.exists():
            st.image(str(AGENT_AVATAR), width=32)
    with title:
        st.markdown(f"### {text}")


def _render_agent_bubble(text: str | None) -> None:
    """Agent speech always uses the same robot avatar."""
    body = str(text or "").strip()
    if not body:
        return
    with st.chat_message("assistant", avatar=AGENT_AVATAR):
        st.markdown(format_agent_markdown(body))


def _render_agent_transcript(*, hide_briefing: bool = False) -> None:
    for message in st.session_state.messages:
        if hide_briefing and _is_hidden_chat_line(message):
            continue
        role = message.get("role") or "assistant"
        if role == "assistant":
            with st.chat_message("assistant", avatar=AGENT_AVATAR):
                st.markdown(format_agent_markdown(message["text"]))
        else:
            with st.chat_message("user", avatar=USER_AVATAR):
                st.markdown(format_agent_markdown(message["text"]))


def _render_typed_answer_box(
    *,
    key_prefix: str,
    hint: str,
    question: str | None = None,
    empty_mode_key: str | None = None,
    empty_offer_key: str | None = None,
) -> None:
    """Input box that sits under the agent's question. Empty Enter opens the offered buttons."""
    with st.form(f"{key_prefix}_type_box", clear_on_submit=True, border=False):
        if question:
            st.markdown(question)
        ask_col, send_col = st.columns([6, 1], gap="small")
        with ask_col:
            typed = st.text_input(
                _tx("type_box_caption"),
                placeholder=hint,
                label_visibility="collapsed",
                key=f"{key_prefix}_type_prompt",
            )
        with send_col:
            sent = st.form_submit_button(_tx("btn_send"), width="stretch", type="primary")
        if sent:
            body = str(typed or "").strip()
            if body:
                st.session_state.pending_user = body
            elif empty_mode_key:
                st.session_state[empty_mode_key] = "buttons"
                if empty_offer_key:
                    st.session_state.messages.append(
                        {"role": "assistant", "text": _tx(empty_offer_key), "artifact": None}
                    )
            st.rerun()


def _render_agent_followups(*, key_prefix: str) -> None:
    """Ask box first, then the three compact task buttons under it."""
    st.divider()
    with st.chat_message("assistant", avatar=AGENT_AVATAR):
        st.markdown(f"**{_tx('continue_agent')}**")
        st.caption(_tx("continue_caption"))
    _render_ask_bar(key_prefix=key_prefix)
    _render_brief_qa()


def _render_ask_bar(*, key_prefix: str) -> None:
    step = st.session_state.get("brief_qa_step")
    if step == "city":
        hint = _tx("placeholder_city")
    elif step == "date":
        hint = _tx("placeholder_date")
    elif st.session_state.get("iz_qa_step") == "scope":
        hint = _tx("placeholder_scope")
    elif st.session_state.get("iz_qa_step") in {"zone", "zone_disambiguate"}:
        hint = _tx("placeholder_zone")
    else:
        hint = _tx("placeholder_ask")
    with st.form(f"{key_prefix}_ask", clear_on_submit=True, border=False):
        ask_col, send_col = st.columns([6, 1], gap="small")
        with ask_col:
            prompt = st.text_input(
                _tx("placeholder_ask"),
                placeholder=hint,
                label_visibility="collapsed",
                key=f"{key_prefix}_prompt",
            )
        with send_col:
            send_chat = st.form_submit_button(_tx("btn_send"), width="stretch", type="primary")
        typed = str(prompt or "").strip()
        if send_chat and typed:
            st.session_state.pending_user = typed
            st.rerun()


QA_SCENARIO_QUESTION = (
    "I can place exactly six sites.\n\n"
    "Which policy should I use? Pick one below, or type it. "
    "This does not change the forecast."
)
QA_TRAVEL_QUESTION = (
    "I still need a travel mode and a time threshold before I run the solver.\n\n"
    "Pick **Drive** or **Walk**, and a threshold in minutes. "
    "I will place the six sites after both are set."
)
QA_THRESHOLDS = (10.0, 15.0, 20.0, 25.0, 30.0)
def _latest_forecast_label() -> str:
    return date_options[0] if date_options else f"{DEMO_FORECAST_DATE} (extrapolation)"


def _latest_forecast_day() -> str:
    return str(_latest_forecast_label()).split(" ")[0]


def _qa_date_question() -> str:
    return _tx("ask_date", latest=_latest_forecast_day())
QA_TASK_QUESTION = (
    "What should I do next?\n\n"
            "The six points are **vaccination sites** — existing GPs, pharmacies, "
            "or mobile stops at car parks. A solver places exactly six of them so "
            "neighbourhoods can reach a site within a travel-time threshold.\n\n"
            "- **Plan 6 vaccination sites** — I place those six vaccination sites. "
    "I first ask which policy (coverage, equity, preventive, or balanced), "
    "then Drive or Walk and the time threshold. Then I show the six points on the map.\n"
    "- **Show the forecast** — I show the predicted infection-rate map, "
    "uncertainty, and how the three graphs mix. I then ask which neighbourhood "
    "to explain with GeoShapley. That explains the forecast, not the site choice.\n"
    "- **Compare four policies** — I keep the same six-site cap and travel rule, "
    "and show how the four policies differ.\n\n"
    "Pick one below, or type it."
)
QA_CONFIRM_TRAIN = (
    "This city does not have a saved model yet. "
    "Should I build one? Edinburgh's saved model and files will not be overwritten."
)
QA_ZONE_QUESTION = (
    "Which **Intermediate Zone** should I explain? "
    "Type the neighbourhood name (for example **Currie West**), or click it on the map. "
    "GeoShapley explains that zone's forecast; it is not a siting score."
)
QA_TASK_BUTTONS = (
    ("Plan 6 sites", "Plan 6 sites"),
    ("Show the forecast", "Show the forecast"),
    ("Compare four policies", "Compare the four allocation policies."),
)


def _parse_city_name(text: str) -> str | None:
    lowered = str(text or "").strip().lower()
    if not lowered:
        return None
    if any(
        word in lowered
        for word in ("glasgow", "格拉斯哥", "glasgae", "glesga", "glasgow city")
    ) or lowered == GLASGOW_CA.lower():
        return GLASGOW_NAME
    if any(
        word in lowered
        for word in ("edinburgh", "爱丁堡", "edinboro", "edinburgh city", "city of edinburgh")
    ) or lowered == EDINBURGH_CA.lower() or re.search(r"\bedin\b", lowered):
        return list(STUDY_AREA_OPTIONS)[0]
    for name in STUDY_AREA_OPTIONS:
        if name.lower() in lowered or lowered in name.lower():
            return name
    return None


def _canonical_date_label(raw: str) -> str | None:
    return extract_forecast_date(
        raw,
        options=date_options,
        latest=_latest_forecast_label(),
    )


def _needs_train_confirm() -> bool:
    if not st.session_state.get("brief_city_ready"):
        return False
    code, _name = map_study_area(st.session_state.brief_study_area)
    if code in {EDINBURGH_CA, "", "UNKNOWN"}:
        return False
    if region_artefacts_ready(code):
        return False
    return not bool(st.session_state.get("brief_confirm_new_region"))


def _apply_interaction_button_style() -> None:
    """Force in-app buttons, dropdown, and page background to Glasgow blues."""
    css = """
    <style>
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"],
    [data-testid="stHeader"], [data-testid="stBottomBlockContainer"],
    section.main, .stMainBlockContainer {
        background-color: #E6F1F8 !important;
    }
    [data-testid="stHeader"] {
        background: #E6F1F8 !important;
    }
    .stApp .stButton button,
    .stApp .stFormSubmitButton button,
    .stApp button[data-testid^="stBaseButton"],
    .stApp [data-testid="stBaseButton-secondary"],
    .stApp [data-testid="stBaseButton-primary"],
    .stApp [data-testid="stBaseButton-secondaryFormSubmit"],
    .stApp [data-testid="stBaseButton-primaryFormSubmit"] {
        background-color: #003865 !important;
        background-image: none !important;
        color: #ffffff !important;
        border: 1px solid #00264A !important;
        font-weight: 700 !important;
    }
    .stApp .stButton button:hover,
    .stApp .stFormSubmitButton button:hover,
    .stApp button[data-testid^="stBaseButton"]:hover {
        background-color: #00264A !important;
        color: #ffffff !important;
        border-color: #001A33 !important;
    }
    .stApp [data-baseweb="select"] > div,
    .stApp [data-testid="stSelectbox"] > div > div {
        background-color: #003865 !important;
        color: #ffffff !important;
        border-color: #00264A !important;
    }
    .stApp [data-baseweb="select"] svg,
    .stApp [data-baseweb="select"] input,
    .stApp [data-baseweb="select"] div {
        color: #ffffff !important;
        fill: #ffffff !important;
    }
    div[class*="st-key-iz_qa_select"] [data-baseweb="select"] > div,
    div[class*="st-key-iz_qa_select"] [data-baseweb="select"] > div > div,
    div[class*="st-key-iz_qa_select"] [data-testid="stSelectbox"] > div > div {
        background-color: #ffffff !important;
        color: #003865 !important;
        border: 1px solid #003865 !important;
    }
    div[class*="st-key-iz_qa_select"] [data-baseweb="select"] svg,
    div[class*="st-key-iz_qa_select"] [data-baseweb="select"] input,
    div[class*="st-key-iz_qa_select"] [data-baseweb="select"] div,
    div[class*="st-key-iz_qa_select"] [data-baseweb="select"] span {
        color: #003865 !important;
        fill: #003865 !important;
        background-color: #ffffff !important;
    }
    .stApp div[class*="st-key-home_to_start"] button,
    .stApp div[class*="st-key-home_to_start"] button[data-testid^="stBaseButton"] {
        background-color: #C8102E !important;
        background-image: none !important;
        color: #ffffff !important;
        border: 1px solid #9B0C24 !important;
        font-weight: 700 !important;
        white-space: normal !important;
        height: auto !important;
        min-height: 2.8rem !important;
        line-height: 1.3 !important;
        padding: 0.45rem 0.7rem !important;
    }
    .stApp div[class*="st-key-home_to_start"] button:hover {
        background-color: #9B0C24 !important;
        border-color: #7A091C !important;
        color: #ffffff !important;
    }
    div[class*="st-key-qa_"] button,
    div[class*="st-key-switch_"] button,
    div[class*="st-key-restart_iz_qa"] button {
        min-height: 3.4rem !important;
        height: auto !important;
        white-space: normal !important;
        line-height: 1.35 !important;
        font-size: 1.02rem !important;
        font-weight: 700 !important;
        padding: 0.55rem 0.75rem !important;
    }
    .stApp div[class*="st-key-qa_task_"] button {
        min-height: 2rem !important;
        height: auto !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        line-height: 1.25 !important;
        padding: 0.28rem 0.4rem !important;
        white-space: normal !important;
    }
    </style>
    """
    if hasattr(st, "html"):
        st.html(css)
    else:
        st.markdown(css, unsafe_allow_html=True)


def _qa_button_style() -> None:
    _apply_interaction_button_style()


def _qa_ask_city() -> None:
    st.session_state.brief_qa_step = "city"
    st.session_state.brief_city_ready = False
    st.session_state.view = "agent"
    st.session_state.messages.append(
        {
            "role": "assistant",
            "text": _tx("ask_city"),
            "artifact": None,
        }
    )


def _qa_ask_date(*, city_picked: str) -> None:
    st.session_state.brief_qa_step = "date"
    st.session_state.brief_date_ready = False
    st.session_state.view = "agent"
    st.session_state.messages.append(
        {
            "role": "assistant",
            "text": _tx("use_city", city=city_display(_lang(), city_picked)) + _qa_date_question(),
            "artifact": None,
        }
    )


def _qa_ask_task() -> None:
    st.session_state.brief_qa_step = "task"
    st.session_state.view = "agent"
    city = st.session_state.brief_study_area
    day = str(st.session_state.brief_date).split(" ")[0]
    st.session_state.messages.append(
        {
            "role": "assistant",
            "text": f"**{city}**, **{day}**. {_tx('ask_task')}",
            "artifact": None,
        }
    )


def _qa_ask_confirm_train() -> None:
    st.session_state.brief_qa_step = "confirm_train"
    st.session_state.view = "agent"
    st.session_state.messages.append({"role": "assistant", "text": _tx("confirm_train"), "artifact": None})


def _iz_names_now() -> dict[str, str]:
    prior = st.session_state.get("pipeline_result")
    area = EDINBURGH_CA
    if prior and prior.get("success"):
        area = str((prior.get("bundle") or {}).get("request", {}).get("area_code") or EDINBURGH_CA)
    gdf = load_zone_boundaries(area)
    gdf = _attach_iz_names(gdf)
    if "iz_code" not in gdf.columns:
        return {}
    names = {}
    for _, row in gdf.iterrows():
        code = str(row["iz_code"])
        name = str(row.get("iz_name") or code).strip() or code
        names[code] = name
    return names


def _match_iz(text: str) -> list[str]:
    return match_zone_names(text, _iz_names_now())


def _apply_forecast_text_choice(text: str) -> bool:
    """Honour typed equivalents of the forecast buttons: region, IZ view, or a named zone."""
    raw = str(text or "").strip()
    if not raw:
        return False
    prior = st.session_state.get("pipeline_result") or {}
    if not prior.get("success"):
        return False
    lowered = raw.lower()
    if text_wants_region_view(raw) or _tx("btn_scope_region").lower() in lowered or _tx("btn_switch_region").lower() in lowered:
        _qa_accept_scope_region()
        return True
    if (
        text_wants_iz_view(raw)
        or _tx("btn_scope_iz").lower() in lowered
        or _tx("btn_another_zone").lower() in lowered
    ):
        _qa_accept_scope_iz()
        return True
    matches = _match_iz(raw)
    if len(matches) == 1:
        _qa_accept_iz(matches[0])
        return True
    if 1 < len(matches) <= 8:
        st.session_state.forecast_scope = "iz"
        st.session_state.iz_qa_step = "zone_disambiguate"
        st.session_state.iz_qa_matches = matches
        names = _iz_names_now()
        listed = ", ".join(f"**{names.get(code, code)}**" for code in matches)
        st.session_state.messages.append(
            {
                "role": "assistant",
                "text": _tx("zone_multi", listed=listed),
                "artifact": "forecast",
            }
        )
        _open_result_page("forecast")
        return True
    return False


def _open_forecast_from_request(text: str, *, intro: str | None = None) -> None:
    """Open forecast using any region / Intermediate Zone already named in the request."""
    pending = str(st.session_state.pop("pending_forecast_text", None) or text or "").strip()
    if _apply_forecast_text_choice(pending):
        return
    _qa_ask_zone(intro=intro)


def _qa_ask_forecast_scope(*, intro: str | None = None) -> None:
    st.session_state.forecast_scope = None
    st.session_state.forecast_input_mode = "type"
    st.session_state.iz_qa_step = "scope"
    st.session_state.iz_qa_matches = []
    st.session_state.iz_picked_by_user = False
    st.session_state.explicit_task = True
    st.session_state.pop("iz_qa_select", None)
    _open_result_page("forecast")
    prefix = f"{intro}\n\n" if intro else ""
    st.session_state.messages.append(
        {"role": "assistant", "text": prefix + _tx("scope_question"), "artifact": "forecast"}
    )


def _qa_accept_scope_region() -> None:
    st.session_state.forecast_scope = "region"
    st.session_state.iz_qa_step = None
    st.session_state.iz_picked_by_user = False
    st.session_state.inspected_iz_code = None
    st.session_state.messages.append(
        {"role": "assistant", "text": _tx("scope_region_accept"), "artifact": "forecast"}
    )
    _open_result_page("forecast")


def _qa_accept_scope_iz() -> None:
    st.session_state.forecast_scope = "iz"
    st.session_state.iz_qa_step = "zone"
    st.session_state.forecast_input_mode = "type"
    st.session_state.iz_picked_by_user = False
    st.session_state.pop("iz_qa_select", None)
    st.session_state.messages.append(
        {"role": "assistant", "text": _tx("zone_question"), "artifact": "forecast"}
    )
    _open_result_page("forecast")


def _qa_ask_zone(*, intro: str | None = None) -> None:
    if st.session_state.get("forecast_scope") == "iz":
        _qa_accept_scope_iz()
        if intro:
            st.session_state.messages[-1]["text"] = f"{intro}\n\n{_tx('zone_question')}"
        return
    _qa_ask_forecast_scope(intro=intro)


def _qa_accept_iz(code: str, *, from_map: bool = False) -> None:
    names = _iz_names_now()
    label = names.get(str(code), str(code))
    st.session_state.inspected_iz_code = str(code)
    st.session_state.iz_inspect_box = str(code)
    st.session_state.iz_picked_by_user = True
    st.session_state.forecast_scope = "iz"
    st.session_state.iz_qa_step = None
    st.session_state.iz_qa_matches = []
    if from_map:
        st.session_state.messages.append({"role": "user", "text": label, "artifact": None})
    st.session_state.messages.append(
        {
            "role": "assistant",
                "text": _tx("zone_accept", name=label),
            "artifact": "forecast",
        }
    )
    _open_result_page("forecast")


def _continue_iz_qa_from_text(text: str, intent: dict) -> bool:
    step = st.session_state.get("iz_qa_step")
    if step not in {"scope", "zone", "zone_disambiguate"}:
        return False
    picked_mode = _text_picks_input_mode(text)
    if picked_mode == "buttons":
        st.session_state.forecast_input_mode = "buttons"
        offer = "show_scope_options" if step == "scope" else "show_zone_options"
        st.session_state.messages.append({"role": "assistant", "text": _tx(offer), "artifact": "forecast"})
        return True
    if picked_mode == "type":
        st.session_state.forecast_input_mode = "type"
        return True
    if not st.session_state.get("forecast_input_mode"):
        st.session_state.forecast_input_mode = "type"
    if step == "scope":
        if text_wants_region_view(text) or _tx("btn_scope_region").lower() in str(text or "").lower():
            _qa_accept_scope_region()
            return True
        if text_wants_iz_view(text) or any(
            word in str(text or "").lower()
            for word in ("intermediate", "neighbourhood", "neighborhood", "中间区", "街区")
        ):
            if not text_wants_region_view(text):
                _qa_accept_scope_iz()
                return True
        matches = _match_iz(text)
        if len(matches) == 1:
            _qa_accept_iz(matches[0])
            return True
        if matches:
            st.session_state.forecast_scope = "iz"
            st.session_state.iz_qa_step = "zone"
            return _continue_iz_qa_from_text(text, {"intent": "explain_zone"})
        st.session_state.messages.append(
            {
                "role": "assistant",
                "text": _tx("scope_need"),
                "artifact": "forecast",
            }
        )
        return True
    matches = list(intent.get("iz_code") and [str(intent["iz_code"])] or [])
    if not matches:
        matches = _match_iz(text)
    if intent.get("intent") in {"plan", "allocation", "compare", "help", "notes"} and not matches:
        if intent.get("intent") == "help":
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "text": _chat_reply(text, step=step, chat=intent.get("chat")),
                    "artifact": "forecast",
                }
            )
            return True
        st.session_state.iz_qa_step = None
        return False
    if intent.get("intent") == "forecast" and not matches:
        st.session_state.iz_qa_step = None
        return False
    if step == "zone_disambiguate":
        allowed = {str(code) for code in st.session_state.get("iz_qa_matches") or []}
        if allowed:
            matches = [code for code in matches if code in allowed] or matches
    if len(matches) == 1:
        _qa_accept_iz(matches[0])
        return True
    if 1 < len(matches) <= 8:
        st.session_state.iz_qa_step = "zone_disambiguate"
        st.session_state.iz_qa_matches = matches
        names = _iz_names_now()
        listed = ", ".join(f"**{names.get(code, code)}**" for code in matches)
        st.session_state.messages.append(
            {
                "role": "assistant",
                "text": _tx("zone_multi", listed=listed),
                "artifact": "forecast",
            }
        )
        return True
    if len(matches) > 8:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "text": _tx("zone_too_many", n=len(matches)),
                "artifact": "forecast",
            }
        )
        return True
    st.session_state.messages.append(
        {
            "role": "assistant",
            "text": _chat_reply(text, step="zone", chat=intent.get("chat")),
            "artifact": "forecast",
        }
    )
    return True


def _render_iz_qa(*, key_prefix: str = "iz") -> None:
    """Ask in the input box first. Empty Enter opens the agent's buttons or full IZ list."""
    step = st.session_state.get("iz_qa_step")
    scope = st.session_state.get("forecast_scope")
    asking_iz = step in {"zone", "zone_disambiguate"}
    asking_scope = step == "scope" or scope is None
    if scope == "region" and not asking_iz:
        return
    if scope == "iz" and st.session_state.get("iz_picked_by_user") and not asking_iz:
        return
    mode = st.session_state.get("forecast_input_mode") or "type"
    if asking_scope and not asking_iz:
        if mode == "type":
            _render_typed_answer_box(
                key_prefix=f"scope_type_{key_prefix}",
                hint=_tx("placeholder_type_scope"),
                empty_mode_key="forecast_input_mode",
                empty_offer_key="show_scope_options",
            )
            return
        _qa_button_style()
        st.markdown(_tx("show_scope_options"))
        if st.button(
            _tx("btn_scope_region"),
            width="stretch",
            key=f"qa_scope_region_{key_prefix}",
            type="primary",
        ):
            st.session_state.messages.append(
                {"role": "user", "text": _tx("btn_scope_region"), "artifact": None}
            )
            _qa_accept_scope_region()
            st.rerun()
        if st.button(
            _tx("btn_scope_iz"),
            width="stretch",
            key=f"qa_scope_iz_{key_prefix}",
            type="primary",
        ):
            st.session_state.messages.append(
                {"role": "user", "text": _tx("btn_scope_iz"), "artifact": None}
            )
            _qa_accept_scope_iz()
            st.rerun()
        return
    if mode == "type" and step != "zone_disambiguate":
        _render_typed_answer_box(
            key_prefix=f"zone_type_{key_prefix}",
            hint=_tx("placeholder_type_scope"),
            empty_mode_key="forecast_input_mode",
            empty_offer_key="show_zone_options",
        )
        return
    names = _iz_names_now()
    if not names:
        return
    with st.chat_message("assistant", avatar=AGENT_AVATAR):
        st.markdown(f"**{_tx('zone_offer')}**")
        if step == "zone_disambiguate":
            matches = list(st.session_state.get("iz_qa_matches") or [])
            if matches:
                st.caption(_tx("zone_offer_more"))
                cols = st.columns(2)
                for i, code in enumerate(matches):
                    label = names.get(code, code)
                    if cols[i % 2].button(
                        label,
                        width="stretch",
                        key=f"qa_iz_{key_prefix}_{code}",
                        type="primary",
                    ):
                        st.session_state.messages.append({"role": "user", "text": label, "artifact": None})
                        _qa_accept_iz(code)
                        st.rerun()
        placeholder = _tx("zone_dropdown_placeholder")
        ordered = sorted(names.items(), key=lambda item: (str(item[1]).lower(), str(item[0])))
        labels = [placeholder]
        codes_by_label: dict[str, str] = {}
        used = set()
        for code, name in ordered:
            label = str(name)
            if label in used:
                label = f"{name} ({code})"
            used.add(label)
            codes_by_label[label] = code
            labels.append(label)
        picked = st.selectbox(
            _tx("zone_dropdown"),
            labels,
            index=0,
            key=f"iz_qa_select_{key_prefix}",
        )
        if picked != placeholder and picked in codes_by_label:
            code = codes_by_label[picked]
            if code != st.session_state.get("inspected_iz_code") or asking_iz:
                st.session_state.messages.append({"role": "user", "text": picked, "artifact": None})
                _qa_accept_iz(code)
                st.rerun()


def _chat_step() -> str:
    if st.session_state.get("alloc_qa_step"):
        return str(st.session_state.alloc_qa_step)
    if st.session_state.get("iz_qa_step") in {"scope", "zone", "zone_disambiguate"}:
        return str(st.session_state.iz_qa_step)
    brief = st.session_state.get("brief_qa_step")
    if brief in {"city", "date", "confirm_train", "task"}:
        return str(brief)
    if st.session_state.get("view") == "results" and st.session_state.get("pipeline_result"):
        return "results"
    if not st.session_state.get("brief_city_ready"):
        return "city"
    if not st.session_state.get("brief_date_ready"):
        return "date"
    return "task"


def _chat_reply(text: str, *, step: str | None = None, chat: str | None = None) -> str:
    from agent.dashboard_bridge import narrate_chat

    return narrate_chat(
        text,
        lang=_lang(),
        chat=chat,
        step=step or _chat_step(),
        latest=_latest_forecast_day(),
    )


def _launch_pending_or_intent(intent_name: str | None, *, silent_user: bool = False) -> None:
    name = intent_name or st.session_state.get("pending_brief_task")
    st.session_state.pending_brief_task = None
    st.session_state.brief_qa_step = None
    if name == "plan":
        st.session_state.pending_user = "Plan 6 sites"
        st.session_state.pending_from_brief = False
    elif name == "allocation":
        st.session_state.pending_user = "Show the allocation"
        st.session_state.pending_from_brief = False
    elif name == "forecast":
        st.session_state.pending_user = "Show the forecast"
    elif name == "compare":
        st.session_state.pending_user = "Compare the four allocation policies."
    else:
        st.session_state.brief_qa_step = "task"
        return
    if silent_user:
        st.session_state.pending_from_brief = True
    st.rerun()


def _qa_accept_city(name: str, *, extra_date: str | None = None, extra_intent: str | None = None) -> None:
    code, official = map_study_area(name)
    if code == "UNKNOWN":
        st.session_state.messages.append(
            {
                "role": "assistant",
                "text": _tx("city_unknown"),
                "artifact": None,
            }
        )
        st.session_state.brief_qa_step = "city"
        return
    prev_code, _ = map_study_area(st.session_state.get("brief_study_area"))
    st.session_state.brief_study_area = official
    st.session_state.brief_city_ready = True
    st.session_state.brief_confirm_new_region = False
    st.session_state.alloc_qa_step = None
    st.session_state.alloc_qa_travel = None
    st.session_state.alloc_qa_threshold = None
    st.session_state.alloc_input_mode = None
    if prev_code != code:
        st.session_state.pipeline_result = None
        st.session_state.applied_brief = None
        st.session_state.iz_qa_step = None
        st.session_state.iz_qa_matches = []
        st.session_state.inspected_iz_code = None
        st.session_state.forecast_scope = None
    if extra_date:
        label = _canonical_date_label(extra_date)
        if label:
            st.session_state.brief_date = label
            st.session_state.brief_date_ready = True
    if extra_intent and not st.session_state.get("brief_date_ready"):
        st.session_state.brief_date = _latest_forecast_label()
        st.session_state.brief_date_ready = True
    if extra_intent:
        st.session_state.pending_brief_task = extra_intent
    if not st.session_state.get("brief_date_ready"):
        _qa_ask_date(city_picked=official)
        return
    if extra_intent and _needs_train_confirm():
        _qa_ask_confirm_train()
        return
    if extra_intent:
        _launch_pending_or_intent(extra_intent, silent_user=True)
        return
    _qa_ask_task()


def _qa_accept_date(raw: str, *, extra_intent: str | None = None) -> None:
    label = _canonical_date_label(raw)
    if not label:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "text": _tx("date_bad"),
                "artifact": None,
            }
        )
        st.session_state.brief_qa_step = "date"
        return
    prev_day = str(st.session_state.get("brief_date") or "").split()[0]
    new_day = str(label).split()[0]
    st.session_state.brief_date = label
    st.session_state.brief_date_ready = True
    if prev_day != new_day:
        st.session_state.forecast_scope = None
        st.session_state.iz_qa_step = None
        st.session_state.iz_picked_by_user = False
    if extra_intent:
        st.session_state.pending_brief_task = extra_intent
    if extra_intent and _needs_train_confirm():
        _qa_ask_confirm_train()
        return
    if extra_intent:
        _launch_pending_or_intent(extra_intent)
        return
    _qa_ask_task()


def _continue_brief_qa_from_text(text: str, intent: dict) -> bool:
    """Handle typed answers for city, date, task, or new-city confirm."""
    step = st.session_state.get("brief_qa_step")
    if step not in {"city", "date", "task", "confirm_train"}:
        return False
    city = _parse_city_name(text)
    date = _canonical_date_label(text)
    intent_name = intent.get("intent") if intent.get("intent") in {"plan", "allocation", "forecast", "compare"} else None
    if step == "city":
        if city:
            if intent_name == "forecast":
                st.session_state.pending_forecast_text = text
            _qa_accept_city(city, extra_date=date, extra_intent=intent_name)
            return True
        if date:
            st.session_state.brief_date = date
            st.session_state.brief_date_ready = True
            if intent_name:
                st.session_state.pending_brief_task = intent_name
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "text": _tx("chat_date_need_city", date=str(date).split()[0]),
                    "artifact": None,
                }
            )
            return True
        if intent_name:
            st.session_state.pending_brief_task = intent_name
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "text": _tx("chat_task_need_city"),
                    "artifact": None,
                }
            )
            return True
        st.session_state.messages.append(
            {
                "role": "assistant",
                "text": _chat_reply(text, step="city", chat=intent.get("chat")),
                "artifact": None,
            }
        )
        return True
    if step == "date":
        if date:
            if intent_name == "forecast":
                st.session_state.pending_forecast_text = text
            _qa_accept_date(date, extra_intent=intent_name)
            return True
        if city:
            _qa_accept_city(city, extra_date=date, extra_intent=intent_name)
            return True
        if intent_name:
            _qa_accept_date(_latest_forecast_label(), extra_intent=intent_name)
            return True
        st.session_state.messages.append(
            {
                "role": "assistant",
                "text": _chat_reply(text, step="date", chat=intent.get("chat")),
                "artifact": None,
            }
        )
        return True
    if step == "confirm_train":
        lowered = str(text or "").lower()
        chat_kind = intent.get("chat")
        strong_yes = any(word in lowered for word in ("yes", "allow", "build", "confirm", "是"))
        strong_no = any(word in lowered for word in ("no", "not", "cancel", "another", "否", "不", "换"))
        if chat_kind in {"greet", "thanks", "farewell", "identity", "wellbeing", "capabilities", "other"} and not strong_yes and not strong_no:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "text": _chat_reply(text, step="confirm_train", chat=chat_kind),
                    "artifact": None,
                }
            )
            return True
        if strong_yes or any(word in lowered for word in ("ok", "好", "可以")):
            st.session_state.brief_confirm_new_region = True
            _launch_pending_or_intent(st.session_state.get("pending_brief_task"))
            return True
        if strong_no:
            st.session_state.brief_confirm_new_region = False
            _qa_ask_city()
            return True
        st.session_state.messages.append(
            {
                "role": "assistant",
                "text": _chat_reply(text, step="confirm_train", chat=chat_kind),
                "artifact": None,
            }
        )
        return True
    if intent_name:
        if city:
            _qa_accept_city(city, extra_date=date, extra_intent=intent_name)
        elif date:
            _qa_accept_date(date, extra_intent=intent_name)
        else:
            _launch_pending_or_intent(intent_name, silent_user=True)
        return True
    recovered = _resolve_user_intent(text, intent)
    recovered_name = recovered.get("intent") if recovered.get("intent") in {"plan", "allocation", "forecast", "compare"} else None
    if recovered_name:
        if city:
            _qa_accept_city(city, extra_date=date, extra_intent=recovered_name)
        else:
            _launch_pending_or_intent(recovered_name, silent_user=True)
        return True
    if city:
        _qa_accept_city(city, extra_date=date, extra_intent=intent_name)
        return True
    if date and not intent_name:
        _qa_accept_date(date)
        return True
    return False


def _render_brief_qa() -> None:
    step = st.session_state.get("brief_qa_step")
    if step not in {"city", "date", "task", "confirm_train"}:
        if (
            st.session_state.get("brief_city_ready")
            and st.session_state.get("brief_date_ready")
            and st.session_state.get("view") == "results"
        ):
            step = "task"
        else:
            return
    _qa_button_style()
    if step == "city":
        return
    if step == "date":
        latest = _latest_forecast_label()
        day = _latest_forecast_day()
        if st.button(_tx("btn_latest", day=day), width="stretch", key="qa_date_latest", type="primary"):
            st.session_state.messages.append({"role": "user", "text": latest, "artifact": None})
            _qa_accept_date(latest)
            st.rerun()
        return
    if step == "confirm_train":
        yes, no = st.columns(2)
        if yes.button(_tx("btn_train_yes"), width="stretch", key="qa_train_yes", type="primary"):
            st.session_state.messages.append({"role": "user", "text": _tx("btn_train_yes"), "artifact": None})
            st.session_state.brief_confirm_new_region = True
            _launch_pending_or_intent(st.session_state.get("pending_brief_task"))
        if no.button(_tx("btn_train_no"), width="stretch", key="qa_train_no", type="primary"):
            st.session_state.messages.append({"role": "user", "text": _tx("btn_train_no"), "artifact": None})
            st.session_state.brief_confirm_new_region = False
            _qa_ask_city()
            st.rerun()
        return
    cols = st.columns(3)
    task_btns = (
        (_tx("btn_plan"), "Plan 6 sites", "plan"),
        (_tx("btn_forecast"), "Show the forecast", "forecast"),
        (_tx("btn_compare"), "Compare the four allocation policies.", "compare"),
    )
    for col, (shown, payload, intent_name) in zip(cols, task_btns):
        if col.button(shown, width="stretch", key=f"qa_task_{intent_name}", type="primary"):
            st.session_state.pending_user = payload
            if _needs_train_confirm():
                st.session_state.messages.append({"role": "user", "text": shown, "artifact": None})
                st.session_state.pending_brief_task = intent_name
                _qa_ask_confirm_train()
                st.rerun()
            st.session_state.brief_qa_step = None
            st.rerun()


def _qa_placing_text(mode: str, threshold: float) -> str:
    return _tx("placing", mode=travel_display(_lang(), mode), threshold=int(threshold))


def _qa_reset_travel_picks() -> None:
    st.session_state.alloc_qa_travel = None
    st.session_state.alloc_qa_threshold = None


def _text_picks_input_mode(text: str) -> str | None:
    lowered = str(text or "").lower()
    if any(
        word in lowered
        for word in ("自己输入", "直接输入", "我来输入", "我输入", "type my", "i'll type", "i will type", "type it")
    ):
        return "type"
    if any(
        word in lowered
        for word in ("点选", "点击", "按钮", "给出的范围", "互动按钮", "click", "use the button", "the options", "the range")
    ):
        return "buttons"
    return None


def _accept_input_mode(mode: str, *, session_key: str, hint_key: str) -> None:
    st.session_state[session_key] = mode
    if mode == "buttons":
        return
    # Do not echo the hint again — the input box already sits under the question.


def _render_input_mode_choice(*, key_prefix: str, session_key: str, hint_type: str, hint_buttons: str) -> str | None:
    mode = st.session_state.get(session_key)
    if mode in {"type", "buttons"}:
        return str(mode)
    _qa_button_style()
    with st.chat_message("assistant", avatar=AGENT_AVATAR):
        st.markdown(f"**{_tx('input_mode_offer')}**")
        if st.button(_tx("btn_input_type"), width="stretch", key=f"qa_input_type_{key_prefix}", type="primary"):
            st.session_state.messages.append({"role": "user", "text": _tx("btn_input_type"), "artifact": None})
            _accept_input_mode("type", session_key=session_key, hint_key=hint_type)
            st.rerun()
        if st.button(_tx("btn_input_buttons"), width="stretch", key=f"qa_input_buttons_{key_prefix}", type="primary"):
            st.session_state.messages.append({"role": "user", "text": _tx("btn_input_buttons"), "artifact": None})
            _accept_input_mode("buttons", session_key=session_key, hint_key=hint_buttons)
            st.rerun()
    return None


def _qa_ask_scenario() -> None:
    st.session_state.alloc_qa_step = "scenario"
    st.session_state.alloc_input_mode = "type"
    _qa_reset_travel_picks()
    st.session_state.explicit_task = True
    _open_result_page("allocation")
    st.session_state.messages.append({"role": "assistant", "text": _tx("scenario_question"), "artifact": "allocation"})


def _qa_ask_travel(*, scenario_picked: str) -> None:
    st.session_state.alloc_qa_step = "travel"
    st.session_state.alloc_input_mode = "type"
    _qa_reset_travel_picks()
    st.session_state.explicit_task = True
    _open_result_page("allocation")
    st.session_state.messages.append(
        {
            "role": "assistant",
            "text": _tx("use_policy", policy=scenario_display(_lang(), scenario_picked)) + _tx("travel_question"),
            "artifact": "allocation",
        }
    )


def _qa_finish_and_allocate() -> None:
    travel_choice = st.session_state.alloc_qa_travel
    threshold_choice = st.session_state.alloc_qa_threshold
    if travel_choice:
        st.session_state.brief_travel_label = travel_choice
    if threshold_choice is not None:
        st.session_state.brief_threshold = float(threshold_choice)
    st.session_state.override_scenario = None
    st.session_state.alloc_qa_step = None
    _qa_reset_travel_picks()
    st.session_state.pending_user = "Show the allocation"
    st.session_state.force_realloc = True


def _qa_accept_scenario(scenario_key_or_label: str) -> None:
    try:
        key = map_scenario(scenario_key_or_label)
        label = SCENARIO_LABELS[key]
    except Exception:
        label = scenario_key_or_label
        if label not in SCENARIO_LABELS.values():
            _qa_ask_scenario()
            return
    st.session_state.brief_scenario_label = label
    st.session_state.override_scenario = None
    _qa_ask_travel(scenario_picked=label)


def _maybe_begin_siting_qa(intent: dict, text: str, from_brief: bool) -> bool:
    """Start stepwise siting Q&A instead of dumping every control at once."""
    if st.session_state.get("force_realloc"):
        return False
    if intent["intent"] == "plan":
        wants = True
    elif intent["intent"] == "allocation":
        prior = st.session_state.get("pipeline_result") or {}
        if prior.get("success") and (prior.get("allocation") or {}).get("selected_sites"):
            return False
        if from_brief:
            return False
        wants = True
    else:
        wants = False
    if not wants:
        return False
    if intent.get("scenario") and not from_brief:
        _qa_accept_scenario(str(intent["scenario"]))
    else:
        _qa_ask_scenario()
    return True


def _continue_siting_qa_from_text(text: str, intent: dict) -> bool:
    """Handle a typed answer while the siting Q&A is open."""
    step = st.session_state.get("alloc_qa_step")
    if not step:
        return False
    lowered = str(text or "").lower()
    picked_mode = _text_picks_input_mode(text)
    if picked_mode == "buttons":
        st.session_state.alloc_input_mode = "buttons"
        offer = "show_policy_options" if step == "scenario" else "show_travel_options"
        st.session_state.messages.append({"role": "assistant", "text": _tx(offer), "artifact": "allocation"})
        return True
    if picked_mode == "type":
        st.session_state.alloc_input_mode = "type"
        return True
    if not st.session_state.get("alloc_input_mode"):
        st.session_state.alloc_input_mode = "type"
    if step == "scenario":
        picked = intent.get("scenario")
        if not picked:
            for key, label in SCENARIO_LABELS.items():
                if key in lowered or label.lower() in lowered:
                    picked = key
                    break
        if not picked:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "text": _chat_reply(text, step="scenario", chat=intent.get("chat")),
                    "artifact": None,
                }
            )
            return True
        _qa_accept_scenario(str(picked))
        return True
    if intent.get("intent") == "help" and not any(
        word in lowered for word in ("walk", "步行", "drive", "开车", "驾车")
    ) and not any(re.search(rf"\b{int(value)}\b", str(text or "")) for value in QA_THRESHOLDS):
        st.session_state.messages.append(
            {
                "role": "assistant",
                "text": _chat_reply(text, step="travel", chat=intent.get("chat")),
                "artifact": None,
            }
        )
        return True
    mode = st.session_state.get("alloc_qa_travel")
    if "walk" in lowered or "步行" in lowered:
        mode = list(TRAVEL_MODE_LABELS.values())[1]
    elif "drive" in lowered or "开车" in lowered or "驾车" in lowered:
        mode = list(TRAVEL_MODE_LABELS.values())[0]
    threshold = st.session_state.get("alloc_qa_threshold")
    for value in QA_THRESHOLDS:
        if re.search(rf"\b{int(value)}\b", str(text or "")):
            threshold = float(value)
            break
    st.session_state.alloc_qa_travel = mode
    st.session_state.alloc_qa_threshold = threshold
    if mode and threshold is not None:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "text": _qa_placing_text(mode, threshold),
                "artifact": None,
            }
        )
        _qa_finish_and_allocate()
        st.rerun()
    missing = []
    if not mode:
        missing.append("驾车或步行" if _lang() == "zh" else "Drive or Walk")
    if threshold is None:
        missing.append("时限（10、15、20、25 或 30 分钟）" if _lang() == "zh" else "a threshold (10, 15, 20, 25 or 30 min)")
    st.session_state.messages.append(
        {
            "role": "assistant",
            "text": _tx("travel_need_both", missing=" 和 ".join(missing) if _lang() == "zh" else " and ".join(missing)),
            "artifact": None,
        }
    )
    return True


def _render_siting_qa() -> None:
    step = st.session_state.get("alloc_qa_step")
    if step not in {"scenario", "travel"}:
        return
    input_mode = st.session_state.get("alloc_input_mode") or "type"
    if input_mode == "type":
        _render_typed_answer_box(
            key_prefix=f"siting_type_{step}",
            hint=_tx("placeholder_type_scenario" if step == "scenario" else "placeholder_type_travel"),
            empty_mode_key="alloc_input_mode",
            empty_offer_key="show_policy_options" if step == "scenario" else "show_travel_options",
        )
        return
    if input_mode != "buttons":
        return
    _qa_button_style()
    if step == "scenario":
        st.markdown(_tx("show_policy_options"))
        cols = st.columns(2)
        for i, label in enumerate(SCENARIO_LABELS.values()):
            shown = scenario_display(_lang(), label)
            if cols[i % 2].button(shown, width="stretch", key=f"qa_scen_{label}", type="primary"):
                st.session_state.messages.append({"role": "user", "text": shown, "artifact": None})
                _qa_accept_scenario(label)
                st.rerun()
        return
    mode_cols = st.columns(2)
    for col, label in zip(mode_cols, TRAVEL_MODE_LABELS.values()):
        selected = st.session_state.get("alloc_qa_travel") == label
        shown = travel_display(_lang(), label)
        if col.button(
            f"{'● ' if selected else ''}{shown}",
            width="stretch",
            type="primary",
            key=f"qa_mode_{label}",
        ):
            st.session_state.alloc_qa_travel = label
            st.session_state.messages.append({"role": "user", "text": shown, "artifact": None})
            if st.session_state.get("alloc_qa_threshold") is not None:
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "text": _qa_placing_text(label, st.session_state.alloc_qa_threshold),
                        "artifact": None,
                    }
                )
                _qa_finish_and_allocate()
            else:
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "text": _tx("travel_need_mode", value=shown),
                        "artifact": None,
                    }
                )
            st.rerun()
    tcols = st.columns(len(QA_THRESHOLDS))
    for col, value in zip(tcols, QA_THRESHOLDS):
        selected = st.session_state.get("alloc_qa_threshold") == value
        if col.button(
            f"{'● ' if selected else ''}{int(value)}",
            width="stretch",
            type="primary",
            key=f"qa_thr_{int(value)}",
        ):
            st.session_state.alloc_qa_threshold = float(value)
            st.session_state.messages.append(
                {"role": "user", "text": f"{int(value)} {_tx('label_min')}", "artifact": None}
            )
            if st.session_state.get("alloc_qa_travel"):
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "text": _qa_placing_text(st.session_state.alloc_qa_travel, value),
                        "artifact": None,
                    }
                )
                _qa_finish_and_allocate()
            else:
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "text": _tx("travel_need_threshold", value=int(value)),
                        "artifact": None,
                    }
                )
            st.rerun()

def _resolve_user_intent(text: str, intent: dict) -> dict:
    """If the parser called the request small-talk, recover the planning task from the wording."""
    name = str(intent.get("intent") or "help")
    if name in {"plan", "allocation", "forecast", "compare", "notes", "explain_site", "explain_zone"}:
        return intent
    try:
        if looks_like_compare(text):
            return {"intent": "compare"}
        if looks_like_forecast(text):
            return {"intent": "forecast"}
        if looks_like_allocation(text):
            return {"intent": "allocation"}
        if looks_like_plan(text):
            return {"intent": "plan"}
    except Exception:
        lowered = str(text or "").lower()
        if any(word in lowered for word in ("vaccination site", "distribution of", "接种点", "疫苗点")):
            return {"intent": "allocation"}
        if any(word in lowered for word in ("forecast", "infection", "预测", "预报", "感染率")):
            return {"intent": "forecast"}
    return intent


def handle_user_turn(text: str) -> None:
    st.session_state.explicit_task = True
    _note_user_language(text)
    from_brief = bool(st.session_state.pop("pending_from_brief", False)) or "task brief" in text.lower()
    prior = st.session_state.get("pipeline_result")
    site_ids, iz_codes = _known_ids(prior)
    intent = _resolve_user_intent(text, parse_agent_intent(text, site_ids=site_ids, iz_codes=iz_codes))
    if intent["intent"] in {"forecast", "compare", "notes", "explain_zone"}:
        st.session_state.alloc_qa_step = None
        st.session_state.alloc_qa_travel = None
        st.session_state.alloc_qa_threshold = None
    if intent.get("scenario") and not from_brief:
        st.session_state.override_scenario = intent["scenario"]
    if from_brief:
        st.session_state.override_scenario = None
        st.session_state.alloc_qa_step = None
        st.session_state.alloc_qa_travel = None
        st.session_state.alloc_qa_threshold = None

    brief_step = st.session_state.get("brief_qa_step")
    iz_step = st.session_state.get("iz_qa_step")
    city_hit = _parse_city_name(text)
    date_hit = _canonical_date_label(text)
    task_intents = {"plan", "allocation", "forecast", "compare"}
    if intent["intent"] == "help" and (city_hit or date_hit):
        if looks_like_allocation(text):
            intent = {"intent": "allocation"}
        elif city_hit and date_hit:
            intent = {"intent": "forecast"}
        elif date_hit and looks_like_forecast(text):
            intent = {"intent": "forecast"}
    if city_hit and intent["intent"] in task_intents:
        st.session_state.messages.append({"role": "user", "text": text, "artifact": None})
        if intent["intent"] == "forecast":
            st.session_state.pending_forecast_text = text
        _qa_accept_city(
            city_hit,
            extra_date=date_hit,
            extra_intent=intent["intent"],
        )
        return
    if date_hit and intent["intent"] in task_intents:
        if intent["intent"] == "forecast":
            st.session_state.pending_forecast_text = text
        st.session_state.messages.append({"role": "user", "text": text, "artifact": None})
        if st.session_state.get("brief_city_ready"):
            _qa_accept_date(date_hit, extra_intent=intent["intent"])
        else:
            st.session_state.brief_date = date_hit
            st.session_state.brief_date_ready = True
            st.session_state.pending_brief_task = intent["intent"]
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "text": _tx("chat_date_need_city", date=str(date_hit).split()[0]),
                    "artifact": None,
                }
            )
            st.session_state.brief_qa_step = "city"
        return
    if brief_step in {"city", "date", "confirm_train", "task"} and text:
        st.session_state.messages.append({"role": "user", "text": text, "artifact": None})
        if _continue_brief_qa_from_text(text, intent):
            return
    if iz_step in {"scope", "zone", "zone_disambiguate"} and text:
        if brief_step not in {"city", "date", "confirm_train", "task"}:
            st.session_state.messages.append({"role": "user", "text": text, "artifact": None})
        if _continue_iz_qa_from_text(text, intent):
            return

    in_qa = bool(st.session_state.get("alloc_qa_step"))
    starting_qa = (
        not in_qa
        and not st.session_state.get("force_realloc")
        and intent["intent"] in {"plan", "allocation"}
    )
    logged = brief_step in {"city", "date", "confirm_train", "task"} or iz_step in {
        "scope",
        "zone",
        "zone_disambiguate",
    }
    if starting_qa:
        user_text = None if (from_brief or logged) else text
    else:
        user_text = None if logged else text
    if user_text:
        st.session_state.messages.append(
            {"role": "user", "text": user_text, "artifact": None}
        )

    if st.session_state.get("alloc_qa_step"):
        if _continue_siting_qa_from_text(text, intent):
            if st.session_state.get("pending_user"):
                st.rerun()
            return

    if not from_brief and _apply_forecast_text_choice(text):
        return

    wants_run = intent["intent"] in {"plan", "allocation", "forecast", "compare"}
    if wants_run and not st.session_state.get("force_realloc"):
        if not st.session_state.get("brief_city_ready"):
            st.session_state.pending_brief_task = intent["intent"]
            if st.session_state.get("brief_qa_step") != "city":
                _qa_ask_city()
            return
        if not st.session_state.get("brief_date_ready"):
            st.session_state.pending_brief_task = intent["intent"]
            _qa_ask_date(city_picked=st.session_state.brief_study_area)
            return
        if _needs_train_confirm():
            st.session_state.pending_brief_task = intent["intent"]
            _qa_ask_confirm_train()
            return
        st.session_state.brief_qa_step = None

    if _maybe_begin_siting_qa(intent, text, from_brief):
        return

    run_scenario = _active_scenario()
    needs_tools = intent["intent"] in {"plan", "forecast", "compare", "allocation", "explain_site", "explain_zone"}
    brief_matches = st.session_state.get("applied_brief") == current_brief
    force_realloc = bool(st.session_state.pop("force_realloc", False))
    reuse = (
        prior
        and prior.get("success")
        and brief_matches
        and intent["intent"] in {"plan", "forecast", "compare", "allocation", "explain_site", "explain_zone"}
        and not intent.get("scenario")
        and not force_realloc
    )
    result = prior
    if (
        (not result or not result.get("success"))
        and area_code not in {EDINBURGH_CA, "", "UNKNOWN"}
        and glasgow_ready
    ):
        hydrated = hydrate_region_ui(area_code)
        if hydrated and hydrated.get("success"):
            result = hydrated
            prior = hydrated
            st.session_state.pipeline_result = hydrated
            st.session_state.applied_brief = current_brief
            brief_matches = True
            reuse = (
                prior.get("success")
                and brief_matches
                and intent["intent"] in {"plan", "forecast", "compare", "allocation", "explain_site", "explain_zone"}
                and not intent.get("scenario")
                and not force_realloc
            )
    if needs_tools and not reuse:
        with st.spinner(_tx("spinner")):
            result = run_agent_pipeline(
                study_area,
                selected_date,
                run_scenario,
                travel_mode,
                float(travel_threshold),
                tuple(eligible_types),
                bool(confirm_new_region) or bool(glasgow_ready),
            )
        st.session_state.pipeline_result = result
        st.session_state.applied_brief = current_brief

    if intent["intent"] == "help":
        chat_kind = intent.get("chat")
        if result and result.get("success") and chat_kind in {None, "other"}:
            matches = _match_iz(text)
            if len(matches) == 1:
                _qa_accept_iz(matches[0])
                return
            if matches:
                st.session_state.forecast_scope = "iz"
                st.session_state.iz_qa_step = "zone"
                st.session_state.iz_qa_matches = []
                _continue_iz_qa_from_text(text, {"intent": "explain_zone"})
                return
        st.session_state.messages.append(
            {"role": "assistant", "text": _chat_reply(text, chat=chat_kind), "artifact": None}
        )
        return
    if not result:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "text": _tx("no_pipeline"),
                "artifact": None,
            }
        )
        return
    if not result.get("success"):
        st.session_state.messages.append(
            {"role": "assistant", "text": narrate_planning(result, lang=_lang()), "artifact": "notes"}
        )
        return

    selected_now, _ = _known_ids(result)
    if st.session_state.get("inspected_site_id") not in selected_now:
        st.session_state.inspected_site_id = selected_now[0] if selected_now else None

    if intent["intent"] == "plan":
        text = narrate_planning(result, lang=_lang())
        st.session_state.messages.append(
            {"role": "assistant", "text": text, "artifact": "allocation"}
        )
        _open_result_page("allocation")
    elif intent["intent"] == "forecast" or intent["intent"] == "explain_zone":
        if intent.get("iz_code"):
            _qa_accept_iz(str(intent["iz_code"]))
        else:
            _open_forecast_from_request(text, intro=_tx("forecast_intro"))
    elif intent["intent"] == "compare":
        st.session_state.messages.append(
            {
                "role": "assistant",
                "text": _tx("compare_intro"),
                "artifact": "compare",
            }
        )
        _open_result_page("compare")
    elif intent["intent"] == "notes":
        st.session_state.messages.append(
            {
                "role": "assistant",
                "text": _tx("notes_intro"),
                "artifact": "notes",
            }
        )
        _open_result_page("notes")
    elif intent["intent"] == "allocation":
        st.session_state.messages.append(
            {
                "role": "assistant",
                "text": _tx("alloc_intro"),
                "artifact": "allocation",
            }
        )
        _open_result_page("allocation")
    else:
        site_id = intent.get("site_id")
        if site_id not in selected_now:
            site_id = st.session_state.get("inspected_site_id")
        if site_id not in selected_now:
            site_id = selected_now[0] if selected_now else None
        if site_id:
            st.session_state.inspected_site_id = site_id
        st.session_state.messages.append(
            {
                "role": "assistant",
                "text": agent.explain_site_selection(site_id, result, lang=_lang())
                if site_id
                else _tx("alloc_intro"),
                "artifact": "allocation",
            }
        )
        _open_result_page("allocation")


if st.session_state.get("pending_user"):
    handle_user_turn(st.session_state.pop("pending_user"))

res = st.session_state.get("pipeline_result")
gdf_plot = None
plottable = []
sites = []
site_ids = []
iz_codes = []
iz_order = []
diag = {}
forecast = {}
zones = {}
n_high_unc = 0
if res and res.get("success"):
    diag = res["allocation"]["diagnostics"]
    sites = res["allocation"]["selected_sites"]
    iz_map = res["allocation"]["iz_assignments"]
    forecast = res["forecast"]
    zones = forecast["zone_forecasts"]
    n_high_unc = sum(1 for row in zones.values() if str(row.get("uncertainty_flag") or "") == "high")
    site_ids = [site["site_id"] for site in sites]
    iz_codes = sorted(zones.keys())
    if st.session_state.get("inspected_sites_fingerprint") != tuple(site_ids):
        st.session_state.inspected_sites_fingerprint = tuple(site_ids)
        st.session_state.inspected_site_id = site_ids[0]
    if st.session_state.get("iz_picked_by_user"):
        if st.session_state.get("inspected_iz_code") not in iz_codes:
            st.session_state.iz_picked_by_user = False
            st.session_state.inspected_iz_code = None
            st.session_state.iz_inspect_box = None
    elif iz_codes and st.session_state.get("inspected_iz_code") not in iz_codes:
        st.session_state.inspected_iz_code = None
        st.session_state.iz_inspect_box = None
    map_area = str((res.get("bundle") or {}).get("request", {}).get("area_code") or EDINBURGH_CA)
    MAP_CENTER.update(MAP_CENTERS.get(map_area, MAP_CENTERS[EDINBURGH_CA]))
    try:
        gdf_plot = load_zone_boundaries(map_area).copy()
    except Exception:
        gdf_plot = None
    if gdf_plot is not None and not gdf_plot.empty:
        gdf_plot = _attach_iz_names(gdf_plot)
        gdf_plot["iz_code"] = gdf_plot["iz_code"].astype(str)
        gdf_plot["Predicted rate"] = gdf_plot["iz_code"].map(
            lambda z: float(zones.get(z, {}).get("predicted_rate") or 0.0)
        )
        gdf_plot["Uncertainty (σ)"] = gdf_plot["iz_code"].map(
            lambda z: float(zones.get(z, {}).get("predicted_sigma") or 0.0)
        )
        gdf_plot["Uncertainty flag"] = gdf_plot["iz_code"].map(
            lambda z: (
                _tx("flag_high")
                if str(zones.get(z, {}).get("uncertainty_flag") or "").lower() == "high"
                else (_tx("flag_normal") if zones.get(z, {}).get("uncertainty_flag") else "")
            )
        )
        gdf_plot["Coverage"] = gdf_plot["iz_code"].map(
            lambda z: "Covered" if iz_map.get(z, {}).get("is_covered") else "Unserved"
        )
        gdf_plot["Assigned site"] = gdf_plot["iz_code"].map(
            lambda z: str(iz_map.get(z, {}).get("assigned_site_id") or "Unserved")
        )
        gdf_plot["Travel time (min)"] = gdf_plot["iz_code"].map(
            lambda z: iz_map.get(z, {}).get("travel_time_min")
        )
        keep = [
            "iz_code",
            "iz_name",
            "geometry",
            "Predicted rate",
            "Uncertainty (σ)",
            "Uncertainty flag",
            "Coverage",
            "Assigned site",
            "Travel time (min)",
        ]
        gdf_plot = gdf_plot[[col for col in keep if col in gdf_plot.columns]]
        iz_order = gdf_plot["iz_code"].astype(str).tolist()
    plottable = [site for site in sites if site.get("easting") is not None and site.get("northing") is not None]
    iz_codes = sorted(iz_codes, key=lambda code: (_iz_label(code).lower(), str(code)))


def _render_forecast_maps(*, scope: str) -> None:
    hover = {
        "Predicted rate": ":.1f",
        "Uncertainty (σ)": ":.1f",
        "Uncertainty flag": True,
        "Coverage": False,
        "Assigned site": False,
        "Travel time (min)": False,
    }
    current_site = st.session_state.get("inspected_site_id")
    st.caption(_tx("forecast_maps"))
    st.markdown(f"**{_tx('predicted_rate')}**")
    st.caption(_tx("red_points"))
    fig_rate = _choropleth(gdf_plot, "Predicted rate", hover, scale="Viridis")
    _add_allocated_sites(fig_rate, plottable, highlight_id=current_site)
    fig_rate.update_layout(height=480, showlegend=False)
    rate_event = st.plotly_chart(
        go.Figure(fig_rate),
        width="stretch",
        on_select="rerun",
        selection_mode="points",
        key="forecast_map_rate_sites",
        theme=None,
    )
    st.markdown(f"**{_tx('uncertainty')}**")
    fig_unc = _choropleth(gdf_plot, "Uncertainty (σ)", hover, scale="Plasma")
    _add_allocated_sites(fig_unc, plottable, highlight_id=current_site)
    fig_unc.update_layout(height=480, showlegend=False)
    unc_event = st.plotly_chart(
        go.Figure(fig_unc),
        width="stretch",
        on_select="rerun",
        selection_mode="points",
        key="forecast_map_sigma_sites",
        theme=None,
    )
    for event in (rate_event, unc_event):
        clicked_site = site_id_from_map_event(event, plottable)
        clicked_iz = iz_code_from_map_event(event, iz_order)
        if clicked_site:
            clicked_iz = _iz_code_under_site(clicked_site, plottable, gdf_plot) or clicked_iz
        if not clicked_iz and not clicked_site:
            continue
        site_changed = bool(clicked_site) and clicked_site != st.session_state.get("inspected_site_id")
        iz_changed = bool(clicked_iz) and clicked_iz != st.session_state.get("iz_inspect_box")
        if clicked_site:
            st.session_state.inspected_site_id = clicked_site
        if clicked_iz:
            if st.session_state.get("iz_qa_step") in {"scope", "zone", "zone_disambiguate"} or scope == "region":
                _qa_accept_iz(clicked_iz, from_map=True)
                st.rerun()
            st.session_state.inspected_iz_code = clicked_iz
            st.session_state.iz_inspect_box = clicked_iz
            st.session_state.iz_picked_by_user = True
            st.session_state.forecast_scope = "iz"
        if site_changed or iz_changed:
            st.rerun()
        break


def _render_iz_detail() -> None:
    iz_choice = st.session_state.get("inspected_iz_code")
    if iz_choice not in iz_codes:
        return
    st.session_state.inspected_iz_code = iz_choice
    if st.button(_tx("btn_another_zone"), key="restart_iz_qa", type="primary", width="stretch"):
        _qa_ask_zone()
        st.rerun()
    if st.button(_tx("btn_switch_region"), key="switch_to_region", type="primary", width="stretch"):
        st.session_state.messages.append({"role": "user", "text": _tx("btn_switch_region"), "artifact": None})
        _qa_accept_scope_region()
        st.rerun()
    zone_row = zones.get(iz_choice) or {}
    z1, z2, z3 = st.columns(3)
    rate = zone_row.get("predicted_rate")
    sigma = zone_row.get("predicted_sigma")
    flag_raw = str(zone_row.get("uncertainty_flag") or "")
    flag_shown = _tx("flag_high") if flag_raw.lower() == "high" else (_tx("flag_normal") if flag_raw else "—")
    z1.metric(_tx("predicted_rate"), f"{float(rate):.1f}" if rate is not None and pd.notna(rate) else "—")
    z2.metric(_tx("uncertainty"), f"{float(sigma):.1f}" if sigma is not None and pd.notna(sigma) else "—")
    z3.metric(_tx("uncertainty_flag"), flag_shown)
    render_geoshapley(forecast, zones, df_shapley)


def render_forecast_artifact() -> None:
    _render_iz_qa(key_prefix="map")
    scope = st.session_state.get("forecast_scope")
    if not scope:
        return
    if gdf_plot is None or getattr(gdf_plot, "empty", True):
        if scope == "region":
            render_graph_fusion()
        return
    if scope == "iz" and not st.session_state.get("iz_picked_by_user"):
        return
    if forecast.get("is_unverified"):
        st.warning(_tx("unverified"))
    if scope == "region":
        _render_forecast_maps(scope=scope)
        render_graph_fusion()
        st.subheader(_tx("region_summary"))
        rates = [float(row.get("predicted_rate") or 0) for row in zones.values()]
        r1, r2, r3 = st.columns(3)
        r1.metric(_tx("region_zones"), f"{len(zones)}")
        mean_rate = sum(rates) / len(rates) if rates else 0
        r2.metric(_tx("predicted_rate"), f"{mean_rate:.1f}")
        r3.metric(_tx("region_high_unc"), f"{n_high_unc}")
        if st.button(_tx("btn_scope_iz"), key="switch_to_iz", type="primary"):
            st.session_state.messages.append({"role": "user", "text": _tx("btn_scope_iz"), "artifact": None})
            _qa_accept_scope_iz()
            st.rerun()
        return
    _render_iz_detail()
    _render_forecast_maps(scope=scope)
    render_glossary()


def render_allocation_artifact() -> None:
    if st.session_state.get("alloc_qa_step") in {"scenario", "travel"}:
        _render_siting_qa()
        return
    st.caption(
        _tx(
            "policy_line",
            policy=_display_scenario_label(),
            threshold=float(travel_threshold),
            travel=travel_display(_lang(), travel_label),
        )
    )
    if st.button(_tx("btn_change_policy"), key="restart_siting_qa", type="primary"):
        _qa_ask_scenario()
        st.rerun()
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric(_tx("covered_pop"), f"{diag['covered_population']:,}")
    k2.metric(_tx("izs_covered"), f"{diag['covered_zones']} / {diag['total_zones']}", f"{diag['coverage_percentage']}%")
    k3.metric(_tx("mean_travel"), f"{diag['mean_travel_time_min']} {_tx('label_min')}")
    k4.metric(_tx("max_travel"), f"{diag['max_travel_time_min']} {_tx('label_min')}")
    k5.metric(_tx("unserved"), f"{diag['unserved_zones']}")
    st.caption(_tx("solver_caption", n=n_high_unc))
    current_id = st.session_state.get("inspected_site_id")
    if current_id not in site_ids and site_ids:
        current_id = site_ids[0]
        st.session_state.inspected_site_id = current_id
    st.markdown(f"**{_tx('coverage')}**")
    fig_alloc = _choropleth(
        gdf_plot,
        "Coverage",
        {
            "Coverage": True,
            "Assigned site": True,
            "Travel time (min)": True,
            "Predicted rate": False,
            "Uncertainty (σ)": False,
            "Uncertainty flag": False,
        },
        discrete=True,
    )
    _add_allocated_sites(fig_alloc, plottable, highlight_id=current_id)
    fig_alloc.update_layout(height=480)
    alloc_event = st.plotly_chart(
        fig_alloc,
        width="stretch",
        on_select="rerun",
        selection_mode="points",
        key="allocation_map",
    )
    st.markdown(f"**{_tx('uncertainty')}**")
    st.caption(_tx("unc_map_caption", n=n_high_unc))
    fig_unc = _choropleth(
        gdf_plot,
        "Uncertainty (σ)",
        {
            "Uncertainty (σ)": ":.1f",
            "Uncertainty flag": True,
            "Predicted rate": ":.1f",
            "Coverage": True,
            "Assigned site": True,
            "Travel time (min)": False,
        },
        scale="Plasma",
    )
    _add_allocated_sites(fig_unc, plottable, highlight_id=current_id)
    fig_unc.update_layout(height=480)
    unc_event = st.plotly_chart(
        fig_unc,
        width="stretch",
        on_select="rerun",
        selection_mode="points",
        key="allocation_uncertainty_map",
    )
    clicked = site_id_from_map_event(alloc_event, plottable) or site_id_from_map_event(
        unc_event, plottable
    )
    if clicked and clicked != current_id:
        st.session_state.inspected_site_id = clicked
        st.rerun()
    st.caption(_tx("click_site"))
    button_cols = st.columns(len(site_ids) or 1)
    for col, site in zip(button_cols, sites):
        sid = site["site_id"]
        selected = sid == current_id
        if col.button(
            f"{'● ' if selected else ''}{sid}",
            key=f"site_btn_{sid}",
            type="primary",
            width="stretch",
            help=f"{site['site_name']} ({site_type_display(_lang(), label_of(SITE_TYPE_LABELS, site['site_type']))})",
        ):
            if sid != current_id:
                st.session_state.inspected_site_id = sid
                st.rerun()
    _render_agent_bubble(agent.explain_site_selection(current_id, res, lang=_lang()))


def render_notes_artifact() -> None:
    st.subheader(_tx("tool_log"))
    if res and res.get("logs"):
        for log in res["logs"]:
            st.code(
                plain_agent_text(f"[{log['step']}]\n{json.dumps(log['details'], indent=2, default=str)}")
            )


_apply_interaction_button_style()
_render_brand_bar()
messages = st.session_state.messages
qa_on_alloc = (
    st.session_state.get("view") == "results"
    and (st.session_state.get("result_artifact") == "allocation")
    and st.session_state.get("alloc_qa_step") in {"scenario", "travel"}
)
qa_on_forecast = (
    st.session_state.get("view") == "results"
    and (st.session_state.get("result_artifact") == "forecast")
    and st.session_state.get("iz_qa_step") in {"scope", "zone", "zone_disambiguate"}
)
qa_on_chat = qa_on_alloc or qa_on_forecast
show_results = qa_on_chat or (
    st.session_state.get("view") == "results"
    and st.session_state.get("explicit_task")
    and res
    and res.get("success")
)

if show_results:
    top = st.columns([1, 5])
    with top[0]:
        if st.button(_tx("btn_back"), type="primary", width="stretch", key="back_to_agent"):
            st.session_state.view = "agent"
            st.session_state.brief_qa_step = "task"
            st.rerun()
    if qa_on_chat:
        _agent_heading(_tx("label_agent"))
        st.caption(_tx("caption_reading", city=city_display(_lang(), study_area), date=selected_date))
        if qa_on_alloc:
            _render_agent_transcript(hide_briefing=True)
            render_allocation_artifact()
        else:
            last = next((msg for msg in reversed(messages) if msg.get("role") == "assistant"), None)
            if last:
                _render_agent_bubble(last.get("text") or "")
            render_forecast_artifact()
    else:
        st.title(_tx("title_result"))
        st.caption(
            f"**{city_display(_lang(), study_area)}** · {selected_date} · {_display_scenario_label()}"
        )
        last = next((msg for msg in reversed(messages) if msg.get("role") == "assistant"), None)
        if last:
            _render_agent_bubble(last.get("text") or "")
        artifact = st.session_state.get("result_artifact") or (last or {}).get("artifact")
        if artifact == "forecast":
            render_forecast_artifact()
        elif artifact == "allocation":
            render_allocation_artifact()
        elif artifact == "compare":
            frame = res["comparison_df"].copy()
            if _lang() == "zh":
                if "Scenario" in frame.columns:
                    frame["Scenario"] = frame["Scenario"].map(lambda v: scenario_display("zh", str(v)))
                frame = frame.rename(
                    columns={
                        "Scenario": _tx("col_scenario"),
                        "Covered IZs": _tx("col_covered_iz"),
                        "Coverage %": _tx("col_coverage_pct"),
                        "Covered Pop": _tx("col_covered_pop"),
                        "Mean Travel Time": _tx("col_mean_tt"),
                        "Max Travel Time": _tx("col_max_tt"),
                        "Unserved IZs": _tx("col_unserved_iz"),
                    }
                )
            st.dataframe(frame, width="stretch", hide_index=True)
        elif artifact == "notes":
            render_notes_artifact()
    if not qa_on_chat:
        _render_agent_followups(key_prefix="result")
else:
    st.title(_tx("title_agent"))
    if st.session_state.get("brief_city_ready"):
        st.caption(_tx("caption_reading", city=city_display(_lang(), study_area), date=selected_date))
    else:
        st.markdown(_tx("home_coverage", latest=_latest_forecast_day()))
        first = st.session_state.messages[0] if st.session_state.messages else None
        if first and first.get("role") == "assistant" and len(st.session_state.messages) == 1:
            first["text"] = _tx("greeting")
    _agent_heading(_tx("label_agent"))
    _render_agent_transcript()
    _render_ask_bar(key_prefix="agent")
    _render_brief_qa()

