"""Adapt the GeoMind Streamlit contract onto the local planning agent.

The language model does not choose sites. Allocation comes from
allocation.engine via agent.planning.
"""

from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd

from agent.planning import run_planning
from allocation.contracts import (
    EDINBURGH_CA,
    N_SITES,
    SCENARIO_ALIASES,
    SCENARIO_LABELS,
)
from common.errors import ModelError
from common.utils import LOCAL_AUTHORITY_NAME, PANEL_CSV, project_root
from presentation.display_labels import SITE_TYPE_LABELS, TRAVEL_MODE_LABELS, label_of
from presentation.ui_i18n import scenario_display, site_type_display, t, travel_display

UI_SCENARIOS = (
    "balanced",
    "coverage priority",
    "equity priority",
    "preventive priority",
)

_IZ_RE = re.compile(r"\bS0200\d{4}\b", re.I)
_SITE_RE = re.compile(r"\b(?:GP|PH|MS)_\d+\b", re.I)

_COMPARE_WORDS = (
    "compare",
    "comparison",
    "contrast",
    "versus",
    " vs ",
    "four polic",
    "four scenario",
    "four policies",
    "which policy",
    "which policies",
    "better policy",
    "policy difference",
    "比较",
    "对比",
    "四种",
    "四个政策",
    "哪个政策",
    "哪种政策",
    "政策差别",
    "政策差异",
)
_NOTES_WORDS = (
    "glossary",
    "what does",
    "what do the labels",
    "alpha weight",
    "audit log",
    "tool log",
    "术语",
    "标签含义",
    "名词解释",
)
_FORECAST_WORDS = (
    "forecast",
    "prediction",
    "predict",
    "projected",
    "projection",
    "uncertainty",
    "geoshapley",
    "shapley",
    "infection rate",
    "infection-rate",
    "case rate",
    "incidence",
    "predicted rate",
    "rate map",
    "risk map",
    "choropleth",
    "how bad",
    "how many cases",
    "outbreak",
    "epidemic",
    "covid map",
    "covid forecast",
    "预测",
    "预报",
    "预估",
    "推演",
    "感染率",
    "发病率",
    "疫情",
    "不确定性",
    "结果",
    "results",
    "result",
    "看图",
    "预报图",
    "预测图",
    "热力图",
)
_ALLOCATION_WORDS = (
    "allocation",
    "show the sites",
    "show the site",
    "selected sites",
    "where did you",
    "where are the site",
    "where the site",
    "where are sites",
    "where is the site",
    "vaccination site",
    "vaccination sites",
    "vaccine site",
    "vaccine sites",
    "jab site",
    "clinic location",
    "clinic map",
    "site map",
    "site locations",
    "site location",
    "site distribution",
    "spatial distribution",
    "distribution of",
    "layout of site",
    "placed the site",
    "placed sites",
    "the six sites",
    "the 6 sites",
    "those six sites",
    "those 6 sites",
    "the six points",
    "red points",
    "get vaccinated",
    "get a vaccine",
    "where to vaccinate",
    "where can i vaccinate",
    "pharmacy site",
    "gp site",
    "选中点",
    "已选",
    "候选点",
    "选址图",
    "接种点分布",
    "点位分布",
    "接种点在哪",
    "接种点在",
    "疫苗点",
    "疫苗接种点",
    "接种地点",
    "接种位置",
    "放在哪",
    "点位图",
    "六个点",
    "6 个点",
    "6个点",
    "红点",
)
_PLAN_WORDS = (
    "plan 6",
    "plan six",
    "place 6",
    "place six",
    "allocate",
    "siting",
    "run the agent",
    "task brief",
    "choose sites",
    "choose 6",
    "pick 6",
    "pick six",
    "find 6",
    "find six",
    "place sites",
    "place the site",
    "sitings",
    "规划",
    "选址",
    "计划",
    "跑一遍",
    "布置",
    "设点",
    "放点",
    "放 6",
    "放6",
    "六个接种",
    "6个接种",
    "6 个接种",
)
_COVERAGE_WORDS = ("coverage", "cover more", "maximi", "覆盖", "尽量覆盖")
_EQUITY_WORDS = ("equity", "equitable", "fairness", "fair access", "公平", "公正")
_PREVENTIVE_WORDS = ("preventive", "prevention", "preventative", "prevent ", "预防")
_BALANCED_WORDS = ("balanced", "balance", "均衡", "平衡", "折中")


def _has_any(text: str, words: tuple[str, ...]) -> bool:
    return any(word in text for word in words)


def looks_like_forecast(text: str) -> bool:
    return _has_any(str(text or "").lower(), _FORECAST_WORDS)


def looks_like_allocation(text: str) -> bool:
    return _has_any(str(text or "").lower(), _ALLOCATION_WORDS)


def looks_like_compare(text: str) -> bool:
    return _has_any(str(text or "").lower(), _COMPARE_WORDS)


def looks_like_plan(text: str) -> bool:
    lowered = str(text or "").lower()
    return _has_any(lowered, _PLAN_WORDS) or bool(re.search(r"\bplan\b", lowered))

AGENT_GREETING = t("en", "greeting")

AGENT_HELP = t("en", "help")

_CHAT_GREET = (
    "你好",
    "您好",
    "嗨",
    "哈喽",
    "早上好",
    "下午好",
    "晚上好",
    "hello",
    "hi",
    "hey",
    "good morning",
    "good afternoon",
    "good evening",
)
_CHAT_THANKS = ("thank", "thanks", "thx", "谢谢", "感谢", "多谢", "thank you")
_CHAT_BYE = ("goodbye", "good bye", "see you", "bye", "再见", "拜拜", "下次见")
_CHAT_WHO = ("who are you", "what are you", "你是谁", "你是什么", "你叫什么", "介绍一下你")
_CHAT_CAN = (
    "what can you",
    "what do you do",
    "how do you work",
    "how does this",
    "你能做什么",
    "你会什么",
    "怎么用",
    "能做什么",
    "有什么功能",
    "帮帮我",
    "怎么办",
)
_CHAT_HOW = ("how are you", "how's it going", "你好吗", "最近好吗", "忙吗")
_CHAT_ACK = ("ok", "okay", "好", "好的", "嗯", "行", "可以", "哦", "喔", "明白", "收到")


def classify_chat(text: str) -> str:
    """Label small-talk. Does not route planning tools or choose sites."""
    raw = str(text or "").strip()
    lowered = raw.lower().rstrip("!！。.~？?，,")
    if not lowered:
        return "other"
    if lowered in _CHAT_GREET or lowered in {"hi!", "hello!"}:
        return "greet"
    if len(lowered) <= 16 and any(
        lowered.startswith(prefix) for prefix in ("hello", "hi ", "hey ", "你好", "您好", "嗨")
    ):
        return "greet"
    if any(word in lowered for word in _CHAT_WHO):
        return "identity"
    if any(word in lowered for word in _CHAT_HOW):
        return "wellbeing"
    if any(word in lowered for word in _CHAT_CAN) or lowered in {"help", "帮助", "帮忙"}:
        return "capabilities"
    if any(word in lowered for word in _CHAT_THANKS):
        return "thanks"
    if any(word in lowered for word in _CHAT_BYE):
        return "farewell"
    if lowered in _CHAT_ACK:
        return "ack"
    return "other"


def _chat_snippet(text: str, limit: int = 36) -> str:
    raw = " ".join(str(text or "").split()).replace("{", "").replace("}", "")
    if not raw:
        return "…"
    if len(raw) > limit:
        return raw[:limit] + "…"
    return raw


def narrate_chat(
    text: str,
    *,
    lang: str = "en",
    chat: str | None = None,
    step: str | None = None,
    latest: str | None = None,
) -> str:
    """First-person reply to chit-chat. Never invents sites, years, or geography."""
    kind = chat or classify_chat(text)
    social_key = {
        "greet": "chat_greet",
        "thanks": "chat_thanks",
        "farewell": "chat_bye",
        "identity": "chat_who",
        "capabilities": "chat_can",
        "wellbeing": "chat_how",
        "ack": "chat_ok",
    }.get(kind, "chat_other")
    if social_key == "chat_other":
        social = t(lang, "chat_other", snippet=_chat_snippet(text))
    else:
        social = t(lang, social_key)
    nudge_key = {
        "city": "chat_nudge_city",
        "date": "chat_nudge_date",
        "confirm_train": "chat_nudge_confirm",
        "task": "chat_nudge_task",
        "zone": "chat_nudge_zone",
        "zone_disambiguate": "chat_nudge_zone",
        "scenario": "chat_nudge_scenario",
        "travel": "chat_nudge_travel",
        "results": "chat_nudge_result",
    }.get(str(step or ""), "chat_nudge_task")
    if nudge_key == "chat_nudge_date":
        nudge = t(lang, nudge_key, latest=latest or "YYYY-MM-DD")
    else:
        nudge = t(lang, nudge_key)
    return f"{social}\n\n{nudge}"


def plain_agent_text(text: str) -> str:
    """Strip internal names (U10, checkpoint, trained) from chat copy."""
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


_ISO_DATE = re.compile(r"(20\d{2})[-/.](\d{1,2})[-/.](\d{1,2})")
_CN_DATE = re.compile(r"(20\d{2})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日?")
_CN_MD = re.compile(r"(?<!\d)(\d{1,2})\s*月\s*(\d{1,2})\s*日?")
_EN_MONTHS = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sept": 9,
    "sep": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}
_EN_MONTH_ALT = "|".join(sorted(_EN_MONTHS, key=len, reverse=True))
_EN_DMY = re.compile(rf"(\d{{1,2}})\s+({_EN_MONTH_ALT})\.?,?\s+(20\d{{2}})", re.I)
_EN_MDY = re.compile(rf"({_EN_MONTH_ALT})\.?\s+(\d{{1,2}}),?\s+(20\d{{2}})", re.I)


def _match_date_option(day: str, options: list[str] | None) -> str:
    for opt in options or []:
        if str(opt).startswith(day):
            return str(opt)
    return day


def extract_forecast_date(
    text: str,
    *,
    options: list[str] | None = None,
    latest: str | None = None,
) -> str | None:
    """Read a forecast day from free text. Accepts ISO, Chinese, and English months."""
    raw = str(text or "").strip()
    if not raw:
        return None
    if options and raw in options:
        return raw
    hit = _CN_DATE.search(raw) or _ISO_DATE.search(raw)
    if hit:
        year, month, day = hit.group(1), hit.group(2), hit.group(3)
        packed = f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
        return _match_date_option(packed, options)
    en_dmy = _EN_DMY.search(raw)
    if en_dmy:
        day_n, month_name, year = en_dmy.group(1), en_dmy.group(2), en_dmy.group(3)
        packed = f"{int(year):04d}-{_EN_MONTHS[month_name.lower()]:02d}-{int(day_n):02d}"
        return _match_date_option(packed, options)
    en_mdy = _EN_MDY.search(raw)
    if en_mdy:
        month_name, day_n, year = en_mdy.group(1), en_mdy.group(2), en_mdy.group(3)
        packed = f"{int(year):04d}-{_EN_MONTHS[month_name.lower()]:02d}-{int(day_n):02d}"
        return _match_date_option(packed, options)
    md = _CN_MD.search(raw)
    if md and (options or latest):
        suffix = f"-{int(md.group(1)):02d}-{int(md.group(2)):02d}"
        latest_day = str(latest or "").split()[0]
        if latest_day.endswith(suffix):
            return latest
        matches = [str(opt) for opt in (options or []) if str(opt).split()[0].endswith(suffix)]
        if matches:
            return matches[0]
    lowered = raw.lower()
    if any(word in lowered for word in ("latest", "newest", "extrapolation", "最新")):
        return latest
    return None


def match_zone_names(text: str, names: dict[str, str]) -> list[str]:
    """Find Intermediate Zones mentioned in a sentence, not only an exact label."""
    raw = str(text or "").strip()
    if not raw:
        return []
    lowered = raw.lower()
    if not names:
        hit = re.search(r"\bS0200\d{4}\b", raw, re.I)
        return [hit.group(0).upper()] if hit else []
    exact = [
        code
        for code, name in names.items()
        if code.lower() == lowered or str(name).lower() == lowered
    ]
    if exact:
        return exact
    code_hit = re.search(r"\bS0200\d{4}\b", raw, re.I)
    if code_hit:
        code = code_hit.group(0).upper()
        if code in names:
            return [code]
    scored: list[tuple[int, str]] = []
    for code, name in names.items():
        label = str(name).lower().strip()
        if len(label) >= 4 and label in lowered:
            scored.append((len(label), str(code)))
        elif len(lowered) >= 4 and lowered in label:
            scored.append((len(lowered), str(code)))
    if not scored:
        return []
    best = max(item[0] for item in scored)
    return [code for length, code in scored if length == best]


def text_wants_region_view(text: str) -> bool:
    lowered = str(text or "").lower()
    if any(
        word in lowered
        for word in ("按所选中间区", "selected intermediate zone view", "one intermediate zone")
    ):
        return False
    return any(
        word in lowered
        for word in (
            "whole-region",
            "whole region",
            "city-wide",
            "citywide",
            "all intermediate zones",
            "全市",
            "全区域",
            "全城",
            "整市",
            "整个区域",
            "全部中间区",
            "改回全区域",
            "back to whole",
        )
    )


def text_wants_iz_view(text: str) -> bool:
    if text_wants_region_view(text):
        return False
    lowered = str(text or "").lower()
    return any(
        word in lowered
        for word in (
            "selected intermediate",
            "one intermediate",
            "按中间区",
            "按所选",
            "单个中间区",
            "一个中间区",
            "某个中间区",
            "某个街区",
            "换一个",
            "重新选择中间区",
            "another zone",
            "another intermediate",
            "interzone",
        )
    )


def parse_agent_intent(
    text: str,
    *,
    site_ids: list[str] | None = None,
    iz_codes: list[str] | None = None,
) -> dict[str, Any]:
    """Map a short user request onto a planning tool. Does not choose sites."""
    raw = str(text or "").strip()
    lowered = raw.lower()
    known_sites = [str(site) for site in (site_ids or [])]
    for site_id in known_sites:
        if site_id.lower() in lowered:
            return {"intent": "explain_site", "site_id": site_id}
    site_hit = _SITE_RE.search(raw)
    if site_hit:
        return {"intent": "explain_site", "site_id": site_hit.group(0)}
    iz_hit = _IZ_RE.search(raw)
    if iz_hit:
        code = iz_hit.group(0).upper()
        known_iz = {str(code) for code in (iz_codes or [])}
        if not known_iz or code in known_iz:
            return {"intent": "explain_zone", "iz_code": code}

    scenario = None
    if _has_any(lowered, _COVERAGE_WORDS):
        scenario = "coverage priority"
    elif _has_any(lowered, _EQUITY_WORDS):
        scenario = "equity priority"
    elif _has_any(lowered, _PREVENTIVE_WORDS):
        scenario = "preventive priority"
    elif _has_any(lowered, _BALANCED_WORDS):
        scenario = "balanced"

    if _has_any(lowered, _COMPARE_WORDS):
        return {"intent": "compare", "scenario": scenario}
    if _has_any(lowered, _NOTES_WORDS):
        return {"intent": "notes"}
    if _has_any(lowered, _FORECAST_WORDS):
        return {"intent": "forecast"}
    planning = _has_any(lowered, _PLAN_WORDS) or bool(re.search(r"\bplan\b", lowered))
    if _has_any(lowered, _ALLOCATION_WORDS):
        if planning:
            return {"intent": "plan", "scenario": scenario}
        return {"intent": "allocation", "scenario": scenario}
    if planning:
        return {"intent": "plan", "scenario": scenario}
    if scenario:
        return {"intent": "plan", "scenario": scenario}
    return {"intent": "help", "chat": classify_chat(raw)}


def narrate_planning(result: dict[str, Any], *, lang: str = "en") -> str:
    """First-person tool trace. The language model still does not pick sites."""
    if not result.get("success"):
        msg = result.get("message") or t(lang, "no_pipeline")
        if lang == "zh":
            return f"选址前我必须停下。{msg}"
        return f"I had to stop before allocation. {msg}"
    forecast = result.get("forecast") or {}
    diag = (result.get("allocation") or {}).get("diagnostics") or {}
    sites = (result.get("allocation") or {}).get("selected_sites") or []
    names = ", ".join(f"{row['site_id']} ({row['site_name']})" for row in sites)
    policy_en = SCENARIO_LABELS.get(diag.get("scenario"), diag.get("scenario"))
    policy = scenario_display(lang, str(policy_en))
    covered_pop = diag.get("covered_population")
    pop_text = f"{int(covered_pop):,}" if covered_pop is not None else "—"
    request = (result.get("bundle") or {}).get("request") or {}
    compat = result.get("compat") or {}
    mode = str(compat.get("mode") or "inference")
    area_name = request.get("area_name") or compat.get("area_code") or ("this city" if lang != "zh" else "这座城市")
    if lang == "zh":
        from presentation.ui_i18n import city_display

        area_name = city_display("zh", str(area_name))
    travel_shown = travel_display(
        lang,
        TRAVEL_MODE_LABELS.get(str(diag.get("travel_mode") or "").lower(), str(diag.get("travel_mode") or "")),
    )
    label = str(forecast.get("label") or "")
    reused_existing = "reused" in label.lower() or "existing" in label.lower()
    if lang == "zh":
        if mode == "new_region_training" and reused_existing:
            step1 = f"**1. 模型。** 我加载了 {area_name} 已保存的模型，没有重建。"
        elif mode == "new_region_training":
            step1 = f"**1. 模型。** 我只为 {area_name} 建了模型。爱丁堡的模型没有改动。"
        else:
            step1 = "**1. 模型。** 爱丁堡，2011 中间区。我使用已保存的模型，没有重建。"
        forecast_when = forecast.get("forecast_date") or forecast.get("label") or "选定日期"
        return (
            "任务已完成。\n\n"
            f"{step1}\n\n"
            f"**2. 预报。** {forecast_when}，{forecast.get('total_zones')} 个街区。\n\n"
            f"**3. 选址。** 求解器放置了 {diag.get('fixed_site_count', N_SITES)} 个点，"
            f"{diag.get('travel_time_threshold_min')} 分钟 {travel_shown}，"
            f"政策 **{policy}**。\n\n"
            f"**4. 结果。** 覆盖 {diag.get('covered_zones')}/{diag.get('total_zones')} 个中间区"
            f"（{diag.get('coverage_percentage')}%），人口 {pop_text}。"
            f"平均出行 {diag.get('mean_travel_time_min')} 分钟。"
            f"未覆盖中间区：{diag.get('unserved_zones')}。\n\n"
            f"选中接种点：{names}。\n\n"
            "可以让我查看预测、比较四种政策，或解释某个接种点/街区。"
        )
    if mode == "new_region_training" and reused_existing:
        step1 = (
            f"**1. Model.** I loaded the saved model for {area_name}. "
            "I did not rebuild it."
        )
    elif mode == "new_region_training":
        step1 = (
            f"**1. Model.** I built a model for {area_name} only. "
            "Edinburgh's model was not changed."
        )
    else:
        step1 = (
            "**1. Model.** City of Edinburgh, 2011 Intermediate Zones. "
            "I used the saved model. I did not rebuild it."
        )
    forecast_when = forecast.get("forecast_date") or forecast.get("label") or "the selected date"
    return (
        "I finished the task.\n\n"
        f"{step1}\n\n"
        f"**2. Forecast.** {forecast_when}, {forecast.get('total_zones')} zones.\n\n"
        f"**3. Allocation.** Solver placed {diag.get('fixed_site_count', N_SITES)} sites, "
        f"{diag.get('travel_time_threshold_min')} min {travel_shown}, "
        f"policy **{policy}**.\n\n"
        f"**4. Result.** Covered {diag.get('covered_zones')}/{diag.get('total_zones')} IZs "
        f"({diag.get('coverage_percentage')}%), population {pop_text}. "
        f"Mean travel {diag.get('mean_travel_time_min')} min. "
        f"Unserved IZs: {diag.get('unserved_zones')}.\n\n"
        f"Selected vaccination sites: {names}.\n\n"
        "Ask me to show the forecast, compare the four policies, or explain one site or zone."
    )


def brief_fingerprint(
    *,
    study_area: str,
    forecast_date: str,
    scenario: str,
    travel_mode: str,
    travel_threshold: float,
    eligible_types: list[str] | tuple[str, ...],
    confirm_new_region: bool,
) -> tuple:
    """Identity of the left-hand brief. Used to decide whether the Agent must re-run."""
    return (
        str(study_area),
        str(forecast_date),
        str(scenario),
        str(travel_mode),
        round(float(travel_threshold), 1),
        tuple(sorted(str(item) for item in eligible_types)),
        bool(confirm_new_region),
    )


def plan_from_brief_text(
    *,
    study_area: str,
    forecast_date: str,
    scenario_label: str,
    travel_threshold: float,
    travel_label: str,
    eligible_labels: list[str] | tuple[str, ...],
) -> str:
    """User-turn text that binds the left brief to a planning task."""
    types = ", ".join(str(item) for item in eligible_labels) if eligible_labels else "none"
    return (
        f"Plan 6 vaccination sites using the task brief: {study_area}, {forecast_date}, "
        f"{scenario_label}, {float(travel_threshold):.0f} min {travel_label}, eligible {types}."
    )


def map_scenario(label: str) -> str:
    key = str(label or "balanced").strip().lower()
    if key not in SCENARIO_ALIASES:
        raise ModelError(f"Unknown scenario {label!r}.", code="invalid_config")
    return SCENARIO_ALIASES[key]


def map_priority_population(label: str) -> str:
    text = str(label or "all").strip().lower()
    if text in {"", "all", "total population", "total"}:
        return "all"
    raise ModelError(
        "Priority population splits are not available. The panel only has total IZ population.",
        code="invalid_config",
    )


GLASGOW_CA = "S12000049"
GLASGOW_NAME = "Glasgow City"
STUDY_AREA_OPTIONS = (LOCAL_AUTHORITY_NAME, GLASGOW_NAME)


def map_study_area(study_area: str) -> tuple[str, str]:
    name = str(study_area or LOCAL_AUTHORITY_NAME).strip() or LOCAL_AUTHORITY_NAME
    lowered = name.lower()
    if "glasgow" in lowered or name == GLASGOW_CA:
        return GLASGOW_CA, GLASGOW_NAME
    if "edinburgh" in lowered or name == EDINBURGH_CA:
        return EDINBURGH_CA, LOCAL_AUTHORITY_NAME
    return "UNKNOWN", name


def _population_by_iz(area_code: str | None = None) -> dict[str, float]:
    code = str(area_code or EDINBURGH_CA).strip()
    if code not in {EDINBURGH_CA, "", "UNKNOWN"}:
        path = project_root() / "data" / "results" / "regions" / code / "covid" / "fill1.csv"
    else:
        path = project_root() / "data" / "results" / PANEL_CSV
    if not path.exists():
        return {}
    panel = pd.read_csv(path)
    date_col = "Date" if "Date" in panel.columns else None
    if date_col:
        last = panel[date_col].max()
        panel = panel.loc[panel[date_col] == last]
    zone_col = next((name for name in ("IntZone", "iz_code") if name in panel.columns), None)
    if zone_col is None or "Population" not in panel.columns:
        return {}
    pop = panel[[zone_col, "Population"]].drop_duplicates(zone_col)
    return {
        str(row[0]): float(row[1])
        for row in pop.itertuples(index=False)
        if pd.notna(row[1])
    }


def _site_coords(area_code: str | None = None) -> dict[str, dict[str, float]]:
    try:
        from data.candidate_sites import load_candidate_sites

        sites = load_candidate_sites(area_code=area_code or EDINBURGH_CA)
    except Exception:
        return {}
    out: dict[str, dict[str, float]] = {}
    for row in sites.itertuples(index=False):
        easting = getattr(row, "easting", None)
        northing = getattr(row, "northing", None)
        if pd.isna(easting) or pd.isna(northing):
            continue
        out[str(row.site_id)] = {"easting": float(easting), "northing": float(northing)}
    return out


def _zone_forecasts(forecast: dict[str, Any]) -> dict[str, dict[str, Any]]:
    path = forecast.get("forecast_path")
    if not path:
        return {}
    frame = pd.read_csv(path)
    if "iz_code" not in frame.columns:
        return {}
    if "predicted_rate" not in frame.columns:
        for alias in ("predicted_mu_original", "predicted_mean", "display_mean"):
            if alias in frame.columns:
                frame["predicted_rate"] = frame[alias]
                break
    if "predicted_sigma" not in frame.columns and "predicted_sigma_original" in frame.columns:
        frame["predicted_sigma"] = frame["predicted_sigma_original"]
    records = frame.to_dict(orient="records")
    return {str(row["iz_code"]): row for row in records}


def _apply_region_alpha(zones: dict[str, dict[str, Any]], area_code: str | None) -> None:
    code = str(area_code or "").strip()
    if not zones or not code or code in {EDINBURGH_CA, "UNKNOWN"}:
        return
    from agent.region_training import region_rolling_alpha_path

    path = region_rolling_alpha_path(code)
    if not path.is_file():
        return
    frame = pd.read_csv(path)
    if frame.empty:
        return
    last = frame.iloc[-1]
    values = {}
    for key in ("alpha_geo", "alpha_transport", "alpha_mobility"):
        if key not in last.index or pd.isna(last[key]):
            return
        values[key] = float(last[key])
    for row in zones.values():
        row.update(values)


def _alpha_weights(zones: dict[str, dict[str, Any]]) -> dict[str, float]:
    if not zones:
        return {"alpha_geo": None, "alpha_transport": None, "alpha_mobility": None}
    row = next(iter(zones.values()))
    return {
        "alpha_geo": float(row["alpha_geo"]) if pd.notna(row.get("alpha_geo")) else None,
        "alpha_transport": float(row["alpha_transport"]) if pd.notna(row.get("alpha_transport")) else None,
        "alpha_mobility": float(row["alpha_mobility"]) if pd.notna(row.get("alpha_mobility")) else None,
    }


def bundle_to_ui(bundle: dict[str, Any]) -> dict[str, Any]:
    """Convert run_planning() output to the GeoMind dashboard payload."""
    logs = [
        {"step": "1. Compatibility & Routing Check", "details": bundle.get("compatibility") or {}},
        {"step": "2. Forecast Retrieval", "details": bundle.get("forecast") or {}},
    ]
    if bundle.get("status") != "ok":
        blockers = bundle.get("blockers") or []
        message = (blockers[0] or {}).get("message") if blockers else bundle.get("forecast_label")
        if not message:
            message = (bundle.get("compatibility") or {}).get("reason") or "Planning pipeline paused."
        return {
            "success": False,
            "error_type": "REQUIRES_USER_CONFIRMATION" if bundle.get("status") == "needs_confirmation" else "PIPELINE_ERROR",
            "message": message,
            "logs": logs + [{"step": "blocked", "details": blockers}],
        }

    allocation = bundle.get("allocation") or {}
    forecast = bundle.get("forecast") or {}
    request = bundle.get("request") or {}
    metrics = allocation.get("metrics") or {}
    reasons = allocation.get("selection_reasons") or {}
    assignments = allocation.get("assignments") or []
    zones = _zone_forecasts(forecast)
    _apply_region_alpha(zones, request.get("area_code"))
    pops = _population_by_iz(request.get("area_code"))
    coords = _site_coords(request.get("area_code"))
    n_iz = int(metrics.get("n_iz") or len(assignments) or len(zones) or 0)
    covered = int(metrics.get("iz_covered") or 0)

    iz_map: dict[str, dict[str, Any]] = {}
    assigned_counts: dict[str, int] = {}
    assigned_pop: dict[str, float] = {}
    for row in assignments:
        iz_code = str(row.get("iz_code"))
        served = bool(row.get("served"))
        site_id = row.get("site_id")
        travel = row.get("travel_time_min")
        iz_map[iz_code] = {
            "assigned_site_id": site_id if served and site_id else "Unserved",
            "travel_time_min": round(float(travel), 1) if travel is not None and pd.notna(travel) else None,
            "is_covered": served,
        }
        if served and site_id:
            assigned_counts[str(site_id)] = assigned_counts.get(str(site_id), 0) + 1
            assigned_pop[str(site_id)] = assigned_pop.get(str(site_id), 0.0) + pops.get(iz_code, 0.0)

    selected_sites = []
    for site in allocation.get("selected_sites") or []:
        site_id = str(site["site_id"])
        xy = coords.get(site_id, {})
        selected_sites.append(
            {
                "site_id": site_id,
                "site_name": site.get("site_name") or site_id,
                "site_type": site.get("site_type") or "facility",
                "easting": xy.get("easting"),
                "northing": xy.get("northing"),
                "assigned_iz_count": assigned_counts.get(site_id, 0),
                "served_population": int(round(assigned_pop.get(site_id, 0.0))),
                "selection_reason": reasons.get(site_id) or "Recorded by the deterministic allocator.",
            }
        )

    mean_tt = metrics.get("mean_travel_time_min")
    max_tt = metrics.get("max_travel_time_min")
    diagnostics = {
        "scenario": request.get("scenario") or allocation.get("scenario"),
        "fixed_site_count": len(selected_sites),
        "travel_mode": request.get("travel_mode") or allocation.get("travel_mode"),
        "travel_time_threshold_min": request.get("travel_time_threshold_min")
        or allocation.get("travel_time_threshold_min"),
        "total_zones": n_iz,
        "covered_zones": covered,
        "unserved_zones": int(metrics.get("unserved_iz") or max(n_iz - covered, 0)),
        "covered_population": int(round(float(metrics.get("population_covered") or 0))),
        "unserved_population": int(round(float(metrics.get("unserved_population") or 0))),
        "mean_travel_time_min": round(float(mean_tt), 1) if mean_tt is not None else None,
        "max_travel_time_min": round(float(max_tt), 1) if max_tt is not None else None,
        "coverage_percentage": round((covered / n_iz) * 100, 1) if n_iz else 0.0,
        "message": (allocation.get("diagnostics") or {}).get("message"),
        "warnings": (allocation.get("diagnostics") or {}).get("warnings") or [],
    }

    comparison_rows = []
    for row in (bundle.get("comparison") or {}).get("scenarios") or []:
        mets = row.get("metrics") or {}
        n = int(mets.get("n_iz") or n_iz or 0)
        iz_cov = mets.get("iz_covered")
        comparison_rows.append(
            {
                "Scenario": SCENARIO_LABELS.get(row.get("scenario"), row.get("label") or row.get("scenario")),
                "Covered IZs": f"{iz_cov} / {n}" if iz_cov is not None else "—",
                "Coverage %": f"{round((iz_cov / n) * 100, 1)}%" if iz_cov is not None and n else "—",
                "Covered Pop": f"{int(round(float(mets['population_covered']))):,}"
                if mets.get("population_covered") is not None
                else "—",
                "Mean Travel Time": f"{round(float(mets['mean_travel_time_min']), 1)} min"
                if mets.get("mean_travel_time_min") is not None
                else "—",
                "Max Travel Time": f"{round(float(mets['max_travel_time_min']), 1)} min"
                if mets.get("max_travel_time_min") is not None
                else "—",
                "Unserved IZs": mets.get("unserved_iz"),
            }
        )

    logs.extend(
        [
            {
                "step": "3. Deterministic Location Allocation",
                "details": {
                    "sites_allocated": len(selected_sites),
                    "coverage_pct": diagnostics["coverage_percentage"],
                    "covered_pop": diagnostics["covered_population"],
                    "invented": allocation.get("invented"),
                },
            },
            {"step": "4. Policy Comparison", "details": f"{len(comparison_rows)} scenarios under n_sites={N_SITES}."},
        ]
    )
    alphas = _alpha_weights(zones)
    return {
        "success": True,
        "compat": bundle.get("compatibility"),
        "forecast": {
            "status": forecast.get("status"),
            "forecast_date": forecast.get("target_date") or request.get("forecast_date"),
            "checkpoint_used": forecast.get("checkpoint_id") or (bundle.get("checkpoint") or {}).get("checkpoint_id"),
            "total_zones": len(zones),
            "alpha_weights": alphas,
            "is_unverified": forecast.get("forecast_status") == "unverified_extrapolation",
            "forecast_status": forecast.get("forecast_status"),
            "label": forecast.get("label") or bundle.get("forecast_label"),
            "zone_forecasts": zones,
            "geoshapley_path": forecast.get("geoshapley_path"),
            "forecast_path": forecast.get("forecast_path"),
        },
        "allocation": {
            "selected_sites": selected_sites,
            "iz_assignments": iz_map,
            "diagnostics": diagnostics,
        },
        "comparison_df": pd.DataFrame(comparison_rows),
        "logs": logs,
        "bundle": bundle,
    }


def hydrate_region_ui(area_code: str) -> dict[str, Any] | None:
    """Load a saved city planning bundle. Does not retrain."""
    from agent.region_training import region_output_dir

    code = str(area_code or "").strip()
    if not code or code in {EDINBURGH_CA, "UNKNOWN"}:
        return None
    path = region_output_dir(code) / "planning" / "latest.json"
    if not path.is_file():
        return None
    try:
        bundle = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    ui = bundle_to_ui(bundle)
    return ui if ui.get("success") else None


class PlanningAgent:
    """UI-facing control layer. Numbers come from registered planning tools."""

    def __init__(self) -> None:
        self.execution_log: list[dict[str, Any]] = []

    def run_planning_pipeline(
        self,
        study_area: str,
        forecast_date: str,
        scenario: str,
        travel_mode: str,
        travel_time_threshold: float,
        eligible_site_types: list[str],
        priority_population: str,
        user_confirmed_retraining: bool = False,
    ) -> dict[str, Any]:
        try:
            area_code, area_name = map_study_area(study_area)
            payload = {
                "area_code": area_code,
                "area_name": area_name,
                "forecast_date": forecast_date,
                "scenario": map_scenario(scenario),
                "travel_mode": travel_mode,
                "travel_time_threshold_min": travel_time_threshold,
                "eligible_site_types": eligible_site_types,
                "priority_population": map_priority_population(priority_population),
                "confirm_rolling_update": user_confirmed_retraining,
                "confirm_new_region_training": user_confirmed_retraining,
            }
            result = bundle_to_ui(run_planning(payload))
        except ModelError as error:
            result = {
                "success": False,
                "error_type": getattr(error, "code", None) or "PIPELINE_ERROR",
                "message": str(error),
                "logs": [{"step": "validation", "details": {"error": str(error)}}],
            }
        self.execution_log = list(result.get("logs") or [])
        return result

    def explain_site_selection(self, site_id: str, pipeline_output: dict[str, Any], *, lang: str = "en") -> str:
        if not pipeline_output.get("success"):
            return (
                "规划尚未完成，无法生成说明。"
                if lang == "zh"
                else "Cannot generate explanation: the planning pipeline did not complete."
            )
        sites = pipeline_output["allocation"]["selected_sites"]
        diag = pipeline_output["allocation"]["diagnostics"]
        target = next((site for site in sites if site["site_id"] == site_id), None)
        if target is None:
            target = sites[0] if sites else None
        if target is None:
            return "求解器还没有放置疫苗接种点。" if lang == "zh" else "The solver has not placed vaccination sites yet."
        facility = site_type_display(lang, label_of(SITE_TYPE_LABELS, target["site_type"]))
        policy = scenario_display(lang, SCENARIO_LABELS.get(diag["scenario"], diag["scenario"]))
        travel_shown = travel_display(
            lang,
            TRAVEL_MODE_LABELS.get(str(diag.get("travel_mode") or "").lower(), str(diag.get("travel_mode") or "")),
        )
        if lang == "zh":
            return (
                f"### 接种点审计：{target['site_name']} (`{target['site_id']}`)\n\n"
                f"- **设施类型：** {facility}\n"
                f"- **分配的中间区数：** {target['assigned_iz_count']}\n"
                f"- **服务人口（面板总人口）：** {target['served_population']:,}\n"
                f"- **当前政策：** {policy}\n"
                f"- **出行约束：** {diag['travel_time_threshold_min']} 分钟（{travel_shown}）\n"
                f"- **求解器理由：** {target['selection_reason']}\n\n"
                "> 接种点由确定性求解器选出。GeoShapley 和图 α 是解释，不是选址分数。"
            )
        return (
            f"### Vaccination site audit: {target['site_name']} (`{target['site_id']}`)\n\n"
            f"- **Facility type:** {facility}\n"
            f"- **Intermediate zones assigned:** {target['assigned_iz_count']}\n"
            f"- **Population served (panel total):** {target['served_population']:,}\n"
            f"- **Active policy:** {policy}\n"
            f"- **Mobility constraints:** {diag['travel_time_threshold_min']} min ({travel_shown})\n"
            f"- **Allocator reason:** {target['selection_reason']}\n\n"
            "> Vaccination sites are chosen by the deterministic solver. "
            "GeoShapley and graph α are explanations, not siting scores."
        )
