"""One planning page. Does not invent sites or IZ metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from allocation.contracts import FORECAST_CACHE, N_SITES, SCENARIO_LABELS
from common.utils import project_root, resolve_project_path, write_json

PAGE_DIR = Path("data") / "results" / "planning"


def write_planning_page(bundle: dict[str, Any], output_dir: str | Path | None = None) -> dict[str, Any]:
    folder = Path(output_dir) if output_dir is not None else project_root() / PAGE_DIR
    folder.mkdir(parents=True, exist_ok=True)
    json_path = write_json(bundle, folder / "latest.json")
    forecast_rows = _forecast_rows(bundle)
    if forecast_rows:
        pd.DataFrame(forecast_rows).to_csv(folder / "iz_risk.csv", index=False)
    html_path = folder / "index.html"
    html_path.write_text(_html(bundle, forecast_rows), encoding="utf-8")
    return {
        "status": "ok",
        "page": str(html_path),
        "json": str(json_path),
        "n_iz": len(forecast_rows),
        "n_sites_selected": int((bundle.get("allocation") or {}).get("n_sites_selected") or 0),
    }


def _forecast_rows(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    path = (bundle.get("forecast") or {}).get("forecast_path")
    file_path = resolve_project_path(path) if path else project_root() / FORECAST_CACHE
    if not file_path.exists():
        return []
    frame = pd.read_csv(file_path)
    keep = [
        name
        for name in (
            "iz_code",
            "predicted_rate",
            "predicted_sigma",
            "calibrated95_lower",
            "calibrated95_upper",
            "alpha_geo",
            "alpha_transport",
            "alpha_mobility",
            "forecast_status",
        )
        if name in frame.columns
    ]
    return frame[keep].to_dict(orient="records")


def _html(bundle: dict[str, Any], forecast_rows: list[dict[str, Any]]) -> str:
    request = bundle.get("request") or {}
    allocation = bundle.get("allocation") or {}
    comparison = (bundle.get("comparison") or {}).get("scenarios") or []
    payload = json.dumps(
        {
            "request": request,
            "status": bundle.get("status"),
            "model_action": bundle.get("model_action"),
            "forecast_label": bundle.get("forecast_label"),
            "forecast": forecast_rows,
            "allocation": allocation,
            "comparison": comparison,
            "n_sites": N_SITES,
            "scenario_labels": SCENARIO_LABELS,
        },
        default=str,
    )
    selected = allocation.get("selected_sites") or []
    metrics = allocation.get("metrics") or {}
    scenario = request.get("scenario") or "balanced"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>OASIS planning — City of Edinburgh</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; color: #1b1b1b; background: #f7f7f5; }}
    h1, h2 {{ font-weight: 600; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    .card {{ background: #fff; border: 1px solid #ddd; padding: 16px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ text-align: left; padding: 6px 8px; border-bottom: 1px solid #eee; }}
    .note {{ color: #555; font-size: 13px; }}
    .warn {{ border-left: 4px solid #b36b00; padding-left: 12px; }}
    button {{ margin-right: 8px; }}
    #detail {{ min-height: 4em; }}
  </style>
</head>
<body>
  <h1>Intervention planning</h1>
  <p class="note">Six sites are selected by the deterministic allocator, not by the language model. Number of sites is fixed at {N_SITES}.</p>
  <p class="warn">{bundle.get("forecast_label") or ""}</p>
  <div class="grid">
    <section class="card">
      <h2>Planning controls</h2>
      <table>
        <tr><th>Study area</th><td>{request.get("area_name")} ({request.get("area_code")})</td></tr>
        <tr><th>Forecast date</th><td>{request.get("forecast_date")}</td></tr>
        <tr><th>Scenario</th><td>{SCENARIO_LABELS.get(scenario, scenario)}</td></tr>
        <tr><th>Travel mode</th><td>{request.get("travel_mode")}</td></tr>
        <tr><th>Travel-time threshold</th><td>{request.get("travel_time_threshold_min")} min</td></tr>
        <tr><th>Eligible site types</th><td>{", ".join(request.get("eligible_site_types") or [])}</td></tr>
        <tr><th>Priority population</th><td>{request.get("priority_population")}</td></tr>
        <tr><th>Sites to select</th><td>{N_SITES} (fixed)</td></tr>
      </table>
    </section>
    <section class="card">
      <h2>Core indicators</h2>
      <p class="note">Status: {bundle.get("status")}. Model action: {bundle.get("model_action")}.</p>
      <table>
        <tr><th>Population covered</th><td>{metrics.get("population_covered")}</td></tr>
        <tr><th>IZs covered</th><td>{metrics.get("iz_covered")}</td></tr>
        <tr><th>Mean travel time</th><td>{metrics.get("mean_travel_time_min")}</td></tr>
        <tr><th>Max travel time</th><td>{metrics.get("max_travel_time_min")}</td></tr>
        <tr><th>Unserved IZs</th><td>{metrics.get("unserved_iz")}</td></tr>
      </table>
      <p class="note">{(allocation.get("diagnostics") or {}).get("message") or ""}</p>
    </section>
  </div>
  <section class="card" style="margin-top:16px">
    <h2>Main map / IZ risk</h2>
    <p class="note">IZ polygons will use the 2011 layer when the map GeoJSON is attached. Predicted risk for {len(forecast_rows)} IZs is listed below.</p>
    <div id="risk"></div>
  </section>
  <section class="card" style="margin-top:16px">
    <h2>Selected sites</h2>
    <div id="sites"></div>
    <h3>Site details</h3>
    <div id="detail" class="note">Click a selected site to see recorded assignment details. Reasons come from the allocator, not the language model.</div>
  </section>
  <section class="card" style="margin-top:16px">
    <h2>Scenario comparison</h2>
    <p class="note">Same six-site count and travel-time threshold for coverage, equity, preventive and balanced.</p>
    <div id="compare"></div>
  </section>
  <script>
    const DATA = {payload};
    function cell(v) {{ return v === null || v === undefined || v === "" ? "—" : v; }}
    const risk = document.getElementById("risk");
    const top = (DATA.forecast || []).slice().sort((a,b) => (b.predicted_rate||0)-(a.predicted_rate||0)).slice(0,12);
    risk.innerHTML = "<table><tr><th>IZ</th><th>Predicted rate</th><th>Sigma</th><th>Status</th></tr>" +
      top.map(r => `<tr><td>${{r.iz_code}}</td><td>${{cell(r.predicted_rate && r.predicted_rate.toFixed(2))}}</td><td>${{cell(r.predicted_sigma && r.predicted_sigma.toFixed(2))}}</td><td>${{cell(r.forecast_status)}}</td></tr>`).join("") +
      "</table><p class='note'>Showing the 12 highest predicted rates. Full table: iz_risk.csv.</p>";
    const sites = DATA.allocation.selected_sites || [];
    const sitesEl = document.getElementById("sites");
    if (!sites.length) {{
      sitesEl.innerHTML = "<p class='note'>No sites displayed. The allocator is not wired, so six sites were not invented.</p>";
    }} else {{
      sitesEl.innerHTML = sites.map((s,i) => {{
        const id = s.site_id || s;
        const name = s.site_name || id;
        return `<button type="button" data-i="${{i}}">${{name}}</button>`;
      }}).join("");
      sitesEl.querySelectorAll("button").forEach(btn => btn.addEventListener("click", () => {{
        const s = sites[Number(btn.dataset.i)];
        const id = s.site_id || s;
        const reason = (DATA.allocation.selection_reasons || {{}})[id] || "No recorded reason from the allocator.";
        const assigned = (DATA.allocation.assignments || []).filter(a => (a.site_id || "") === id);
        document.getElementById("detail").innerHTML =
          `<p><strong>${{s.site_name || id}}</strong> (${{s.site_type || "unknown type"}})</p>` +
          `<p>Assigned IZs: ${{assigned.map(a => a.iz_code).join(", ") || "—"}}</p>` +
          `<p>Recorded reason: ${{reason}}</p>`;
      }}));
    }}
    const cmp = document.getElementById("compare");
    cmp.innerHTML = "<table><tr><th>Scenario</th><th>Status</th><th>Sites</th><th>Covered IZs</th></tr>" +
      (DATA.comparison || []).map(r => `<tr><td>${{r.label}}</td><td>${{r.status}}</td><td>${{r.n_sites_selected}}</td><td>${{cell((r.metrics||{{}}).iz_covered)}}</td></tr>`).join("") +
      "</table>";
  </script>
</body>
</html>
"""
