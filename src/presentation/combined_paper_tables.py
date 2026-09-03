"""Build combined Edinburgh–Glasgow paper tables. Does not overwrite website_article_v1."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from allocation.contracts import N_SITES, SCENARIO_LABELS, SCENARIOS
from common.utils import project_relative_path, project_root
from presentation.website_export import write_article_tables

OUT_RELATIVE = "data/results/exports/paper_tables_combined_v1"
EDI_ARTICLE = Path("data/results/exports/website_article_v1/article")
GLA_ARTICLE = Path("data/results/exports/website_article_glasgow_v1/article")


def _read(rel: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(project_root() / rel / name)


def _forecast_rows() -> pd.DataFrame:
    edi = _read(EDI_ARTICLE, "table02_overall_performance.csv")
    gla = _read(GLA_ARTICLE, "table02_overall_performance.csv")
    edi.insert(0, "city", "Edinburgh")
    gla.insert(0, "city", "Glasgow")
    keep = gla["method"].isin(["Persistence", "Rolling 65/10/25 model"])
    gla = gla.loc[keep]
    edi = edi.loc[edi["method"].isin(["Persistence", "Rolling 65/10/25 model"])]
    out = pd.concat([edi, gla], ignore_index=True)
    out["n_iz"] = out["city"].map({"Edinburgh": 111, "Glasgow": 136})
    return out


def _interval_rows() -> pd.DataFrame:
    edi = _read(EDI_ARTICLE, "table04_uncertainty_intervals.csv")
    gla = _read(GLA_ARTICLE, "table04_uncertainty_intervals.csv")
    edi.insert(0, "city", "Edinburgh")
    gla.insert(0, "city", "Glasgow")
    both = pd.concat([edi, gla], ignore_index=True)
    return both.loc[both["interval_type"].isin(["calibrated_80", "calibrated_95"])].copy()


def _period_rows() -> pd.DataFrame:
    edi = _read(EDI_ARTICLE, "table03_performance_by_period.csv")
    gla = _read(GLA_ARTICLE, "table03_performance_by_period.csv")
    edi.insert(0, "city", "Edinburgh")
    gla.insert(0, "city", "Glasgow")
    return pd.concat([edi, gla], ignore_index=True)


def _alpha_rows() -> pd.DataFrame:
    """Last rolling update only (U10). Do not average U01–U10."""
    rows = []
    for city, rel in (("Edinburgh", EDI_ARTICLE), ("Glasgow", GLA_ARTICLE)):
        frame = _read(rel, "table05a_alpha_by_checkpoint.csv")
        last = frame.loc[frame["update_id"].astype(str) == "U10"]
        if last.empty:
            last = frame.iloc[[-1]]
        row = last.iloc[0]
        rows.append(
            {
                "city": city,
                "update_id": str(row["update_id"]),
                "forecast_start": row["forecast_start"],
                "forecast_end": row["forecast_end"],
                "alpha_geo": float(row["alpha_geo"]),
                "alpha_transport": float(row["alpha_transport"]),
                "alpha_mobility": float(row["alpha_mobility"]),
            }
        )
    return pd.DataFrame(rows)


def _allocation_rows() -> pd.DataFrame:
    from allocation.metrics import allocation_metrics

    root = project_root()
    cities = (
        ("Edinburgh", "S12000036", root / "data/results/allocation", root / "data/results/simd_iz.csv"),
        ("Glasgow", "S12000049", root / "data/results/regions/S12000049/allocation", root / "data/results/regions/S12000049/simd_iz.csv"),
    )
    rows = []
    for city, _code, alloc_root, simd_path in cities:
        simd = pd.read_csv(simd_path)
        iz = simd.rename(columns={"IntZone": "iz_code", "total_population": "population"})[
            ["iz_code", "population"]
        ].copy()
        iz["iz_code"] = iz["iz_code"].astype(str)
        for scenario in SCENARIOS:
            assign_path = alloc_root / scenario / "assignments.csv"
            if not assign_path.is_file():
                raise FileNotFoundError(f"Missing {assign_path}")
            assignments = pd.read_csv(assign_path)
            assignments["iz_code"] = assignments["iz_code"].astype(str)
            mets = allocation_metrics(iz, assignments)
            n_iz = int(mets.get("n_iz") or 0)
            covered = int(mets.get("iz_covered") or 0)
            n_sites = int(assignments.loc[assignments["served"].astype(bool), "site_id"].nunique())
            rows.append(
                {
                    "city": city,
                    "policy": SCENARIO_LABELS[scenario],
                    "n_sites": n_sites or N_SITES,
                    "population_covered": mets.get("population_covered"),
                    "iz_covered": covered,
                    "n_iz": n_iz,
                    "coverage_pct": None if not n_iz else 100.0 * covered / n_iz,
                    "mean_travel_min": mets.get("mean_travel_time_min"),
                    "max_travel_min": mets.get("max_travel_time_min"),
                    "unserved_iz": mets.get("unserved_iz"),
                    "status": "ok",
                }
            )
    return pd.DataFrame(rows)


def _fmt_int(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "---"
    return f"{int(round(float(value))):,}"


def _fmt_float(value: Any, digits: int) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "---"
    return f"{float(value):.{digits}f}"


def write_overleaf(path: Path, forecast: pd.DataFrame, intervals: pd.DataFrame, periods: pd.DataFrame, alpha: pd.DataFrame, allocation: pd.DataFrame | None) -> None:
    def frow(city: str, method: str) -> pd.Series:
        hit = forecast.loc[(forecast["city"] == city) & (forecast["method"] == method)]
        return hit.iloc[0]

    def irow(city: str, kind: str) -> pd.Series:
        hit = intervals.loc[(intervals["city"] == city) & (intervals["interval_type"] == kind)]
        return hit.iloc[0]

    def prow(city: str, name: str) -> pd.Series:
        hit = periods.loc[(periods["city"] == city) & (periods["period_name"] == name)]
        return hit.iloc[0]

    alloc_block = "% Allocation numbers were not available when this file was written.\n"
    if allocation is not None and not allocation.empty:
        lines = []
        for _, row in allocation.iterrows():
            lines.append(
                f"{row['city']} & {row['policy']} & {int(row['n_sites'])} & "
                f"{_fmt_int(row['population_covered'])} & "
                f"{int(row['iz_covered'])}/{int(row['n_iz'])} & "
                f"{_fmt_float(row['coverage_pct'], 1)} & "
                f"{_fmt_float(row['mean_travel_min'], 1)} & "
                f"{_fmt_float(row['max_travel_min'], 1)} & "
                f"{int(row['unserved_iz'])} \\\\"
            )
        alloc_rows = "\n".join(lines)
        alloc_block = rf"""
\begin{{table*}}[t]
  \caption{{Six-site vaccination allocation on 4 March 2023 (unverified forecast), drive, 20-minute threshold. Coverage and equity can both reach 100\% IZ cover while selecting different sites. Numbers are solver outputs, not a unique moral optimum.}}
  \label{{tab:allocation}}
  \small
  \centering
  \begin{{tabular}}{{llrrrrrrr}}
    \toprule
    City & Policy & Sites & Pop.\ covered & IZs covered & Cov. (\%) & Mean min & Max min & Unserved IZs \\
    \midrule
    {alloc_rows}
    \bottomrule
  \end{{tabular}}
\end{{table*}}
"""

    e_p, e_r = frow("Edinburgh", "Persistence"), frow("Edinburgh", "Rolling 65/10/25 model")
    g_p, g_r = frow("Glasgow", "Persistence"), frow("Glasgow", "Rolling 65/10/25 model")
    e80, e95 = irow("Edinburgh", "calibrated_80"), irow("Edinburgh", "calibrated_95")
    g80, g95 = irow("Glasgow", "calibrated_80"), irow("Glasgow", "calibrated_95")
    e_w, e_l = prow("Edinburgh", "declining_or_wave_period"), prow("Edinburgh", "late_stable_period")
    g_w, g_l = prow("Glasgow", "declining_or_wave_period"), prow("Glasgow", "late_stable_period")

    a_e = alpha.loc[alpha["city"] == "Edinburgh"].iloc[0]
    a_g = alpha.loc[alpha["city"] == "Glasgow"].iloc[0]

    tex = rf"""% Paste into an ACM acmart paper (Overleaf: acmart + booktabs).
% Requires: \usepackage{{booktabs}}
% Do not pool Edinburgh and Glasgow into one MAE.

\begin{{table}}[t]
  \caption{{Retrospective 7-day-ahead rate forecasts on the 65/10/25 test window. Persistence assumes no change from the issue-day rate. Cities are not pooled (111 vs 136 Intermediate Zones). 4 March 2023 is excluded.}}
  \label{{tab:forecast}}
  \small
  \centering
  \begin{{tabular}}{{llrrrrr}}
    \toprule
    City & Method & MAE & Skill & RMSE & $R^2$ & Days \\
    \midrule
    Edinburgh & Persistence & {e_p.MAE:.2f} & 0.000 & {e_p.RMSE:.2f} & {e_p.R2:.2f} & {int(e_p.n_unique_target_dates)} \\
    Edinburgh & Rolling DCRNN & {e_r.MAE:.2f} & {e_r.MAE_skill:.3f} & {e_r.RMSE:.2f} & {e_r.R2:.2f} & {int(e_r.n_unique_target_dates)} \\
    Glasgow & Persistence & {g_p.MAE:.2f} & 0.000 & {g_p.RMSE:.2f} & {g_p.R2:.2f} & {int(g_p.n_unique_target_dates)} \\
    Glasgow & Rolling DCRNN & {g_r.MAE:.2f} & {g_r.MAE_skill:.3f} & {g_r.RMSE:.2f} & {g_r.R2:.2f} & {int(g_r.n_unique_target_dates)} \\
    \bottomrule
  \end{{tabular}}
\end{{table}}

\begin{{table}}[t]
  \caption{{Empirical coverage of calibrated prediction intervals on the same retrospective cells. These are not exchangeability guarantees.}}
  \label{{tab:intervals}}
  \small
  \centering
  \begin{{tabular}}{{llrrr}}
    \toprule
    City & Interval & Nominal & Observed & Mean width \\
    \midrule
    Edinburgh & Calibrated 80\% & 0.80 & {e80.observed_coverage:.3f} & {e80.mean_interval_width:.1f} \\
    Edinburgh & Calibrated 95\% & 0.95 & {e95.observed_coverage:.3f} & {e95.mean_interval_width:.1f} \\
    Glasgow & Calibrated 80\% & 0.80 & {g80.observed_coverage:.3f} & {g80.mean_interval_width:.1f} \\
    Glasgow & Calibrated 95\% & 0.95 & {g95.observed_coverage:.3f} & {g95.mean_interval_width:.1f} \\
    \bottomrule
  \end{{tabular}}
\end{{table}}

% Optional if space remains. Late-period MAE skill is near zero or negative.
\begin{{table}}[t]
  \caption{{Rolling-model MAE by period versus persistence. The late window is more stable and the skill gain shrinks.}}
  \label{{tab:period}}
  \small
  \centering
  \begin{{tabular}}{{llrrrr}}
    \toprule
    City & Period & Model MAE & Persist.\ MAE & Skill & Days \\
    \midrule
    Edinburgh & Decline/wave & {e_w.model_MAE:.2f} & {e_w.persistence_MAE:.2f} & {e_w.MAE_skill:.3f} & {int(e_w.n_unique_target_dates)} \\
    Edinburgh & Late stable & {e_l.model_MAE:.2f} & {e_l.persistence_MAE:.2f} & {e_l.MAE_skill:.3f} & {int(e_l.n_unique_target_dates)} \\
    Glasgow & Decline/wave & {g_w.model_MAE:.2f} & {g_w.persistence_MAE:.2f} & {g_w.MAE_skill:.3f} & {int(g_w.n_unique_target_dates)} \\
    Glasgow & Late stable & {g_l.model_MAE:.2f} & {g_l.persistence_MAE:.2f} & {g_l.MAE_skill:.3f} & {int(g_l.n_unique_target_dates)} \\
    \bottomrule
  \end{{tabular}}
\end{{table}}

\begin{{table}}[t]
  \caption{{Last rolling-update graph-fusion weights (U10). The same checkpoint is used for the last retrospective days and the 4 March 2023 map. $\alpha>0$, $\sum\alpha=1$. Not a risk share and not a siting score.}}
  \label{{tab:alpha}}
  \small
  \centering
  \begin{{tabular}}{{llrrr}}
    \toprule
    City & Update & $\alpha_{{\mathrm{{geo}}}}$ & $\alpha_{{\mathrm{{transport}}}}$ & $\alpha_{{\mathrm{{mobility}}}}$ \\
    \midrule
    Edinburgh & {a_e.update_id} & {float(a_e.alpha_geo):.3f} & {float(a_e.alpha_transport):.3f} & {float(a_e.alpha_mobility):.3f} \\
    Glasgow & {a_g.update_id} & {float(a_g.alpha_geo):.3f} & {float(a_g.alpha_transport):.3f} & {float(a_g.alpha_mobility):.3f} \\
    \bottomrule
  \end{{tabular}}
\end{{table}}
{alloc_block}
"""
    path.write_text(tex.strip() + "\n", encoding="utf-8")


def export_combined_paper_tables() -> dict[str, Any]:
    out = project_root() / OUT_RELATIVE
    out.mkdir(parents=True, exist_ok=True)
    forecast = _forecast_rows()
    intervals = _interval_rows()
    periods = _period_rows()
    alpha = _alpha_rows()
    allocation = None
    allocation_error = None
    try:
        allocation = _allocation_rows()
    except Exception as exc:  # keep forecast tables even if siting cache is incomplete
        allocation_error = str(exc)

    tables = {
        "table_forecast_both_cities": forecast,
        "table_intervals_both_cities": intervals,
        "table_period_both_cities": periods,
        "table_alpha_last_update": alpha,
    }
    if allocation is not None:
        tables["table_allocation_both_cities"] = allocation
    write_article_tables(out / "article", tables)
    write_overleaf(out / "overleaf_paper_tables.tex", forecast, intervals, periods, alpha, allocation)
    manifest = {
        "export_dir": project_relative_path(out),
        "did_not_overwrite_edinburgh": True,
        "paper_use": {
            "include": ["table_forecast_both_cities", "table_allocation_both_cities", "table_alpha_last_update"],
            "if_space": ["table_intervals_both_cities", "table_period_both_cities"],
            "omit_from_4page": ["table01_split", "table05a_alpha_by_update", "table05b_alpha_mean", "table06_geoshapley"],
        },
        "allocation_error": allocation_error,
    }
    (out / "EXPORT_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    print(json.dumps(export_combined_paper_tables(), indent=2))
