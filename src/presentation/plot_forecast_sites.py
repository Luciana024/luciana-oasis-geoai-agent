"""Paper maps: 2023-03-04 predicted rate choropleth + six vaccination sites.

Does not overwrite website_article_v1, rolling_v1, or data/raw.
Sites come from cached allocation CSVs; they are not chosen here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patheffects import withStroke

from allocation.contracts import DEMO_FORECAST_DATE, N_SITES, SCENARIO_LABELS
from common.errors import ModelError
from common.utils import project_root

OUT_DIRS = (
    Path("docs/figures"),
    Path("data/results/exports/paper_tables_combined_v1/figures"),
)
PROTECTED = (
    Path("data/results/exports/website_article_v1"),
    Path("data/results/model/rolling_v1_split65_10_25"),
    Path("data/raw"),
)
TARGET_CRS = "EPSG:27700"
CMAP = "YlOrRd"
SITE_STYLES = {
    "gp": {"marker": "o", "facecolor": "#102a43", "label": "GP practice"},
    "pharmacy": {"marker": "s", "facecolor": "#014d4e", "label": "Pharmacy"},
    "mobile_stop": {"marker": "^", "facecolor": "#3d1a5c", "label": "Mobile stop"},
}
DEFAULT_SCENARIO = "balanced"


@dataclass(frozen=True)
class CityMapSpec:
    key: str
    label: str
    n_iz: int
    forecast: Path
    boundaries: Path
    selected_sites: Path
    candidates: Path
    rate_column: str = "predicted_rate"


def default_city_specs(scenario: str = DEFAULT_SCENARIO) -> tuple[CityMapSpec, ...]:
    if scenario not in SCENARIO_LABELS:
        raise ModelError(f"Unknown allocation scenario {scenario}.", code="invalid_config")
    return (
        CityMapSpec(
            key="edinburgh",
            label="Edinburgh",
            n_iz=111,
            forecast=Path("data/results/exports/website_article_v1/website/future_forecast_20230304.csv"),
            boundaries=Path("data/results/exports/website_article_v1/website/edinburgh_iz_boundaries.geojson"),
            selected_sites=Path(f"data/results/allocation/{scenario}/selected_sites.csv"),
            candidates=Path("data/results/candidate_sites/S12000036/merged_candidate_sites.csv"),
        ),
        CityMapSpec(
            key="glasgow",
            label="Glasgow",
            n_iz=136,
            forecast=Path("data/results/regions/S12000049/forecast_for_allocation.csv"),
            boundaries=Path("data/results/regions/S12000049/planning/iz_boundaries.geojson"),
            selected_sites=Path(f"data/results/regions/S12000049/allocation/{scenario}/selected_sites.csv"),
            candidates=Path("data/results/candidate_sites/S12000049/merged_candidate_sites.csv"),
        ),
    )


def _as_root(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path


def _assert_writable(dest: Path, root: Path) -> None:
    resolved = dest.resolve()
    for rel in PROTECTED:
        protected = (root / rel).resolve()
        if resolved == protected or protected in resolved.parents:
            raise ModelError(
                f"Refusing to write map figures under protected path {rel}.",
                code="invalid_config",
            )


def _read_forecast(path: Path, rate_column: str, n_iz: int) -> pd.DataFrame:
    if not path.is_file():
        raise ModelError(f"Forecast table missing: {path}", code="missing_dataset")
    frame = pd.read_csv(path)
    if "iz_code" not in frame.columns:
        raise ModelError(f"Forecast has no iz_code: {path}", code="invalid_config")
    if "target_report_date" in frame.columns:
        frame = frame.loc[frame["target_report_date"].astype(str) == DEMO_FORECAST_DATE].copy()
    if rate_column not in frame.columns:
        raise ModelError(
            f"Forecast has no {rate_column}; will not invent a predicted rate.",
            code="missing_dataset",
            details={"path": str(path)},
        )
    frame["iz_code"] = frame["iz_code"].astype(str)
    frame = frame.dropna(subset=[rate_column]).drop_duplicates("iz_code", keep="first")
    if len(frame) != n_iz:
        raise ModelError(
            f"Forecast has {len(frame)} IZs, expected {n_iz}.",
            code="node_order_mismatch",
            details={"path": str(path)},
        )
    return frame[["iz_code", rate_column]].rename(columns={rate_column: "predicted_rate"})


def _read_sites(selected_path: Path, candidates_path: Path) -> pd.DataFrame:
    if not selected_path.is_file():
        raise ModelError(f"Selected-site table missing: {selected_path}", code="missing_dataset")
    if not candidates_path.is_file():
        raise ModelError(f"Candidate-site table missing: {candidates_path}", code="missing_dataset")
    selected = pd.read_csv(selected_path)
    candidates = pd.read_csv(candidates_path)
    if "site_id" not in selected.columns:
        raise ModelError(f"selected_sites has no site_id: {selected_path}", code="invalid_config")
    need = {"site_id", "easting", "northing"}
    missing_cols = need - set(candidates.columns)
    if missing_cols:
        raise ModelError(
            f"Candidate table missing {sorted(missing_cols)}; will not invent coordinates.",
            code="missing_dataset",
        )
    selected["site_id"] = selected["site_id"].astype(str)
    candidates["site_id"] = candidates["site_id"].astype(str)
    keep = ["site_id", "easting", "northing"]
    if "site_type" in candidates.columns:
        keep.append("site_type")
    if "site_name" in candidates.columns:
        keep.append("site_name")
    merged = selected[["site_id"]].merge(candidates[keep], on="site_id", how="left")
    if "site_name" in selected.columns:
        merged["site_name"] = selected["site_name"].astype(str).values
    if "site_type" in selected.columns:
        merged["site_type"] = selected["site_type"].astype(str).values
    if len(merged) != N_SITES:
        raise ModelError(
            f"Need exactly {N_SITES} selected sites, found {len(merged)}.",
            code="invalid_config",
            details={"path": str(selected_path)},
        )
    if merged["easting"].isna().any() or merged["northing"].isna().any():
        lost = merged.loc[merged["easting"].isna() | merged["northing"].isna(), "site_id"].tolist()
        raise ModelError(
            "Selected sites are missing coordinates; will not invent them.",
            code="missing_dataset",
            details={"site_id": lost},
        )
    merged["map_index"] = range(1, len(merged) + 1)
    if "site_type" not in merged.columns:
        merged["site_type"] = "gp"
    merged["site_type"] = merged["site_type"].fillna("gp").astype(str)
    if "site_name" not in merged.columns:
        merged["site_name"] = merged["site_id"]
    return merged


def _allocation_caption(selected_path: Path, scenario: str) -> str:
    policy = SCENARIO_LABELS.get(scenario, scenario)
    result_path = selected_path.parent / "result.json"
    extra = ""
    if result_path.is_file():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        mode = payload.get("travel_mode") or ""
        threshold = payload.get("travel_time_threshold_min")
        if mode and threshold is not None:
            extra = f" · {mode} {int(round(float(threshold)))} min"
    return f"{policy} · {N_SITES} sites{extra}"


def _load_geo(spec: CityMapSpec, root: Path):
    import geopandas as gpd

    forecast = _read_forecast(_as_root(spec.forecast, root), spec.rate_column, spec.n_iz)
    bounds_path = _as_root(spec.boundaries, root)
    if not bounds_path.is_file():
        raise ModelError(f"IZ boundaries missing: {bounds_path}", code="missing_dataset")
    boundaries = gpd.read_file(bounds_path)
    code_col = next(
        (name for name in ("iz_code", "IntZone", "IZ_CODE", "InterZone") if name in boundaries.columns),
        None,
    )
    if code_col is None:
        raise ModelError(f"Boundaries have no IZ code column: {bounds_path}", code="invalid_config")
    boundaries["iz_code"] = boundaries[code_col].astype(str)
    if boundaries.crs is None:
        raise ModelError("IZ boundaries have no CRS; will not assume one.", code="invalid_config")
    geo = boundaries[["iz_code", "geometry"]].to_crs(TARGET_CRS).merge(forecast, on="iz_code", how="inner")
    if len(geo) != spec.n_iz:
        raise ModelError(
            f"Map join produced {len(geo)} IZs, expected {spec.n_iz}.",
            code="node_order_mismatch",
            details={"city": spec.label},
        )
    sites = _read_sites(_as_root(spec.selected_sites, root), _as_root(spec.candidates, root))
    site_gdf = gpd.GeoDataFrame(
        sites,
        geometry=gpd.points_from_xy(sites["easting"], sites["northing"]),
        crs=TARGET_CRS,
    )
    return geo, site_gdf


def _scale_bar(ax, length_m: float = 5000.0) -> None:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    x0 = xmin + 0.06 * (xmax - xmin)
    y0 = ymin + 0.05 * (ymax - ymin)
    ax.plot([x0, x0 + length_m], [y0, y0], color="0.15", linewidth=1.8, solid_capstyle="butt", zorder=6)
    ax.plot([x0, x0], [y0 - 0.008 * (ymax - ymin), y0 + 0.008 * (ymax - ymin)], color="0.15", linewidth=1.2, zorder=6)
    ax.plot(
        [x0 + length_m, x0 + length_m],
        [y0 - 0.008 * (ymax - ymin), y0 + 0.008 * (ymax - ymin)],
        color="0.15",
        linewidth=1.2,
        zorder=6,
    )
    ax.text(
        x0 + length_m / 2.0,
        y0 + 0.018 * (ymax - ymin),
        "5 km",
        ha="center",
        va="bottom",
        fontsize=7,
        color="0.15",
    )


def _draw_sites(ax, sites) -> None:
    stroke = withStroke(linewidth=2.2, foreground="white")
    for row in sites.itertuples(index=False):
        style = SITE_STYLES.get(str(row.site_type), SITE_STYLES["gp"])
        ax.scatter(
            [row.geometry.x],
            [row.geometry.y],
            s=62,
            marker=style["marker"],
            facecolor=style["facecolor"],
            edgecolor="white",
            linewidths=0.9,
            zorder=5,
        )
        ax.annotate(
            str(row.map_index),
            (row.geometry.x, row.geometry.y),
            textcoords="offset points",
            xytext=(6, 5),
            fontsize=7,
            fontweight="bold",
            color="0.12",
            zorder=6,
            path_effects=[stroke],
        )


def _site_legend_handles() -> list[Line2D]:
    handles = []
    for style in SITE_STYLES.values():
        handles.append(
            Line2D(
                [0],
                [0],
                marker=style["marker"],
                color="none",
                markerfacecolor=style["facecolor"],
                markeredgecolor="white",
                markeredgewidth=0.8,
                markersize=8,
                label=style["label"],
            )
        )
    return handles


def _draw_city(ax, geo, sites, title: str, subtitle: str, vmin: float, vmax: float, *, show_ylabel: bool) -> None:
    geo.plot(
        column="predicted_rate",
        cmap=CMAP,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
        edgecolor="white",
        linewidth=0.22,
        legend=False,
        missing_kwds={"color": "#dddddd"},
    )
    _draw_sites(ax, sites)
    _scale_bar(ax)
    ax.set_axis_off()
    ax.set_title(title, fontsize=10, pad=6)
    ax.text(0.5, -0.02, subtitle, transform=ax.transAxes, ha="center", va="top", fontsize=7.5, color="0.25")
    if show_ylabel:
        pass


def _save(fig, dest: Path, stem: str, written: dict[str, str], label: str) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    pdf = dest / f"{stem}.pdf"
    png = dest / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    written[str(pdf)] = label
    written[str(png)] = label


def plot_forecast_sites(
    *,
    scenario: str = DEFAULT_SCENARIO,
    specs: tuple[CityMapSpec, ...] | None = None,
    out_dirs: tuple[Path, ...] | None = None,
    root: Path | None = None,
) -> dict[str, str]:
    """Write per-city and combined choropleth PDFs. Shared colour scale across cities."""
    import geopandas as gpd  # noqa: F401  — fail fast if missing

    base = root or project_root()
    city_specs = specs or default_city_specs(scenario)
    destinations = out_dirs or OUT_DIRS
    loaded: list[tuple[CityMapSpec, Any, Any, str]] = []
    rates: list[float] = []
    site_rows: list[dict[str, Any]] = []
    for spec in city_specs:
        geo, sites = _load_geo(spec, base)
        caption = _allocation_caption(_as_root(spec.selected_sites, base), scenario)
        loaded.append((spec, geo, sites, caption))
        rates.extend(geo["predicted_rate"].astype(float).tolist())
        for row in sites.itertuples(index=False):
            site_rows.append(
                {
                    "city": spec.label,
                    "scenario": scenario,
                    "map_index": int(row.map_index),
                    "site_id": row.site_id,
                    "site_name": row.site_name,
                    "site_type": row.site_type,
                    "easting": float(row.easting),
                    "northing": float(row.northing),
                }
            )
    vmin = float(min(rates))
    vmax = float(max(rates))
    norm = Normalize(vmin=vmin, vmax=vmax)

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    written: dict[str, str] = {}
    n = len(loaded)
    fig, axes = plt.subplots(1, n, figsize=(3.7 * n + 0.7, 4.35), constrained_layout=True)
    if n == 1:
        axes = [axes]
    for ax, (spec, geo, sites, caption) in zip(axes, loaded):
        _draw_city(
            ax,
            geo,
            sites,
            f"{spec.label} ({spec.n_iz} IZs)",
            f"Predicted 7-day infection rate · {DEMO_FORECAST_DATE}\n{caption}",
            vmin,
            vmax,
            show_ylabel=False,
        )
    sm = plt.cm.ScalarMappable(norm=norm, cmap=CMAP)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.03, pad=0.02)
    cbar.set_label("Predicted 7-day rate per 100,000")
    fig.legend(
        handles=_site_legend_handles(),
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.08),
        fontsize=8,
    )

    singles = []
    for spec, geo, sites, caption in loaded:
        one, ax = plt.subplots(figsize=(4.15, 4.85), constrained_layout=True)
        _draw_city(
            ax,
            geo,
            sites,
            f"{spec.label} ({spec.n_iz} IZs)",
            f"Predicted 7-day infection rate · {DEMO_FORECAST_DATE}\n{caption}",
            vmin,
            vmax,
            show_ylabel=False,
        )
        sm_one = plt.cm.ScalarMappable(norm=norm, cmap=CMAP)
        sm_one.set_array([])
        cbar_one = one.colorbar(sm_one, ax=ax, fraction=0.046, pad=0.03)
        cbar_one.set_label("Predicted 7-day rate per 100,000")
        ax.legend(handles=_site_legend_handles(), loc="lower right", frameon=True, fontsize=7, framealpha=0.92)
        singles.append((spec, one))

    site_table = pd.DataFrame(site_rows)
    for dest_rel in destinations:
        dest = dest_rel if dest_rel.is_absolute() else base / dest_rel
        _assert_writable(dest, base)
        _save(fig, dest, "fig_map_forecast_sites_both", written, "both")
        for spec, one in singles:
            _save(one, dest, f"fig_map_forecast_sites_{spec.key}", written, spec.label)
        csv_path = dest / "fig_map_forecast_sites_selected.csv"
        site_table.to_csv(csv_path, index=False)
        written[str(csv_path)] = "sites"

    plt.close(fig)
    for _, one in singles:
        plt.close(one)
    return written


if __name__ == "__main__":
    paths = plot_forecast_sites()
    for path in paths:
        print(path)
