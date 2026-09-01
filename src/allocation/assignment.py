"""Greedy site selection: covering and/or weighted travel-time reduction."""

from __future__ import annotations

import pandas as pd

UNREACHABLE = 1.0e6


def travel_cover_scores(
    travel_wide: pd.DataFrame,
    threshold: float,
    *,
    kind: str = "binary",
) -> pd.DataFrame:
    """IZ×site covering scores in [0, 1]. NaN travel times score 0."""
    wide = travel_wide.astype(float)
    if kind == "binary":
        return (wide.le(threshold) & wide.notna()).astype(float)
    if kind == "fractional":
        scores = (1.0 - wide / float(threshold)).clip(lower=0.0)
        return scores.where(wide.notna(), 0.0)
    raise ValueError(f"Unknown cover score kind {kind!r}.")


def greedy_cover(
    coverable: pd.DataFrame,
    weights: pd.Series,
    n_sites: int,
    travel_wide: pd.DataFrame | None = None,
    *,
    primary: str = "cover",
    threshold: float | None = None,
) -> tuple[list[str], dict[str, str], list[float]]:
    """Pick p recorded sites. Does not invent IDs.

    primary='cover' (Coverage-priority): maximise newly covered demand; if that
    is tied or already zero, pick the site that most reduces weighted travel time.
    Does not pad leftover slots by sorting site_id (that always preferred GP_*
    over MS_* car parks).

    primary='access' (equity, balanced): greedy p-median on the
    scenario weights, counting only IZ–site times within the threshold.
    A car park nearer a high-weight IZ can beat a GP.
    """
    scores = coverable.astype(float).fillna(0.0).clip(lower=0.0)
    demand = weights.reindex(scores.index).fillna(0.0).astype(float)
    remaining = demand.copy()
    times = (
        travel_wide if travel_wide is not None else pd.DataFrame(index=scores.index, columns=scores.columns, dtype=float)
    )
    times = times.reindex(index=scores.index, columns=scores.columns).astype(float)
    if threshold is not None:
        times = times.where(times.le(float(threshold)))
    current_t = pd.Series(UNREACHABLE, index=scores.index, dtype=float)
    available = [str(col) for col in scores.columns]
    selected: list[str] = []
    reasons: dict[str, str] = {}
    gains: list[float] = []
    for step in range(1, n_sites + 1):
        if not available:
            break
        best_site = None
        best_pair = None
        for site_id in available:
            cover_gain, access_gain = _site_gains(
                remaining, demand, scores[site_id], times[site_id], current_t
            )
            pair = (access_gain, cover_gain) if primary == "access" else (cover_gain, access_gain)
            if (
                best_pair is None
                or pair > best_pair
                or (pair == best_pair and (best_site is None or site_id < best_site))
            ):
                best_pair = pair
                best_site = site_id
        assert best_site is not None
        cover_gain, access_gain = _site_gains(
            remaining, demand, scores[best_site], times[best_site], current_t
        )
        remaining = remaining * (1.0 - scores[best_site].reindex(remaining.index).fillna(0.0).clip(upper=1.0))
        site_t = times[best_site].reindex(current_t.index)
        current_t = pd.concat([current_t, site_t.fillna(UNREACHABLE)], axis=1).min(axis=1)
        selected.append(best_site)
        available.remove(best_site)
        reported = float(cover_gain if primary == "cover" else access_gain)
        gains.append(reported)
        n_positive = int((scores[best_site].fillna(0.0) > 0).sum())
        reasons[best_site] = (
            f"Step {step} ({primary}): covering gain {cover_gain:.4f}, "
            f"travel-time reduction {access_gain:.4f}; "
            f"{n_positive} IZs within threshold of this site."
        )
    return selected, reasons, gains


def _site_gains(
    remaining: pd.Series,
    demand: pd.Series,
    cover_col: pd.Series,
    time_col: pd.Series,
    current_t: pd.Series,
) -> tuple[float, float]:
    cover = cover_col.reindex(remaining.index).fillna(0.0)
    cover_gain = float((remaining * cover).sum())
    new_t = time_col.reindex(current_t.index)
    improved = (current_t - new_t).where(new_t.notna(), 0.0).clip(lower=0.0)
    access_gain = float((demand.reindex(improved.index).fillna(0.0) * improved.fillna(0.0)).sum())
    return cover_gain, access_gain


def assign_iz_to_sites(
    travel_wide: pd.DataFrame,
    selected: list[str],
    threshold: float,
) -> pd.DataFrame:
    """Nearest selected site within threshold. Unreachable IZs stay unserved."""
    if not selected:
        return pd.DataFrame(columns=["iz_code", "site_id", "travel_time_min", "served"])
    subset = travel_wide.reindex(columns=selected)
    nearest = subset.idxmin(axis=1)
    minutes = subset.min(axis=1)
    served = minutes.notna() & (minutes <= threshold)
    out = pd.DataFrame(
        {
            "iz_code": travel_wide.index.astype(str),
            "site_id": nearest.where(served).astype("string"),
            "travel_time_min": minutes.where(served),
            "served": served.astype(bool),
        }
    )
    return out.reset_index(drop=True)
