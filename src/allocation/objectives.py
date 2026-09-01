"""IZ demand weights for the four planning scenarios.

Deprived and underserved are not the same thing, and they are not binary classes.
Both come from SIMD 2020v2 aggregated to 2011 Intermediate Zones:

- deprived: income_rate = share of people who are income-deprived
- underserved: pt_gp_min = public-transport minutes to a GP

Preventive does not spread weight over every IZ. It only scores IZs that are
high predicted risk and/or high uncertainty (forecast uncertainty_flag, else
the 90th percentile of predicted_sigma). Other IZs get weight 0.
"""

from __future__ import annotations

import pandas as pd

from common.errors import ModelError

HIGH_RISK_QUANTILE = 0.75
HIGH_SIGMA_QUANTILE = 0.90

SCENARIO_WEIGHTS = {
    "coverage": (
        "Maximise newly covered population within 20 minutes; leftover slots "
        "reduce population-weighted travel time (car parks can compete)."
    ),
    "equity": (
        "Greedy p-median on equity weights: nearer sites to income-deprived "
        "and GP-underserved IZs, including mobile_stop car parks."
    ),
    "preventive": (
        "Cover and pull sites toward IZs with high predicted_rate and/or "
        "high uncertainty (uncertainty_flag=high). Other IZs have weight 0."
    ),
    "balanced": (
        "Equal mix of unit(coverage), unit(equity) and unit(preventive hot spots)."
    ),
}


def iz_demand_weights(iz: pd.DataFrame, scenario: str) -> pd.Series:
    """One non-negative weight per IZ. Missing values fail; zeros stay zeros."""
    parts = scenario_components(iz)
    if scenario == "coverage":
        weights = parts["coverage"]
    elif scenario == "equity":
        weights = parts["equity"]
    elif scenario == "preventive":
        weights = parts["preventive"]
    elif scenario == "balanced":
        weights = _unit(parts["coverage"]) + _unit(parts["equity"]) + _unit(parts["preventive"])
    else:
        raise ModelError(f"Unknown allocation scenario {scenario!r}.", code="invalid_config")
    if (weights < 0).any():
        raise ModelError("Demand weights must be non-negative.", code="invalid_config")
    weights.name = "demand_weight"
    return weights


def scenario_components(iz: pd.DataFrame) -> dict[str, pd.Series]:
    pop = pd.to_numeric(iz["population"], errors="coerce")
    income = pd.to_numeric(iz["income_rate"], errors="coerce").clip(lower=0)
    gp_min = pd.to_numeric(iz["pt_gp_min"], errors="coerce").clip(lower=0)
    rate = pd.to_numeric(iz["predicted_rate"], errors="coerce").clip(lower=0)
    sigma = pd.to_numeric(iz["predicted_sigma"], errors="coerce").clip(lower=0)
    if pop.isna().any() or income.isna().any() or gp_min.isna().any() or rate.isna().any():
        raise ModelError(
            "Demand weights require population, income_rate, pt_gp_min and predicted_rate.",
            code="missing_dataset",
        )
    high_risk, high_unc = _hotspot_masks(iz, rate, sigma)
    income_rel = _relative_to_mean(income)
    access_rel = _relative_to_mean(gp_min)
    coverage = pop.astype(float)
    equity = pop.astype(float) * income_rel * access_rel
    risk_hot = pop.astype(float) * _relative_to_mean(rate)
    risk_hot = risk_hot.where(high_risk, 0.0)
    unc_hot = pop.astype(float) * _relative_to_mean(sigma)
    unc_hot = unc_hot.where(high_unc, 0.0)
    preventive = _unit(risk_hot) + _unit(unc_hot)
    if float(preventive.sum()) <= 0:
        raise ModelError(
            "Preventive weights are all zero: no IZ is high-risk or high-uncertainty.",
            code="missing_dataset",
        )
    return {
        "coverage": coverage,
        "equity": equity,
        "risk": risk_hot,
        "uncertainty": unc_hot,
        "preventive": preventive,
        "high_risk": high_risk,
        "high_uncertainty": high_unc,
    }


def attach_priority_labels(iz: pd.DataFrame) -> pd.DataFrame:
    """Display labels. Quantiles and uncertainty_flag; not invented classes."""
    out = iz.copy()
    income = pd.to_numeric(out["income_rate"], errors="coerce")
    gp_min = pd.to_numeric(out["pt_gp_min"], errors="coerce")
    rate = pd.to_numeric(out["predicted_rate"], errors="coerce").clip(lower=0)
    sigma = pd.to_numeric(out["predicted_sigma"], errors="coerce").clip(lower=0)
    high_risk, high_unc = _hotspot_masks(out, rate, sigma)
    out["more_income_deprived_than_city_median"] = income > income.median()
    out["more_gp_access_underserved_than_city_median"] = gp_min > gp_min.median()
    out["high_predicted_risk"] = high_risk
    out["high_uncertainty"] = high_unc
    return out


def attach_equity_labels(iz: pd.DataFrame) -> pd.DataFrame:
    return attach_priority_labels(iz)


def _hotspot_masks(iz: pd.DataFrame, rate: pd.Series, sigma: pd.Series) -> tuple[pd.Series, pd.Series]:
    high_risk = rate >= rate.quantile(HIGH_RISK_QUANTILE)
    if "uncertainty_flag" in iz.columns:
        high_unc = iz["uncertainty_flag"].astype(str).str.lower().eq("high")
    else:
        high_unc = sigma >= sigma.quantile(HIGH_SIGMA_QUANTILE)
    return high_risk.fillna(False), high_unc.reindex(rate.index).fillna(False)


def _relative_to_mean(series: pd.Series) -> pd.Series:
    mean = float(series.mean())
    if mean <= 0:
        return pd.Series(1.0, index=series.index)
    return series.astype(float) / mean


def _unit(series: pd.Series) -> pd.Series:
    total = float(series.sum())
    if total <= 0:
        return pd.Series(0.0, index=series.index)
    return series.astype(float) / total
