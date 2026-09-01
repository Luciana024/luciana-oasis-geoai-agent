"""Coverage metrics from IZ assignments. No filled-in travel times."""

from __future__ import annotations

import pandas as pd


def allocation_metrics(iz: pd.DataFrame, assignments: pd.DataFrame) -> dict[str, float | int | None]:
    iz_tbl = iz.reset_index()
    if "iz_code" not in iz_tbl.columns:
        iz_tbl = iz_tbl.rename(columns={iz_tbl.columns[0]: "iz_code"})
    joined = assignments.merge(iz_tbl, on="iz_code", how="left")
    served = joined.loc[joined["served"].astype(bool)]
    unserved = joined.loc[~joined["served"].astype(bool)]
    times = pd.to_numeric(served["travel_time_min"], errors="coerce").dropna()
    pop = pd.to_numeric(joined["population"], errors="coerce")
    return {
        "population_covered": float(pd.to_numeric(served["population"], errors="coerce").sum()) if len(served) else 0.0,
        "iz_covered": int(len(served)),
        "mean_travel_time_min": float(times.mean()) if len(times) else None,
        "max_travel_time_min": float(times.max()) if len(times) else None,
        "unserved_population": float(pd.to_numeric(unserved["population"], errors="coerce").sum()) if len(unserved) else 0.0,
        "unserved_iz": int(len(unserved)),
        "n_iz": int(len(joined)),
        "population_total": float(pop.sum()) if pop.notna().any() else None,
    }
