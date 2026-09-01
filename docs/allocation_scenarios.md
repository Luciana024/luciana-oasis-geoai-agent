# Four allocation scenarios — definition

Location: `docs/allocation_scenarios.md`  
Machine-readable twin: `configs/allocation.yaml`

This note defines how six intervention sites are chosen for City of Edinburgh. Read **Section 1** first. The language model does not choose sites.

In one sentence: from the recorded candidate-site table, a deterministic greedy covering pick **six** sites under four demand weights (coverage, equity, preventive, balanced), then assigns each 2011 Intermediate Zone (IZ) to the nearest selected site within a **20-minute drive**.

---

## 1. Glossary (read this first)

### Geography and sites

| Term | Meaning |
|---|---|
| **IZ** | 2011 Intermediate Zone. Edinburgh (`CA=S12000036`) has **111** zones. Join key: `iz_code`. |
| **Candidate site** | A row already in `data/results/candidate_sites/S12000036/merged_candidate_sites.csv`. Types: `gp`, `pharmacy`, `mobile_stop`. |
| **mobile_stop** | A public car park kept after oasis-v4 OSM tag filters. Provisional, not a confirmed clinic. |
| **n_sites** | Fixed at **6**. Not a user input. Sites are never invented. |

### Demand ingredients (not the same thing)

| Term | Field | Meaning |
|---|---|---|
| **Population** | `Population` on the last date in `data/results/panel.csv` | People living in the IZ. Same denominator as the infection rate. **Not** SIMD 2020 `total_population`. |
| **Deprived** | SIMD 2020v2 `income_rate` | Share of people who are income-deprived. Continuous. **Not** a yes/no class. |
| **Underserved** | SIMD 2020v2 `pt_gp_min` | Public-transport minutes to a GP (current access). Continuous. **Not** our OSM drive/walk matrix. |
| **Predicted risk** | `predicted_rate` | U10 forecast of the rolling seven-day rate, **per 100,000**, for **2023-03-04**. |
| **Uncertainty** | `predicted_sigma` | Model uncertainty for that forecast. Larger = less sure. |

Deprived and underserved can disagree: an IZ may be income-deprived and close to a GP, or well-off and far from a GP.

We **do not** label IZs as “SIMD 20% most deprived”. That official rank is Data Zone geography, not 2011 IZ. City-median flags (`more_income_deprived_than_city_median`, `more_gp_access_underserved_than_city_median`) are display labels only. They are not a siting rule.

The COVID panel has **no age-band population**, so we do not allocate by 65+ or other subgroups.

### Travel time

| Term | Meaning |
|---|---|
| **Travel time** \(t_{ij}\) | Minutes from IZ \(i\) centroid to site \(j\) on the OSM graph. File: `data/results/travel_time/S12000036/travel_time_matrix.csv`. |
| **Default mode** | `drive`. Assumed speed **30 km/h**. Walk is **4.5 km/h**. These are oasis-v4 urban averages, **not** official travel-to-work or GTFS times. |
| **Threshold** | **20 minutes**. After sites are chosen, an IZ is **served** only if its nearest selected site is ≤ 20 minutes. Otherwise it is unserved; the time is left missing, not filled. |

### Notation

- \(\mathrm{mean}(x)\): unweighted mean of \(x\) over the 111 IZs. If the mean is 0, the relative term is 1.
- \(\mathrm{unit}(x) = x / \sum x\) over the 111 IZs. If the sum is 0, all weights are 0.
- Covering score \(c_{ij}\) is how much site \(j\) counts toward IZ \(i\) **while choosing sites**. Assignment after selection is always the binary 20-minute rule, in every scenario.

### What this definition does not use

GeoShapley, graph \(\alpha\), and persistence MAE are explanation or accuracy tools. They are **not** siting scores.

The 4 March 2023 forecast is an **unverified extrapolation** (U10; panel ends 25 February). Do not score allocation against a published 4 March rate; there is none.

---

## 2. Shared decision (all four scenarios)

1. Eligible sites = every `site_id` in the candidate table of types `gp`, `pharmacy`, `mobile_stop`.
2. Pick exactly six of those IDs by greedy covering (Section 4). Ties: smaller `site_id`.
3. Assign each IZ to the **nearest selected site** with \(t_{ij} \le 20\) minutes (drive).
4. Report population covered, IZs covered, mean/max travel time among served IZs, and unserved population. Do not fill unserved times.

Only the **demand weight** \(w_i\) and the **covering score** \(c_{ij}\) change by scenario.

---

## 3. The four scenarios

### 3.1 Coverage-priority

**Objective:** maximise the population covered within the travel-time threshold.

| | |
|---|---|
| Demand | \(w_i = \mathrm{Population}_i\) |
| Covering score | \(c_{ij} = 1\) if \(t_{ij} \le 20\), else \(0\) |
| Ignores | `income_rate`, `pt_gp_min`, `predicted_rate`, `predicted_sigma` |

Plain language: treat every resident equally. Prefer sites that bring the largest number of people inside 20 minutes who are not already covered. If covering gain is tied or already zero, pick the site that most reduces population-weighted travel time. Leftover slots are **not** filled by sorting `site_id` (that always preferred `GP_*` over `MS_*` car parks).

### 3.2 Equity-priority

**Objective:** prioritise deprived and currently underserved IZs.

| | |
|---|---|
| Deprived | `income_rate` (higher = more income-deprived) |
| Underserved | `pt_gp_min` (higher = longer public-transport time to a GP today) |
| Demand | \(w_i = \mathrm{Population}_i \times \dfrac{\mathrm{income\_rate}_i}{\mathrm{mean}(\mathrm{income\_rate})} \times \dfrac{\mathrm{pt\_gp\_min}_i}{\mathrm{mean}(\mathrm{pt\_gp\_min})}\) |
| Site pick | greedy p-median: add the candidate that most reduces \(\sum_i w_i \min_{j \in S} t_{ij}\) |

Plain language: a resident in a more income-deprived **and** worse-access IZ pulls sites closer to them. A car park nearer that IZ can beat a GP that is farther away. Binary 20-minute covering is **not** the primary score here (it saturates Edinburgh in three sites and collapsed this scenario onto coverage).

### 3.3 Preventive-priority

**Objective:** prioritise IZs with high predicted risk **and/or** high uncertainty.

An IZ is a preventive hotspot if **either**:

| Rule | Field | Cut |
|---|---|---|
| High predicted risk | `predicted_rate` | at or above the **75th percentile** among the 111 IZs |
| High uncertainty | `uncertainty_flag` | `high` (U10: σ above the 90th percentile of that version). If the flag is missing, σ at or above the 90th percentile of the 111 IZs |

All other IZs get **weight 0**. They are not used to choose sites in this scenario.

| | High-risk IZs | High-uncertainty IZs |
|---|---|---|
| Demand piece | \(\mathrm{Population} \times (\mathrm{rate}/\mathrm{mean\ rate})\) | \(\mathrm{Population} \times (\sigma/\mathrm{mean\ }\sigma)\) |
| Combined | \(w_i = \mathrm{unit}(\mathrm{risk\_hot}) + \mathrm{unit}(\mathrm{unc\_hot})\) | |

An IZ that is only high-σ still counts (the “or”). An IZ that is both gets both pieces.

**Site pick:** greedy covering of this hotspot demand (not city-wide p-median). Spreading \(\mathrm{Population}\times\mathrm{rate}\) over all 111 zones made preventive look like coverage/equity and sat the same six sites.

### 3.4 Balanced

**Objective:** jointly consider coverage, equity, and the preventive hotspots (risk and uncertainty).

| | |
|---|---|
| Demand | \(w_i = \mathrm{unit}(w^{\mathrm{cov}}) + \mathrm{unit}(w^{\mathrm{eq}}) + \mathrm{unit}(w^{\mathrm{prev}})\) |
| Site pick | greedy p-median on that mix |

`w^{\mathrm{prev}}\) is the hotspot vector from 3.3, not a smooth rate over every IZ.

---

## 4. Solver (how the six sites are picked)

Eligible types are `gp`, `pharmacy`, and `mobile_stop`. Parking is not excluded; it loses only if its travel-time reduction (or covering gain) is worse.

**Coverage-priority** — six greedy steps on total population (covering, then travel-time reduction).

**Preventive-priority** — the same covering steps, but demand is **only** high-risk and/or high-uncertainty IZs.

**Equity, balanced** — greedy p-median on that scenario’s \(w_i\).

Then assign IZs as in Section 2. Do not invent IDs to pad to six.

---

## 5. Files the solver may read

| Role | Path |
|---|---|
| Forecast (risk, \(\sigma\)) | `data/results/exports/website_article_v1/website/future_forecast_20230304.csv` |
| Population | `data/results/panel.csv` |
| Deprivation / GP access | `data/results/simd_iz.csv` |
| Candidate sites | `data/results/candidate_sites/S12000036/merged_candidate_sites.csv` |
| Travel time | `data/results/travel_time/S12000036/travel_time_matrix.csv` |

Missing IZ population, SIMD, forecast rate, candidate `site_id`, or travel-time overlap is a hard fail. Values are not invented.

Allocation outputs (once run) go to `data/results/allocation/<scenario>/`.
