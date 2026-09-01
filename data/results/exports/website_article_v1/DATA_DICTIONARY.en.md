# Results guide (for collaborators)

Location: `data/results/exports/website_article_v1/`

Chinese version: `DATA_DICTIONARY.md`

This note is for collaborators who did not run the model. Read **Section 1 (glossary)** first, then look up files and column names.

In one sentence: for 111 neighbourhoods in Edinburgh, we use the past 7 days of the rolling seven-day infection rate to forecast the rolling seven-day rate **7 days ahead**, with uncertainty every day and an explanation for the last (unverified) day only.

---

## 1. Glossary (read this first)

### Geography

| Term | Meaning |
|---|---|
| **IZ** | Intermediate Zone, a Scottish neighbourhood unit. Edinburgh uses the **2011** geography: **111** zones. |
| **iz_code** | Zone code, e.g. `S02001576`. Maps, forecasts and explanations all join on this key. |
| **node_index** | The zone’s index in the model order, 0 to 110. Do not reorder. |

### What we forecast

| Term | Meaning |
|---|---|
| **Rolling seven-day infection rate** | The published “last 7 days positive rate”, unit: **per 100,000**. This is **not** daily new cases. |
| **issue_date** | The last day the model is allowed to see. Think of it as “today”. |
| **target_report_date** | The day being forecast. Always **issue_date + 7 days**. |
| **observed_rate** | The rate that was later published for the target day. Needed to score accuracy. |
| **predicted_rate** | The model’s forecast for the target day. |

Example: if the issue date is 2023-02-18, the target date is 2023-02-25.  
The model uses rates from 12–18 February to forecast the rolling seven-day rate for 25 February.

### What U01 and U10 mean

The test period is about nine months. We do **not** train once and freeze the weights. We refit about every **28 days**. There are 10 updates:

**U01, U02, …, U10** = 1st update, 2nd update, …, 10th (final) update.

Treat them as “model versions”.

| ID | Issue dates it covers (approx.) | Plain language |
|---|---|---|
| U01 | 2022-05-31 to 2022-06-27 | First version in the test period |
| U02 | 2022-06-28 to 2022-07-25 | Second version |
| … | … | … |
| **U10** | Labelled retrospective use through 2023-02-18; the 4 March extrapolation also uses this version | **Last version** |

For a given day’s map, use **that day’s U0x**. Do not average all ten versions and then explain that day.

**checkpoint** = the saved weights for that version. In these tables `checkpoint_id` and `update_id` are the same label (U01–U10).

**Rolling** means this updating protocol **on the test set**, not on the training set. Train / validation dates are not in the website date list.

### Two kinds of date — do not mix them

| Type | What it is | Is there a published truth? | Can we score accuracy? |
|---|---|---|---|
| **Retrospective** | 2022-06-07 to 2023-02-25, **264** target days | Yes | Yes. These are the paper metrics. |
| **Unverified extrapolation** | **2023-03-04 only** | **No**. The panel ends 25 February. | **No** MAE, R² or coverage |

4 March uses **U10** because 25 February still falls inside U10’s scheduled 28-day live window.

### How the forecast is built (intuition)

1. Start from today’s rate \(Y_t\).  
2. The model predicts only the **change** \(\Delta\) (how much higher or lower in 7 days).  
3. Forecast rate = today’s rate + change.  
4. **Persistence baseline**: pretend the rate in 7 days is still today’s rate. Used to ask whether the model beats “no change”.

### What alpha (α) is

The model mixes three relationship graphs:

- **alpha_geo**: whether zones are neighbours  
- **alpha_transport**: directed road / public-transport graph  
- **alpha_mobility**: commuting OD (not real-time flow)

The three numbers are **positive and sum to 1**. For U10 they are about 0.32 / 0.32 / 0.36.

**Correct:** “This version relies a little more on the mobility graph.”  
**Incorrect:** “Mobility explains 36% of cases / 36% of vaccination demand.”

Alpha is **not** a risk map and **not** an allocation weight.

### Uncertainty

| Term | Meaning |
|---|---|
| **predicted_sigma (σ)** | How unsure the model is. Larger = more uncertain. Maps should colour this continuously. |
| **80% / 95% interval** | “We think the truth probably lies in this band.” Calibrated bands are the ones to report, not the raw Gaussian bands. |
| **uncertainty_flag** | `high` if σ is above the **90th percentile of σ for that model version**; otherwise `normal`. Optional overlay, not required. |

Uncertainty exists for **every one of the 264 retrospective days**, and also for 4 March.  
4 March has no observation, so you can show “how unsure the model is” but you cannot check whether the interval covered the truth.

### What GeoShapley is

It answers: **for this zone’s forecast**, relative to a city-wide median baseline, which socio-economic variables push the forecast up or down?

- Not causal (do not say “poverty caused infections”)  
- Does not explain neural embedding dimensions  
- Different from alpha  
- **Currently computed only for 4 March, U10**. The 264 retrospective days have no GeoShapley (it was turned off during rolling to save compute). We do **not** recommend computing all 264 days; a few staged dates would be enough if needed.

### Accuracy metrics (paper tables)

| Metric | Plain meaning | Better when… |
|---|---|---|
| **MAE** | Average absolute miss (per 100,000) | Smaller. Rolling **46.53**, persistence **50.66** |
| **MAE skill** | How much better than persistence: `1 − model MAE / persistence MAE` | Positive = better than “no change”. Rolling **+0.081** (about 8% better) |
| **RMSE** | Also an error score; large misses are penalised more | Smaller |
| **R²** | Share of target-day variation explained vs predicting the mean | Closer to 1. Rolling **0.67** |
| **bias** | Average over- or under-prediction | Near 0. Positive = systematically too high |
| **coverage** | Share of truths that fall inside the forecast interval | Nominal 80% covers about 82% in the test set; nominal 95% about 95% |

These numbers come from **retrospective test days only**. They do **not** include 4 March.

---

## 2. What each folder is for

```
website_article_v1/
  website/                  → future website (one row per zone and day)
  article/                  → paper tables (rounded for display)
  article/full_precision/   → same tables, full precision
  DATA_DICTIONARY.md        → Chinese guide
  DATA_DICTIONARY.en.md     → this file
  EXPORT_MANIFEST.json      → file list and checksums
```

Use `website/` for the webpage. Use `article/table01` … `table06` for the paper.

---

## 3. Each file: purpose, how it was made, column names

### 1) `website/retrospective_predictions.csv` (main retrospective table)

**One row:** one target day × one zone.  
**Rows:** 264 days × 111 zones = 29,304.  
**How:** forecast with that day’s model version (U01–U10). Published values exist, so errors can be computed.  
**Use:** after picking a date, draw predicted / observed / error / uncertainty maps; compare the model with persistence.

| Column | Meaning |
|---|---|
| `issue_date` | Cut-off day (last day the model may see) |
| `target_report_date` | Day being forecast (issue_date + 7) |
| `update_id` | Which model version, U01–U10 |
| `checkpoint_id` | Same as `update_id` |
| `iz_code` | Zone code |
| `node_index` | Zone index 0–110 |
| `observed_rate` | True rolling seven-day rate on the target day |
| `anchor_rate_y_t` | True rate on the issue day (starting point) |
| `predicted_delta` | Predicted change over 7 days |
| `predicted_rate` | Forecast = starting point + change |
| `predicted_sigma` | Uncertainty (larger = less sure) |
| `calibrated80_lower` / `upper` | Calibrated ~80% interval |
| `calibrated95_lower` / `upper` | Calibrated ~95% interval |
| `uncertainty_flag` | `high` if unusually uncertain for that version; else `normal` |
| `persistence_prediction` | “Assume no change” forecast; equals the starting point |
| `model_error` | prediction − truth (positive = too high) |
| `model_absolute_error` | Absolute error (≥ 0) |
| `persistence_absolute_error` | Absolute error of “assume no change” |
| `alpha_geo` | Geographic-graph weight for this version |
| `alpha_transport` | Transport-graph weight |
| `alpha_mobility` | Mobility-graph weight |
| `node_order_hash` | Fingerprint of the 111-zone order |
| `forecast_status` | Always `retrospective_evaluation` (labelled test) |
| `observed_target_available` | Always true |

### 2) `website/future_forecast_20230304.csv` (4 March, unverified)

**One row:** one zone on 2023-03-04 (111 rows).  
**How:** the panel ends 25 February. **U10** reads 19–25 February and forecasts 4 March. No retraining.  
**Use:** a separate website day; if discussing “next-week vaccination sites”, use `predicted_rate` and σ from this file as the risk layer.  
**Missing:** truth and errors. Do not score accuracy.

Extra columns vs the retrospective file:

| Column | Meaning |
|---|---|
| `predicted_variance` | σ squared |
| `include_in_metrics` | Always false: do not put this day in accuracy tables |
| `forecast_status` | `unverified_extrapolation` |
| `observed_rate` and all error columns | Empty: no published 4 March value |

### 3) `website/geoshapley.csv` (explanation, one day only)

**One row:** 4 March × one zone × one explanation component.  
**How:** change only that zone’s socio-economic variables and location, see how the forecast moves, then split the movement into parts.  
**Use:** click a zone to see what pushed the forecast up or down.  
**Missing:** the other 264 retrospective days.

| Column | Meaning |
|---|---|
| `component` | `baseline` = city-median starting forecast; `main` = a variable on its own; `location` = location; `location_interaction` = location × variable |
| `feature_name` | Which variable; see table below |
| `shapley_value` | How much this part adds to the forecast rate (can be negative) |
| `predicted_rate` | Model forecast for that zone |
| `reconstructed_prediction` | Sum of parts; should match the forecast |
| `additivity_error` | Reconstruction gap (near 0 here) |
| `explanation_scope` | `target_iz_local`: this zone only |
| `alpha_*` | U10 graph weights (context, not Shapley values) |

`feature_name` labels:

| Name in the file | Plain meaning |
|---|---|
| `baseline` | Forecast if this zone sat at city-wide medians |
| `income_deprivation` | Income deprivation |
| `employment_deprivation` | Employment deprivation |
| `higher_education` | Higher education |
| `overcrowding` | Housing overcrowding |
| `crime` | Neighbourhood crime |
| `public_transport_time_to_gp` | Public-transport time to a GP |
| `location` | Geographic location (easting and northing together) |
| `location_x_income_deprivation` etc. | Extra location × variable term, not a repeat of the main effect |

Parts should add up to the zone forecast. This is an additive decomposition, not “poverty caused infections”. Do not use it automatically as a siting score.

### 4) `website/rolling_alpha.csv` (graph weights for ten versions)

**One row:** one U0x.  
**Use:** a small chart of how the three graph weights change across updates.

| Column | Meaning |
|---|---|
| `update_id` | U01–U10 |
| `update_date` | Date this version starts |
| `forecast_start` / `forecast_end` | Issue-date range this version covers in the labelled test |
| `alpha_geo` / `alpha_transport` / `alpha_mobility` | Three graph weights, sum to 1 |
| `selected_epoch` | Training epoch that was selected (internal) |
| `training_window_days` | Each update fits on the past 730 days |
| `checkpoint_checksum` | Hash of the weight file |

### 5) `website/date_selector.csv`

Website date menu: the 264 retrospective issue dates only.  
Do **not** add train or validation dates. Do **not** mix 4 March into the same ordinary menu (label it separately as unverified).

### 6) `website/edinburgh_iz_boundaries.geojson`

111 zone polygons, web CRS **EPSG:4326** (longitude/latitude). Join to forecasts on `iz_code`.

### 7) `website/model_metadata.json`

Footer text for the website: forecast definition, graph meanings, calibration, explanation limits.

---

## 4. Paper tables (`article/`)

Numbers in `article/` are rounded for Word. The same tables at full precision are in `article/full_precision/`.

### table01 How the data are split

Time is cut earliest to latest: train → validation (half to pick the training epoch, half to calibrate uncertainty) → test.

| Column | Meaning |
|---|---|
| `split_name` | Which slice |
| `fraction` | Share of all target days |
| `target_start_date` / `target_end_date` | Target-day range |
| `n_unique_target_dates` | Number of distinct target days |
| `n_valid_iz_date_cells` | Zone × day cells (about 111 × days) |
| `mean_infection_rate` | Mean rate in this slice |
| `std_infection_rate` | How spread out the rate is |
| `mean_target_delta` | Average 7-day change |
| `std_target_delta` | How spread out the change is |

Test slice: 2022-06-07 to 2023-02-25. That is the website date range and the accuracy tables.

### table02 Which method is more accurate (full test)

Three rows: persistence (“assume no change”), fixed model (trained once), rolling model (U01–U10, the main result).

See MAE, skill, RMSE, R², bias above.  
Rolling MAE 46.53, about 8% better than persistence.

### table03 By epidemic period

The test period is split again:

- **Declining / wave period** 2022-06-07 to 2022-09-19  
- **Late stable period** from **2022-09-20** to 2023-02-25  
- **Overall**

Late-period MAE skill is negative: “assume no change” is often more stable then. Do not write that the model wins throughout.

### table04 How good is the uncertainty

Retrospective only.  
`raw_*` = interval from a Gaussian formula (comparison only).  
`calibrated_*` = interval after validation-period calibration (report these).

| Column | Meaning |
|---|---|
| `nominal_coverage` | The 80% or 95% we claim |
| `observed_coverage` | Share of truths actually inside the band (0 to 1) |
| `mean/median_interval_width` | How wide the band is |
| `mean_predicted_sigma` | Average uncertainty |
| `corr_sigma_absolute_error` | When σ is large, is the error also large? (about 0.63) |
| `gaussian_nll` | Probabilistic score; smaller is better |

### table05a / 05b Graph weights

- **05a:** each U01–U10’s own three alphas. Use 05a for a specific day.  
- **05b:** average and spread across ten updates. You may write “mixed, mobility slightly higher”. **Do not** use 05b to explain a single forecast day.

### table06 4 March explanation summary

One row per socio-economic variable, plus location.

| Column | Meaning |
|---|---|
| `mean_absolute_main_effect` | Average strength of the main effect (ignoring sign) |
| `mean_signed_main_effect` | Average push up or down |
| `mean_absolute_location_interaction` | Extra location × variable strength; blank on the location row |
| `positive_effect_iz_fraction` | Share of 111 zones pushed up |
| `negative_effect_iz_fraction` | Share pushed down |

Example: income deprivation pushes about half of zones up and half down, so do not write that it raises risk city-wide.

---
