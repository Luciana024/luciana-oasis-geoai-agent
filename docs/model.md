# Model and Analytical Method

This document defines the final forecasting, explanation, and vaccination-site planning method used by the Luciana GeoAI Health Agent.

## 1. Study scope

The validated study areas are:

- City of Edinburgh (`S12000036`), with 111 2011 Intermediate Zones
- Glasgow City (`S12000049`), with 136 2011 Intermediate Zones

For every Intermediate Zone, the forecasting task uses seven observed rolling seven-day COVID-19 infection rates to predict the rolling seven-day rate reported seven days after the issue date:

```text
[Y(t-6), ..., Y(t)] -> Y(t+7)
```

The model produces one forecast per Intermediate Zone, not a seven-step forecast sequence. Rates are reported per 100,000 population.

## 2. Temporal data and split

The complete source timeline begins on 8 March 2020 and ends on 25 February 2023. Windows with missing required dates are excluded rather than filled with zero.

The final experiment orders valid samples by target date and applies a chronological 65/10/25 split:

- 65% training
- 10% validation
- 25% testing

The validation period is divided chronologically into checkpoint-selection and uncertainty-calibration subsets. Test observations are never used for training, checkpoint selection, calibration, or uncertainty-threshold selection.

The archived configuration is stored in `configs/model.yaml` and `configs/model_split65_10_25.yaml`. Frozen temporal arrays, dates, scalers, and node orders are stored under `data/results/forecast/` for Edinburgh and `data/results/regions/S12000049/forecast_split65_10_25/` for Glasgow.

## 3. Spatial units and contextual variables

The canonical node key is the 2011 Intermediate Zone code. Boundaries, centroids, forecasts, explanations, and planning tables are joined using this code. Node order is validated at each analytical stage and is not inferred from row position.

Six contextual variables are derived from the Scottish Index of Multiple Deprivation 2020v2 indicators:

- Income deprivation rate
- Employment deprivation rate
- Higher-education entry indicator
- Housing overcrowding rate
- Crime rate
- Public-transport time to a GP

Projected Intermediate Zone centroid coordinates provide the geographic-location context. Static variables and coordinates are scaled using parameters fitted without test leakage.

## 4. Graph construction

The model uses three graph representations with the same canonical Intermediate Zone node order.

### Geographic graph

The geographic graph represents spatial relationships derived from the 2011 Intermediate Zone geography.

### Transport graph

The transport graph represents network-distance relationships derived from the OpenStreetMap and Geofabrik road, walking, cycling, and rail layers. These graph distances are not the same as the separate site-planning travel-time matrix.

### Mobility graph

The mobility graph is a directed graph derived from origin-destination flows. It represents average movement relationships rather than real-time pandemic mobility. Sparse origin-destination pairs are retained as genuine sparsity rather than completed with invented flows.

Geographic relationships may be symmetric. Transport and mobility directions are preserved. The system does not silently remove a graph at inference time; the loaded checkpoint must match the declared graph set.

## 5. Adaptive graph fusion

The three graph components share one learned global fusion vector:

```text
alpha = softmax(theta)
```

Each alpha value is positive and the values sum to one. Forward and backward diffusion supports are constructed separately and remain separate after fusion. The same alpha vector is used by the contextual graph encoder and the temporal forecasting encoder.

Alpha values describe learned model weighting. They are not infection rates, causal effects, or policy scores. The dashboard therefore presents them only when a user requests the advanced graph-mix explanation.

## 6. Forecasting architecture

The forecasting backbone is an encoder-only Diffusion Convolutional Recurrent Neural Network (DCRNN). Each recurrent step combines the current infection-rate input, the previous hidden state, and directed diffusion over the fused graph supports.

After seven input steps, the final node-level hidden representation is passed to a univariate probabilistic prediction head. The head returns:

- Predicted mean
- Predicted variance
- Predicted standard deviation
- Raw Gaussian intervals
- Empirically calibrated intervals when calibration is available

Variance is constrained to be positive using a softplus transformation. Training minimises univariate Gaussian negative log likelihood. This component is accurately described as a **UQGNN-inspired univariate probabilistic prediction head**; the project does not claim to reproduce the full multivariate UQGNN architecture.

## 7. Rolling evaluation and calibration

The final rolling evaluation uses a 730-day training window and ten archived updates, labelled U01 to U10. Each update records its checkpoint, node order, graph set, alpha values, predictions, and metrics.

Prediction intervals are calibrated only after checkpoint selection, using the reserved validation-calibration subset. The project reports mean absolute error, root mean squared error, coefficient of determination, and empirical interval coverage. Persistence is used as the simpler reference forecast for performance comparison.

Expected rounded retrospective metrics are:

| City | Model MAE | Persistence MAE | Model RMSE | Model R2 |
|---|---:|---:|---:|---:|
| Edinburgh | 46.53 | 50.66 | 69.06 | 0.67 |
| Glasgow | 44.73 | 47.97 | 63.32 | 0.60 |

The 4 March 2023 operational layer is an unverified extrapolation because the observed panel ends on 25 February 2023. It is excluded from retrospective accuracy metrics.

## 8. GeoShapley explanation

GeoShapley explains a selected Intermediate Zone forecast using six socioeconomic and accessibility variables plus geographic location as a joint player. Explanations are target-local: the selected zone's contextual inputs change across coalitions while the remaining graph context is preserved.

For six contextual variables and one joint location player, the exact explanation evaluates 128 coalitions. The decomposition contains:

- Baseline
- Six contextual main effects
- Intrinsic location effect
- Six location-context interaction effects

The components reconstruct the selected area's prediction within numerical tolerance.

### Baseline

The Baseline is the model's prediction starting point. The selected area's own contextual and location information is replaced by the reference situation. Contextual and location contributions then move the forecast above or below that starting point to produce the final prediction.

GeoShapley explains model behaviour; it is not used as the vaccination-site allocation score and must not be interpreted as causal inference.

## 9. Candidate vaccination sites

The planning system selects only from recorded candidate tables. It does not invent locations.

Candidate types are:

- GP practices from Public Health Scotland records
- Community pharmacies from Public Health Scotland dispenser records
- Provisional mobile stops derived from eligible OpenStreetMap public car parks

GP and pharmacy postcodes are geocoded using postcodes.io, with an eligible nearby named OpenStreetMap feature used when the recorded matching criteria are met. Car parks are filtered for access, parking type, capacity, meaningful name, park-and-ride status, duplication, and airport exclusion. A mobile stop is a possible planning location, not a confirmed clinic.

Frozen candidate tables and provenance records are stored under `data/results/candidate_sites/`.

## 10. Site travel-time matrix

Site-planning travel time is calculated from each official 2011 Intermediate Zone centroid to every candidate site on OpenStreetMap driving and walking networks. Origins and destinations are snapped to their nearest network nodes, and NetworkX Dijkstra shortest paths are calculated using edge travel time.

The fixed planning assumptions are:

- Driving speed: 30 km/h
- Walking speed: 4.5 km/h

These values are planning assumptions, not observed journeys, live congestion, Google Maps results, GTFS schedules, or official travel-to-work measurements. Unreachable pairs remain missing rather than being assigned an invented value.

The matrices and their provenance are stored under `data/results/travel_time/`.

## 11. Six-site allocation

Every policy selects exactly six candidate sites. After selection, each Intermediate Zone is assigned to its nearest selected site if that site is within the chosen travel-time threshold. Areas outside the threshold remain unserved.

The policies represent different planning perspectives:

- **Coverage:** prioritises the largest population reached within the travel-time threshold.
- **Equity:** gives greater priority to income-deprived areas and areas with poorer public-transport access to a GP.
- **Preventive:** gives greater priority to higher predicted infection risk or uncertainty.
- **Balanced:** combines coverage, equity, and preventive considerations.

The deterministic greedy solver reports the selected sites, assignments, served and unserved population, and travel-time summaries. The policy comparison is decision support, not a declaration that one value system is universally correct.

Detailed allocation definitions are provided in [allocation_scenarios.md](allocation_scenarios.md).

## 12. Frozen results and reproducibility

The GitHub repository contains the final checkpoints, temporal arrays, graph supports, predictions, calibration metadata, GeoShapley outputs, candidate sites, travel-time matrices, allocation results, and dashboard exports. These files allow reviewers to run the dashboard and inspect the submitted results without retraining.

Some raw public-health and origin-destination data are distributed separately under controlled conditions because of access, copyright, licensing, and privacy constraints. They are needed only to rebuild and train the models from the beginning. See the repository-level [REVIEWER_GUIDE.md](../REVIEWER_GUIDE.md), [REPRODUCING.md](../REPRODUCING.md), and [ARTIFACTS.md](../ARTIFACTS.md).

## 13. Interpretation boundaries

- Forecasts are uncertain estimates, not guaranteed outcomes.
- GeoShapley describes model contributions, not causal effects.
- Alpha values describe learned graph weighting, not graph quality or policy importance.
- Travel times use simplified fixed-speed assumptions.
- Mobile stops require feasibility review and approval.
- Allocation results depend on the user's chosen values and constraints.
- Final decisions require public-health expertise, local knowledge, community consultation, and statutory approval.
