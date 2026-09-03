# GeoAI Health Agent Interface and Function Guide

This guide explains the sections of the Luciana GeoAI Health Agent, what each function does, and how the functions relate to one another. It is intended for reviewers, collaborators, public-health users, and readers without a technical GeoAI background.

## 1. Purpose of the Agent

The GeoAI Health Agent supports value-sensitive public-health planning. It combines stored infection forecasts, model uncertainty, population, deprivation, healthcare accessibility, candidate vaccination sites, and estimated travel times.

The system does not claim that one allocation is universally best or fairest. Users choose the planning perspective and travel constraint. The Agent then shows the consequences of that choice and selects exactly six recorded candidate sites.

The outputs are decision support. They do not replace public-health expertise, community consultation, operational feasibility assessment, or statutory approval.

## 2. Home page

The home page introduces the Agent and provides the starting controls for the workflow.

### About values, policies and responsible use

This panel explains why site allocation depends on planning priorities. It introduces the four policy perspectives and clarifies the limitations of automated decision support.

### City selection

Users select one of the two study areas:

- City of Edinburgh
- Glasgow City

The Agent keeps each city's models, spatial units, candidate sites, travel matrices, and results separate.

### Date selection and validation

The source timeline begins on 8 March 2020. The date selector lists only dates for which the application has a stored forecast result. A user can select a date from the list or enter an accepted date where free-text input is available.

Invalid dates and unsupported formats are rejected before the analytical workflow proceeds. The application does not silently replace an unavailable date with another date.

## 3. Floating Terminology Guide

The **Terminology Guide** remains on the right side of the page while the user scrolls. Its compact tab expands when the user hovers over it or gives it keyboard focus.

The guide provides plain-language definitions for concepts including:

- Intermediate Zone
- Whole region
- Scottish Index of Multiple Deprivation
- Baseline
- GeoShapley
- Prediction uncertainty
- Alpha graph-mix weights
- GP and pharmacy candidate sites
- Mobile stops
- Travel-time threshold

The Agent can also ask whether the user would like terminology support during the conversation.

## 4. The three main tasks

The application presents three related but independently selectable tasks. Task 1 provides forecasting and explanation. Tasks 2 and 3 use stored forecasts and planning inputs, so users do not have to run Task 1 manually before using them.

### Task 1: Show the forecast

Task 1 displays predicted infection rates and uncertainty. Users choose one of two spatial views.

#### Whole region

The Whole region view presents all Intermediate Zones in the selected city and provides a regional summary. It is intended for comparing spatial patterns across the city.

After the Whole region result is displayed, the interface moves directly to an Agent follow-up question about the alpha graph-mix result instead of returning the user to the top of the page.

#### Intermediate Zone

The Intermediate Zone view allows the user to select one neighbourhood-level spatial unit. It presents more detailed information for that selected area, including its forecast, uncertainty, and model explanation.

The difference between the Whole region and Intermediate Zone views is intentional: the first summarises the complete study area, while the second provides local detail.

### Task 2: Plan 6 vaccination sites

Task 2 selects exactly six sites from recorded candidate locations. Candidate types include GP practices, community pharmacies, and provisional mobile stops based on eligible public car parks.

The user selects a planning policy and a travel-time constraint. The Agent then combines the selected policy with population, predicted infection risk, uncertainty, deprivation, healthcare accessibility, and estimated travel times as required by that policy.

The output includes the selected sites, spatial assignments, population coverage, served and unserved areas, and travel-time summaries.

### Task 3: Compare four policies

Task 3 holds the six-site limit and travel rule constant and compares all four planning policies. It helps users understand how changing public-health priorities changes the selected locations and resulting outcomes.

The comparison is not a ranking of good and bad policies. It exposes the trade-offs created by different planning values.

## 5. Planning policies

Each policy button includes a short explanation so that users do not need specialist knowledge before making a selection.

### Coverage

Coverage prioritises reaching the largest possible number of residents within the selected travel-time limit.

### Equity

Equity gives greater priority to areas with higher income deprivation and poorer public-transport access to a GP, using indicators derived from the Scottish Index of Multiple Deprivation.

### Preventive

Preventive planning gives greater priority to areas with higher predicted infection risk or prediction uncertainty.

### Balanced

Balanced planning combines coverage, equity, and preventive considerations rather than optimising only one perspective.

## 6. Travel controls and results

Users choose the travel mode and time threshold used by the planning workflow. The stored travel-time matrices estimate journeys from each 2011 Intermediate Zone centroid to each candidate site on the OpenStreetMap network.

- Driving assumes an average speed of 30 km/h.
- Walking assumes an average speed of 4.5 km/h.

These are planning estimates, not observed journeys, live traffic conditions, public-transport schedules, or Google Maps travel times. An Intermediate Zone is counted as served when its nearest selected site is within the chosen threshold.

## 7. Agent follow-up interactions

Technical explanations use progressive disclosure: the Agent asks before presenting details that are not necessary for the core decision task.

### Terminology question

The Agent can offer the user a terminology guide. The user can accept the explanation or continue without it.

### Alpha graph-mix question

After a Whole region forecast, the Agent asks whether the user wants to see the alpha graph-mix results. Two explicit choices are provided:

- **Yes, show alpha results**
- **No, return to the forecast**

The alpha result is not displayed before the user accepts. After viewing it, the user receives options to return to the Whole region forecast or choose an Intermediate Zone.

## 8. Alpha graph-mix explanation

The prediction model uses three graph-based representations:

- Geographic relationships between neighbouring areas
- Transport-network relationships
- Origin-destination mobility relationships

Alpha values describe the model's learned relative weighting of these graph components. They are model-interpretation information and are not themselves infection rates or policy scores. For this reason, the graph-mix visualisation is shown only after the user requests the advanced explanation.

## 9. GeoShapley and Baseline

GeoShapley explains how socioeconomic, accessibility, and geographic factors move a local prediction away from its reference point.

The **Baseline** is the model's prediction starting point. At the Baseline, the model uses an ordinary reference situation instead of the selected area's own socioeconomic and location details. The Agent then considers factors such as:

- Income deprivation
- Employment deprivation
- Housing overcrowding
- Public-transport time to a GP
- Crime
- Higher education
- Geographic location

Each factor can move the prediction above or below the Baseline. Together, these adjustments produce the selected area's final predicted value. GeoShapley is an explanation of the prediction; it is not used directly to select vaccination sites.

## 10. Conversation and result history

The application preserves previous conversation messages and generated results during the user's session. Results are associated with the message that produced them rather than being collected in an unrelated panel at the bottom of the page.

Users can select an earlier conversation entry to reopen its corresponding result. Returning to the Luciana home view does not intentionally erase the session history.

The history is session-specific. It is not a shared database of another user's conversations, and a new browser session begins with its own history.

## 11. Data and reproducibility

The dashboard reads trained checkpoints and frozen processed outputs included in the GitHub repository. Reviewers do not need to retrain the models or download restricted raw data to reproduce the displayed interface and results.

The separate controlled raw-data bundle is required only for rebuilding and training the analytical pipeline from the beginning. See [REVIEWER_GUIDE.md](REVIEWER_GUIDE.md), [REPRODUCING.md](REPRODUCING.md), and [ARTIFACTS.md](ARTIFACTS.md) for installation instructions, reproduction boundaries, and controlled-data conditions.

## 12. Interpretation limitations

- Forecasts contain uncertainty and should not be treated as certain future outcomes.
- Candidate mobile stops are provisional locations, not approved clinics.
- Estimated travel times use fixed average speeds and do not represent real-time conditions.
- Policy outputs depend on the priorities selected by the user.
- The 4 March 2023 planning layer is an unverified operational extrapolation and is excluded from retrospective accuracy evaluation.
- Final decisions require domain expertise, local knowledge, community engagement, feasibility checks, and appropriate approval.
