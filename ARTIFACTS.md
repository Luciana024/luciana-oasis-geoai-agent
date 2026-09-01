# External artefact manifest

Large, licensed or generated artefacts are distributed separately from the
source archive. Preserve the following relative path families when assembling
the official reproduction bundle.

| Path family | Contents | Required for |
|---|---|---|
| `data/raw/covid/` | PHS source extracts | data preparation |
| `data/raw/boundaries/` | 2011 IZ boundaries | graph, map and operational export |
| `data/raw/deprivation/` | SIMD source tables | contextual features and equity |
| `data/raw/mobility/` | city OD matrices | mobility graph |
| `data/raw/roads/` | Geofabrik/OSM road layers | road graph and travel time |
| `data/raw/gp/` | local GP source files when API is not used | candidate preparation |
| `data/raw/candidate_sites/` | recorded candidate-site inputs | allocation |
| `data/results/forecast/` | frozen S1 tensors, dates, scalers and node order | training/evaluation |
| `data/results/graph/` | three graph matrices and validation reports | model training |
| `data/results/model/` | checkpoints, calibration and rolling predictions | evaluation/operational inference |
| `data/results/candidate_sites/` | validated city candidate tables | allocation |
| `data/results/travel_time/` | IZ-to-site travel matrices | allocation |
| `data/results/regions/S12000049/` | isolated Glasgow model/planning outputs | Glasgow reproduction |
| `data/results/exports/` | frozen website/article tables | dashboard/result verification |

Every official data archive should include a machine-generated SHA-256
manifest and a provenance note stating the source URL or custodian, retrieval
date, licence/redistribution status and any preprocessing already applied.

Do not upload caches, Python bytecode, local logs, `.pytest_cache`, temporary
public-URL scripts, or the historical `old_code` research prototype.
