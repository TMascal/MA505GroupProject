# MA505 Group Project — Causal Analysis of Aircraft Accident Fatalities
Read me file generated completely with Claude Sonnet-4.6, reviewed by Tim Mascal, April 24, 2026.

## Purpose

This project estimates the causal effect of various factors (pilot error, weather, mechanical failure, sabotage, etc.) on the lethality of aircraft accidents. It uses causal discovery (FCI) to learn a graph from observational data, then applies DoWhy for identification and OLS for estimation.

## Pipeline Overview

```
data/Airplane_Crashes_and_Fatalities_Since_1908.csv
        │
        ▼
  data_labeling.py          — classify causes, subregion, flight type
        │
        ▼
data/labeled_accidents.csv
        │
        ├──► evaluate_classifier.py   — validate labels against manual review
        │
        ▼
  causal_dowhy.py           — FCI discovery → PAG → DoWhy → OLS estimates
        │
        ▼
  output/dowhy/pag.png      — rendered causal graph
  output/pag_kci.pkl        — cached PAG (reused on subsequent runs)
```

### Step 1 — Label the data

```bash
python data_labeling.py
```

Reads the raw CSV and produces `data/labeled_accidents.csv` with columns:
`Date, Time, Location, Subregion, Operator, FlightType, Cause, Aboard, Fatalities, FatalityRate`

### Step 2 — (Optional) Evaluate classifier accuracy

```bash
python evaluate_classifier.py
```

Compares algorithm labels against `data/manual_review_sample_accidents.csv` and prints per-category precision, recall, and F1.

### Step 3 — Run causal analysis

```bash
python causal_dowhy.py
```

Runs FCI with chi-square independence tests (or loads a cached PAG), converts the result to a DoWhy graph, and prints OLS causal effect estimates for each cause on `high_lethality`.

## File Descriptions

| File | Description |
|---|---|
| `data_labeling.py` | `AccidentCauseClassifier` (keyword-based cause labeling) and `LocationSubregionClassifier` (UN M49 subregion mapping). Run directly to regenerate `labeled_accidents.csv`. |
| `evaluate_classifier.py` | Reads `manual_review_sample_accidents.csv` and computes TP/TN/FP/FN, precision, recall, F1 per cause category. |
| `causal_dowhy.py` | Full causal pipeline: data preparation, background knowledge encoding, FCI discovery, PAG-to-DAG conversion, DoWhy identification, OLS estimation, and subregion breakdown. |

## Data Files

| File | Description |
|---|---|
| `data/Airplane_Crashes_and_Fatalities_Since_1908.csv` | Raw input — all recorded airplane crashes since 1908 |
| `data/labeled_accidents.csv` | Output of `data_labeling.py` — used by causal pipeline |
| `data/manual_review_sample_accidents.csv` | Hand-labeled sample for classifier validation |

## Cause Categories

Labels assigned by `AccidentCauseClassifier`, checked in priority order:

1. `sabotage` — hijacking, explosives, bombs, suicide
2. `shot_down` — enemy fire, anti-aircraft, gunfire
3. `fire` — fire, flames, explosion
4. `fuel` — fuel exhaustion, starvation, out of fuel
5. `mechanical` — engine failure, landing gear, icing, structural
6. `weather` — fog, wind, turbulence, visibility, storms
7. `collision` — mid-air collision with another aircraft
8. `cfit` — Controlled Flight Into Terrain (mountains, terrain, ocean)
9. `pilot_error` — lost control, stall, nosedive, improper procedures
10. `undetermined` — disappeared, never found, cause unknown
11. `unknown` — no matching keywords

Multiple causes are allowed (comma-separated). `undetermined` is dropped if a more specific cause also matched.

## Causal Model Assumptions

Background knowledge encoded in `causal_dowhy.py`:

- **Exogenous roots**: `subregion`, `weather`, `is_military` — nothing in the graph causes these.
- **Required edges**: `pilot_error → cfit`, `pilot_error → mechanical`
- **Forbidden edges**: `cfit → pilot_error`, `mechanical → pilot_error`, `fire → sabotage`, `high_lethality → *`
- **Outcome**: `high_lethality` (binary: FatalityRate > 0.5) cannot cause anything.

Rows excluded before discovery: unknown/undetermined cause, solo flights (Aboard=1, Fatalities=1), Aboard=0, unknown subregion.

## Required Libraries

Install with pip:

```bash
pip install pandas numpy networkx graphviz statsmodels dowhy causal-learn
```

| Library | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `networkx` | Graph representation for DoWhy |
| `graphviz` | Rendering the causal graph to PNG |
| `statsmodels` | OLS regression for causal estimation |
| `dowhy` | Causal identification (backdoor/frontdoor/IV) |
| `causal-learn` | FCI causal discovery algorithm |

**Note:** `graphviz` also requires the Graphviz system binaries in addition to the Python package. On Windows, download from [graphviz.org](https://graphviz.org/download/) and add to PATH. On Linux/macOS: `apt install graphviz` / `brew install graphviz`.

Standard library modules used: `re`, `os`, `pickle`, `warnings`.

## Output

- `output/dowhy/pag.png` — Rendered causal graph. Observed nodes are plain ellipses; latent confounders are grey dashed ellipses.
- `output/pag_kci.pkl` — Cached PAG. Delete this file to force a fresh FCI run.