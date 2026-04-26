# AI Simulation Algorithm

This document explains how wildfire spread simulation is computed in this project.

## Scope

The simulation is a research/demo algorithm. It is **not** a calibrated operational fire behavior model.

The full workflow has two layers:

1. Core spread model in [spread.py](c:/Codex/wildfire_digital_twin/app/models/spread.py)
2. App-level orchestration in [app.py](c:/Codex/wildfire_digital_twin/app/app.py)

## Inputs

Primary inputs:

- FIRMS detections (`latitude`, `longitude`, `frp`, etc.)
- Weather:
  - wind speed (km/h)
  - wind direction (degrees)
  - relative humidity (%)
- Optional raster/context layers:
  - terrain slope and elevation
  - NDVI-like vegetation proxy
  - fuel moisture proxy
  - infrastructure (roads, power lines, buildings)

## High-Level Pipeline

1. Fetch FIRMS detections for the requested bbox/time range.
2. User selects which detections to simulate.
3. For each selected detection, build a local window of `0.1 deg x 0.1 deg` (`+- 0.05 deg` lat/lon).
4. Run one local simulation at a time (sequentially), each with Monte Carlo runs.
5. Convert local rasters to map points and aggregate all local outputs.
6. Render map layers and summary metrics.

## Core Spread Model (`SpreadPredictor`)

### 1. Surrogate ignition model training

On predictor construction, a synthetic training set is generated (random samples of fuel, dryness, slope, wind, FRP, hotspot density, infrastructure density).

A target spread probability `y` is computed from weighted components:

- fuel and dryness (largest weights)
- slope, wind alignment/speed
- FRP intensity
- hotspot density
- infrastructure densities

Then `RandomForestRegressor` is trained on this synthetic dataset.

Notes:

- This is a surrogate model, not physics-based combustion.
- It is deterministic for a fixed seed.

### 2. Environment field creation

The model builds gridded fields:

- fuel
- dryness
- slope

Then optionally blends:

- NDVI into fuel
- fuel moisture into dryness
- terrain slope override from real terrain layer

### 3. Fire seeding

FIRMS detections are projected to grid indices and used to create:

- `seed_fire` (initial burning cells)
- `frp_grid`
- local hotspot density raster

### 4. Monte Carlo simulation loop

For each run:

1. Start from seeded cells.
2. For each burning cell, evaluate 8-neighbor spread.
3. Build feature vector per candidate neighbor:
   - fuel, dryness, slope
   - wind speed
   - direction alignment with wind
   - FRP source intensity
   - hotspot density
   - structure/road/power density
4. Predict spread probability with surrogate RF model.
5. Ignite neighbor stochastically (`rng.random() < p`).
6. Continue for configured step count.

Run outputs are accumulated into burn frequency.

### 5. Final outputs per local area

- `burn_probability` = burned_count / completed_runs
- `risk_score` = weighted combination of:
  - burn probability
  - dryness
  - fuel
  - slope
  - infrastructure exposure
  - FRP signal
  - normalized elevation

Risk is clipped to `[0, 100]`.

## App-Level Orchestration (Area-by-Area)

The app no longer runs one giant simulation over the full bbox.

Instead:

1. Build local jobs from selected FIRMS detections.
2. For each job:
   - collect nearby detections in `+-0.05 deg`
   - run local `SpreadPredictor.simulate(...)`
   - convert local rasters to points (`burn_pts`, `risk_pts`, `terrain_pts`)
3. Aggregate all local points and summary stats.

Progress is reported as local areas completed (with intra-area partial updates).

## Stopping and Progress

- A stop signal can be requested from UI.
- Simulation stops cleanly between iterations.
- Output reports whether run stopped early.

Metadata fields:

- `requested_areas`
- `completed_areas`
- `requested_runs_per_area`
- `stopped_early`

## Caching and Fallbacks

Data fetches are cached in memory and on disk (`data_cache/`).

Service fallbacks:

- FIRMS endpoint fallback (`/usfs/api/...` then `/api/...`)
- NDVI offline synthetic fallback when USDA service fails
- Terrain fallback when OpenTopoData rate-limits/fails
- Optional demo hotspots when FIRMS returns no detections

## Known Limitations

- Not physically calibrated (no crown fire, spotting, fuel models, suppression, or detailed moisture physics).
- Uses synthetic surrogate training data.
- Local-area split may underrepresent long-range interactions between distant fire clusters.
- Outputs are useful for exploratory visualization, not operational incident decisions.
