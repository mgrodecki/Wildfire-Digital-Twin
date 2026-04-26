# Wildfire AI Dashboard Prototype - Terrain, NDVI, and Infrastructure Edition

This laptop-friendly Streamlit app combines:

- live NASA FIRMS active fire detections
- open weather data from Open-Meteo
- terrain/elevation from OpenTopoData
- NDVI/fuel-moisture from USDA VegScape (with offline fallback)
- infrastructure overlays from OpenStreetMap Overpass
- a lightweight AI-assisted Monte Carlo spread model

## Features

### Terrain
- Fetches elevation samples for the selected bounding box from OpenTopoData
- Derives slope in degrees from sampled DEM values
- Feeds slope and elevation into risk scoring and map display

### NDVI / fuel moisture
- Pulls NDVI visualization tiles from USDA VegScape WMS
- Converts relative greenness to a normalized NDVI-like raster
- Derives a fuel-moisture proxy used in spread simulation
- Falls back to a synthetic offline NDVI proxy if the remote WMS is unavailable

### Infrastructure
- Pulls roads, overhead power lines, and buildings from OpenStreetMap Overpass
- Converts features into map overlays and exposure rasters
- Adds infrastructure exposure weighting into wildfire risk score

### FIRMS fetch diagnostics
- Uses USFS FIRMS API first, then global FIRMS API fallback
- Surfaces fetch failures clearly instead of silently treating all HTTP failures as "no active fires"

## Project structure

```text
app/
  app.py
  services/
    firms.py
    weather.py
    terrain.py
    ndvi.py
    infrastructure.py
  models/
    spread.py
  utils/
    geo.py
tests/
requirements.txt
```

## Setup

Windows PowerShell:

```powershell
py -3 -m pip install -r requirements.txt
```

Linux/macOS:

```bash
python -m pip install -r requirements.txt
```

Set your NASA FIRMS key before running:

Linux/macOS:

```bash
export FIRMS_MAP_KEY="your_key_here"
```

Windows PowerShell:

```powershell
$env:FIRMS_MAP_KEY="your_key_here"
```

## Run the dashboard

CLI:

```powershell
py -3 -m streamlit run app/app.py
```

VS Code:

- Run task: `Run dashboard`
- Stop task: `Stop dashboard`

Default local URL:

- `http://localhost:8501`

## Testing

Run tests:

```powershell
py -3 -m pytest -q
```

Current suite includes smoke tests for:

- `utils.geo`
- `services.firms`
- `services.weather` (mocked HTTP)
- `services.ndvi` fallback generation
- `models.spread`

## Simulation Algorithm

- Detailed documentation: [AI_SIMULATION_ALGORITHM.md](c:/Codex/wildfire_digital_twin/docs/AI_SIMULATION_ALGORITHM.md)
- Covers:
  - surrogate model logic
  - Monte Carlo spread steps
  - area-by-area orchestration around selected FIRMS detections
  - progress/stop behavior
  - outputs, assumptions, and limitations

## Troubleshooting

### "No active fires returned for the selected area/time window."

- This can be real (no detections for selected bbox/time/datasets)
- Optional fallback is available in the sidebar: `Use demo hotspots when FIRMS is empty`

### FIRMS returns data on the NASA page but app shows none

- Verify `FIRMS_MAP_KEY` is set in the same shell/session that launched Streamlit
- Check selected datasets and day range
- The app now reports explicit FIRMS fetch failures in UI when all datasets fail

### NDVI layer unavailable

- If USDA VegScape is unreachable (DNS/network/service issues), the app now uses an offline synthetic NDVI fallback and continues simulation

## Important notes

- This is a research/demo prototype, not an operational fire-behavior model
- Overpass requests can be slow for very large bounding boxes
- For production use, replace NDVI proxy logic with direct calibrated satellite raster workflows
