# Wildfire AI Dashboard Prototype — Terrain, NDVI, and Infrastructure Edition

This laptop-friendly Streamlit app combines:

- live NASA FIRMS active fire detections
- open weather data from Open-Meteo
- real elevation-derived terrain from OpenTopoData
- a real NDVI/fuel-moisture proxy layer from USDA VegScape NDVI WMS
- infrastructure overlays from OpenStreetMap Overpass
- a lightweight AI-assisted Monte Carlo spread model

## New layers added

### Terrain
- Fetches elevation samples for the selected bounding box from OpenTopoData.
- Derives slope in degrees from the sampled DEM grid.
- Feeds slope and elevation into the risk model and map display.

### NDVI / fuel moisture
- Pulls a recent NDVI visualization tile from USDA VegScape WMS for the bbox.
- Converts relative greenness to a normalized NDVI-like raster.
- Derives a fuel-moisture proxy used in spread simulation.

### Infrastructure
- Pulls roads, overhead power lines, and building footprints from OpenStreetMap using Overpass API.
- Converts them into map overlays and infrastructure exposure rasters.
- Adds exposure weighting into the wildfire risk score.

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
requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

Set your NASA FIRMS key first.

Linux/macOS:

```bash
export FIRMS_MAP_KEY="your_key_here"
```

Windows PowerShell:

```powershell
$env:FIRMS_MAP_KEY="your_key_here"
```

## Important notes

- This is still a research/demo prototype, not an operational fire-behavior model.
- The NDVI layer is implemented as a lightweight greenness proxy from a live WMS image so the app stays runnable on a laptop.
- For production use, replace the NDVI module with direct MODIS/Sentinel raster ingestion and use a calibrated dead/live fuel moisture workflow.
- Overpass requests can be slow for very large bounding boxes.
