from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, List

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

from models.spread import SpreadConfig, SpreadPredictor
from services.firms import BBox, DEFAULT_DATASETS, FirmsClient
from services.infrastructure import OverpassClient
from services.ndvi import VegScapeClient
from services.terrain import OpenTopoDataClient
from services.weather import OpenMeteoClient
from utils.geo import raster_to_points

st.set_page_config(page_title="Wildfire AI Dashboard", layout="wide")

DEFAULT_BBOX = BBox(west=-125.0, south=32.0, east=-113.0, north=43.0)


@st.cache_data(ttl=60 * 20, show_spinner=False)
def fetch_fires(map_key: str, bbox_tuple: tuple[float, float, float, float], datasets: List[str], day_range: int) -> pd.DataFrame:
    return FirmsClient(map_key=map_key).fetch_many(bbox=BBox(*bbox_tuple), datasets=datasets, day_range=day_range)


@st.cache_data(ttl=60 * 30, show_spinner=False)
def fetch_weather(lat: float, lon: float):
    wx = OpenMeteoClient()
    return wx.fetch_current(lat, lon), wx.fetch_hourly(lat, lon, hours=24)


@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_terrain(bbox_tuple: tuple[float, float, float, float], grid_size: int):
    return OpenTopoDataClient().fetch_grid(BBox(*bbox_tuple), grid_size=grid_size)


@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_ndvi(bbox_tuple: tuple[float, float, float, float], grid_size: int):
    return VegScapeClient().fetch_grid(BBox(*bbox_tuple), grid_size=grid_size)


def fallback_ndvi(grid_size: int):
    return VegScapeClient.fallback_grid(grid_size=grid_size)


@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_infrastructure(bbox_tuple: tuple[float, float, float, float]):
    return OverpassClient().fetch(BBox(*bbox_tuple))


def _simulation_worker(
    context: dict[str, Any],
    progress_state: dict[str, Any],
    stop_event: threading.Event,
) -> None:
    try:
        predictor = SpreadPredictor(
            SpreadConfig(
                grid_size=context["grid_size"],
                steps=context["sim_steps"],
                runs=context["sim_runs"],
            )
        )

        def on_progress(done_runs: int, total_runs: int) -> None:
            progress_state["done"] = done_runs
            progress_state["total"] = total_runs

        result = predictor.simulate(
            fires=context["fires"],
            wind_speed_kmh=context["current_weather"].wind_speed_kmh,
            wind_dir_deg=context["current_weather"].wind_direction_deg,
            humidity_pct=context["current_weather"].humidity_pct,
            terrain_slope=(context["terrain"].slope_deg if context["terrain"] else None),
            elevation=(context["terrain"].elevation_m if context["terrain"] else None),
            ndvi=(context["ndvi"].ndvi if context["ndvi"] else None),
            fuel_moisture=(context["ndvi"].fuel_moisture if context["ndvi"] else None),
            infrastructure={
                "roads": context["infra"].roads if context["infra"] is not None else pd.DataFrame(),
                "power_lines": context["infra"].power_lines if context["infra"] is not None else pd.DataFrame(),
                "buildings": context["infra"].buildings if context["infra"] is not None else pd.DataFrame(),
            },
            grid_size=context["grid_size"],
            steps=context["sim_steps"],
            runs=context["sim_runs"],
            progress_callback=on_progress,
            should_stop=stop_event.is_set,
        )
        progress_state["result"] = result
    except Exception as exc:
        progress_state["error"] = str(exc)


def make_demo_fires(bbox: BBox, count: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    lats = rng.uniform(bbox.south, bbox.north, count)
    lons = rng.uniform(bbox.west, bbox.east, count)
    frp = rng.uniform(8.0, 65.0, count)
    now = pd.Timestamp(datetime.now(timezone.utc)).floor("min")
    acq_date = now.strftime("%Y-%m-%d")
    acq_time = now.strftime("%H%M")
    return pd.DataFrame(
        {
            "latitude": lats,
            "longitude": lons,
            "brightness": rng.uniform(300.0, 380.0, count),
            "bright_ti5": rng.uniform(260.0, 320.0, count),
            "frp": frp,
            "scan": np.full(count, 0.5),
            "track": np.full(count, 0.5),
            "acq_date": [acq_date] * count,
            "acq_time": [acq_time] * count,
            "timestamp_utc": [now] * count,
            "satellite": ["DEMO"] * count,
            "instrument": ["SIM"] * count,
            "confidence": ["demo"] * count,
            "version": ["demo"] * count,
            "daynight": ["D"] * count,
            "dataset": ["DEMO_SYNTHETIC"] * count,
        }
    )


def build_map(
    fires: pd.DataFrame,
    burn_pts: pd.DataFrame,
    risk_pts: pd.DataFrame,
    terrain_pts: pd.DataFrame,
    roads: pd.DataFrame,
    power_lines: pd.DataFrame,
    buildings: pd.DataFrame,
    center_lat: float,
    center_lon: float,
):
    layers = []

    if not terrain_pts.empty:
        layers.append(
            pdk.Layer(
                "GridCellLayer",
                data=terrain_pts,
                get_position="[longitude, latitude]",
                get_elevation="elevation_m",
                elevation_scale=1,
                cell_size=4000,
                extruded=False,
                pickable=True,
                opacity=0.2,
                get_fill_color="[80 + slope_deg*3, 120, 180, 80]",
            )
        )

    if not risk_pts.empty:
        layers.append(
            pdk.Layer(
                "HeatmapLayer",
                data=risk_pts,
                get_position='[longitude, latitude]',
                get_weight="risk_score",
                radiusPixels=45,
                intensity=1,
                threshold=0.08,
                opacity=0.5,
            )
        )

    if not roads.empty:
        layers.append(
            pdk.Layer(
                "PathLayer",
                data=roads,
                get_path="path",
                get_width=6,
                get_color="[120, 200, 255, 110]",
                pickable=True,
            )
        )

    if not power_lines.empty:
        layers.append(
            pdk.Layer(
                "PathLayer",
                data=power_lines,
                get_path="path",
                get_width=5,
                get_color="[255, 210, 0, 180]",
                pickable=True,
            )
        )

    if not buildings.empty:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=buildings,
                get_position='[longitude, latitude]',
                get_radius=120,
                get_fill_color='[255,255,255,70]',
                pickable=True,
            )
        )

    if not burn_pts.empty:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=burn_pts,
                get_position='[longitude, latitude]',
                get_radius="burn_probability * 5000",
                get_fill_color="[255, 120, 0, 140]",
                pickable=True,
            )
        )

    if not fires.empty:
        fires = fires.copy()
        fires["frp_radius"] = np.clip(fires["frp"].fillna(10).astype(float) * 80, 1500, 25000)
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=fires,
                get_position='[longitude, latitude]',
                get_radius="frp_radius",
                get_fill_color="[255, 30, 0, 180]",
                pickable=True,
            )
        )

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=6, pitch=35),
        tooltip={
            "html": (
                "<b>Lat:</b> {latitude}<br/>"
                "<b>Lon:</b> {longitude}<br/>"
                "<b>Burn prob:</b> {burn_probability}<br/>"
                "<b>Risk:</b> {risk_score}<br/>"
                "<b>Elevation:</b> {elevation_m}<br/>"
                "<b>Slope:</b> {slope_deg}<br/>"
                "<b>FRP:</b> {frp}<br/>"
                "<b>Satellite:</b> {satellite}<br/>"
                "<b>Time UTC:</b> {timestamp_utc}<br/>"
                "<b>Name:</b> {name}"
            )
        },
        map_style="mapbox://styles/mapbox/dark-v10",
    )
    st.pydeck_chart(deck, use_container_width=True)


def main() -> None:
    st.title("Real-Time Wildfire Dashboard with Terrain, NDVI, and Infrastructure")
    st.caption("Live FIRMS hotspots + real terrain + NDVI/fuel moisture proxy + roads/power/buildings + AI spread forecast")

    st.session_state.setdefault("sim_thread", None)
    st.session_state.setdefault("sim_stop_event", None)
    st.session_state.setdefault("sim_progress", {"done": 0, "total": 0, "result": None, "error": None})
    st.session_state.setdefault("sim_context", None)
    st.session_state.setdefault("sim_running", False)

    if st.session_state.sim_running:
        thread = st.session_state.sim_thread
        if thread is None or not thread.is_alive():
            st.session_state.sim_running = False

    with st.sidebar:
        st.header("Controls")
        map_key = st.text_input("NASA FIRMS MAP_KEY", value=os.getenv("FIRMS_MAP_KEY", ""), type="password")
        west = st.number_input("West", value=float(DEFAULT_BBOX.west), format="%.4f")
        south = st.number_input("South", value=float(DEFAULT_BBOX.south), format="%.4f")
        east = st.number_input("East", value=float(DEFAULT_BBOX.east), format="%.4f")
        north = st.number_input("North", value=float(DEFAULT_BBOX.north), format="%.4f")
        day_range = st.slider("Days of FIRMS detections", min_value=1, max_value=10, value=1)
        datasets = st.multiselect("Datasets", options=DEFAULT_DATASETS, default=DEFAULT_DATASETS[:3])
        grid_size = st.slider("Environment grid size", min_value=32, max_value=128, value=64, step=16)
        sim_steps = st.slider("Simulation steps", min_value=6, max_value=36, value=18)
        sim_runs = st.slider("Monte Carlo runs", min_value=20, max_value=200, value=100)
        risk_threshold = st.slider("Risk overlay threshold", min_value=0.0, max_value=100.0, value=30.0)
        load_terrain = st.toggle("Load real terrain", value=True)
        load_ndvi = st.toggle("Load NDVI / fuel moisture", value=True)
        load_infra = st.toggle("Load roads / power / buildings", value=True)
        demo_fires_on_empty = st.toggle("Use demo hotspots when FIRMS is empty", value=True)
        run_sidebar_button = st.button("Run dashboard", type="primary", disabled=st.session_state.sim_running)
        stop_sim_button = st.button("Stop AI spread simulation", disabled=not st.session_state.sim_running)

    if stop_sim_button and st.session_state.sim_running and st.session_state.sim_stop_event is not None:
        st.session_state.sim_stop_event.set()
        st.warning("Stop requested. Finishing current Monte Carlo iteration...")

    run_main_button = False
    if not run_sidebar_button and not st.session_state.sim_running:
        st.caption("Enter a FIRMS MAP_KEY, adjust the region, then click Run dashboard below or in the sidebar.")
        run_main_button = st.button("Run dashboard", type="primary")

    if run_sidebar_button or run_main_button:
        if not map_key:
            st.error("A NASA FIRMS MAP_KEY is required.")
            return

        bbox = (west, south, east, north)
        bbox_obj = BBox(*bbox)
        center_lat, center_lon = bbox_obj.center()

        try:
            with st.spinner("Fetching live fire detections..."):
                fires = fetch_fires(map_key, bbox, datasets, day_range)
        except Exception as exc:
            st.error(f"FIRMS fetch failed: {exc}")
            st.caption(
                "Verify your MAP_KEY, selected datasets, and service availability. "
                "If this persists, compare against the USFS FIRMS API endpoint for the same bbox/day range."
            )
            return
        if fires.empty:
            if demo_fires_on_empty:
                fires = make_demo_fires(bbox_obj, count=10)
                st.warning("No active fires returned for the selected area/time window. Using demo hotspots for simulation.")
            else:
                st.warning("No active fires returned for the selected area/time window.")
                return

        with st.spinner("Fetching weather..."):
            current_weather, hourly_weather = fetch_weather(center_lat, center_lon)

        terrain = ndvi = infra = None
        if load_terrain:
            try:
                with st.spinner("Fetching terrain..."):
                    terrain = fetch_terrain(bbox, grid_size)
            except Exception as exc:
                st.warning(f"Terrain layer unavailable: {exc}")
        if load_ndvi:
            try:
                with st.spinner("Fetching NDVI / fuel moisture..."):
                    ndvi = fetch_ndvi(bbox, grid_size)
            except Exception:
                ndvi = fallback_ndvi(grid_size)
                st.warning("NDVI layer unavailable from USDA VegScape. Using offline fallback NDVI proxy for this run.")
        if load_infra:
            try:
                with st.spinner("Fetching infrastructure..."):
                    infra = fetch_infrastructure(bbox)
            except Exception as exc:
                st.warning(f"Infrastructure layer unavailable: {exc}")

        context: dict[str, Any] = {
            "fires": fires,
            "current_weather": current_weather,
            "hourly_weather": hourly_weather,
            "terrain": terrain,
            "ndvi": ndvi,
            "infra": infra,
            "center_lat": center_lat,
            "center_lon": center_lon,
            "risk_threshold": risk_threshold,
            "grid_size": grid_size,
            "sim_steps": sim_steps,
            "sim_runs": sim_runs,
        }

        stop_event = threading.Event()
        progress_state: dict[str, Any] = {"done": 0, "total": sim_runs, "result": None, "error": None}
        thread = threading.Thread(
            target=_simulation_worker,
            args=(context, progress_state, stop_event),
            daemon=True,
        )
        thread.start()
        st.session_state.sim_thread = thread
        st.session_state.sim_stop_event = stop_event
        st.session_state.sim_progress = progress_state
        st.session_state.sim_context = context
        st.session_state.sim_running = True

    if st.session_state.sim_running:
        progress_state = st.session_state.sim_progress
        done_runs = int(progress_state.get("done", 0))
        total_runs = int(progress_state.get("total", 0) or 0)
        progress_pct = min(1.0, done_runs / total_runs) if total_runs > 0 else 0.0
        st.progress(progress_pct, text=f"AI spread simulation progress: {done_runs}/{total_runs} runs")
        status_text = "Stopping simulation..." if (
            st.session_state.sim_stop_event is not None and st.session_state.sim_stop_event.is_set()
        ) else "AI spread simulation is running..."
        st.caption(status_text)
        time.sleep(0.5)
        st.rerun()
        return

    progress_state = st.session_state.sim_progress
    context = st.session_state.sim_context
    if progress_state.get("error"):
        st.error(f"AI spread simulation failed: {progress_state['error']}")
        return
    if context is None or progress_state.get("result") is None:
        return

    result = progress_state["result"]
    fires = context["fires"]
    current_weather = context["current_weather"]
    hourly_weather = context["hourly_weather"]
    terrain = context["terrain"]
    ndvi = context["ndvi"]
    infra = context["infra"]
    center_lat = context["center_lat"]
    center_lon = context["center_lon"]
    risk_threshold = context["risk_threshold"]
    grid_size = context["grid_size"]

    burn_pts = raster_to_points(result["burn_probability"], result["extent"], "burn_probability", threshold=0.10, stride=1)
    risk_pts = raster_to_points(result["risk_score"], result["extent"], "risk_score", threshold=risk_threshold, stride=1)
    terrain_pts = pd.DataFrame()
    if terrain is not None:
        terrain_pts = raster_to_points(result["elevation"], result["extent"], "elevation_m", threshold=0.0, stride=max(1, grid_size // 24))
        slope_pts = raster_to_points(result["slope"], result["extent"], "slope_deg", threshold=0.0, stride=max(1, grid_size // 24))
        if not slope_pts.empty:
            terrain_pts = terrain_pts.merge(slope_pts, on=["latitude", "longitude"], how="left")

    roads = infra.roads if infra is not None else pd.DataFrame()
    power_lines = infra.power_lines if infra is not None else pd.DataFrame()
    buildings = infra.buildings if infra is not None else pd.DataFrame()
    sim_meta = result.get("simulation", {})
    if sim_meta.get("stopped_early"):
        st.warning(
            "AI spread simulation was stopped early. "
            f"Completed {sim_meta.get('completed_runs', 0)} of {sim_meta.get('requested_runs', 0)} runs."
        )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Detections", f"{len(fires):,}")
    c2.metric("Max FRP", f"{fires['frp'].fillna(0).max():.1f}")
    c3.metric("Wind", f"{current_weather.wind_speed_kmh:.1f} km/h")
    c4.metric("Humidity", f"{current_weather.humidity_pct:.0f}%")
    c5.metric("Buildings", f"{len(buildings):,}" if not buildings.empty else "0")

    build_map(fires, burn_pts, risk_pts, terrain_pts, roads, power_lines, buildings, center_lat, center_lon)

    tab1, tab2, tab3, tab4 = st.tabs(["Active fires", "Weather", "Model output", "Layers"])

    with tab1:
        show_cols = ["timestamp_utc", "satellite", "instrument", "latitude", "longitude", "frp", "brightness", "confidence", "daynight", "dataset"]
        st.dataframe(fires[show_cols], use_container_width=True, height=400)

    with tab2:
        st.write(
            {
                "time_utc": current_weather.time,
                "temperature_c": current_weather.temperature_c,
                "humidity_pct": current_weather.humidity_pct,
                "wind_speed_kmh": current_weather.wind_speed_kmh,
                "wind_direction_deg": current_weather.wind_direction_deg,
                "precip_mm": current_weather.precip_mm,
            }
        )
        st.line_chart(hourly_weather.set_index("time")[["temperature_2m", "relative_humidity_2m", "wind_speed_10m"]], use_container_width=True)

    with tab3:
        burn_prob = result["burn_probability"]
        risk = result["risk_score"]
        st.write(
            {
                "burn_prob_mean": float(burn_prob.mean()),
                "burn_prob_max": float(burn_prob.max()),
                "risk_mean": float(risk.mean()),
                "risk_max": float(risk.max()),
                "terrain_enabled": terrain is not None,
                "ndvi_enabled": ndvi is not None,
                "infrastructure_enabled": infra is not None,
                "requested_runs": int(sim_meta.get("requested_runs", 0)),
                "completed_runs": int(sim_meta.get("completed_runs", 0)),
                "stopped_early": bool(sim_meta.get("stopped_early", False)),
            }
        )
        st.caption("Research/demo prototype only. Real incident support requires calibrated fuels, terrain, weather downscaling, and operational QA.")

    with tab4:
        if terrain is not None:
            st.write({"terrain_grid_shape": terrain.elevation_m.shape, "max_elevation_m": float(np.nanmax(terrain.elevation_m))})
        if ndvi is not None:
            st.write({"ndvi_source": ndvi.source, "mean_ndvi_like": float(np.nanmean(ndvi.ndvi)), "mean_fuel_moisture": float(np.nanmean(ndvi.fuel_moisture))})
        st.write({"roads": len(roads), "power_lines": len(power_lines), "buildings": len(buildings)})
        if not buildings.empty:
            st.dataframe(buildings.head(200), use_container_width=True)


if __name__ == "__main__":
    main()
