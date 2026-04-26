from __future__ import annotations

import hashlib
import logging
import os
import pickle
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
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
CACHE_DIR = Path(__file__).resolve().parents[1] / "data_cache"
LOG_DIR = Path(__file__).resolve().parents[1] / "run_logs"
SIM_LOGGER_LOCK = threading.Lock()
MAX_SELECTION_ROWS = 5000
MAX_MAP_BURN_POINTS = 60000
MAX_MAP_RISK_POINTS = 60000
MAX_MAP_TERRAIN_POINTS = 40000
MAX_MAP_FIRE_POINTS = 20000
MAX_MAP_LINE_FEATURES = 4000
MAX_MAP_BUILDING_POINTS = 15000
MAX_TABLE_ROWS = 20000


def _get_sim_logger() -> logging.Logger:
    logger = logging.getLogger("wildfire_simulation")
    with SIM_LOGGER_LOCK:
        logger.setLevel(logging.INFO)
        logger.propagate = False
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        formatter = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
        target_file = str((LOG_DIR / "simulation.log").resolve())

        has_file_handler = False
        has_stream_handler = False
        duplicate_handlers: list[logging.Handler] = []
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler) and str(Path(handler.baseFilename).resolve()) == target_file:
                if not has_file_handler:
                    has_file_handler = True
                else:
                    duplicate_handlers.append(handler)
            elif isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                if not has_stream_handler:
                    has_stream_handler = True
                else:
                    duplicate_handlers.append(handler)

        for handler in duplicate_handlers:
            logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass

        if not has_file_handler:
            file_handler = logging.FileHandler(LOG_DIR / "simulation.log", encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        if not has_stream_handler:
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)
    return logger


def _cache_file(prefix: str, key_parts: dict[str, Any]) -> Path:
    payload = repr(sorted(key_parts.items())).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{prefix}_{digest}.pkl"


def _cache_load(prefix: str, key_parts: dict[str, Any], ttl_seconds: int) -> Any | None:
    path = _cache_file(prefix, key_parts)
    if not path.exists():
        return None
    age = time.time() - path.stat().st_mtime
    if age > ttl_seconds:
        return None
    try:
        with path.open("rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _cache_save(prefix: str, key_parts: dict[str, Any], value: Any) -> Any:
    path = _cache_file(prefix, key_parts)
    with path.open("wb") as f:
        pickle.dump(value, f)
    return value


def _downsample_df(df: pd.DataFrame, max_rows: int, sort_by: str | None = None) -> pd.DataFrame:
    if df is None or df.empty or len(df) <= max_rows:
        return df
    out = df
    if sort_by and sort_by in out.columns:
        out = out.sort_values(sort_by, ascending=False)
        return out.head(max_rows).reset_index(drop=True)
    return out.sample(n=max_rows, random_state=42).reset_index(drop=True)


@st.cache_data(ttl=60 * 20, show_spinner=False)
def fetch_fires(map_key: str, bbox_tuple: tuple[float, float, float, float], datasets: List[str], day_range: int) -> pd.DataFrame:
    key = {
        "map_key": map_key,
        "bbox": tuple(float(x) for x in bbox_tuple),
        "datasets": tuple(datasets),
        "day_range": int(day_range),
    }
    cached = _cache_load("fires", key, ttl_seconds=60 * 20)
    if cached is not None:
        return cached
    value = FirmsClient(map_key=map_key).fetch_many(bbox=BBox(*bbox_tuple), datasets=datasets, day_range=day_range)
    return _cache_save("fires", key, value)


@st.cache_data(ttl=60 * 30, show_spinner=False)
def fetch_weather(lat: float, lon: float):
    key = {"lat": round(float(lat), 5), "lon": round(float(lon), 5), "hours": 24}
    cached = _cache_load("weather", key, ttl_seconds=60 * 30)
    if cached is not None:
        return cached
    wx = OpenMeteoClient()
    value = (wx.fetch_current(lat, lon), wx.fetch_hourly(lat, lon, hours=24))
    return _cache_save("weather", key, value)


@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_terrain(bbox_tuple: tuple[float, float, float, float], grid_size: int):
    key = {"bbox": tuple(float(x) for x in bbox_tuple), "grid_size": int(grid_size)}
    cached = _cache_load("terrain", key, ttl_seconds=60 * 60)
    if cached is not None:
        return cached
    value = OpenTopoDataClient().fetch_grid(BBox(*bbox_tuple), grid_size=grid_size)
    return _cache_save("terrain", key, value)


def fallback_terrain(bbox_tuple: tuple[float, float, float, float], grid_size: int):
    return OpenTopoDataClient.fallback_grid(BBox(*bbox_tuple), grid_size=grid_size)


@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_ndvi(bbox_tuple: tuple[float, float, float, float], grid_size: int):
    key = {"bbox": tuple(float(x) for x in bbox_tuple), "grid_size": int(grid_size)}
    cached = _cache_load("ndvi", key, ttl_seconds=60 * 60)
    if cached is not None:
        return cached
    value = VegScapeClient().fetch_grid(BBox(*bbox_tuple), grid_size=grid_size)
    return _cache_save("ndvi", key, value)


def fallback_ndvi(grid_size: int):
    return VegScapeClient.fallback_grid(grid_size=grid_size)


@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_infrastructure(bbox_tuple: tuple[float, float, float, float]):
    key = {"bbox": tuple(float(x) for x in bbox_tuple)}
    cached = _cache_load("infrastructure", key, ttl_seconds=60 * 60)
    if cached is not None:
        return cached
    value = OverpassClient().fetch(BBox(*bbox_tuple))
    return _cache_save("infrastructure", key, value)


def _simulation_worker(
    context: dict[str, Any],
    progress_state: dict[str, Any],
    stop_event: threading.Event,
) -> None:
    logger = _get_sim_logger()
    sim_id = context.get("sim_id", "unknown")
    try:
        progress_state["status"] = "Preparing AI spread model..."
        logger.info("sim_id=%s stage=prepare grid_size=%s runs_per_area=%s steps=%s", sim_id, context["grid_size"], context["sim_runs"], context["sim_steps"])
        predictor = SpreadPredictor(
            SpreadConfig(
                grid_size=context["grid_size"],
                steps=context["sim_steps"],
                runs=context["sim_runs"],
            )
        )
        logger.info("sim_id=%s stage=model_ready", sim_id)

        jobs = _build_local_fire_jobs(context["fires"], half_span_deg=0.05)
        total_jobs = len(jobs)
        progress_state["done"] = 0
        progress_state["total"] = total_jobs
        progress_state["status"] = "Running area-by-area simulations..."
        logger.info("sim_id=%s stage=run total_areas=%s", sim_id, total_jobs)

        burn_pts_all: list[pd.DataFrame] = []
        risk_pts_all: list[pd.DataFrame] = []
        terrain_pts_all: list[pd.DataFrame] = []
        burn_means: list[float] = []
        burn_maxes: list[float] = []
        risk_means: list[float] = []
        risk_maxes: list[float] = []
        completed_areas = 0
        stopped_early = False

        for area_idx, job in enumerate(jobs, start=1):
            if stop_event.is_set():
                stopped_early = True
                break

            progress_state["status"] = f"Simulating area {area_idx}/{total_jobs}..."
            last_logged_local_run = {"value": 0}

            def on_local_progress(done_runs: int, total_runs: int) -> None:
                # Surface partial progress within each local area so UI doesn't stay at 0.
                partial = (done_runs / max(total_runs, 1))
                progress_state["done"] = (area_idx - 1) + partial
                progress_state["total"] = total_jobs
                log_every_local = max(1, total_runs // 10)
                if done_runs == 1 or done_runs == total_runs or done_runs - last_logged_local_run["value"] >= log_every_local:
                    last_logged_local_run["value"] = done_runs
                    logger.info(
                        "sim_id=%s area=%s/%s local_runs=%s/%s",
                        sim_id,
                        area_idx,
                        total_jobs,
                        done_runs,
                        total_runs,
                    )

            local_result = predictor.simulate(
                fires=job["fires"],
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
                progress_callback=on_local_progress,
                should_stop=stop_event.is_set,
            )
            if local_result.get("simulation", {}).get("stopped_early"):
                stopped_early = True

            local_burn_pts = raster_to_points(local_result["burn_probability"], local_result["extent"], "burn_probability", threshold=0.10, stride=1)
            if not local_burn_pts.empty:
                local_burn_pts["area_index"] = area_idx
                burn_pts_all.append(local_burn_pts)

            local_risk_pts = raster_to_points(
                local_result["risk_score"],
                local_result["extent"],
                "risk_score",
                threshold=context["risk_threshold"],
                stride=1,
            )
            if not local_risk_pts.empty:
                local_risk_pts["area_index"] = area_idx
                risk_pts_all.append(local_risk_pts)

            if context["terrain"] is not None:
                local_terrain_pts = raster_to_points(
                    local_result["elevation"],
                    local_result["extent"],
                    "elevation_m",
                    threshold=0.0,
                    stride=max(1, context["grid_size"] // 24),
                )
                slope_pts = raster_to_points(
                    local_result["slope"],
                    local_result["extent"],
                    "slope_deg",
                    threshold=0.0,
                    stride=max(1, context["grid_size"] // 24),
                )
                if not slope_pts.empty:
                    local_terrain_pts = local_terrain_pts.merge(slope_pts, on=["latitude", "longitude"], how="left")
                if not local_terrain_pts.empty:
                    local_terrain_pts["area_index"] = area_idx
                    terrain_pts_all.append(local_terrain_pts)

            burn_means.append(float(local_result["burn_probability"].mean()))
            burn_maxes.append(float(local_result["burn_probability"].max()))
            risk_means.append(float(local_result["risk_score"].mean()))
            risk_maxes.append(float(local_result["risk_score"].max()))

            completed_areas = area_idx
            progress_state["done"] = completed_areas
            log_every = max(1, total_jobs // 20)
            if area_idx == 1 or area_idx == total_jobs or area_idx % log_every == 0:
                logger.info("sim_id=%s area_progress=%s/%s", sim_id, area_idx, total_jobs)

            if stop_event.is_set():
                stopped_early = True
                break

        if completed_areas == 0:
            raise RuntimeError("AI spread simulation stopped before completing any local-area simulations.")

        result = {
            "burn_pts": pd.concat(burn_pts_all, ignore_index=True) if burn_pts_all else pd.DataFrame(),
            "risk_pts": pd.concat(risk_pts_all, ignore_index=True) if risk_pts_all else pd.DataFrame(),
            "terrain_pts": pd.concat(terrain_pts_all, ignore_index=True) if terrain_pts_all else pd.DataFrame(),
            "summary": {
                "burn_prob_mean": float(np.mean(burn_means)),
                "burn_prob_max": float(np.max(burn_maxes)),
                "risk_mean": float(np.mean(risk_means)),
                "risk_max": float(np.max(risk_maxes)),
            },
            "simulation": {
                "requested_areas": total_jobs,
                "completed_areas": completed_areas,
                "stopped_early": stopped_early,
                "requested_runs_per_area": context["sim_runs"],
            },
        }

        progress_state["result"] = result
        progress_state["status"] = "Simulation complete."
        sim_meta = result["simulation"]
        logger.info(
            "sim_id=%s stage=complete completed_areas=%s requested_areas=%s stopped_early=%s",
            sim_id,
            sim_meta.get("completed_areas"),
            sim_meta.get("requested_areas"),
            sim_meta.get("stopped_early"),
        )
    except Exception as exc:
        progress_state["error"] = str(exc)
        logger.exception("sim_id=%s stage=error message=%s", sim_id, exc)


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


def _build_local_fire_jobs(fires: pd.DataFrame, half_span_deg: float = 0.05) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    base = fires.reset_index(drop=True)
    for idx, row in base.iterrows():
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        lat_min, lat_max = lat - half_span_deg, lat + half_span_deg
        lon_min, lon_max = lon - half_span_deg, lon + half_span_deg
        local = base[
            base["latitude"].between(lat_min, lat_max)
            & base["longitude"].between(lon_min, lon_max)
        ].copy()
        if local.empty:
            local = pd.DataFrame([row])
        jobs.append(
            {
                "index": idx + 1,
                "center_lat": lat,
                "center_lon": lon,
                "fires": local.reset_index(drop=True),
            }
        )
    return jobs


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
                cell_size=3000,
                extruded=False,
                pickable=True,
                opacity=0.16,
                get_fill_color="[120 + slope_deg*2, 95 + slope_deg*1.3, 70, 70]",
            )
        )

    if not risk_pts.empty:
        layers.append(
            pdk.Layer(
                "HeatmapLayer",
                data=risk_pts,
                get_position='[longitude, latitude]',
                get_weight="risk_score",
                radiusPixels=38,
                intensity=0.9,
                threshold=0.06,
                opacity=0.42,
            )
        )

    if not roads.empty:
        layers.append(
            pdk.Layer(
                "PathLayer",
                data=roads,
                get_path="path",
                get_width=4,
                get_color="[110, 110, 110, 150]",
                pickable=True,
            )
        )

    if not power_lines.empty:
        layers.append(
            pdk.Layer(
                "PathLayer",
                data=power_lines,
                get_path="path",
                get_width=3,
                get_color="[52, 92, 155, 170]",
                pickable=True,
            )
        )

    if not buildings.empty:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=buildings,
                get_position='[longitude, latitude]',
                get_radius=90,
                get_fill_color='[90, 90, 90, 55]',
                pickable=True,
            )
        )

    if not burn_pts.empty:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=burn_pts,
                get_position='[longitude, latitude]',
                get_radius="1000 + burn_probability * 3500",
                get_fill_color="[255, 110, 20, 85]",
                pickable=True,
            )
        )

    if not fires.empty:
        fires = fires.copy()
        frp_val = fires["frp"].fillna(10).astype(float)
        fires["frp_radius"] = np.clip(fires["frp"].fillna(10).astype(float) * 70, 1800, 22000)
        fires["fire_r"] = 255
        fires["fire_g"] = np.clip(220 - frp_val, 40, 220).astype(int)
        fires["fire_b"] = 25
        fires["fire_a"] = 210
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=fires,
                get_position='[longitude, latitude]',
                get_radius="frp_radius",
                get_fill_color="[fire_r, fire_g, fire_b, fire_a]",
                get_line_color="[35, 20, 0, 200]",
                stroked=True,
                line_width_min_pixels=1,
                line_width_max_pixels=2,
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
        map_style="https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json",
    )
    st.pydeck_chart(deck, use_container_width=True)


def main() -> None:
    st.title("Real-Time Wildfire Dashboard with Terrain, NDVI, and Infrastructure")
    st.caption("Live FIRMS hotspots + real terrain + NDVI/fuel moisture proxy + roads/power/buildings + AI spread forecast")
    sim_progress_slot = st.empty()
    sim_status_slot = st.empty()

    st.session_state.setdefault("sim_thread", None)
    st.session_state.setdefault("sim_stop_event", None)
    st.session_state.setdefault("sim_progress", {"done": 0, "total": 0, "result": None, "error": None})
    st.session_state.setdefault("sim_context", None)
    st.session_state.setdefault("sim_running", False)
    st.session_state.setdefault("sim_launching", False)
    st.session_state.setdefault("pending_selection", None)

    if st.session_state.sim_running:
        thread = st.session_state.sim_thread
        if thread is None or not thread.is_alive():
            st.session_state.sim_running = False

    # Render progress/status early on every rerun to avoid visible flicker.
    progress_state = st.session_state.sim_progress
    done_areas = float(progress_state.get("done", 0.0))
    total_areas = int(progress_state.get("total", 0) or 0)
    if st.session_state.sim_running:
        safe_total = max(1, total_areas)
        progress_pct = min(1.0, done_areas / safe_total)
        sim_progress_slot.progress(progress_pct, text=f"AI spread simulation progress: {done_areas:.1f}/{safe_total} local areas")
        status_text = "Stopping simulation..." if (
            st.session_state.sim_stop_event is not None and st.session_state.sim_stop_event.is_set()
        ) else str(progress_state.get("status") or "AI spread simulation is running...")
        sim_status_slot.caption(status_text)
    elif st.session_state.sim_launching:
        sim_progress_slot.progress(0.0, text="AI spread simulation progress: 0.0/1 local areas")
        sim_status_slot.caption("Preparing simulation inputs...")
    elif progress_state.get("result") is not None:
        safe_total = max(1, total_areas)
        shown_done = done_areas if done_areas > 0 else float(safe_total)
        progress_pct = min(1.0, shown_done / safe_total)
        sim_progress_slot.progress(progress_pct, text=f"AI spread simulation progress: {shown_done:.1f}/{safe_total} local areas")
        sim_status_slot.caption(str(progress_state.get("status") or "Simulation complete."))

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
        busy = st.session_state.sim_running or st.session_state.sim_launching or (st.session_state.pending_selection is not None)
        run_sidebar_button = st.button("Run dashboard", type="primary", disabled=busy)
        stop_sim_button = st.button("Stop AI spread simulation", disabled=not (st.session_state.sim_running or st.session_state.sim_launching))

    if stop_sim_button and st.session_state.sim_running and st.session_state.sim_stop_event is not None:
        st.session_state.sim_stop_event.set()
        _get_sim_logger().info("sim_id=%s stage=stop_requested", st.session_state.get("sim_id", "unknown"))
        st.warning("Stop requested. Finishing current Monte Carlo iteration...")

    run_main_button = False
    if not run_sidebar_button and not (st.session_state.sim_running or st.session_state.sim_launching):
        st.caption("Enter a FIRMS MAP_KEY, adjust the region, then click Run dashboard below or in the sidebar.")
        run_main_button = st.button("Run dashboard", type="primary")

    if run_sidebar_button or run_main_button:
        if st.session_state.sim_running or st.session_state.sim_launching:
            st.warning("A simulation is already in progress.")
            return
        st.session_state.sim_launching = True
        try:
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

            st.session_state.pending_selection = {
                "request_id": int(time.time() * 1000),
                "bbox": bbox,
                "center_lat": center_lat,
                "center_lon": center_lon,
                "fires": fires.reset_index(drop=True),
                "risk_threshold": risk_threshold,
                "grid_size": grid_size,
                "sim_steps": sim_steps,
                "sim_runs": sim_runs,
                "load_terrain": load_terrain,
                "load_ndvi": load_ndvi,
                "load_infra": load_infra,
            }
            st.info("FIRMS detections fetched. Select detections below, then click Start simulation for selected detections.")
        finally:
            st.session_state.sim_launching = False

    pending = st.session_state.pending_selection
    if pending is not None and not st.session_state.sim_running and not st.session_state.sim_launching:
        st.subheader("Select FIRMS Detections To Simulate")
        fires_all = pending["fires"].copy().reset_index(drop=True)
        if len(fires_all) > MAX_SELECTION_ROWS:
            fires_all = _downsample_df(fires_all, MAX_SELECTION_ROWS, sort_by="frp")
            st.warning(f"Showing top {MAX_SELECTION_ROWS:,} detections by FRP for selection.")
        display = fires_all.copy()
        display["simulate"] = True
        if "timestamp_utc" in display.columns:
            display["timestamp_utc"] = display["timestamp_utc"].astype(str)
        cols = [
            "simulate",
            "latitude",
            "longitude",
            "frp",
            "brightness",
            "satellite",
            "instrument",
            "confidence",
            "daynight",
            "dataset",
            "timestamp_utc",
        ]
        cols = [c for c in cols if c in display.columns]
        selector_key = f"detection_selector_{pending['request_id']}"
        edited = st.data_editor(
            display[cols],
            use_container_width=True,
            hide_index=True,
            key=selector_key,
            disabled=[c for c in cols if c != "simulate"],
        )
        selected_count = int(edited["simulate"].fillna(False).sum()) if "simulate" in edited.columns else 0
        st.caption(f"Selected detections: {selected_count}/{len(display)}")
        c_start, c_cancel = st.columns(2)
        start_selected = c_start.button("Start simulation for selected detections", type="primary")
        cancel_selected = c_cancel.button("Cancel selection")

        if cancel_selected:
            st.session_state.pending_selection = None
            if selector_key in st.session_state:
                del st.session_state[selector_key]
            st.rerun()
            return

        if start_selected:
            selected_mask = edited["simulate"].fillna(False).to_numpy() if "simulate" in edited.columns else np.array([], dtype=bool)
            fires = fires_all.loc[selected_mask].reset_index(drop=True) if selected_mask.size else pd.DataFrame()
            if fires.empty:
                st.error("Please select at least one detection.")
                return

            st.session_state.sim_launching = True
            try:
                bbox = pending["bbox"]
                center_lat = pending["center_lat"]
                center_lon = pending["center_lon"]
                grid_size = int(pending["grid_size"])
                sim_steps = int(pending["sim_steps"])
                sim_runs = int(pending["sim_runs"])
                risk_threshold = float(pending["risk_threshold"])

                with st.spinner("Fetching weather..."):
                    current_weather, hourly_weather = fetch_weather(center_lat, center_lon)

                terrain = ndvi = infra = None
                if bool(pending["load_terrain"]):
                    try:
                        with st.spinner("Fetching terrain..."):
                            terrain = fetch_terrain(bbox, grid_size)
                    except Exception:
                        terrain = fallback_terrain(bbox, grid_size)
                        st.warning("Terrain layer unavailable from OpenTopoData (rate limit or service issue). Using offline fallback terrain for this run.")
                if bool(pending["load_ndvi"]):
                    try:
                        with st.spinner("Fetching NDVI / fuel moisture..."):
                            ndvi = fetch_ndvi(bbox, grid_size)
                    except Exception:
                        ndvi = fallback_ndvi(grid_size)
                        st.warning("NDVI layer unavailable from USDA VegScape. Using offline fallback NDVI proxy for this run.")
                if bool(pending["load_infra"]):
                    try:
                        with st.spinner("Fetching infrastructure..."):
                            infra = fetch_infrastructure(bbox)
                    except Exception as exc:
                        st.warning(f"Infrastructure layer unavailable: {exc}")

                context: dict[str, Any] = {
                    "sim_id": int(time.time() * 1000),
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
                progress_state: dict[str, Any] = {
                    "done": 0,
                    "total": len(fires),
                    "result": None,
                    "error": None,
                    "status": "Queued...",
                }
                thread = threading.Thread(
                    target=_simulation_worker,
                    args=(context, progress_state, stop_event),
                    daemon=True,
                )
                thread.start()
                st.session_state.sim_id = context["sim_id"]
                _get_sim_logger().info("sim_id=%s stage=thread_started", context["sim_id"])
                st.session_state.sim_thread = thread
                st.session_state.sim_stop_event = stop_event
                st.session_state.sim_progress = progress_state
                st.session_state.sim_context = context
                st.session_state.sim_running = True
                st.session_state.pending_selection = None
                if selector_key in st.session_state:
                    del st.session_state[selector_key]
            finally:
                st.session_state.sim_launching = False
        return

    if st.session_state.sim_launching:
        return

    if st.session_state.sim_running:
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
    burn_pts = _downsample_df(result.get("burn_pts", pd.DataFrame()), MAX_MAP_BURN_POINTS, sort_by="burn_probability")
    risk_pts = _downsample_df(result.get("risk_pts", pd.DataFrame()), MAX_MAP_RISK_POINTS, sort_by="risk_score")
    terrain_pts = _downsample_df(result.get("terrain_pts", pd.DataFrame()), MAX_MAP_TERRAIN_POINTS, sort_by="slope_deg")

    roads = _downsample_df(infra.roads if infra is not None else pd.DataFrame(), MAX_MAP_LINE_FEATURES)
    power_lines = _downsample_df(infra.power_lines if infra is not None else pd.DataFrame(), MAX_MAP_LINE_FEATURES)
    buildings = _downsample_df(infra.buildings if infra is not None else pd.DataFrame(), MAX_MAP_BUILDING_POINTS)
    fires_map = _downsample_df(fires, MAX_MAP_FIRE_POINTS, sort_by="frp")
    sim_meta = result.get("simulation", {})
    if sim_meta.get("stopped_early"):
        st.warning(
            "AI spread simulation was stopped early. "
            f"Completed {sim_meta.get('completed_areas', 0)} of {sim_meta.get('requested_areas', 0)} local areas."
        )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Detections", f"{len(fires):,}")
    c2.metric("Max FRP", f"{fires['frp'].fillna(0).max():.1f}")
    c3.metric("Wind", f"{current_weather.wind_speed_kmh:.1f} km/h")
    c4.metric("Humidity", f"{current_weather.humidity_pct:.0f}%")
    c5.metric("Buildings", f"{len(buildings):,}" if not buildings.empty else "0")

    build_map(fires_map, burn_pts, risk_pts, terrain_pts, roads, power_lines, buildings, center_lat, center_lon)

    tab1, tab2, tab3, tab4 = st.tabs(["Active fires", "Weather", "Model output", "Layers"])

    with tab1:
        show_cols = ["timestamp_utc", "satellite", "instrument", "latitude", "longitude", "frp", "brightness", "confidence", "daynight", "dataset"]
        fires_table = fires.copy()
        if "timestamp_utc" in fires_table.columns:
            fires_table = fires_table.sort_values("timestamp_utc", ascending=False)
        if len(fires_table) > MAX_TABLE_ROWS:
            st.caption(f"Showing {MAX_TABLE_ROWS:,} of {len(fires):,} detections.")
            fires_table = fires_table.head(MAX_TABLE_ROWS)
        st.dataframe(fires_table[show_cols], use_container_width=True, height=400)

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
        summary = result.get("summary", {})
        st.write(
            {
                "burn_prob_mean": float(summary.get("burn_prob_mean", 0.0)),
                "burn_prob_max": float(summary.get("burn_prob_max", 0.0)),
                "risk_mean": float(summary.get("risk_mean", 0.0)),
                "risk_max": float(summary.get("risk_max", 0.0)),
                "terrain_enabled": terrain is not None,
                "ndvi_enabled": ndvi is not None,
                "infrastructure_enabled": infra is not None,
                "requested_areas": int(sim_meta.get("requested_areas", 0)),
                "completed_areas": int(sim_meta.get("completed_areas", 0)),
                "requested_runs_per_area": int(sim_meta.get("requested_runs_per_area", 0)),
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
