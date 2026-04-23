from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from utils.geo import normalize_array, points_to_density_raster, resample_to_shape


@dataclass
class SpreadConfig:
    grid_size: int = 96
    steps: int = 18
    runs: int = 120
    seed: int = 42


class SpreadPredictor:
    def __init__(self, config: SpreadConfig | None = None) -> None:
        self.config = config or SpreadConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.model = self._fit_surrogate()

    def _fit_surrogate(self) -> RandomForestRegressor:
        n = 9000
        rng = self.rng
        fuel = rng.uniform(0.05, 1.0, n)
        dryness = rng.uniform(0.0, 1.0, n)
        slope = rng.uniform(0.0, 45.0, n)
        wind_speed = rng.uniform(0.0, 80.0, n)
        alignment = rng.uniform(-1.0, 1.0, n)
        frp = rng.uniform(0.0, 500.0, n)
        hotspot_density = rng.uniform(0.0, 1.0, n)
        structure_density = rng.uniform(0.0, 1.0, n)
        road_density = rng.uniform(0.0, 1.0, n)
        power_density = rng.uniform(0.0, 1.0, n)

        y = (
            0.02
            + 0.28 * fuel
            + 0.24 * dryness
            + 0.10 * (slope / 45.0)
            + 0.16 * (wind_speed / 80.0) * (0.5 + 0.5 * alignment)
            + 0.08 * np.tanh(frp / 120.0)
            + 0.11 * hotspot_density
            + 0.05 * structure_density
            + 0.03 * road_density
            + 0.03 * power_density
        )
        y += rng.normal(0, 0.025, n)
        y = np.clip(y, 0.0, 0.98)

        X = pd.DataFrame(
            {
                "fuel": fuel,
                "dryness": dryness,
                "slope": slope,
                "wind_speed": wind_speed,
                "alignment": alignment,
                "frp": frp,
                "hotspot_density": hotspot_density,
                "structure_density": structure_density,
                "road_density": road_density,
                "power_density": power_density,
            }
        )
        model = RandomForestRegressor(
            n_estimators=180,
            max_depth=14,
            min_samples_leaf=3,
            random_state=self.config.seed,
            n_jobs=-1,
        )
        model.fit(X, y)
        return model

    def simulate(
        self,
        fires: pd.DataFrame,
        wind_speed_kmh: float,
        wind_dir_deg: float,
        humidity_pct: float,
        terrain_slope: np.ndarray | None = None,
        elevation: np.ndarray | None = None,
        ndvi: np.ndarray | None = None,
        fuel_moisture: np.ndarray | None = None,
        infrastructure: dict[str, pd.DataFrame] | None = None,
        grid_size: int | None = None,
        steps: int | None = None,
        runs: int | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> dict:
        grid_size = grid_size or self.config.grid_size
        steps = steps or self.config.steps
        runs = runs or self.config.runs
        if fires.empty:
            raise ValueError("fires dataframe is empty")

        lat_min, lat_max = fires["latitude"].min(), fires["latitude"].max()
        lon_min, lon_max = fires["longitude"].min(), fires["longitude"].max()
        lat_pad = max((lat_max - lat_min) * 0.05, 0.05)
        lon_pad = max((lon_max - lon_min) * 0.05, 0.05)
        lat_min, lat_max = lat_min - lat_pad, lat_max + lat_pad
        lon_min, lon_max = lon_min - lon_pad, lon_max + lon_pad
        extent = {"lat_min": lat_min, "lat_max": lat_max, "lon_min": lon_min, "lon_max": lon_max}

        fuel, dryness, slope = self._make_environment(
            grid_size=grid_size,
            humidity_pct=humidity_pct,
            terrain_slope=terrain_slope,
            ndvi=ndvi,
            fuel_moisture=fuel_moisture,
        )
        seed_fire, frp_grid, density_grid = self._seed_from_detections(fires, grid_size, extent)

        infrastructure = infrastructure or {}
        building_density = points_to_density_raster(infrastructure.get("buildings", pd.DataFrame()), extent, (grid_size, grid_size), sigma_cells=2)
        road_density = points_to_density_raster(infrastructure.get("roads", pd.DataFrame()), extent, (grid_size, grid_size), sigma_cells=2)
        power_density = points_to_density_raster(infrastructure.get("power_lines", pd.DataFrame()), extent, (grid_size, grid_size), sigma_cells=2)

        burn_sum = np.zeros((grid_size, grid_size), dtype=float)
        completed_runs = 0
        stopped_early = False
        for run_idx in range(runs):
            if should_stop is not None and should_stop():
                stopped_early = True
                break
            burn_sum += self._run_once(
                seed_fire=seed_fire,
                fuel=fuel,
                dryness=dryness,
                slope=slope,
                frp_grid=frp_grid,
                density_grid=density_grid,
                building_density=building_density,
                road_density=road_density,
                power_density=power_density,
                wind_speed_kmh=wind_speed_kmh,
                wind_dir_deg=wind_dir_deg,
                steps=steps,
            )
            completed_runs = run_idx + 1
            if progress_callback is not None:
                progress_callback(completed_runs, runs)

        if completed_runs == 0:
            raise RuntimeError("AI spread simulation stopped before completing any Monte Carlo runs.")

        burn_probability = burn_sum / completed_runs
        elevation_r = resample_to_shape(elevation, (grid_size, grid_size)) if elevation is not None else np.zeros((grid_size, grid_size))
        elev_norm = normalize_array(elevation_r)
        infrastructure_exposure = np.clip(0.5 * building_density + 0.25 * road_density + 0.25 * power_density, 0.0, 1.0)
        risk = np.clip(
            100.0
            * (
                0.42 * burn_probability
                + 0.16 * dryness
                + 0.12 * fuel
                + 0.10 * normalize_array(slope)
                + 0.10 * infrastructure_exposure
                + 0.06 * np.tanh(frp_grid / 180.0)
                + 0.04 * elev_norm
            ),
            0,
            100,
        )

        return {
            "burn_probability": burn_probability,
            "risk_score": risk,
            "extent": extent,
            "seed_fire": seed_fire,
            "fuel": fuel,
            "dryness": dryness,
            "slope": slope,
            "elevation": elevation_r,
            "building_density": building_density,
            "road_density": road_density,
            "power_density": power_density,
            "infrastructure_exposure": infrastructure_exposure,
            "simulation": {
                "requested_runs": runs,
                "completed_runs": completed_runs,
                "stopped_early": stopped_early,
            },
        }

    def _make_environment(
        self,
        grid_size: int,
        humidity_pct: float,
        terrain_slope: np.ndarray | None,
        ndvi: np.ndarray | None,
        fuel_moisture: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.linspace(0, 1, grid_size)
        y = np.linspace(0, 1, grid_size)
        xx, yy = np.meshgrid(x, y, indexing="ij")
        base_fuel = 0.25 + 0.7 * (0.65 * np.sin(xx * 3.2) ** 2 + 0.35 * np.cos(yy * 4.4) ** 2)
        base_dryness = np.clip(1.0 - humidity_pct / 100.0, 0.05, 0.95)
        dryness = np.clip(base_dryness + 0.25 * np.sin((xx + yy) * 5.5), 0.05, 1.0)
        slope = 45 * np.abs(np.gradient(base_fuel)[0])

        if ndvi is not None:
            ndvi_r = np.clip(resample_to_shape(ndvi, (grid_size, grid_size)), 0.0, 1.0)
            fuel = np.clip(0.35 * base_fuel + 0.65 * ndvi_r, 0.0, 1.0)
        else:
            fuel = np.clip(base_fuel, 0.0, 1.0)

        if fuel_moisture is not None:
            fm_r = np.clip(resample_to_shape(fuel_moisture, (grid_size, grid_size)), 0.0, 1.0)
            dryness = np.clip(0.45 * dryness + 0.55 * (1.0 - fm_r), 0.0, 1.0)

        if terrain_slope is not None:
            slope = np.clip(resample_to_shape(terrain_slope, (grid_size, grid_size)), 0.0, 45.0)

        return fuel, dryness, slope

    def _seed_from_detections(self, fires: pd.DataFrame, grid_size: int, extent: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        seed_fire = np.zeros((grid_size, grid_size), dtype=float)
        frp_grid = np.zeros((grid_size, grid_size), dtype=float)
        count_grid = np.zeros((grid_size, grid_size), dtype=float)
        lat_min, lat_max = extent["lat_min"], extent["lat_max"]
        lon_min, lon_max = extent["lon_min"], extent["lon_max"]
        lat_span = max(lat_max - lat_min, 1e-6)
        lon_span = max(lon_max - lon_min, 1e-6)

        for _, row in fires.iterrows():
            i = int((row["latitude"] - lat_min) / lat_span * (grid_size - 1))
            j = int((row["longitude"] - lon_min) / lon_span * (grid_size - 1))
            i = min(max(i, 0), grid_size - 1)
            j = min(max(j, 0), grid_size - 1)
            seed_fire[i, j] = 1.0
            frp = float(row.get("frp", 0.0) or 0.0)
            frp_grid[i, j] = max(frp_grid[i, j], frp)
            count_grid[i, j] += 1.0

        density_grid = count_grid.copy()
        for i in range(grid_size):
            for j in range(grid_size):
                i0, i1 = max(0, i - 2), min(grid_size, i + 3)
                j0, j1 = max(0, j - 2), min(grid_size, j + 3)
                density_grid[i, j] = count_grid[i0:i1, j0:j1].sum()
        density_grid = normalize_array(density_grid)
        return seed_fire, frp_grid, density_grid

    def _run_once(
        self,
        seed_fire: np.ndarray,
        fuel: np.ndarray,
        dryness: np.ndarray,
        slope: np.ndarray,
        frp_grid: np.ndarray,
        density_grid: np.ndarray,
        building_density: np.ndarray,
        road_density: np.ndarray,
        power_density: np.ndarray,
        wind_speed_kmh: float,
        wind_dir_deg: float,
        steps: int,
    ) -> np.ndarray:
        grid_size = seed_fire.shape[0]
        state = seed_fire.copy()
        burned = seed_fire.copy()
        wind_vector = self._wind_unit_vector(wind_dir_deg)
        neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        for _ in range(steps):
            new_state = state.copy()
            burning_cells = np.argwhere(state > 0.5)
            if burning_cells.size == 0:
                break
            for i, j in burning_cells:
                for di, dj in neighbors:
                    ni, nj = i + di, j + dj
                    if ni < 0 or nj < 0 or ni >= grid_size or nj >= grid_size or burned[ni, nj] > 0.5:
                        continue
                    direction = np.array([di, dj], dtype=float)
                    direction /= (np.linalg.norm(direction) or 1.0)
                    alignment = float(np.dot(direction, wind_vector))
                    features = pd.DataFrame(
                        {
                            "fuel": [fuel[ni, nj]],
                            "dryness": [dryness[ni, nj]],
                            "slope": [slope[ni, nj]],
                            "wind_speed": [wind_speed_kmh],
                            "alignment": [alignment],
                            "frp": [frp_grid[i, j]],
                            "hotspot_density": [density_grid[ni, nj]],
                            "structure_density": [building_density[ni, nj]],
                            "road_density": [road_density[ni, nj]],
                            "power_density": [power_density[ni, nj]],
                        }
                    )
                    p = float(self.model.predict(features)[0])
                    if self.rng.random() < p:
                        new_state[ni, nj] = 1.0
                        burned[ni, nj] = 1.0
                new_state[i, j] = 0.0
            state = new_state
        return burned

    @staticmethod
    def _wind_unit_vector(wind_dir_deg: float) -> np.ndarray:
        rad = np.deg2rad((270 - wind_dir_deg) % 360)
        return np.array([np.sin(rad), np.cos(rad)])
