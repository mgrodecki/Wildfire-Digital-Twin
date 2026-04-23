from __future__ import annotations

import numpy as np
import pandas as pd

from models.spread import SpreadConfig, SpreadPredictor


def test_spread_simulate_smoke() -> None:
    fires = pd.DataFrame(
        [
            {"latitude": 35.0, "longitude": -120.0, "frp": 22.0},
            {"latitude": 35.05, "longitude": -119.95, "frp": 15.0},
        ]
    )
    predictor = SpreadPredictor(config=SpreadConfig(grid_size=12, steps=2, runs=2, seed=7))
    result = predictor.simulate(
        fires=fires,
        wind_speed_kmh=20.0,
        wind_dir_deg=270.0,
        humidity_pct=25.0,
        terrain_slope=np.full((12, 12), 8.0),
        elevation=np.linspace(50, 400, 144, dtype=float).reshape(12, 12),
    )
    assert result["burn_probability"].shape == (12, 12)
    assert result["risk_score"].shape == (12, 12)
    assert np.all(result["risk_score"] >= 0.0)
    assert np.all(result["risk_score"] <= 100.0)


def test_spread_requires_nonempty_fires() -> None:
    predictor = SpreadPredictor(config=SpreadConfig(grid_size=8, steps=1, runs=1, seed=1))
    empty = pd.DataFrame(columns=["latitude", "longitude", "frp"])
    try:
        predictor.simulate(
            fires=empty,
            wind_speed_kmh=5.0,
            wind_dir_deg=180.0,
            humidity_pct=40.0,
        )
    except ValueError as exc:
        assert "empty" in str(exc).lower()
    else:
        raise AssertionError("Expected simulate() to raise ValueError for empty fires input")


def test_spread_supports_progress_and_early_stop() -> None:
    fires = pd.DataFrame(
        [
            {"latitude": 35.0, "longitude": -120.0, "frp": 22.0},
            {"latitude": 35.05, "longitude": -119.95, "frp": 15.0},
        ]
    )
    predictor = SpreadPredictor(config=SpreadConfig(grid_size=10, steps=2, runs=6, seed=5))
    progress: list[int] = []
    stop_after = {"value": False}

    def on_progress(done: int, total: int) -> None:
        progress.append(done)
        if done >= 2:
            stop_after["value"] = True

    result = predictor.simulate(
        fires=fires,
        wind_speed_kmh=15.0,
        wind_dir_deg=250.0,
        humidity_pct=35.0,
        grid_size=10,
        steps=2,
        runs=6,
        progress_callback=on_progress,
        should_stop=lambda: stop_after["value"],
    )

    assert progress
    sim_meta = result["simulation"]
    assert sim_meta["completed_runs"] >= 2
    assert sim_meta["completed_runs"] <= 6
    assert sim_meta["stopped_early"] is True
