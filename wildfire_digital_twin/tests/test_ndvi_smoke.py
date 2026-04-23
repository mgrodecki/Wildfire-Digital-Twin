from __future__ import annotations

import numpy as np

from services.ndvi import VegScapeClient


def test_fallback_ndvi_grid_shape_and_bounds() -> None:
    result = VegScapeClient.fallback_grid(grid_size=16)
    assert result.ndvi.shape == (16, 16)
    assert result.fuel_moisture.shape == (16, 16)
    assert np.all(result.ndvi >= 0.0)
    assert np.all(result.ndvi <= 1.0)
    assert np.all(result.fuel_moisture >= 0.0)
    assert np.all(result.fuel_moisture <= 1.0)
