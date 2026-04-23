from __future__ import annotations

import numpy as np
import pandas as pd

from utils.geo import normalize_array, points_to_density_raster, raster_to_points, resample_to_shape


def test_normalize_resample_and_raster_roundtrip() -> None:
    arr = np.array([[0.0, 10.0], [20.0, 30.0]])
    normalized = normalize_array(arr)
    assert normalized.shape == (2, 2)
    assert float(normalized.min()) == 0.0
    assert float(normalized.max()) == 1.0

    resampled = resample_to_shape(arr, (4, 4))
    assert resampled.shape == (4, 4)

    extent = {"lat_min": 30.0, "lat_max": 31.0, "lon_min": -120.0, "lon_max": -119.0}
    points = raster_to_points(normalized, extent, value_name="risk", threshold=0.5)
    assert isinstance(points, pd.DataFrame)
    assert not points.empty
    assert {"latitude", "longitude", "risk"}.issubset(points.columns)


def test_points_to_density_raster_empty_and_nonempty() -> None:
    extent = {"lat_min": 30.0, "lat_max": 31.0, "lon_min": -120.0, "lon_max": -119.0}
    shape = (8, 8)

    empty = points_to_density_raster(pd.DataFrame(), extent, shape)
    assert empty.shape == shape
    assert float(empty.sum()) == 0.0

    points = pd.DataFrame(
        [
            {"latitude": 30.2, "longitude": -119.8},
            {"latitude": 30.21, "longitude": -119.79},
        ]
    )
    density = points_to_density_raster(points, extent, shape)
    assert density.shape == shape
    assert float(density.max()) <= 1.0
