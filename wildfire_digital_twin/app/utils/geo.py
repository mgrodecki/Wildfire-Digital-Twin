from __future__ import annotations

import numpy as np
import pandas as pd


def raster_to_points(
    raster: np.ndarray,
    extent: dict,
    value_name: str,
    threshold: float = 0.0,
    stride: int = 1,
) -> pd.DataFrame:
    nrows, ncols = raster.shape
    lat_vals = np.linspace(extent["lat_min"], extent["lat_max"], nrows)
    lon_vals = np.linspace(extent["lon_min"], extent["lon_max"], ncols)

    rows = []
    for i in range(0, nrows, stride):
        for j in range(0, ncols, stride):
            value = float(raster[i, j])
            if value < threshold:
                continue
            rows.append(
                {
                    "latitude": float(lat_vals[i]),
                    "longitude": float(lon_vals[j]),
                    value_name: value,
                }
            )
    return pd.DataFrame(rows)


def normalize_array(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=float)
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if hi - lo < 1e-9:
        return np.zeros_like(arr, dtype=float)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def resample_to_shape(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    src_rows, src_cols = arr.shape
    dst_rows, dst_cols = shape
    row_idx = np.linspace(0, src_rows - 1, dst_rows).astype(int)
    col_idx = np.linspace(0, src_cols - 1, dst_cols).astype(int)
    return arr[np.ix_(row_idx, col_idx)]


def points_to_density_raster(
    points_df: pd.DataFrame,
    extent: dict,
    shape: tuple[int, int],
    sigma_cells: int = 2,
) -> np.ndarray:
    if points_df.empty:
        return np.zeros(shape, dtype=float)

    rows, cols = shape
    out = np.zeros(shape, dtype=float)
    lat_min, lat_max = extent["lat_min"], extent["lat_max" ]
    lon_min, lon_max = extent["lon_min"], extent["lon_max" ]
    lat_span = max(lat_max - lat_min, 1e-9)
    lon_span = max(lon_max - lon_min, 1e-9)

    for _, row in points_df.iterrows():
        i = int((row["latitude"] - lat_min) / lat_span * (rows - 1))
        j = int((row["longitude"] - lon_min) / lon_span * (cols - 1))
        i = min(max(i, 0), rows - 1)
        j = min(max(j, 0), cols - 1)
        out[i, j] += 1.0

    if sigma_cells <= 0:
        return normalize_array(out)

    kernel_radius = int(max(1, sigma_cells * 2))
    smoothed = np.zeros_like(out)
    for i in range(rows):
        i0, i1 = max(0, i - kernel_radius), min(rows, i + kernel_radius + 1)
        for j in range(cols):
            j0, j1 = max(0, j - kernel_radius), min(cols, j + kernel_radius + 1)
            smoothed[i, j] = out[i0:i1, j0:j1].sum()
    return normalize_array(smoothed)
