from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Iterable

import numpy as np
import requests

from services.firms import BBox


@dataclass(frozen=True)
class TerrainResult:
    elevation_m: np.ndarray
    slope_deg: np.ndarray


class OpenTopoDataClient:
    """Fetches real elevation samples from OpenTopoData and derives slope.

    Default dataset is `mapzen`, which is globally available in OpenTopoData.
    The API accepts batched `locations=lat,lon|lat,lon|...` queries.
    """

    BASE_URL = "https://api.opentopodata.org/v1"
    RETRYABLE_STATUS = {429, 502, 503, 504}

    def __init__(self, dataset: str = "mapzen", timeout: int = 60) -> None:
        self.dataset = dataset
        self.timeout = timeout

    @staticmethod
    def fallback_grid(bbox: BBox, grid_size: int = 40) -> TerrainResult:
        lat_vals = np.linspace(bbox.south, bbox.north, grid_size)
        lon_vals = np.linspace(bbox.west, bbox.east, grid_size)
        yy, xx = np.meshgrid(lat_vals, lon_vals, indexing="ij")
        # Offline-safe smooth synthetic terrain field.
        elev = (
            1300.0
            + 450.0 * np.sin((xx - bbox.west) * np.pi / max(bbox.east - bbox.west, 1e-6))
            + 300.0 * np.cos((yy - bbox.south) * np.pi / max(bbox.north - bbox.south, 1e-6))
        )
        center_lat = (bbox.south + bbox.north) / 2.0
        lat_m = max((bbox.north - bbox.south) * 111_320.0 / max(grid_size - 1, 1), 1.0)
        lon_m = max((bbox.east - bbox.west) * 111_320.0 * np.cos(np.deg2rad(center_lat)) / max(grid_size - 1, 1), 1.0)
        dz_dy, dz_dx = np.gradient(elev, lat_m, lon_m)
        slope = np.degrees(np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2)))
        slope = np.nan_to_num(slope, nan=0.0, posinf=45.0, neginf=0.0)
        return TerrainResult(elevation_m=elev.astype(float), slope_deg=slope.astype(float))

    def fetch_grid(self, bbox: BBox, grid_size: int = 40) -> TerrainResult:
        lat_vals = np.linspace(bbox.south, bbox.north, grid_size)
        lon_vals = np.linspace(bbox.west, bbox.east, grid_size)
        locations: list[tuple[float, float]] = [(float(lat), float(lon)) for lat in lat_vals for lon in lon_vals]
        elevations = self._fetch_locations(locations)
        elev = np.array(elevations, dtype=float).reshape(grid_size, grid_size)

        # Approximate grid spacing in meters for slope derivation.
        center_lat = (bbox.south + bbox.north) / 2.0
        lat_m = max((bbox.north - bbox.south) * 111_320.0 / max(grid_size - 1, 1), 1.0)
        lon_m = max((bbox.east - bbox.west) * 111_320.0 * np.cos(np.deg2rad(center_lat)) / max(grid_size - 1, 1), 1.0)
        dz_dy, dz_dx = np.gradient(elev, lat_m, lon_m)
        slope = np.degrees(np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2)))
        slope = np.nan_to_num(slope, nan=0.0, posinf=45.0, neginf=0.0)
        return TerrainResult(elevation_m=elev, slope_deg=slope)

    def _fetch_locations(self, locations: Iterable[tuple[float, float]], batch_size: int = 100) -> list[float]:
        pairs = list(locations)
        results: list[float] = []
        for i in range(0, len(pairs), batch_size):
            chunk = pairs[i : i + batch_size]
            loc_str = "|".join(f"{lat:.6f},{lon:.6f}" for lat, lon in chunk)
            response = self._request_with_retry(loc_str)
            payload = response.json()
            for item in payload.get("results", []):
                elevation = item.get("elevation")
                results.append(float(elevation) if elevation is not None else np.nan)
        return results

    def _request_with_retry(self, loc_str: str, max_attempts: int = 4) -> requests.Response:
        url = f"{self.BASE_URL}/{self.dataset}"
        last_error: requests.HTTPError | None = None
        for attempt in range(1, max_attempts + 1):
            response = requests.get(
                url,
                params={"locations": loc_str},
                timeout=self.timeout,
            )
            try:
                response.raise_for_status()
                return response
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else None
                last_error = exc
                if status not in self.RETRYABLE_STATUS or attempt == max_attempts:
                    raise
                retry_after = response.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    wait_s = max(1, int(retry_after))
                else:
                    wait_s = min(8, 2 ** (attempt - 1))
                time.sleep(wait_s)
        if last_error is not None:
            raise last_error
        raise RuntimeError("OpenTopoData request failed without an HTTP error.")
