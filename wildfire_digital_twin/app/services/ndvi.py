from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from io import BytesIO

import numpy as np
import requests
from PIL import Image

from services.firms import BBox


@dataclass(frozen=True)
class NdviResult:
    ndvi: np.ndarray
    fuel_moisture: np.ndarray
    source: str


class VegScapeClient:
    """Fetches recent NDVI thumbnails from USDA VegScape and derives fuel-moisture proxy.

    VegScape publishes NDVI products for the contiguous U.S. as WMS layers. This client
    pulls a small PNG for the current viewport and converts the relative greenness signal
    into a normalized NDVI-like layer for lightweight dashboard use.
    """

    BASE_URL = "https://gis1.sc.egov.usda.gov/arcgis/services/fgdc/NDVI_ConUS/MapServer/WMSServer"

    def __init__(self, timeout: int = 60) -> None:
        self.timeout = timeout

    @staticmethod
    def fallback_grid(grid_size: int = 128) -> NdviResult:
        # Offline-safe NDVI proxy so simulation can proceed when WMS is unavailable.
        x = np.linspace(0.0, 1.0, grid_size, dtype=float)
        y = np.linspace(0.0, 1.0, grid_size, dtype=float)
        xx, yy = np.meshgrid(x, y, indexing="ij")
        ndvi_like = np.clip(0.25 + 0.55 * (0.6 * np.sin(xx * 4.0) ** 2 + 0.4 * np.cos(yy * 3.0) ** 2), 0.0, 1.0)
        fuel_moisture = np.clip(0.15 + 0.75 * ndvi_like, 0.0, 1.0)
        return NdviResult(ndvi=ndvi_like, fuel_moisture=fuel_moisture, source="Fallback synthetic NDVI (offline)")

    def fetch_grid(self, bbox: BBox, grid_size: int = 128) -> NdviResult:
        time_hint = (datetime.now(timezone.utc) - timedelta(days=16)).date().isoformat()
        params = {
            "service": "WMS",
            "request": "GetMap",
            "version": "1.3.0",
            "layers": "0",
            "styles": "",
            "crs": "EPSG:4326",
            "bbox": f"{bbox.south},{bbox.west},{bbox.north},{bbox.east}",
            "width": grid_size,
            "height": grid_size,
            "format": "image/png",
            "transparent": "false",
            "time": time_hint,
        }
        response = requests.get(self.BASE_URL, params=params, timeout=self.timeout)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        arr = np.asarray(image).astype(float)
        # Use relative greenness from the WMS colorized tile as a lightweight proxy.
        green = arr[..., 1]
        red = arr[..., 0]
        blue = arr[..., 2]
        ndvi_like = np.clip((green - 0.5 * red - 0.2 * blue) / 255.0, 0.0, 1.0)
        fuel_moisture = np.clip(0.15 + 0.75 * ndvi_like, 0.0, 1.0)
        return NdviResult(ndvi=ndvi_like, fuel_moisture=fuel_moisture, source="USDA VegScape NDVI WMS")
