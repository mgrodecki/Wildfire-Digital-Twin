from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests

from services.firms import BBox


@dataclass(frozen=True)
class InfrastructureBundle:
    roads: pd.DataFrame
    power_lines: pd.DataFrame
    buildings: pd.DataFrame


class OverpassClient:
    BASE_URL = "https://overpass-api.de/api/interpreter"

    def __init__(self, timeout: int = 90) -> None:
        self.timeout = timeout

    def fetch(self, bbox: BBox) -> InfrastructureBundle:
        south, west, north, east = bbox.south, bbox.west, bbox.north, bbox.east
        query = f"""
        [out:json][timeout:60];
        (
          way["highway"]({south},{west},{north},{east});
          way["power"="line"]({south},{west},{north},{east});
          way["building"]({south},{west},{north},{east});
        );
        out geom tags;
        """
        response = requests.post(self.BASE_URL, data=query.encode("utf-8"), timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        elements = payload.get("elements", [])

        roads = self._paths_to_df(elements, feature_type="road", tag_key="highway")
        power = self._paths_to_df(elements, feature_type="power", tag_key="power", tag_value="line")
        buildings = self._buildings_to_df(elements)
        return InfrastructureBundle(roads=roads, power_lines=power, buildings=buildings)

    @staticmethod
    def _paths_to_df(elements: list[dict[str, Any]], feature_type: str, tag_key: str, tag_value: str | None = None) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for el in elements:
            if el.get("type") != "way":
                continue
            tags = el.get("tags", {})
            if tag_key not in tags:
                continue
            if tag_value is not None and tags.get(tag_key) != tag_value:
                continue
            geom = el.get("geometry") or []
            if len(geom) < 2:
                continue
            path = [[pt["lon"], pt["lat"]] for pt in geom if "lat" in pt and "lon" in pt]
            if len(path) < 2:
                continue
            rows.append(
                {
                    "feature_type": feature_type,
                    "name": tags.get("name", tags.get(tag_key, feature_type)),
                    "path": path,
                    "latitude": sum(pt[1] for pt in path) / len(path),
                    "longitude": sum(pt[0] for pt in path) / len(path),
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def _buildings_to_df(elements: list[dict[str, Any]]) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for el in elements:
            if el.get("type") != "way":
                continue
            tags = el.get("tags", {})
            if "building" not in tags:
                continue
            geom = el.get("geometry") or []
            if not geom:
                continue
            coords = [(pt.get("lon"), pt.get("lat")) for pt in geom if "lat" in pt and "lon" in pt]
            coords = [(lon, lat) for lon, lat in coords if lon is not None and lat is not None]
            if not coords:
                continue
            rows.append(
                {
                    "name": tags.get("name", "building"),
                    "latitude": sum(lat for _, lat in coords) / len(coords),
                    "longitude": sum(lon for lon, _ in coords) / len(coords),
                }
            )
        return pd.DataFrame(rows)
