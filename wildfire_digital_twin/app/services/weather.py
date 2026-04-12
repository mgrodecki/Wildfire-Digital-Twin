from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import requests


@dataclass(frozen=True)
class WeatherSnapshot:
    latitude: float
    longitude: float
    time: str
    temperature_c: float
    humidity_pct: float
    wind_speed_kmh: float
    wind_direction_deg: float
    precip_mm: float


class OpenMeteoClient:
    BASE_URL = "https://api.open-meteo.com/v1/forecast"

    def __init__(self, timeout: int = 30) -> None:
        self.timeout = timeout

    def fetch_current(self, latitude: float, longitude: float) -> WeatherSnapshot:
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": ",".join([
                "temperature_2m",
                "relative_humidity_2m",
                "wind_speed_10m",
                "wind_direction_10m",
                "precipitation",
            ]),
            "timezone": "UTC",
            "wind_speed_unit": "kmh",
        }
        response = requests.get(self.BASE_URL, params=params, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        current = payload["current"]
        return WeatherSnapshot(
            latitude=latitude,
            longitude=longitude,
            time=current["time"],
            temperature_c=float(current.get("temperature_2m", 0.0)),
            humidity_pct=float(current.get("relative_humidity_2m", 0.0)),
            wind_speed_kmh=float(current.get("wind_speed_10m", 0.0)),
            wind_direction_deg=float(current.get("wind_direction_10m", 0.0)),
            precip_mm=float(current.get("precipitation", 0.0)),
        )

    def fetch_hourly(self, latitude: float, longitude: float, hours: int = 24) -> pd.DataFrame:
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": ",".join([
                "temperature_2m",
                "relative_humidity_2m",
                "wind_speed_10m",
                "wind_direction_10m",
                "precipitation_probability",
                "precipitation",
            ]),
            "forecast_hours": hours,
            "timezone": "UTC",
            "wind_speed_unit": "kmh",
        }
        response = requests.get(self.BASE_URL, params=params, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        hourly = payload["hourly"]
        return pd.DataFrame(hourly)
