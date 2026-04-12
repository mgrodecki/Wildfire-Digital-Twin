from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from typing import Iterable, Optional

import pandas as pd
import requests

FIRMS_BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
DEFAULT_DATASETS = [
    "VIIRS_NOAA21_NRT",
    "VIIRS_NOAA20_NRT",
    "VIIRS_SNPP_NRT",
    "MODIS_NRT",
]


@dataclass(frozen=True)
class BBox:
    west: float
    south: float
    east: float
    north: float

    def as_firms_area(self) -> str:
        return f"{self.west},{self.south},{self.east},{self.north}"

    def center(self) -> tuple[float, float]:
        return ((self.south + self.north) / 2.0, (self.west + self.east) / 2.0)


class FirmsClient:
    def __init__(self, map_key: str, timeout: int = 60) -> None:
        self.map_key = map_key
        self.timeout = timeout

    def fetch_area(
        self,
        dataset: str,
        bbox: BBox,
        day_range: int = 1,
        date: Optional[str] = None,
    ) -> pd.DataFrame:
        url = f"{FIRMS_BASE}/{self.map_key}/{dataset}/{bbox.as_firms_area()}/{day_range}"
        if date:
            url = f"{url}/{date}"
        response = requests.get(url, timeout=self.timeout)
        response.raise_for_status()
        text = response.text.strip()
        if not text:
            return pd.DataFrame()
        df = pd.read_csv(StringIO(text))
        if df.empty:
            return df
        df["dataset"] = dataset
        return self._normalize(df)

    def fetch_many(
        self,
        bbox: BBox,
        datasets: Optional[Iterable[str]] = None,
        day_range: int = 1,
    ) -> pd.DataFrame:
        datasets = list(datasets or DEFAULT_DATASETS)
        frames: list[pd.DataFrame] = []
        for ds in datasets:
            try:
                df = self.fetch_area(ds, bbox=bbox, day_range=day_range)
                if not df.empty:
                    frames.append(df)
            except requests.HTTPError:
                continue
        if not frames:
            return pd.DataFrame(columns=self._output_columns())
        out = pd.concat(frames, ignore_index=True)
        out = out.sort_values("timestamp_utc", ascending=False).reset_index(drop=True)
        return out

    @staticmethod
    def _normalize(df: pd.DataFrame) -> pd.DataFrame:
        frame = df.copy()

        # Normalize temperature/brightness columns across MODIS and VIIRS.
        if "bright_ti4" in frame.columns and "brightness" not in frame.columns:
            frame["brightness"] = frame["bright_ti4"]
        if "bright_t31" in frame.columns and "bright_ti5" not in frame.columns:
            frame["bright_ti5"] = frame["bright_t31"]

        sat = frame.get("satellite")
        if sat is not None:
            frame["satellite"] = sat.replace({"N20": "NOAA-20", "N21": "NOAA-21", "S": "SNPP", "A": "AQUA", "T": "TERRA"})
        else:
            frame["satellite"] = frame["dataset"]

        frame["acq_time"] = frame["acq_time"].astype(str).str.zfill(4)
        frame["timestamp_utc"] = pd.to_datetime(
            frame["acq_date"].astype(str) + " " + frame["acq_time"],
            format="%Y-%m-%d %H%M",
            utc=True,
            errors="coerce",
        )

        for col in ["frp", "brightness", "scan", "track", "latitude", "longitude"]:
            if col in frame.columns:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")

        if "confidence" not in frame.columns:
            frame["confidence"] = None
        if "daynight" not in frame.columns:
            frame["daynight"] = None
        if "instrument" not in frame.columns:
            frame["instrument"] = None

        return frame[FirmsClient._output_columns()].dropna(subset=["latitude", "longitude", "timestamp_utc"])

    @staticmethod
    def _output_columns() -> list[str]:
        return [
            "latitude",
            "longitude",
            "brightness",
            "bright_ti5",
            "frp",
            "scan",
            "track",
            "acq_date",
            "acq_time",
            "timestamp_utc",
            "satellite",
            "instrument",
            "confidence",
            "version",
            "daynight",
            "dataset",
        ]
