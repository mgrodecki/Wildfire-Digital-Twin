from __future__ import annotations

import pandas as pd
import pytest
import requests

from services.firms import BBox, FirmsClient


def test_bbox_helpers() -> None:
    bbox = BBox(west=-120.0, south=35.0, east=-119.0, north=36.0)
    assert bbox.as_firms_area() == "-120.0,35.0,-119.0,36.0"
    center_lat, center_lon = bbox.center()
    assert center_lat == 35.5
    assert center_lon == -119.5


def test_firms_normalize_basic_viirs_shape() -> None:
    raw = pd.DataFrame(
        [
            {
                "latitude": "35.1",
                "longitude": "-119.9",
                "bright_ti4": "330.5",
                "bright_ti5": "290.2",
                "frp": "12.1",
                "scan": "0.5",
                "track": "0.6",
                "acq_date": "2026-04-22",
                "acq_time": "915",
                "satellite": "N21",
                "instrument": "VIIRS",
                "confidence": "n",
                "version": "2.0NRT",
                "daynight": "D",
                "dataset": "VIIRS_NOAA21_NRT",
            }
        ]
    )
    normalized = FirmsClient._normalize(raw)
    assert not normalized.empty
    assert "timestamp_utc" in normalized.columns
    assert "brightness" in normalized.columns
    assert normalized.iloc[0]["satellite"] == "NOAA-21"


class _FakeResponse:
    def __init__(self, status_code: int, text: str) -> None:
        self.status_code = status_code
        self.text = text

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error", response=self)


def test_fetch_many_raises_when_all_datasets_fail(monkeypatch) -> None:
    def fake_get(*args, **kwargs):
        return _FakeResponse(400, "Invalid MAP_KEY.")

    monkeypatch.setattr("services.firms.requests.get", fake_get)
    client = FirmsClient(map_key="bad-key")

    with pytest.raises(RuntimeError) as exc:
        client.fetch_many(BBox(west=-125.0, south=32.0, east=-113.0, north=43.0), datasets=["VIIRS_NOAA21_NRT"], day_range=1)

    assert "FIRMS request failed" in str(exc.value)
    assert "Invalid MAP_KEY" in str(exc.value)


def test_fetch_area_tries_fallback_endpoint(monkeypatch) -> None:
    calls = {"count": 0}
    csv_text = (
        "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_ti5,frp,daynight\n"
        "35.1,-119.9,330.5,0.5,0.6,2026-04-22,0915,N21,VIIRS,n,2.0NRT,290.2,12.1,D\n"
    )

    def fake_get(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return _FakeResponse(404, "Not found")
        return _FakeResponse(200, csv_text)

    monkeypatch.setattr("services.firms.requests.get", fake_get)
    client = FirmsClient(map_key="test-key")
    out = client.fetch_area("VIIRS_NOAA21_NRT", BBox(west=-125.0, south=32.0, east=-113.0, north=43.0), day_range=1)
    assert not out.empty
    assert calls["count"] == 2
