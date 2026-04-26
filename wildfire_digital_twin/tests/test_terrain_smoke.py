from __future__ import annotations

import requests

from services.firms import BBox
from services.terrain import OpenTopoDataClient


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None, headers: dict | None = None) -> None:
        self.status_code = status_code
        self._payload = payload or {}
        self.headers = headers or {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error", response=self)

    def json(self) -> dict:
        return self._payload


def test_terrain_fallback_grid_shape_and_slope_bounds() -> None:
    bbox = BBox(west=-111.0, south=41.0, east=-103.0, north=43.0)
    result = OpenTopoDataClient.fallback_grid(bbox, grid_size=12)
    assert result.elevation_m.shape == (12, 12)
    assert result.slope_deg.shape == (12, 12)
    assert float(result.slope_deg.min()) >= 0.0
    assert float(result.slope_deg.max()) <= 90.0


def test_terrain_retries_on_429(monkeypatch) -> None:
    calls = {"count": 0}

    def fake_sleep(_seconds: float) -> None:
        return None

    def fake_get(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return _FakeResponse(429, headers={"Retry-After": "1"})
        return _FakeResponse(200, payload={"results": [{"elevation": 1234.0}]})

    monkeypatch.setattr("services.terrain.time.sleep", fake_sleep)
    monkeypatch.setattr("services.terrain.requests.get", fake_get)
    client = OpenTopoDataClient(timeout=1)
    values = client._fetch_locations([(41.0, -111.0)], batch_size=1)
    assert values == [1234.0]
    assert calls["count"] == 2
