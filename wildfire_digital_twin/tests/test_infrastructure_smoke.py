from __future__ import annotations

import requests

from services.firms import BBox
from services.infrastructure import OverpassClient


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None) -> None:
        self.status_code = status_code
        self._payload = payload or {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error", response=self)

    def json(self) -> dict:
        return self._payload


def test_overpass_retries_on_406(monkeypatch) -> None:
    calls = {"count": 0}

    def fake_post(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return _FakeResponse(406)
        return _FakeResponse(200, {"elements": []})

    monkeypatch.setattr("services.infrastructure.requests.post", fake_post)
    client = OverpassClient(timeout=1)
    out = client.fetch(BBox(west=-125.0, south=32.0, east=-113.0, north=43.0))
    assert out.roads.empty
    assert out.power_lines.empty
    assert out.buildings.empty
    assert calls["count"] == 2
