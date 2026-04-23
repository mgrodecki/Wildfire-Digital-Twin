from __future__ import annotations

from typing import Any

from services.weather import OpenMeteoClient


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


def test_openmeteo_fetch_current_parses_payload(monkeypatch) -> None:
    payload = {
        "current": {
            "time": "2026-04-22T12:00",
            "temperature_2m": 21.5,
            "relative_humidity_2m": 35,
            "wind_speed_10m": 18.0,
            "wind_direction_10m": 250,
            "precipitation": 0.1,
        }
    }

    def fake_get(*args, **kwargs):
        return _FakeResponse(payload)

    monkeypatch.setattr("services.weather.requests.get", fake_get)
    client = OpenMeteoClient(timeout=1)
    snapshot = client.fetch_current(latitude=35.0, longitude=-120.0)
    assert snapshot.time == "2026-04-22T12:00"
    assert snapshot.temperature_c == 21.5
    assert snapshot.humidity_pct == 35.0
