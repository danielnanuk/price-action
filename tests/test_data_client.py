"""Tests for pa.data.client — Massive API HTTP wrapper."""

from __future__ import annotations

from datetime import UTC, date, datetime

import httpx
import pytest
from pa.data.client import MassiveAPIError, MassiveClient


def test_fetch_aggregates_returns_bars() -> None:
    """Mock returning 2 daily bars."""
    transport = httpx.MockTransport(
        lambda req: httpx.Response(
            200,
            json={
                "ticker": "AAPL",
                "results": [
                    {
                        "t": 1619395200000,  # 2021-04-26
                        "o": 134.83,
                        "h": 134.85,
                        "l": 132.81,
                        "c": 134.32,
                        "v": 66905680,
                        "vw": 133.91,
                    },
                    {
                        "t": 1619481600000,  # 2021-04-27
                        "o": 135.01,
                        "h": 135.41,
                        "l": 134.11,
                        "c": 134.39,
                        "v": 66362094,
                        "vw": 134.78,
                    },
                ],
            },
        )
    )
    client = MassiveClient(
        api_key="dummy",
        base_url="https://api.massive.com",
        transport=transport,
    )
    bars = client.fetch_aggregates(
        ticker="AAPL",
        start=date(2021, 4, 26),
        end=date(2021, 4, 27),
    )
    assert len(bars) == 2
    assert bars[0].open == 134.83
    assert bars[0].date == datetime(2021, 4, 26, tzinfo=UTC)


def test_fetch_aggregates_raises_on_5xx() -> None:
    transport = httpx.MockTransport(lambda req: httpx.Response(503))
    client = MassiveClient(
        api_key="dummy",
        base_url="https://api.massive.com",
        transport=transport,
        max_retries=1,
    )
    with pytest.raises(MassiveAPIError):
        client.fetch_aggregates(
            ticker="AAPL",
            start=date(2021, 4, 26),
            end=date(2021, 4, 27),
        )
