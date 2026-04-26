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


def test_fetch_aggregates_follows_pagination() -> None:
    """When the API returns next_url, the client must fetch all chunks."""
    call_count = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return httpx.Response(
                200,
                json={
                    "ticker": "AAPL",
                    "results": [
                        {
                            "t": 1619395200000,
                            "o": 1.0,
                            "h": 2.0,
                            "l": 0.5,
                            "c": 1.5,
                            "v": 100,
                            "vw": 1.2,
                        },
                    ],
                    "next_url": "https://api.massive.com/v2/aggs/page2",
                },
            )
        if call_count["n"] == 2:
            return httpx.Response(
                200,
                json={
                    "ticker": "AAPL",
                    "results": [
                        {
                            "t": 1619481600000,
                            "o": 2.0,
                            "h": 3.0,
                            "l": 1.5,
                            "c": 2.5,
                            "v": 200,
                            "vw": 2.2,
                        },
                    ],
                    "next_url": "https://api.massive.com/v2/aggs/page3",
                },
            )
        return httpx.Response(
            200,
            json={
                "ticker": "AAPL",
                "results": [
                    {
                        "t": 1619568000000,
                        "o": 3.0,
                        "h": 4.0,
                        "l": 2.5,
                        "c": 3.5,
                        "v": 300,
                        "vw": 3.2,
                    },
                ],
                # no next_url => terminal
            },
        )

    transport = httpx.MockTransport(handler)
    client = MassiveClient(api_key="dummy", base_url="https://api.massive.com", transport=transport)
    bars = client.fetch_aggregates(
        ticker="AAPL",
        start=date(2021, 4, 26),
        end=date(2021, 4, 28),
    )
    assert len(bars) == 3
    assert call_count["n"] == 3
    assert bars[0].open == 1.0
    assert bars[2].close == 3.5
