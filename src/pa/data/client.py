"""HTTP client for Massive API (Polygon-compatible aggregates endpoint)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


class MassiveAPIError(Exception):
    """Raised when the Massive API returns an unrecoverable error."""


@dataclass(frozen=True, slots=True)
class Bar:
    date: datetime  # Full timestamp (UTC). Daily bars use 00:00 UTC; intraday carries time.
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: float


class MassiveClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        transport: httpx.BaseTransport | None = None,
        max_retries: int = 3,
        timeout_s: float = 30.0,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._max_retries = max_retries
        self._client = httpx.Client(
            transport=transport,
            timeout=timeout_s,
            headers={"Authorization": f"Bearer {api_key}"},
        )

    def fetch_aggregates(
        self,
        *,
        ticker: str,
        start: date,
        end: date,
        timespan: str = "day",
        multiplier: int = 1,
        adjusted: bool = True,
    ) -> list[Bar]:
        """Fetch all aggregate bars for the ticker over [start, end].

        Follows Polygon-style pagination via `next_url` so the caller always gets
        the full range, not just the first chunk. Tier-specific per-call caps
        (e.g. ~1000 bars per response on Starter) are handled transparently.
        """
        initial_url = (
            f"{self._base_url}/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/"
            f"{start.isoformat()}/{end.isoformat()}"
        )
        initial_params: dict[str, str | int] = {
            "adjusted": str(adjusted).lower(),
            "sort": "asc",
            "limit": 50000,
        }

        all_bars: list[Bar] = []
        url: str | None = initial_url
        params: dict[str, str | int] | None = initial_params
        while url is not None:
            payload = self._get_page(url, params)
            results = payload.get("results") or []
            all_bars.extend(_bar_from_dict(b) for b in results)
            next_url = payload.get("next_url")
            url = next_url if isinstance(next_url, str) else None
            params = None  # next_url has cursor + params encoded
        return all_bars

    def _get_page(self, url: str, params: dict[str, str | int] | None) -> dict[str, Any]:
        @retry(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=30),
            retry=retry_if_exception_type((httpx.HTTPError, MassiveAPIError)),
            reraise=True,
        )
        def _do() -> dict[str, Any]:
            resp = self._client.get(url, params=params)
            if resp.status_code >= 500:
                raise MassiveAPIError(f"Server {resp.status_code}: {resp.text[:200]}")
            if resp.status_code == 429:
                raise MassiveAPIError("Rate limited")
            resp.raise_for_status()
            return resp.json()  # type: ignore[no-any-return]

        return _do()

    def close(self) -> None:
        self._client.close()


def _bar_from_dict(d: dict[str, float | int]) -> Bar:
    ts_ms = int(d["t"])
    bar_ts = datetime.fromtimestamp(ts_ms / 1000, tz=UTC)
    return Bar(
        date=bar_ts,
        open=float(d["o"]),
        high=float(d["h"]),
        low=float(d["l"]),
        close=float(d["c"]),
        volume=int(d["v"]),
        vwap=float(d["vw"]),
    )
