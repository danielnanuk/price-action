"""Parquet-backed OHLCV cache; one file per (ticker, start, end)."""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from datetime import date
from pathlib import Path

import pandas as pd

from pa.data.client import MassiveClient
from pa.types import OHLCV_COLS


class OhlcvCache:
    def __init__(self, *, cache_dir: Path, client: MassiveClient) -> None:
        self._dir = cache_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._client = client

    def _key(self, ticker: str, start: date, end: date, timespan: str, multiplier: int) -> Path:
        h = hashlib.sha256(
            f"{ticker}|{start.isoformat()}|{end.isoformat()}|{multiplier}{timespan}".encode()
        ).hexdigest()[:12]
        return self._dir / f"{ticker}_{h}.parquet"

    def fetch_ohlcv(
        self,
        ticker: str,
        start: date,
        end: date,
        *,
        timespan: str = "day",
        multiplier: int = 1,
    ) -> pd.DataFrame:
        path = self._key(ticker, start, end, timespan, multiplier)
        if path.exists():
            return pd.read_parquet(path)

        bars = self._client.fetch_aggregates(
            ticker=ticker, start=start, end=end, timespan=timespan, multiplier=multiplier
        )
        if not bars:
            df = pd.DataFrame(columns=OHLCV_COLS)
        else:
            df = pd.DataFrame([asdict(b) for b in bars])[OHLCV_COLS]
            df["date"] = pd.to_datetime(df["date"])

        df.to_parquet(path, index=False)
        return df
