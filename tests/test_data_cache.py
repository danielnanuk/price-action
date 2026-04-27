"""Tests for pa.data.cache — parquet OHLCV cache layer."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
from pa.data.cache import OhlcvCache
from pa.data.client import Bar


def _fake_bars() -> list[Bar]:
    return [
        Bar(datetime(2021, 4, 26), 100.0, 101.0, 99.0, 100.5, 1000, 100.2),
        Bar(datetime(2021, 4, 27), 100.5, 102.0, 100.0, 101.5, 1100, 101.0),
    ]


def test_cache_miss_calls_client(tmp_path: Path) -> None:
    client = MagicMock()
    client.fetch_aggregates.return_value = _fake_bars()
    cache = OhlcvCache(cache_dir=tmp_path, client=client)
    df = cache.fetch_ohlcv("AAPL", date(2021, 4, 26), date(2021, 4, 27))
    assert len(df) == 2
    assert list(df.columns) == ["date", "open", "high", "low", "close", "volume", "vwap"]
    client.fetch_aggregates.assert_called_once()


def test_cache_hit_skips_client(tmp_path: Path) -> None:
    client = MagicMock()
    client.fetch_aggregates.return_value = _fake_bars()
    cache = OhlcvCache(cache_dir=tmp_path, client=client)
    cache.fetch_ohlcv("AAPL", date(2021, 4, 26), date(2021, 4, 27))
    cache.fetch_ohlcv("AAPL", date(2021, 4, 26), date(2021, 4, 27))
    assert client.fetch_aggregates.call_count == 1


def test_cache_persists_as_parquet(tmp_path: Path) -> None:
    client = MagicMock()
    client.fetch_aggregates.return_value = _fake_bars()
    cache = OhlcvCache(cache_dir=tmp_path, client=client)
    cache.fetch_ohlcv("AAPL", date(2021, 4, 26), date(2021, 4, 27))
    files = list(tmp_path.glob("*.parquet"))
    assert len(files) == 1
    df = pd.read_parquet(files[0])
    assert len(df) == 2
