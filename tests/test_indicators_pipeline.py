"""Tests for the indicators-pipeline glue function."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pa.indicators import compute_indicators
from pa.types import INDICATOR_COLS


def _synthetic_ohlcv(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    closes = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame(
        {
            "date": pd.date_range("2021-04-26", periods=n, freq="B"),
            "open": closes - 0.1,
            "high": closes + 0.5,
            "low": closes - 0.5,
            "close": closes,
            "volume": rng.integers(1_000_000, 5_000_000, n),
            "vwap": closes,
        }
    )


def test_pipeline_produces_required_columns() -> None:
    ohlcv = _synthetic_ohlcv(100)
    out = compute_indicators(ohlcv)
    assert "date" in out.columns
    for col in INDICATOR_COLS:
        assert col in out.columns
    assert len(out) == len(ohlcv)


def test_pipeline_no_extra_columns() -> None:
    ohlcv = _synthetic_ohlcv(100)
    out = compute_indicators(ohlcv)
    assert set(out.columns) == {"date", *INDICATOR_COLS}
