"""Tests for pa.indicators.atr — Wilder's Average True Range."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pa.indicators.atr import atr


def test_atr_constant_bars_zero() -> None:
    df = pd.DataFrame(
        {
            "high": [10.0] * 30,
            "low": [10.0] * 30,
            "close": [10.0] * 30,
        }
    )
    out = atr(df["high"], df["low"], df["close"], n=14)
    assert (out.dropna() == 0.0).all()


def test_atr_first_n_are_nan() -> None:
    rng = np.random.default_rng(1)
    n_bars = 50
    closes = pd.Series(rng.normal(loc=100, scale=1, size=n_bars))
    highs = closes + 1
    lows = closes - 1
    out = atr(highs, lows, closes, n=14)
    assert out.iloc[:13].isna().all()
    assert not np.isnan(out.iloc[14])


def test_atr_known_value() -> None:
    """Hand computed: TR = max(h-l, |h-prev_close|, |l-prev_close|)."""
    high = pd.Series([10.0, 11.0, 12.0, 13.0])
    low = pd.Series([9.0, 9.5, 10.0, 11.0])
    close = pd.Series([9.5, 10.5, 11.5, 12.5])
    out = atr(high, low, close, n=2)
    # TR series: [1.0, 1.5 (11-9.5), 2.0 (12-10), 2.0 (13-11)]
    # ATR2 (Wilder) seeded at index 1 = mean(TR[0:2]) = 1.25
    # ATR2 at index 2 = (1.25*(2-1) + 2.0) / 2 = 1.625
    # ATR2 at index 3 = (1.625*(2-1) + 2.0) / 2 = 1.8125
    assert np.isnan(out.iloc[0])
    assert out.iloc[1] == pytest.approx(1.25)
    assert out.iloc[2] == pytest.approx(1.625)
    assert out.iloc[3] == pytest.approx(1.8125)
