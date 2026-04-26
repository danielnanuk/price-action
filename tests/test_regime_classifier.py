"""Tests for pa.regime.classifier."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pa.indicators import compute_indicators
from pa.regime.classifier import classify_regime
from pa.types import Regime


def _trending_up_ohlcv(n: int = 80) -> pd.DataFrame:
    closes = np.linspace(100, 130, n) + np.random.default_rng(0).normal(0, 0.3, n)
    return pd.DataFrame(
        {
            "date": pd.date_range("2021-04-26", periods=n, freq="B"),
            "open": closes,
            "high": closes + 0.5,
            "low": closes - 0.5,
            "close": closes,
            "volume": [1_000_000] * n,
            "vwap": closes,
        }
    )


def _trending_down_ohlcv(n: int = 80) -> pd.DataFrame:
    closes = np.linspace(130, 100, n) + np.random.default_rng(0).normal(0, 0.3, n)
    return pd.DataFrame(
        {
            "date": pd.date_range("2021-04-26", periods=n, freq="B"),
            "open": closes,
            "high": closes + 0.5,
            "low": closes - 0.5,
            "close": closes,
            "volume": [1_000_000] * n,
            "vwap": closes,
        }
    )


def _flat_ohlcv(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    closes = 100 + rng.normal(0, 0.5, n)
    return pd.DataFrame(
        {
            "date": pd.date_range("2021-04-26", periods=n, freq="B"),
            "open": closes,
            "high": closes + 0.5,
            "low": closes - 0.5,
            "close": closes,
            "volume": [1_000_000] * n,
            "vwap": closes,
        }
    )


def test_uptrend_classified_as_bull() -> None:
    ohlcv = _trending_up_ohlcv()
    indicators = compute_indicators(ohlcv)
    regimes = classify_regime(ohlcv, indicators)
    valid = regimes["regime"].dropna().tail(20)
    assert (valid == Regime.BULL_TREND).mean() > 0.7


def test_downtrend_classified_as_bear() -> None:
    ohlcv = _trending_down_ohlcv()
    indicators = compute_indicators(ohlcv)
    regimes = classify_regime(ohlcv, indicators)
    valid = regimes["regime"].dropna().tail(20)
    assert (valid == Regime.BEAR_TREND).mean() > 0.7


def test_flat_classified_as_range() -> None:
    ohlcv = _flat_ohlcv()
    indicators = compute_indicators(ohlcv)
    regimes = classify_regime(ohlcv, indicators)
    valid = regimes["regime"].dropna().tail(20)
    assert (valid == Regime.TRADING_RANGE).mean() > 0.5


def test_every_bar_has_regime_label() -> None:
    ohlcv = _trending_up_ohlcv()
    indicators = compute_indicators(ohlcv)
    regimes = classify_regime(ohlcv, indicators)
    # Bars before EMA50 warmup may be `transitional`; none should be NaN.
    assert regimes["regime"].notna().all()
