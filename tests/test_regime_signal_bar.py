"""Tests for pa.regime.signal_bar — quality score for entry signal candles."""

from __future__ import annotations

import pandas as pd
from pa.indicators import compute_indicators
from pa.regime.signal_bar import signal_bar_score


def _ohlcv_with_known_bars() -> pd.DataFrame:
    # Index 0: full body bull bar (best)
    # Index 1: doji (worst)
    # Index 2: bull bar with big upper wick (mediocre)
    return pd.DataFrame(
        {
            "date": pd.date_range("2021-04-26", periods=60, freq="B"),
            "open": [100.0] * 60,
            "high": [101.0] * 60,
            "low": [99.0] * 60,
            "close": [100.5] * 60,
            "volume": [1_000_000] * 60,
            "vwap": [100.0] * 60,
        }
    )


def test_score_is_between_0_and_1() -> None:
    ohlcv = _ohlcv_with_known_bars()
    ohlcv.loc[0, ["open", "high", "low", "close"]] = [100.0, 101.0, 100.0, 101.0]
    ohlcv.loc[1, ["open", "high", "low", "close"]] = [100.0, 101.0, 99.0, 100.0]
    ohlcv.loc[2, ["open", "high", "low", "close"]] = [100.0, 102.0, 99.5, 100.4]
    indicators = compute_indicators(ohlcv)
    scores = signal_bar_score(ohlcv, indicators)
    assert (scores.dropna().between(0.0, 1.0)).all()


def test_full_body_bull_scores_higher_than_doji() -> None:
    ohlcv = _ohlcv_with_known_bars()
    ohlcv.loc[0, ["open", "high", "low", "close"]] = [100.0, 101.0, 100.0, 101.0]
    ohlcv.loc[1, ["open", "high", "low", "close"]] = [100.0, 101.0, 99.0, 100.0]
    indicators = compute_indicators(ohlcv)
    scores = signal_bar_score(ohlcv, indicators)
    assert scores.iloc[0] > scores.iloc[1]
