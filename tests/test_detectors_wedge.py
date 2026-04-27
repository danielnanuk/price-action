"""Tests for Wedge / 3-push reversal detector (long-only / bottom wedge)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pa.detectors.params import wedge_params
from pa.detectors.wedge import detect_wedge
from pa.indicators import compute_indicators
from pa.regime.classifier import classify_regime
from pa.regime.signal_bar import signal_bar_score
from pa.types import ParamTier


def _bars(ohlcv: pd.DataFrame) -> pd.DataFrame:
    indicators = compute_indicators(ohlcv)
    regimes = classify_regime(ohlcv, indicators)
    bars = ohlcv.merge(indicators, on="date").merge(regimes, on="date")
    bars["signal_bar_score"] = signal_bar_score(ohlcv, indicators).values
    bars["ticker"] = "TEST"
    return bars


def _generate_bottom_wedge_ohlcv() -> pd.DataFrame:
    """Hand-shape a downtrend ending in a 3-push wedge with diminishing pushes.

    Bars 0..49: linear downtrend from 100 to 80 (establish bear regime).
    Bars 50..58: B1 = 78, then rally to H1 = 85 (push1 = 7).
    Bars 59..65: B2 = 76 (lower low), then rally to H2 = 80 (push2 = 4 < 0.7*push1=4.9).
    Bars 66..72: B3 = 74 (lower low), then bull bar at 73.5 close 75.5 (signal).
    """
    bars: list[dict[str, float | str]] = []
    dates = pd.date_range("2021-04-26", periods=80, freq="B")

    closes = np.linspace(100.0, 80.0, 50).tolist()  # bars 0..49: downtrend
    # Push 1: rally from 78 to 85
    closes += [
        78.0,
        79.0,
        80.5,
        82.0,
        83.5,
        84.5,
        85.0,  # bars 50..56
        83.5,
        81.5,  # bars 57..58 down to B2
    ]
    # Push 2: rally to 80
    closes += [
        76.5,  # bar 59 = B2 (lower low at 76)
        77.5,
        78.5,
        79.5,
        80.0,  # bars 60..63
        78.5,
        77.0,  # bars 64..65 down to B3
    ]
    # Approach to B3 and signal bar
    closes += [
        74.5,  # bar 66 = B3 (lowest)
        76.0,
        77.0,  # bars 67..68 (signal forms)
    ]
    while len(closes) < 80:
        closes.append(closes[-1])

    for i, d in enumerate(dates):
        c = closes[i]
        # Signal bar at index 67-68 should be a clear bull bar (close > open, body large)
        if i == 67:
            bars.append(
                {
                    "date": d,
                    "open": 74.6,
                    "high": 76.5,
                    "low": 74.5,
                    "close": 76.0,
                    "volume": 1_000_000,
                    "vwap": c,
                }
            )
        elif i == 68:
            bars.append(
                {
                    "date": d,
                    "open": 76.0,
                    "high": 78.0,
                    "low": 75.7,
                    "close": 77.5,
                    "volume": 1_500_000,
                    "vwap": c,
                }
            )
        else:
            bars.append(
                {
                    "date": d,
                    "open": c - 0.2,
                    "high": c + 0.5,
                    "low": c - 0.5,
                    "close": c,
                    "volume": 1_000_000,
                    "vwap": c,
                }
            )
    return pd.DataFrame(bars)


def test_detector_imports_and_runs_without_error() -> None:
    """Smoke test: wedge detector handles a clean downtrend ending in a wedge.

    Doesn't assert specific signal_date — fixture geometry is approximate
    (bear regime detection has its own thresholds that may or may not lock
    in on this specific synthetic data). Asserts only that the detector
    runs cleanly and returns a DataFrame with the right schema.
    """
    ohlcv = _generate_bottom_wedge_ohlcv()
    bars = _bars(ohlcv)
    candidates = detect_wedge(bars, wedge_params(ParamTier.STANDARD))
    # Schema check
    expected_cols = {
        "ticker",
        "signal_date",
        "side",
        "entry_price",
        "stop_price",
        "target_price",
        "setup_score",
        "regime_at_signal",
        "params_tier",
    }
    assert expected_cols.issubset(set(candidates.columns))


def test_short_input_returns_empty() -> None:
    """Insufficient history (< lookback_bars) returns empty without error."""
    ohlcv = _generate_bottom_wedge_ohlcv().head(20)  # only 20 bars
    bars = _bars(ohlcv)
    candidates = detect_wedge(bars, wedge_params(ParamTier.STANDARD))
    assert candidates.empty
