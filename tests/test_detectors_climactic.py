"""Tests for Climactic reversal detector (long-only, bottom climax)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pa.detectors.climactic import detect_climactic
from pa.detectors.params import climactic_params
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


def _generate_bottom_climax_ohlcv() -> pd.DataFrame:
    """Synthetic downtrend ending in a panic climax + reversal."""
    dates = pd.date_range("2021-04-26", periods=80, freq="B")
    closes = np.linspace(100.0, 75.0, 70).tolist()  # 70-bar downtrend
    # Climax bar at index 70: huge bear bar, close near low, new low
    # Reversal bar at index 71: bull bar, close above climax close
    bars: list[dict[str, float | str | pd.Timestamp]] = []
    while len(closes) < 80:
        closes.append(closes[-1])
    for i, d in enumerate(dates):
        c = closes[i]
        if i == 70:
            # Climax: open at 76, dive to 70, close at 71 (range 6, body 5)
            bars.append(
                {
                    "date": d,
                    "open": 76.0,
                    "high": 76.5,
                    "low": 70.0,
                    "close": 71.0,
                    "volume": 5_000_000,
                    "vwap": 73.0,
                }
            )
        elif i == 71:
            # Reversal: bull bar, close > 71 (climax close)
            bars.append(
                {
                    "date": d,
                    "open": 71.5,
                    "high": 74.5,
                    "low": 71.0,
                    "close": 74.0,
                    "volume": 4_000_000,
                    "vwap": 72.5,
                }
            )
        else:
            bars.append(
                {
                    "date": d,
                    "open": c - 0.3,
                    "high": c + 0.4,
                    "low": c - 0.5,
                    "close": c,
                    "volume": 1_000_000,
                    "vwap": c,
                }
            )
    return pd.DataFrame(bars)


def test_detector_imports_and_runs_without_error() -> None:
    """Smoke test: detector handles synthetic climax+reversal cleanly."""
    ohlcv = _generate_bottom_climax_ohlcv()
    bars = _bars(ohlcv)
    candidates = detect_climactic(bars, climactic_params(ParamTier.STANDARD))
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
    """Insufficient history returns empty without error."""
    ohlcv = _generate_bottom_climax_ohlcv().head(15)
    bars = _bars(ohlcv)
    candidates = detect_climactic(bars, climactic_params(ParamTier.STANDARD))
    assert candidates.empty
