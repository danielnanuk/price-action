"""Tests for TCL (Trend Channel Line) overshoot detector (long-only)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pa.detectors.params import tcl_params
from pa.detectors.tcl import detect_tcl
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


def _generate_tcl_ohlcv() -> pd.DataFrame:
    """Synthetic downtrend with TCL overshoot at the latest swing low."""
    dates = pd.date_range("2021-04-26", periods=80, freq="B")
    closes = np.linspace(100.0, 75.0, 70).tolist()
    bars: list[dict[str, float | str | pd.Timestamp]] = []
    while len(closes) < 80:
        closes.append(closes[-1])
    for i, d in enumerate(dates):
        c = closes[i]
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


def test_detector_imports_and_runs() -> None:
    ohlcv = _generate_tcl_ohlcv()
    bars = _bars(ohlcv)
    candidates = detect_tcl(bars, tcl_params(ParamTier.STANDARD))
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
    ohlcv = _generate_tcl_ohlcv().head(20)
    bars = _bars(ohlcv)
    candidates = detect_tcl(bars, tcl_params(ParamTier.STANDARD))
    assert candidates.empty
