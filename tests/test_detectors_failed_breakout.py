"""Tests for Failed Breakout reversal detector."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pa.detectors.failed_breakout import detect_failed_breakout
from pa.detectors.params import failed_breakout_params
from pa.indicators import compute_indicators
from pa.regime.classifier import classify_regime
from pa.regime.signal_bar import signal_bar_score
from pa.types import ParamTier

from tests.fixtures.loader import load_fixture

FIX_DIR = Path(__file__).parent / "fixtures" / "failed_breakout"


def _bars(ohlcv: pd.DataFrame) -> pd.DataFrame:
    indicators = compute_indicators(ohlcv)
    regimes = classify_regime(ohlcv, indicators)
    bars = ohlcv.merge(indicators, on="date").merge(regimes, on="date")
    bars["signal_bar_score"] = signal_bar_score(ohlcv, indicators).values
    bars["ticker"] = "TEST"
    return bars


def test_up_breakout_failure_detected() -> None:
    fix = load_fixture(FIX_DIR / "positive_up_fail_01.json")
    candidates = detect_failed_breakout(
        _bars(fix.ohlcv), failed_breakout_params(ParamTier.STANDARD)
    )
    assert not candidates.empty
    detected = set(pd.to_datetime(candidates["signal_date"]))
    for d in fix.expected_signal_dates:
        assert d in detected


def test_real_breakout_not_detected_as_failure() -> None:
    fix = load_fixture(FIX_DIR / "negative_real_breakout.json")
    candidates = detect_failed_breakout(
        _bars(fix.ohlcv), failed_breakout_params(ParamTier.STANDARD)
    )
    assert candidates.empty


def test_target_is_closer_of_2r_or_range_low() -> None:
    """For SHORT: target = max(entry - 2R, range_low). This test asserts that
    target is never lower than 2R target (always >= due to max())."""
    fix = load_fixture(FIX_DIR / "positive_up_fail_01.json")
    candidates = detect_failed_breakout(
        _bars(fix.ohlcv), failed_breakout_params(ParamTier.STANDARD)
    )
    assert not candidates.empty
    c = candidates.iloc[0]
    risk = c["stop_price"] - c["entry_price"]
    target_2r = c["entry_price"] - 2 * risk
    # Target should never be lower than 2R target (always >= due to max())
    assert c["target_price"] >= target_2r - 1e-6
