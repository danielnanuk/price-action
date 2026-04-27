"""Tests for Bull/Bear Flag breakout detector."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pa.detectors.flag import detect_flag
from pa.detectors.params import flag_params
from pa.indicators import compute_indicators
from pa.regime.classifier import classify_regime
from pa.regime.signal_bar import signal_bar_score
from pa.types import ParamTier

from tests.fixtures.loader import load_fixture

FIX_DIR = Path(__file__).parent / "fixtures" / "flag"


def _bars(ohlcv: pd.DataFrame) -> pd.DataFrame:
    indicators = compute_indicators(ohlcv)
    regimes = classify_regime(ohlcv, indicators)
    bars = ohlcv.merge(indicators, on="date").merge(regimes, on="date")
    bars["signal_bar_score"] = signal_bar_score(ohlcv, indicators).values
    bars["ticker"] = "TEST"
    return bars


def test_bull_flag_detected() -> None:
    fix = load_fixture(FIX_DIR / "positive_bull_01.json")
    candidates = detect_flag(_bars(fix.ohlcv), flag_params(ParamTier.STANDARD))
    assert not candidates.empty
    detected = set(pd.to_datetime(candidates["signal_date"]))
    for d in fix.expected_signal_dates:
        assert d in detected


def test_no_consolidation_not_detected() -> None:
    fix = load_fixture(FIX_DIR / "negative_no_consolidation.json")
    candidates = detect_flag(_bars(fix.ohlcv), flag_params(ParamTier.STANDARD))
    assert candidates.empty


def test_monotonicity() -> None:
    fix = load_fixture(FIX_DIR / "positive_bull_01.json")
    bars = _bars(fix.ohlcv)
    s = set(detect_flag(bars, flag_params(ParamTier.STRICT))["signal_date"])
    m = set(detect_flag(bars, flag_params(ParamTier.STANDARD))["signal_date"])
    lo = set(detect_flag(bars, flag_params(ParamTier.LOOSE))["signal_date"])
    assert s <= m <= lo
