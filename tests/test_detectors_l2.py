"""Tests for L2 (Low-2 two-legged rally in bear trend)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pa.detectors.l2 import detect_l2
from pa.detectors.params import l2_params
from pa.indicators import compute_indicators
from pa.regime.classifier import classify_regime
from pa.regime.signal_bar import signal_bar_score
from pa.types import ParamTier

from tests.fixtures.loader import load_fixture

FIX_DIR = Path(__file__).parent / "fixtures" / "l2"


def _bars(ohlcv: pd.DataFrame) -> pd.DataFrame:
    indicators = compute_indicators(ohlcv)
    regimes = classify_regime(ohlcv, indicators)
    bars = ohlcv.merge(indicators, on="date").merge(regimes, on="date")
    bars["signal_bar_score"] = signal_bar_score(ohlcv, indicators).values
    bars["ticker"] = "TEST"
    return bars


def test_positive_textbook_detected() -> None:
    fix = load_fixture(FIX_DIR / "positive_01_textbook.json")
    candidates = detect_l2(_bars(fix.ohlcv), l2_params(ParamTier.STANDARD))
    detected = set(pd.to_datetime(candidates["signal_date"])) if not candidates.empty else set()
    for d in fix.expected_signal_dates:
        assert d in detected


def test_negative_not_detected() -> None:
    fix = load_fixture(FIX_DIR / "negative_01_no_second_leg.json")
    candidates = detect_l2(_bars(fix.ohlcv), l2_params(ParamTier.STANDARD))
    assert candidates.empty


def test_monotonicity() -> None:
    fix = load_fixture(FIX_DIR / "positive_01_textbook.json")
    bars = _bars(fix.ohlcv)
    strict = set(detect_l2(bars, l2_params(ParamTier.STRICT))["signal_date"])
    standard = set(detect_l2(bars, l2_params(ParamTier.STANDARD))["signal_date"])
    loose = set(detect_l2(bars, l2_params(ParamTier.LOOSE))["signal_date"])
    assert strict <= standard <= loose
