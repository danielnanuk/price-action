"""Tests for H2 (High-2 two-legged pullback in bull trend)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pa.detectors.h2 import detect_h2
from pa.detectors.params import h2_params
from pa.indicators import compute_indicators
from pa.regime.classifier import classify_regime
from pa.regime.signal_bar import signal_bar_score
from pa.types import ParamTier

from tests.fixtures.loader import load_fixture

FIX_DIR = Path(__file__).parent / "fixtures" / "h2"


def _bars_for_detector(ohlcv: pd.DataFrame) -> pd.DataFrame:
    indicators = compute_indicators(ohlcv)
    regimes = classify_regime(ohlcv, indicators)
    bars = ohlcv.merge(indicators, on="date").merge(regimes, on="date")
    bars["signal_bar_score"] = signal_bar_score(ohlcv, indicators).values
    bars["ticker"] = "TEST"
    return bars


def test_positive_textbook_detected() -> None:
    fix = load_fixture(FIX_DIR / "positive_01_textbook.json")
    bars = _bars_for_detector(fix.ohlcv)
    candidates = detect_h2(bars, h2_params(ParamTier.STANDARD))
    assert not candidates.empty
    detected_dates = set(pd.to_datetime(candidates["signal_date"]))
    for d in fix.expected_signal_dates:
        assert d in detected_dates


def test_negative_no_second_leg_not_detected() -> None:
    fix = load_fixture(FIX_DIR / "negative_01_no_second_leg.json")
    bars = _bars_for_detector(fix.ohlcv)
    candidates = detect_h2(bars, h2_params(ParamTier.STANDARD))
    assert candidates.empty


def test_monotonicity_loose_superset_of_standard_superset_of_strict() -> None:
    fix = load_fixture(FIX_DIR / "positive_01_textbook.json")
    bars = _bars_for_detector(fix.ohlcv)
    strict = set(detect_h2(bars, h2_params(ParamTier.STRICT))["signal_date"])
    standard = set(detect_h2(bars, h2_params(ParamTier.STANDARD))["signal_date"])
    loose = set(detect_h2(bars, h2_params(ParamTier.LOOSE))["signal_date"])
    assert strict <= standard
    assert standard <= loose


def test_returns_required_columns() -> None:
    fix = load_fixture(FIX_DIR / "positive_01_textbook.json")
    bars = _bars_for_detector(fix.ohlcv)
    candidates = detect_h2(bars, h2_params(ParamTier.STANDARD))
    if not candidates.empty:
        for col in [
            "ticker",
            "signal_date",
            "side",
            "entry_price",
            "stop_price",
            "target_price",
            "setup_score",
            "regime_at_signal",
            "params_tier",
        ]:
            assert col in candidates.columns
