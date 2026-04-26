"""Tests for Double Top / Double Bottom reversal detector."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pa.detectors.double_top_bottom import detect_double_top_bottom
from pa.detectors.params import double_tb_params
from pa.indicators import compute_indicators
from pa.regime.classifier import classify_regime
from pa.regime.signal_bar import signal_bar_score
from pa.types import ParamTier

from tests.fixtures.loader import load_fixture

FIX_DIR = Path(__file__).parent / "fixtures" / "double_top_bottom"


def _bars(ohlcv: pd.DataFrame) -> pd.DataFrame:
    indicators = compute_indicators(ohlcv)
    regimes = classify_regime(ohlcv, indicators)
    bars = ohlcv.merge(indicators, on="date").merge(regimes, on="date")
    bars["signal_bar_score"] = signal_bar_score(ohlcv, indicators).values
    bars["ticker"] = "TEST"
    return bars


def test_double_top_detected() -> None:
    fix = load_fixture(FIX_DIR / "positive_double_top_01.json")
    candidates = detect_double_top_bottom(_bars(fix.ohlcv), double_tb_params(ParamTier.STANDARD))
    assert not candidates.empty
    detected = set(pd.to_datetime(candidates["signal_date"]))
    for d in fix.expected_signal_dates:
        assert d in detected


def test_single_top_not_detected() -> None:
    fix = load_fixture(FIX_DIR / "negative_single_top.json")
    candidates = detect_double_top_bottom(_bars(fix.ohlcv), double_tb_params(ParamTier.STANDARD))
    assert candidates.empty
