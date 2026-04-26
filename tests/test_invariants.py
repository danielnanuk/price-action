"""Property-based invariants — properties that must hold for ANY valid input."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from pa.backtest import simulate
from pa.detectors.h2 import detect_h2
from pa.detectors.params import h2_params
from pa.indicators import compute_indicators
from pa.regime.classifier import classify_regime
from pa.regime.signal_bar import signal_bar_score
from pa.types import ParamTier

_H2_TEXTBOOK_FIXTURE = Path(__file__).parent / "fixtures" / "h2" / "positive_01_textbook.json"


def _load_h2_textbook_ohlcv() -> pd.DataFrame:
    """Load the deterministic textbook H2 OHLCV used as a scaffold for the
    trade-ledger invariant test. Random walks essentially never produce
    Brooks-style H2 candidates (the detector demands a very specific
    swing-high → 2-down → ≥2-up → ≥2-down → strong-bull-bar sequence in a
    confirmed bull regime), so we need a known-good scaffold that yields at
    least one candidate. The simulator is then exercised via hypothesis-drawn
    parameters.
    """
    fix = json.loads(_H2_TEXTBOOK_FIXTURE.read_text())
    ohlcv = pd.DataFrame(fix["bars"])
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])
    return ohlcv


@st.composite
def random_ohlcv(draw, n: int = 80) -> pd.DataFrame:
    closes = draw(
        st.lists(
            st.floats(min_value=10.0, max_value=500.0, allow_nan=False),
            min_size=n,
            max_size=n,
        )
    )
    return pd.DataFrame(
        {
            "date": pd.date_range("2021-04-26", periods=n, freq="B"),
            "open": closes,
            "high": [c + 0.5 for c in closes],
            "low": [c - 0.5 for c in closes],
            "close": closes,
            "volume": [1_000_000] * n,
            "vwap": closes,
        }
    )


@settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
@given(random_ohlcv())
def test_regime_no_nan(ohlcv: pd.DataFrame) -> None:
    indicators = compute_indicators(ohlcv)
    regimes = classify_regime(ohlcv, indicators)
    assert regimes["regime"].notna().all()


@settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
@given(random_ohlcv(n=120))
def test_h2_monotonicity_loose_superset_standard_superset_strict(
    ohlcv: pd.DataFrame,
) -> None:
    indicators = compute_indicators(ohlcv)
    regimes = classify_regime(ohlcv, indicators)
    bars = ohlcv.merge(indicators, on="date").merge(regimes, on="date")
    bars["signal_bar_score"] = signal_bar_score(ohlcv, indicators).values
    bars["ticker"] = "TEST"

    strict = set(detect_h2(bars, h2_params(ParamTier.STRICT))["signal_date"])
    standard = set(detect_h2(bars, h2_params(ParamTier.STANDARD))["signal_date"])
    loose = set(detect_h2(bars, h2_params(ParamTier.LOOSE))["signal_date"])
    assert strict <= standard, f"strict not subset of standard: {strict - standard}"
    assert standard <= loose, f"standard not subset of loose: {standard - loose}"


@settings(max_examples=15, suppress_health_check=[HealthCheck.too_slow])
@given(
    time_stop_bars=st.integers(min_value=5, max_value=40),
    same_bar_priority=st.sampled_from(["stop_first", "target_first"]),
    tier=st.sampled_from([ParamTier.STANDARD, ParamTier.LOOSE]),
)
def test_trade_ledger_invariants(
    time_stop_bars: int,
    same_bar_priority: str,
    tier: ParamTier,
) -> None:
    """Trade ledger invariants must hold across simulator parameter variations.

    Uses a deterministic textbook H2 OHLCV scaffold (since random walks
    essentially never produce Brooks H2 setups) and lets hypothesis explore
    the simulator's parameter space.
    """
    ohlcv = _load_h2_textbook_ohlcv()
    indicators = compute_indicators(ohlcv)
    regimes = classify_regime(ohlcv, indicators)
    bars = ohlcv.merge(indicators, on="date").merge(regimes, on="date")
    bars["signal_bar_score"] = signal_bar_score(ohlcv, indicators).values
    bars["ticker"] = "TEST"
    cands = detect_h2(bars, h2_params(tier))
    assert not cands.empty, "scaffold must yield at least one H2 candidate"

    trades = simulate(
        cands,
        ohlcv,
        time_stop_bars=time_stop_bars,
        same_bar_priority=same_bar_priority,  # type: ignore[arg-type]
    )
    assert (trades["entry_date"] <= trades["exit_date"]).all()
    assert trades["pnl_r"].notna().all()
    assert (trades["pnl_r"].abs() < 1e6).all()  # finite-ish


@settings(max_examples=10, suppress_health_check=[HealthCheck.too_slow])
@given(random_ohlcv(n=80))
def test_indicators_deterministic(ohlcv: pd.DataFrame) -> None:
    a = compute_indicators(ohlcv)
    b = compute_indicators(ohlcv.copy())
    pd.testing.assert_frame_equal(a, b)
