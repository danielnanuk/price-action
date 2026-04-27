"""Tests for pa.indicators.swing — Brooks-style swing high/low pivots."""

from __future__ import annotations

import pandas as pd
from pa.indicators.swing import swing_pivot


def test_swing_high_detected() -> None:
    """A bar whose high > N bars on each side is a swing high."""
    highs = pd.Series([1.0, 2.0, 3.0, 5.0, 3.5, 2.5, 1.5])
    lows = pd.Series([0.5, 1.5, 2.5, 4.0, 3.0, 2.0, 1.0])
    out = swing_pivot(highs, lows, n=2)
    # Index 3 (high=5.0) should be a swing high (highest within ±2)
    assert out.loc[3, "is_swing_high"]
    assert not out.loc[3, "is_swing_low"]


def test_swing_low_detected() -> None:
    highs = pd.Series([5.0, 4.0, 3.0, 1.0, 3.5, 4.5, 5.5])
    lows = pd.Series([4.5, 3.5, 2.5, 0.5, 3.0, 4.0, 5.0])
    out = swing_pivot(highs, lows, n=2)
    assert out.loc[3, "is_swing_low"]
    assert not out.loc[3, "is_swing_high"]


def test_edges_cannot_be_swings() -> None:
    """First and last n bars cannot be swings (insufficient lookback/forward)."""
    highs = pd.Series([10.0, 9.0, 8.0, 7.0, 6.0])
    lows = pd.Series([9.5, 8.5, 7.5, 6.5, 5.5])
    out = swing_pivot(highs, lows, n=2)
    assert not out.iloc[:2]["is_swing_high"].any()
    assert not out.iloc[:2]["is_swing_low"].any()
    assert not out.iloc[-2:]["is_swing_high"].any()
    assert not out.iloc[-2:]["is_swing_low"].any()


def test_swing_levels_carried_forward() -> None:
    """`swing_high` column = price of most recent confirmed swing high (forward-filled)."""
    highs = pd.Series([1.0, 2.0, 3.0, 5.0, 3.5, 2.5, 1.5])
    lows = pd.Series([0.5, 1.5, 2.5, 4.0, 3.0, 2.0, 1.0])
    out = swing_pivot(highs, lows, n=2)
    # After bar 3 confirms swing high=5.0 at bar 3+n=5, level should appear from there.
    assert out.loc[5, "swing_high"] == 5.0
