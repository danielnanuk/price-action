"""Tests for pa.indicators.bar_anatomy — per-bar geometric features."""

from __future__ import annotations

import pandas as pd
import pytest
from pa.indicators.bar_anatomy import bar_anatomy


def _bar(o: float, h: float, lo: float, c: float) -> pd.DataFrame:
    return pd.DataFrame({"open": [o], "high": [h], "low": [lo], "close": [c]})


def test_full_body_bull_bar() -> None:
    df = _bar(o=10.0, h=11.0, lo=10.0, c=11.0)
    out = bar_anatomy(df)
    row = out.iloc[0]
    assert row["bar_is_bull"] is True or row["bar_is_bull"] == True  # noqa: E712
    assert row["bar_body_pct"] == pytest.approx(1.0)
    assert row["bar_upper_wick_pct"] == pytest.approx(0.0)
    assert row["bar_lower_wick_pct"] == pytest.approx(0.0)
    assert row["bar_close_position"] == pytest.approx(1.0)


def test_doji_bar() -> None:
    df = _bar(o=10.0, h=11.0, lo=9.0, c=10.0)
    out = bar_anatomy(df)
    row = out.iloc[0]
    assert row["bar_body_pct"] == pytest.approx(0.0)
    assert row["bar_close_position"] == pytest.approx(0.5)


def test_zero_range_safely_handled() -> None:
    df = _bar(o=10.0, h=10.0, lo=10.0, c=10.0)
    out = bar_anatomy(df)
    row = out.iloc[0]
    assert row["bar_body_pct"] == 0.0
    assert row["bar_close_position"] == 0.5  # convention for zero-range bars
