"""Tests for pa.backtest.engine — execution simulator."""

from __future__ import annotations

import pandas as pd
import pytest
from pa.backtest.engine import simulate
from pa.types import ExitReason, Side


def _ohlcv(rows: list[tuple[str, float, float, float, float]]) -> pd.DataFrame:
    """rows = [(date, open, high, low, close), ...]"""
    return pd.DataFrame(
        [
            {
                "date": pd.Timestamp(r[0]),
                "open": r[1],
                "high": r[2],
                "low": r[3],
                "close": r[4],
                "volume": 1_000,
                "vwap": (r[2] + r[3]) / 2,
            }
            for r in rows
        ]
    )


def test_long_target_hit() -> None:
    ohlcv = _ohlcv(
        [
            ("2021-04-26", 100, 100.5, 99.5, 100),
            ("2021-04-27", 100, 105, 100, 104),  # high=105 hits target=104
        ]
    )
    cands = pd.DataFrame(
        [
            {
                "ticker": "X",
                "signal_date": pd.Timestamp("2021-04-26"),
                "side": Side.LONG.value,
                "entry_price": 100.0,
                "stop_price": 99.0,
                "target_price": 104.0,
                "setup_score": 0.8,
                "regime_at_signal": "bull_trend",
                "params_tier": "standard",
            }
        ]
    )
    trades = simulate(cands, ohlcv, time_stop_bars=20, same_bar_priority="stop_first")
    assert len(trades) == 1
    t = trades.iloc[0]
    assert t["exit_reason"] == ExitReason.TARGET_HIT.value
    assert t["exit_price"] == pytest.approx(104.0)
    assert t["pnl_r"] == pytest.approx(4.0)


def test_long_stop_hit() -> None:
    ohlcv = _ohlcv(
        [
            ("2021-04-26", 100, 100.5, 99.5, 100),
            ("2021-04-27", 100, 100.5, 98.5, 99),  # low=98.5 hits stop=99
        ]
    )
    cands = pd.DataFrame(
        [
            {
                "ticker": "X",
                "signal_date": pd.Timestamp("2021-04-26"),
                "side": Side.LONG.value,
                "entry_price": 100.0,
                "stop_price": 99.0,
                "target_price": 104.0,
                "setup_score": 0.8,
                "regime_at_signal": "bull_trend",
                "params_tier": "standard",
            }
        ]
    )
    trades = simulate(cands, ohlcv, time_stop_bars=20, same_bar_priority="stop_first")
    t = trades.iloc[0]
    assert t["exit_reason"] == ExitReason.STOP_HIT.value
    assert t["pnl_r"] == pytest.approx(-1.0)


def test_same_bar_stop_first_takes_loss() -> None:
    ohlcv = _ohlcv(
        [
            ("2021-04-26", 100, 100.5, 99.5, 100),
            ("2021-04-27", 100, 105, 98.5, 100),  # both hit
        ]
    )
    cands = pd.DataFrame(
        [
            {
                "ticker": "X",
                "signal_date": pd.Timestamp("2021-04-26"),
                "side": Side.LONG.value,
                "entry_price": 100.0,
                "stop_price": 99.0,
                "target_price": 104.0,
                "setup_score": 0.8,
                "regime_at_signal": "bull_trend",
                "params_tier": "standard",
            }
        ]
    )
    trades = simulate(cands, ohlcv, time_stop_bars=20, same_bar_priority="stop_first")
    t = trades.iloc[0]
    assert t["exit_reason"] == ExitReason.STOP_HIT.value
    assert t["same_bar_ambiguous"] is True or t["same_bar_ambiguous"] == True  # noqa: E712


def test_time_stop_when_neither_hit() -> None:
    # Generate 6 calendar dates starting 2021-04-26 (avoids April-31 month-rollover bug
    # in the verbatim plan snippet which produced f"2021-04-{27+i:02d}" for i in range(5)).
    dates = pd.date_range("2021-04-26", periods=6, freq="D")
    ohlcv = _ohlcv([(d.strftime("%Y-%m-%d"), 100, 100.5, 99.5, 100) for d in dates])
    cands = pd.DataFrame(
        [
            {
                "ticker": "X",
                "signal_date": pd.Timestamp("2021-04-26"),
                "side": Side.LONG.value,
                "entry_price": 100.0,
                "stop_price": 99.0,
                "target_price": 104.0,
                "setup_score": 0.8,
                "regime_at_signal": "bull_trend",
                "params_tier": "standard",
            }
        ]
    )
    trades = simulate(cands, ohlcv, time_stop_bars=3, same_bar_priority="stop_first")
    t = trades.iloc[0]
    assert t["exit_reason"] == ExitReason.TIME_STOP.value


def test_short_target_hit() -> None:
    ohlcv = _ohlcv(
        [
            ("2021-04-26", 100, 100.5, 99.5, 100),
            ("2021-04-27", 100, 100, 95, 96),  # low=95 hits target=96 (short)
        ]
    )
    cands = pd.DataFrame(
        [
            {
                "ticker": "X",
                "signal_date": pd.Timestamp("2021-04-26"),
                "side": Side.SHORT.value,
                "entry_price": 100.0,
                "stop_price": 101.0,
                "target_price": 96.0,
                "setup_score": 0.8,
                "regime_at_signal": "bear_trend",
                "params_tier": "standard",
            }
        ]
    )
    trades = simulate(cands, ohlcv, time_stop_bars=20, same_bar_priority="stop_first")
    t = trades.iloc[0]
    assert t["exit_reason"] == ExitReason.TARGET_HIT.value
    assert t["pnl_r"] == pytest.approx(4.0)
