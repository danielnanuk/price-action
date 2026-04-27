"""Tests for pa.backtest.engine — execution simulator."""

from __future__ import annotations

import pandas as pd
import pytest
from pa.backtest.engine import ExitStrategy, simulate
from pa.types import ExitReason


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


def _ohlcv_with_atr(
    rows: list[tuple[str, float, float, float, float]], atr: float = 1.0
) -> pd.DataFrame:
    """OHLCV with a fixed ATR14 column for trailing-stop tests."""
    df = _ohlcv(rows)
    df["atr14"] = atr
    return df


_DEFAULT = ExitStrategy()  # 2R fixed target, 20-bar time stop, no scale, no trail


def _cand(
    side: str = "long",
    entry: float = 100.0,
    stop: float = 99.0,
    target: float = 104.0,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ticker": "X",
                "signal_date": pd.Timestamp("2021-04-26"),
                "side": side,
                "entry_price": entry,
                "stop_price": stop,
                "target_price": target,
                "setup_score": 0.8,
                "regime_at_signal": "bull_trend",
                "params_tier": "standard",
            }
        ]
    )


def test_long_target_hit() -> None:
    ohlcv = _ohlcv(
        [
            ("2021-04-26", 100, 100.5, 99.5, 100),
            ("2021-04-27", 100, 105, 100, 104),  # high=105 hits target=104
        ]
    )
    trades = simulate(_cand(), ohlcv, strategy=_DEFAULT)
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
    trades = simulate(_cand(), ohlcv, strategy=_DEFAULT)
    t = trades.iloc[0]
    assert t["exit_reason"] == ExitReason.STOP_HIT.value
    assert t["pnl_r"] == pytest.approx(-1.0)


def test_same_bar_stop_first_takes_loss() -> None:
    ohlcv = _ohlcv(
        [
            ("2021-04-26", 100, 100.5, 99.5, 100),
            ("2021-04-27", 100, 105, 98.5, 100),  # both hit in single bar
        ]
    )
    trades = simulate(_cand(), ohlcv, strategy=_DEFAULT)
    t = trades.iloc[0]
    assert t["exit_reason"] == ExitReason.STOP_HIT.value
    assert t["same_bar_ambiguous"] is True or t["same_bar_ambiguous"] == True  # noqa: E712


def test_time_stop_when_neither_hit() -> None:
    dates = pd.date_range("2021-04-26", periods=6, freq="D")
    ohlcv = _ohlcv([(d.strftime("%Y-%m-%d"), 100, 100.5, 99.5, 100) for d in dates])
    short_strategy = ExitStrategy(time_stop_bars=3)
    trades = simulate(_cand(), ohlcv, strategy=short_strategy)
    t = trades.iloc[0]
    assert t["exit_reason"] == ExitReason.TIME_STOP.value


def test_short_target_hit() -> None:
    ohlcv = _ohlcv(
        [
            ("2021-04-26", 100, 100.5, 99.5, 100),
            ("2021-04-27", 100, 100, 95, 96),  # low=95 hits target=96 (short)
        ]
    )
    trades = simulate(
        _cand(side="short", entry=100.0, stop=101.0, target=96.0),
        ohlcv,
        strategy=_DEFAULT,
    )
    t = trades.iloc[0]
    assert t["exit_reason"] == ExitReason.TARGET_HIT.value
    assert t["pnl_r"] == pytest.approx(4.0)


def test_no_fixed_target_keeps_runner_until_trail_or_time_stop() -> None:
    """With use_fixed_target=False, hitting +2R does NOT exit; only stop/trail/time."""
    dates = pd.date_range("2021-04-26", periods=6, freq="D")
    ohlcv = _ohlcv_with_atr(
        [
            (dates[0].strftime("%Y-%m-%d"), 100, 100.5, 99.5, 100),
            (dates[1].strftime("%Y-%m-%d"), 100, 110, 99.7, 109),  # touched +10R
            (dates[2].strftime("%Y-%m-%d"), 109, 110, 108, 108),
            (dates[3].strftime("%Y-%m-%d"), 108, 109, 107, 107),
            (dates[4].strftime("%Y-%m-%d"), 107, 108, 106, 106),
            (dates[5].strftime("%Y-%m-%d"), 106, 107, 105, 105),
        ],
        atr=1.0,
    )
    strategy = ExitStrategy(use_fixed_target=False, trailing_atr_mult=0.0, time_stop_bars=10)
    trades = simulate(_cand(), ohlcv, strategy=strategy)
    t = trades.iloc[0]
    # No fixed target -> reaching 110 (which would be 10R) should NOT trigger TARGET_HIT
    assert t["exit_reason"] != ExitReason.TARGET_HIT.value


def test_scale_at_1r_locks_half_at_one_r() -> None:
    """With scale_at_1r=True, MFE>=1R locks +1R on half; runner exits separately."""
    dates = pd.date_range("2021-04-26", periods=4, freq="D")
    # Bar 1 hits +1R (high=101), then drops back to entry by close.
    # Bar 2 stops out the runner at 99.
    ohlcv = _ohlcv_with_atr(
        [
            (dates[0].strftime("%Y-%m-%d"), 100, 100.5, 99.5, 100),
            (dates[1].strftime("%Y-%m-%d"), 100, 101, 99.7, 100),  # touches +1R
            (dates[2].strftime("%Y-%m-%d"), 100, 100.5, 98.5, 99),  # runner stops at 99
        ],
        atr=1.0,
    )
    strategy = ExitStrategy(scale_at_1r=True)
    trades = simulate(_cand(), ohlcv, strategy=strategy)
    t = trades.iloc[0]
    # Half locked at +1R, runner exits at -1R -> avg = 0
    assert t["pnl_r"] == pytest.approx(0.0)


def test_trailing_atr_keeps_winners_running() -> None:
    """A long that ramps from 100 to 110 with chandelier 3xATR should exit on
    the first deep retracement, not on a fixed 2R target."""
    dates = pd.date_range("2021-04-26", periods=6, freq="D")
    ohlcv = _ohlcv_with_atr(
        [
            (dates[0].strftime("%Y-%m-%d"), 100, 100.5, 99.5, 100),
            (dates[1].strftime("%Y-%m-%d"), 100, 105, 99.7, 104),  # high=105 -> trail to 102
            (dates[2].strftime("%Y-%m-%d"), 104, 110, 103, 109),  # high=110 -> trail to 107
            (dates[3].strftime("%Y-%m-%d"), 109, 109, 106.9, 107),  # 106.9 trips trail
        ],
        atr=1.0,
    )
    strategy = ExitStrategy(use_fixed_target=False, trailing_atr_mult=3.0, time_stop_bars=10)
    trades = simulate(_cand(), ohlcv, strategy=strategy)
    t = trades.iloc[0]
    assert t["exit_reason"] == ExitReason.STOP_HIT.value
    # pnl_r should be > 2R because we let the trade run past the original 4R target
    # (entry 100, initial stop 99, R=1; trail caught around 107 -> +7R)
    assert t["pnl_r"] > 4.0


def test_trailing_requires_atr_column() -> None:
    """Engine must raise if trailing is requested but bars lack atr14."""
    ohlcv = _ohlcv(
        [
            ("2021-04-26", 100, 100.5, 99.5, 100),
            ("2021-04-27", 100, 105, 100, 104),
        ]
    )
    strategy = ExitStrategy(trailing_atr_mult=3.0)
    with pytest.raises(ValueError, match="atr14"):
        simulate(_cand(), ohlcv, strategy=strategy)
