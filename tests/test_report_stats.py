"""Tests for pa.report.stats — aggregate metrics from a trades ledger."""

from __future__ import annotations

import pandas as pd
import pytest
from pa.report.stats import compute_setup_stats, group_by_regime, group_by_year
from pa.types import ExitReason


def _trades(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_basic_metrics_three_trades() -> None:
    trades = _trades(
        [
            {"pnl_r": 2.0, "exit_reason": ExitReason.TARGET_HIT.value, "mae_r": -0.4, "mfe_r": 2.1},
            {"pnl_r": -1.0, "exit_reason": ExitReason.STOP_HIT.value, "mae_r": -1.0, "mfe_r": 0.5},
            {"pnl_r": 2.0, "exit_reason": ExitReason.TARGET_HIT.value, "mae_r": -0.3, "mfe_r": 2.2},
        ]
    )
    stats = compute_setup_stats(trades)
    assert stats["n"] == 3
    assert stats["win_rate"] == pytest.approx(2 / 3)
    assert stats["mean_r"] == pytest.approx(1.0)
    assert stats["profit_factor"] == pytest.approx(4.0)  # gross 4 / gross 1
    assert stats["mean_mae_r"] == pytest.approx((-0.4 + -1.0 + -0.3) / 3)


def test_empty_returns_zero_metrics() -> None:
    stats = compute_setup_stats(pd.DataFrame())
    assert stats["n"] == 0
    assert stats["win_rate"] == 0.0
    assert stats["mean_r"] == 0.0


def test_group_by_year() -> None:
    trades = _trades(
        [
            {
                "signal_date": pd.Timestamp("2022-03-01"),
                "pnl_r": 2.0,
                "exit_reason": ExitReason.TARGET_HIT.value,
                "mae_r": -0.5,
                "mfe_r": 2.1,
            },
            {
                "signal_date": pd.Timestamp("2023-06-15"),
                "pnl_r": -1.0,
                "exit_reason": ExitReason.STOP_HIT.value,
                "mae_r": -1.0,
                "mfe_r": 0.3,
            },
        ]
    )
    by_year = group_by_year(trades)
    assert {2022, 2023} <= set(by_year.index)


def test_group_by_regime() -> None:
    trades = _trades(
        [
            {
                "regime_at_signal": "bull_trend",
                "pnl_r": 2.0,
                "exit_reason": ExitReason.TARGET_HIT.value,
                "mae_r": -0.5,
                "mfe_r": 2.1,
            },
            {
                "regime_at_signal": "trading_range",
                "pnl_r": -1.0,
                "exit_reason": ExitReason.STOP_HIT.value,
                "mae_r": -1.0,
                "mfe_r": 0.3,
            },
        ]
    )
    by_regime = group_by_regime(trades)
    assert {"bull_trend", "trading_range"} <= set(by_regime.index)
