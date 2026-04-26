"""Tests for pa.report.chart — matplotlib annotated K-line chart."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless

import pandas as pd
from pa.report.chart import render_trade_chart


def _ohlcv() -> pd.DataFrame:
    dates = pd.date_range("2021-04-26", periods=30, freq="B")
    closes = [100 + i for i in range(30)]
    return pd.DataFrame(
        {
            "date": dates,
            "open": closes,
            "high": [c + 1 for c in closes],
            "low": [c - 1 for c in closes],
            "close": closes,
            "volume": [1_000_000] * 30,
            "vwap": closes,
        }
    )


def test_render_creates_png(tmp_path: Path) -> None:
    ohlcv = _ohlcv()
    trade = {
        "ticker": "TEST",
        "signal_date": pd.Timestamp("2021-05-10"),
        "entry_date": pd.Timestamp("2021-05-11"),
        "entry_price": 110.5,
        "stop_price": 109.0,
        "target_price": 113.5,
        "exit_date": pd.Timestamp("2021-05-15"),
        "exit_price": 113.5,
        "exit_reason": "target_hit",
        "side": "long",
        "pnl_r": 2.0,
    }
    out_path = tmp_path / "trade.png"
    render_trade_chart(trade=trade, ohlcv=ohlcv, out_path=out_path, lookback=15)
    assert out_path.exists()
    assert out_path.stat().st_size > 1000  # non-empty PNG
