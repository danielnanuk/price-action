"""Tests for HTML report generation (smoke test only — exact markup not asserted)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pa.report.html import build_report


def _trades() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "signal_date": pd.Timestamp("2022-03-01"),
                "entry_date": pd.Timestamp("2022-03-02"),
                "entry_price": 165.0,
                "stop_price": 163.0,
                "target_price": 169.0,
                "exit_date": pd.Timestamp("2022-03-08"),
                "exit_price": 169.0,
                "exit_reason": "target_hit",
                "pnl_r": 2.0,
                "pnl_pct": 0.024,
                "mae_r": -0.4,
                "mfe_r": 2.1,
                "days_held": 6,
                "regime_at_signal": "bull_trend",
                "same_bar_ambiguous": False,
                "params_tier": "standard",
                "side": "long",
            }
        ]
    )


def _ohlcv() -> pd.DataFrame:
    dates = pd.date_range("2022-02-01", periods=40, freq="B")
    return pd.DataFrame(
        {
            "date": dates,
            "open": [165.0] * 40,
            "high": [166.0] * 40,
            "low": [164.0] * 40,
            "close": [165.5] * 40,
            "volume": [1_000_000] * 40,
            "vwap": [165.0] * 40,
        }
    )


def test_build_report_creates_index_and_setup_pages(tmp_path: Path) -> None:
    out = build_report(
        run_id="test_run",
        trades_by_setup_tier={("h2", "standard"): _trades()},
        ohlcv_by_ticker={"AAPL": _ohlcv()},
        out_dir=tmp_path,
        universe="sp500",
        date_start="2021-04-26",
        date_end="2026-04-26",
        samples_per_setup=10,
    )
    assert (out / "index.html").exists()
    assert (out / "h2_standard.html").exists()
    assert (out / "samples").is_dir()
    assert (out / "data.parquet").exists()
