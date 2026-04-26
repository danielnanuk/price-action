"""Visual regression test — chart rendering must remain pixel-stable.

Generate baseline PNGs by running once with REGEN=1 env var.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["text.antialiased"] = False

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pa.report.chart import render_trade_chart  # noqa: E402
from PIL import Image  # noqa: E402

BASE_DIR = Path(__file__).parent / "fixtures" / "visual_baselines"


def _ohlcv() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 50
    closes = 100 + np.cumsum(rng.normal(0.1, 0.5, n))
    return pd.DataFrame(
        {
            "date": pd.date_range("2021-04-26", periods=n, freq="B"),
            "open": closes - 0.1,
            "high": closes + 0.5,
            "low": closes - 0.5,
            "close": closes,
            "volume": [1_000_000] * n,
            "vwap": closes,
        }
    )


def _pixel_diff_pct(a: Path, b: Path) -> float:
    img_a = np.asarray(Image.open(a).convert("RGB"))
    img_b = np.asarray(Image.open(b).convert("RGB"))
    if img_a.shape != img_b.shape:
        return 1.0
    return float((img_a != img_b).mean())


def test_h2_textbook_chart_pixel_stable(tmp_path: Path) -> None:
    ohlcv = _ohlcv()
    trade = {
        "ticker": "TEST",
        "signal_date": pd.Timestamp("2021-05-25"),
        "entry_date": pd.Timestamp("2021-05-26"),
        "entry_price": float(ohlcv.iloc[20]["close"] + 0.5),
        "stop_price": float(ohlcv.iloc[20]["close"] - 1.5),
        "target_price": float(ohlcv.iloc[20]["close"] + 4.5),
        "exit_date": pd.Timestamp("2021-06-04"),
        "exit_price": float(ohlcv.iloc[20]["close"] + 4.5),
        "exit_reason": "target_hit",
        "side": "long",
        "pnl_r": 2.0,
    }
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    baseline = BASE_DIR / "h2_textbook.png"
    candidate = tmp_path / "h2_textbook.png"
    render_trade_chart(trade=trade, ohlcv=ohlcv, out_path=candidate, lookback=15, lookforward=10)

    if os.environ.get("REGEN") == "1" or not baseline.exists():
        candidate.replace(baseline)
        return  # baseline regenerated; treat as pass

    diff = _pixel_diff_pct(candidate, baseline)
    assert diff < 0.01, f"Chart drift: {diff:.4%}"
