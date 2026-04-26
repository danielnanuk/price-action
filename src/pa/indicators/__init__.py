"""Technical indicators for Brooks-style price action."""

from __future__ import annotations

import pandas as pd

from pa.indicators.atr import atr
from pa.indicators.bar_anatomy import bar_anatomy
from pa.indicators.ema import ema
from pa.indicators.swing import swing_pivot

__all__ = ["atr", "bar_anatomy", "compute_indicators", "ema", "swing_pivot"]


def compute_indicators(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Run all indicator computations on an OHLCV frame.

    Input columns required: date, open, high, low, close, volume, vwap.
    Returns a DataFrame with `date` plus all INDICATOR_COLS columns.
    """
    out = pd.DataFrame({"date": ohlcv["date"]})
    out["ema20"] = ema(ohlcv["close"], n=20)
    out["ema50"] = ema(ohlcv["close"], n=50)
    out["atr14"] = atr(ohlcv["high"], ohlcv["low"], ohlcv["close"], n=14)
    swings = swing_pivot(ohlcv["high"], ohlcv["low"], n=2)
    out["swing_high"] = swings["swing_high"]
    out["swing_low"] = swings["swing_low"]
    anatomy = bar_anatomy(ohlcv)
    for col in [
        "bar_body_pct",
        "bar_upper_wick_pct",
        "bar_lower_wick_pct",
        "bar_is_bull",
        "bar_close_position",
    ]:
        out[col] = anatomy[col]
    return out
