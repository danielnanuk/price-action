"""Trend regime classifier using EMA stack + structural HH/HL checks.

Standard tier rules (loose/strict adjust thresholds, not the rule shape):
- bull_trend: close > EMA20 > EMA50, EMA20 slope rising over last 10 bars,
  AND price made at least one HH+HL in last 20 bars
- bear_trend: mirror
- trading_range: |close - EMA20| <= 1.5 * ATR for majority of last 20 bars
- transitional: warmup bars (before EMA50 valid) or borderline conditions
"""

from __future__ import annotations

import pandas as pd

from pa.types import Regime


def classify_regime(
    ohlcv: pd.DataFrame,
    indicators: pd.DataFrame,
    *,
    slope_window: int = 10,
    structure_window: int = 20,
    range_atr_mult: float = 1.5,
) -> pd.DataFrame:
    """Return DataFrame with `regime` (StrEnum value) and `regime_strength` (0..1)."""
    df = pd.DataFrame({"date": ohlcv["date"]})
    close = ohlcv["close"]
    ema20 = indicators["ema20"]
    ema50 = indicators["ema50"]
    atr14 = indicators["atr14"]

    ema20_slope = ema20.diff(slope_window)

    rolling_hh = ohlcv["high"].rolling(structure_window, min_periods=1).max()
    rolling_ll = ohlcv["low"].rolling(structure_window, min_periods=1).min()
    made_higher_high = ohlcv["high"] >= rolling_hh.shift(1)
    made_lower_low = ohlcv["low"] <= rolling_ll.shift(1)
    hh_count = made_higher_high.rolling(structure_window, min_periods=1).sum()
    ll_count = made_lower_low.rolling(structure_window, min_periods=1).sum()

    is_bull = (close > ema20) & (ema20 > ema50) & (ema20_slope > 0) & (hh_count >= 1)
    is_bear = (close < ema20) & (ema20 < ema50) & (ema20_slope < 0) & (ll_count >= 1)
    band = atr14 * range_atr_mult
    in_range_band = (close - ema20).abs() <= band
    range_score = in_range_band.rolling(structure_window, min_periods=1).mean()
    is_range = (~is_bull) & (~is_bear) & (range_score > 0.6)

    regime = pd.Series(Regime.TRANSITIONAL.value, index=close.index, dtype=object)
    regime[is_bull] = Regime.BULL_TREND.value
    regime[is_bear] = Regime.BEAR_TREND.value
    regime[is_range] = Regime.TRADING_RANGE.value
    # Force warmup (pre-EMA50) into TRANSITIONAL — prevents false signals on NaN.
    warmup = ema50.isna() | atr14.isna()
    regime[warmup] = Regime.TRANSITIONAL.value

    # Strength: fraction of the last `structure_window` bars sharing this bar's regime.
    arr = regime.to_numpy()
    strength_vals: list[float] = []
    for idx in range(len(arr)):
        start = max(0, idx - structure_window + 1)
        win = arr[start : idx + 1]
        cur = arr[idx]
        strength_vals.append(float((win == cur).mean()))

    df["regime"] = regime.values
    df["regime_strength"] = strength_vals
    return df
