"""Quality score in [0, 1] for a bar to act as a Brooks 'signal bar'.

A good signal bar (bull or bear) has:
  - Large body relative to range (50%+ ideal)
  - Close in the upper third (bull) / lower third (bear) of range
  - Body size > recent average (bar of significance)
"""

from __future__ import annotations

import pandas as pd


def signal_bar_score(
    ohlcv: pd.DataFrame,
    indicators: pd.DataFrame,
) -> pd.Series:
    body_pct = indicators["bar_body_pct"]
    close_pos = indicators["bar_close_position"]
    is_bull = indicators["bar_is_bull"]

    # Direction-aware close position: bull → close near top, bear → close near bottom
    directional_pos = close_pos.where(is_bull, 1.0 - close_pos)

    # Size relative to recent volatility: body / ATR14
    body_abs = (ohlcv["close"] - ohlcv["open"]).abs()
    size_ratio = (body_abs / indicators["atr14"]).clip(upper=1.0).fillna(0.0)

    score = (0.4 * body_pct) + (0.4 * directional_pos) + (0.2 * size_ratio)
    return score.clip(lower=0.0, upper=1.0)
