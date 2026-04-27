"""Brooks-style swing high/low pivots with N-bar confirmation on each side."""

from __future__ import annotations

import numpy as np
import pandas as pd


def swing_pivot(
    high: pd.Series,
    low: pd.Series,
    *,
    n: int = 2,
) -> pd.DataFrame:
    """Compute swing pivots.

    A bar at index i is a swing high if `high[i]` is strictly greater than
    every `high` value in `[i-n, i-1]` and `[i+1, i+n]`. Symmetric for low.

    Returns DataFrame with:
      - is_swing_high (bool): True at the bar that *is* the swing
      - is_swing_low (bool)
      - swing_high (float): price level of most recent confirmed swing high,
                            forward-filled from confirmation bar (i+n)
      - swing_low (float): symmetric

    Confirmation: a swing at bar i is only "known" at bar i+n (we need n future
    bars to validate). The level is therefore forward-filled starting at i+n.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    h = high.to_numpy()
    lo = low.to_numpy()
    size = len(h)
    is_sh = np.zeros(size, dtype=bool)
    is_sl = np.zeros(size, dtype=bool)
    for i in range(n, size - n):
        left_h = h[i - n : i]
        right_h = h[i + 1 : i + n + 1]
        if h[i] > left_h.max() and h[i] > right_h.max():
            is_sh[i] = True
        left_l = lo[i - n : i]
        right_l = lo[i + 1 : i + n + 1]
        if lo[i] < left_l.min() and lo[i] < right_l.min():
            is_sl[i] = True

    sh_level = np.full(size, np.nan)
    sl_level = np.full(size, np.nan)
    last_sh = np.nan
    last_sl = np.nan
    for i in range(size):
        # confirmed swing at index i-n becomes available at index i
        confirm_idx = i - n
        if confirm_idx >= 0:
            if is_sh[confirm_idx]:
                last_sh = h[confirm_idx]
            if is_sl[confirm_idx]:
                last_sl = lo[confirm_idx]
        sh_level[i] = last_sh
        sl_level[i] = last_sl

    return pd.DataFrame(
        {
            "is_swing_high": pd.Series(is_sh, index=high.index),
            "is_swing_low": pd.Series(is_sl, index=high.index),
            "swing_high": pd.Series(sh_level, index=high.index),
            "swing_low": pd.Series(sl_level, index=high.index),
        }
    )
