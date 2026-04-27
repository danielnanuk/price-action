"""Wilder's ATR — used for stop placement and volatility-aware thresholds."""

from __future__ import annotations

import numpy as np
import pandas as pd


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    *,
    n: int = 14,
) -> pd.Series:
    """Wilder's Average True Range. Output has NaN for indices 0..n-2."""
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    out = pd.Series(np.nan, index=high.index)
    if len(tr) < n:
        return out
    # Wilder seed: mean of first n TR values; place at index n-1.
    seed = tr.iloc[:n].mean()
    out.iloc[n - 1] = seed
    for i in range(n, len(tr)):
        prev_atr = out.iloc[i - 1]
        out.iloc[i] = (prev_atr * (n - 1) + tr.iloc[i]) / n
    # First TR has no prev_close so seed used TR[1..n-1]; mask index 0 NaN explicitly.
    out.iloc[: n - 1] = np.nan
    return out
