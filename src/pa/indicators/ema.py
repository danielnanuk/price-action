"""Exponential moving average — the only smoothing primitive used in pa."""

from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, *, n: int) -> pd.Series:
    """Standard EMA with span=n; first n-1 values are NaN.

    Brooks uses EMA20 / EMA50 to determine trend regime.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    out = series.ewm(span=n, adjust=False).mean()
    out.iloc[: n - 1] = np.nan
    return out
