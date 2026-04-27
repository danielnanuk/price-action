"""Per-bar geometric features: body, wicks, close position, bull/bear."""

from __future__ import annotations

import numpy as np
import pandas as pd


def bar_anatomy(df: pd.DataFrame) -> pd.DataFrame:
    """Add Brooks-style per-bar geometry columns.

    Required input columns: open, high, low, close.
    Returns a DataFrame indexed identically to df with anatomy columns only
    (caller decides whether to concat).
    """
    o, h, lo, c = df["open"], df["high"], df["low"], df["close"]
    bar_range = (h - lo).replace(0.0, np.nan)
    body = (c - o).abs()
    upper_wick = h - c.where(c >= o, o)
    lower_wick = o.where(c >= o, c) - lo

    body_pct = (body / bar_range).fillna(0.0)
    upper_pct = (upper_wick / bar_range).fillna(0.0)
    lower_pct = (lower_wick / bar_range).fillna(0.0)
    close_pos = ((c - lo) / bar_range).fillna(0.5)

    return pd.DataFrame(
        {
            "bar_body_pct": body_pct,
            "bar_upper_wick_pct": upper_pct,
            "bar_lower_wick_pct": lower_pct,
            "bar_is_bull": (c > o),
            "bar_close_position": close_pos,
        },
        index=df.index,
    )
