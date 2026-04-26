"""H2 detector: two-legged pullback in bull regime -> long entry on signal bar.

Algorithm (per spec section 6.1):
  1. Walk forward bar-by-bar. For each bar i in bull_trend regime:
     a. Look back to find a recent swing_high (within 30 bars).
     b. From that swing_high, identify first leg down: >= min_first_leg_bars
        consecutive lower closes ending at first low L1.
     c. Identify rebound: >= 2 bars of upward closes after L1.
     d. Identify second leg down: >= min_second_leg_bars lower closes
        ending at second low L2.
     e. Bar i must be the first bull bar after L2 with signal_bar_score
        >= min_signal_score AND L2 not more than max_second_low_below_first_low_atr
        ATRs below L1.
  2. Emit candidate: entry = high(i) + 1 tick, stop = low(i) - buffer*ATR,
     target = entry + R*target_r_multiple where R = entry - stop.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from pa.detectors.base import CANDIDATE_COLS, SetupParams, empty_candidates
from pa.types import Regime, Side


def detect_h2(bars: pd.DataFrame, params: SetupParams) -> pd.DataFrame:
    out: list[dict[str, object]] = []
    p = params.thresholds
    n = len(bars)
    if n < 5:
        return empty_candidates()

    closes = bars["close"].to_numpy()
    highs = bars["high"].to_numpy()
    lows = bars["low"].to_numpy()
    atr = bars["atr14"].to_numpy()
    regime = bars["regime"].to_numpy()
    regime_strength = bars["regime_strength"].to_numpy()
    is_bull_bar = bars["bar_is_bull"].to_numpy()
    sig_score = bars["signal_bar_score"].to_numpy()
    dates = bars["date"].to_numpy()
    ticker = bars["ticker"].iloc[0] if "ticker" in bars.columns else ""

    min_first = int(p["min_first_leg_bars"])
    min_second = int(p["min_second_leg_bars"])
    min_score = float(p["min_signal_score"])
    max_below = float(p["max_second_low_below_first_low_atr"])
    min_strength = float(p["regime_strength_min"])
    buffer_atr = float(p["stop_atr_buffer"])
    r_mult = float(p["target_r_multiple"])

    for i in range(min_first + min_second + 4, n):
        if regime[i] != Regime.BULL_TREND.value:
            continue
        if regime_strength[i] < min_strength:
            continue
        if not is_bull_bar[i]:
            continue
        if sig_score[i] < min_score:
            continue

        # Locate swing high within last 30 bars before i
        lookback_start = max(0, i - 30)
        window_high_idx = lookback_start + int(highs[lookback_start:i].argmax())
        # First leg: bars after window_high_idx with strictly lower closes
        first_leg_end = _walk_lower_closes(closes, start=window_high_idx, max_idx=i - 1)
        if first_leg_end - window_high_idx < min_first:
            continue
        l1 = lows[first_leg_end]

        # Rebound: at least 2 bars of higher closes
        rebound_end = _walk_higher_closes(closes, start=first_leg_end, max_idx=i - 1, min_bars=2)
        if rebound_end is None:
            continue

        # Second leg: lower closes after rebound, ending before bar i
        second_leg_end = _walk_lower_closes(closes, start=rebound_end, max_idx=i - 1)
        if second_leg_end - rebound_end < min_second:
            continue
        l2 = lows[second_leg_end]

        # L2 may dip below L1 by at most `max_below` ATRs
        atr_i = float(atr[i]) if not pd.isna(atr[i]) else 0.0
        if atr_i == 0:
            continue
        if (l1 - l2) / atr_i > max_below:
            continue

        # Bar i must be the FIRST bull bar after l2_end (i.e., second_leg_end + 1)
        if i != second_leg_end + 1:
            continue

        entry = float(highs[i]) + 0.01
        stop = float(lows[i]) - buffer_atr * atr_i
        risk = entry - stop
        target = entry + r_mult * risk

        out.append(
            {
                "ticker": ticker,
                "signal_date": dates[i],
                "side": Side.LONG.value,
                "entry_price": entry,
                "stop_price": stop,
                "target_price": target,
                "setup_score": float(sig_score[i]),
                "regime_at_signal": regime[i],
                "params_tier": params.tier.value,
            }
        )

    if not out:
        return empty_candidates()
    return pd.DataFrame(out, columns=CANDIDATE_COLS)


def _walk_lower_closes(closes: np.ndarray[Any, Any], *, start: int, max_idx: int) -> int:
    """Walk forward from start, returning index of last consecutive lower close."""
    i = start
    while i + 1 <= max_idx and closes[i + 1] < closes[i]:
        i += 1
    return i


def _walk_higher_closes(
    closes: np.ndarray[Any, Any], *, start: int, max_idx: int, min_bars: int
) -> int | None:
    """Walk forward, return end index after at least `min_bars` higher closes."""
    i = start
    count = 0
    while i + 1 <= max_idx and closes[i + 1] > closes[i]:
        i += 1
        count += 1
        if count >= min_bars:
            return i
    return None
