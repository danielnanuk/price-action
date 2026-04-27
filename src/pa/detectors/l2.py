"""L2 detector: two-legged rally in bear regime -> short entry on signal bar.

Algorithm (mirror of H2):
  1. Walk forward bar-by-bar. For each bar i in bear_trend regime:
     a. Look back to find a recent swing_low (within 30 bars).
     b. From that swing_low, identify first leg up: >= min_first_leg_bars
        consecutive higher closes ending at first high H1.
     c. Identify pullback: >= 2 bars of downward closes after H1.
     d. Identify second leg up: >= min_second_leg_bars higher closes
        ending at second high H2.
     e. Bar i must be the first bear bar after H2 with signal_bar_score
        >= min_signal_score AND H2 not more than
        max_second_high_above_first_high_atr ATRs above H1.
  2. Emit candidate: entry = low(i) - 1 tick, stop = high(i) + buffer*ATR,
     target = entry - R*target_r_multiple where R = stop - entry.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from pa.detectors.base import CANDIDATE_COLS, SetupParams, empty_candidates
from pa.types import Regime, Side


def detect_l2(bars: pd.DataFrame, params: SetupParams) -> pd.DataFrame:
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
    max_above = float(p["max_second_high_above_first_high_atr"])
    min_strength = float(p["regime_strength_min"])
    buffer_atr = float(p["stop_atr_buffer"])
    r_mult = float(p["target_r_multiple"])

    for i in range(min_first + min_second + 4, n):
        if regime[i] != Regime.BEAR_TREND.value:
            continue
        if regime_strength[i] < min_strength:
            continue
        if is_bull_bar[i]:  # need a bear bar for short signal
            continue
        if sig_score[i] < min_score:
            continue

        # Locate swing low within last 30 bars before i
        lookback_start = max(0, i - 30)
        swing_low_idx = lookback_start + int(lows[lookback_start:i].argmin())

        # First leg: bars after swing_low_idx with strictly higher closes
        first_leg_end = _walk_higher_closes(closes, start=swing_low_idx, max_idx=i - 1)
        if first_leg_end - swing_low_idx < min_first:
            continue
        h1 = highs[first_leg_end]

        # Pullback: at least 2 bars of lower closes
        pullback_end = _walk_lower_closes(closes, start=first_leg_end, max_idx=i - 1, min_bars=2)
        if pullback_end is None:
            continue

        # Second leg: higher closes after pullback, ending before bar i
        second_leg_end = _walk_higher_closes(closes, start=pullback_end, max_idx=i - 1)
        if second_leg_end - pullback_end < min_second:
            continue
        h2 = highs[second_leg_end]

        # H2 may exceed H1 by at most `max_above` ATRs
        atr_i = float(atr[i]) if not pd.isna(atr[i]) else 0.0
        if atr_i == 0:
            continue
        if (h2 - h1) / atr_i > max_above:
            continue

        # Bar i must be the FIRST bear bar after h2_end (i.e., second_leg_end + 1)
        if i != second_leg_end + 1:
            continue

        entry = float(lows[i]) - 0.01
        stop = float(highs[i]) + buffer_atr * atr_i
        risk = stop - entry
        target = entry - r_mult * risk

        out.append(
            {
                "ticker": ticker,
                "signal_date": dates[i],
                "side": Side.SHORT.value,
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


def _walk_higher_closes(closes: np.ndarray[Any, Any], *, start: int, max_idx: int) -> int:
    """Walk forward from start, returning index of last consecutive higher close."""
    i = start
    while i + 1 <= max_idx and closes[i + 1] > closes[i]:
        i += 1
    return i


def _walk_lower_closes(
    closes: np.ndarray[Any, Any], *, start: int, max_idx: int, min_bars: int
) -> int | None:
    """Walk forward, return end index after at least `min_bars` lower closes."""
    i = start
    count = 0
    while i + 1 <= max_idx and closes[i + 1] < closes[i]:
        i += 1
        count += 1
        if count >= min_bars:
            return i
    return None
