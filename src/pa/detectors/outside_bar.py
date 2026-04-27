"""Outside Bar reversal detector (long-only, bull OB at bottom).

Brooks's "bull outside bar at bottom": a single bar that fully engulfs
the prior bar's range (high higher, low lower), makes a new local low
intraday, and closes strongly near its high — the textbook V-shaped
intraday reversal.

Long-only by default (bull OB in bear regime — catching bottoms). Top
outside bars (bear OB at top) follow the same regime-mismatch pattern as
Bear Flag/Top Wedge in 2025-2026 mostly-bull window; not implemented yet.

Algorithm (bull OB at bottom):
  1. Bar i in mature bear_trend with regime_strength >= min:
       a. high[i] > high[i-1] AND low[i] < low[i-1] (engulfs prior)
       b. close[i] > open[i] (bull bar)
       c. range = high - low > min_range_atr x ATR (oversized)
       d. (close - low) / range > min_close_pos (close near high)
       e. low[i] = min(lows[i - lookback : i+1]) (intraday new local low)
  2. Entry: high[i] + 1 tick. Stop: low[i] - buffer x ATR. Target: 2R above.

Walk-forward results on daily 5y x S&P 500, scale_trail exit (best-fit):

  tier      IS                       OOS
  strict    +101R PF 2.47 N=239      +20R PF 1.75 N=94
  standard  +122R PF 1.61 N=529      +48R PF 1.65 N=219
  loose     +150R PF 1.38 N=958      +117R PF 1.83 N=405

Strongest single-setup contribution observed in this iteration: standard
OOS +48R (vs Wedge +13R, Climactic +13R).
"""

from __future__ import annotations

import pandas as pd

from pa.detectors.base import CANDIDATE_COLS, SetupParams, empty_candidates
from pa.types import Regime, Side


def detect_outside_bar(bars: pd.DataFrame, params: SetupParams) -> pd.DataFrame:
    out: list[dict[str, object]] = []
    p = params.thresholds
    n = len(bars)
    lookback = int(p["new_low_lookback"])
    if n < lookback + 5:
        return empty_candidates()

    h = bars["high"].to_numpy()
    lo = bars["low"].to_numpy()
    opens = bars["open"].to_numpy()
    closes = bars["close"].to_numpy()
    atr = bars["atr14"].to_numpy()
    regime = bars["regime"].to_numpy()
    regime_strength = bars["regime_strength"].to_numpy()
    sig_score = bars["signal_bar_score"].to_numpy()
    dates = bars["date"].to_numpy()
    ticker = bars["ticker"].iloc[0] if "ticker" in bars.columns else ""

    min_range_atr = float(p["min_range_atr"])
    min_close_pos = float(p["min_close_pos"])
    min_score = float(p["min_signal_score"])
    min_strength = float(p["regime_strength_min"])
    buf = float(p["stop_atr_buffer"])
    r_mult = float(p["target_r_multiple"])

    for i in range(lookback + 1, n):
        atr_i = float(atr[i]) if not pd.isna(atr[i]) else 0.0
        if atr_i == 0 or sig_score[i] < min_score:
            continue
        if regime[i] != Regime.BEAR_TREND.value:
            continue
        if regime_strength[i] < min_strength:
            continue

        # Engulfs prior bar's range
        if not (h[i] > h[i - 1] and lo[i] < lo[i - 1]):
            continue
        # Bull outside bar
        if closes[i] <= opens[i]:
            continue
        range_i = h[i] - lo[i]
        if range_i < min_range_atr * atr_i:
            continue
        # Close near high
        close_pos = (closes[i] - lo[i]) / range_i
        if close_pos < min_close_pos:
            continue
        # Intraday new local low — the engulfing low must be the lowest in the lookback
        window_low = lo[i - lookback : i].min()
        if lo[i] >= window_low:
            continue

        entry = float(h[i]) + 0.01
        stop = float(lo[i]) - buf * atr_i
        risk = entry - stop
        if risk <= 0:
            continue
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
                "regime_at_signal": str(regime[i]),
                "params_tier": params.tier.value,
            }
        )

    if not out:
        return empty_candidates()
    return pd.DataFrame(out, columns=CANDIDATE_COLS)
