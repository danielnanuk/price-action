"""TCL (Trend Channel Line) overshoot reversal detector (long-only).

Brooks's "TCL overshoot": a trend channel line connects swing extremes
on the trend side (highs in bull, lows in bear) and acts as projected
support/resistance. A bar that briefly overshoots the line but closes
back inside = rejection signal → reversal.

Long-only (bottom TCL in bear regime). Top TCL (short) follows the
familiar regime-mismatch pattern in 2025-2026 mostly-bull window;
deferred. Long-side validates IS+OOS:

  tier      IS                       OOS
  strict    +52R PF 1.42 N=300       +2R PF 1.04 N=96
  standard  +69R PF 1.14 N=984       +31R PF 1.18 N=356
  loose     +48R PF 1.04 N=2148      +72R PF 1.18 N=809

Algorithm:
  1. Bar i in mature bear_trend (regime_strength >= min):
       Find the last 2 swing lows (B1, B2) in lookback window.
       Slope = (B2 - B1) / (B2_idx - B1_idx).
       Projected TCL at i: tcl_i = B2 + slope x (i - B2_idx).
  2. Overshoot: low[i] < tcl_i x (1 - min_overshoot_pct).
  3. Rejection: close[i] > tcl_i AND close[i] > open[i] (bull bar).
  4. Entry: high[i] + tick. Stop: low[i] - buffer x ATR.
     Target: 2R above entry.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pa.detectors.base import CANDIDATE_COLS, SetupParams, empty_candidates
from pa.indicators.swing import swing_pivot
from pa.types import Regime, Side


def detect_tcl(bars: pd.DataFrame, params: SetupParams) -> pd.DataFrame:
    out: list[dict[str, object]] = []
    p = params.thresholds
    n = len(bars)
    lookback = int(p["lookback_bars"])
    if n < lookback + 5:
        return empty_candidates()

    h = bars["high"].to_numpy()
    lo = bars["low"].to_numpy()
    closes = bars["close"].to_numpy()
    opens = bars["open"].to_numpy()
    atr = bars["atr14"].to_numpy()
    regime = bars["regime"].to_numpy()
    regime_strength = bars["regime_strength"].to_numpy()
    sig_score = bars["signal_bar_score"].to_numpy()
    dates = bars["date"].to_numpy()
    ticker = bars["ticker"].iloc[0] if "ticker" in bars.columns else ""

    swings = swing_pivot(bars["high"], bars["low"], n=int(p["swing_n"]))
    is_sl = swings["is_swing_low"].to_numpy()

    min_overshoot_pct = float(p["min_overshoot_pct"])
    min_score = float(p["min_signal_score"])
    min_strength = float(p["regime_strength_min"])
    buf = float(p["stop_atr_buffer"])
    r_mult = float(p["target_r_multiple"])

    for i in range(lookback, n):
        atr_i = float(atr[i]) if not pd.isna(atr[i]) else 0.0
        if atr_i == 0 or sig_score[i] < min_score:
            continue
        if regime[i] != Regime.BEAR_TREND.value:
            continue
        if regime_strength[i] < min_strength:
            continue

        window = slice(i - lookback, i)
        sl_indices = np.where(is_sl[window])[0] + (i - lookback)
        if len(sl_indices) < 2:
            continue
        b1_idx, b2_idx = sl_indices[-2], sl_indices[-1]
        if b2_idx == b1_idx:
            continue
        b1 = float(lo[b1_idx])
        b2 = float(lo[b2_idx])
        slope = (b2 - b1) / (b2_idx - b1_idx)
        tcl_i = b2 + slope * (i - b2_idx)
        if tcl_i <= 0:
            continue

        # Overshoot below the line
        if lo[i] > tcl_i * (1 - min_overshoot_pct):
            continue
        # Rejection: close back above the line AND bull bar
        if closes[i] <= tcl_i:
            continue
        if closes[i] <= opens[i]:
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
