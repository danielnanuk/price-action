"""Wedge / 3-push reversal detector.

Brooks's wedge: 3 progressive higher-highs (top wedge) or lower-lows (bottom
wedge), with each push smaller than the prior. The 3rd push fails -> reversal.

Long-only by default (bottom wedge in bear-trend). Top wedge had IS edge
in the 2021-2024 daily 5y backtest (PF 1.07-1.28) but FAILED OOS in
2025-2026 (PF 0.65-0.87) — same regime-mismatch pattern as Bear Flag,
since 2025+ has been mostly bullish and short reversals get squeezed.
Top wedge code path retired here; revive once a real bear cycle enters
the dataset.

Algorithm (bottom wedge, the live path):
  1. For each bar i in mature bear_trend (regime_strength >= min):
     a. Look back lookback_bars; collect the last 3 swing lows B1 > B2 > B3
        (strictly decreasing) and the swing highs H1, H2 between them.
     b. Push amplitudes:
          push1 = H1 - B2  (height of the rally between B2 and B1)
          push2 = H2 - B3  (height of the rally between B3 and B2)
        Require push2 / push1 <= max_push_ratio (real momentum decay).
     c. Combined push significance: push1 + push2 >= min_pushes_total_atr * ATR
        (ensures the wedge sits inside a meaningful trend, not chop).
     d. Signal bar i: bull bar within 1-3 bars of B3, close > B3 (rejection).
  2. Entry: high(i) + 1 tick. Stop: B3 - buffer * ATR. Target: 2R above entry.

Validated on daily 5y x S&P 500 with walk-forward (split 2025-01-01):

  tier      IS                       OOS
  strict    +37R PF 1.40 N=227       +5.5R PF 1.13 N=96
  standard  +86R PF 1.28 N=718       +13.4R PF 1.12 N=248
  loose     +109R PF 1.15 N=1696     +9.4R PF 1.04 N=560
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pa.detectors.base import CANDIDATE_COLS, SetupParams, empty_candidates
from pa.indicators.swing import swing_pivot
from pa.types import Regime, Side


def detect_wedge(bars: pd.DataFrame, params: SetupParams) -> pd.DataFrame:
    out: list[dict[str, object]] = []
    p = params.thresholds
    n = len(bars)
    lookback = int(p["lookback_bars"])
    if n < lookback + 5:
        return empty_candidates()

    h = bars["high"].to_numpy()
    lo = bars["low"].to_numpy()
    closes = bars["close"].to_numpy()
    atr = bars["atr14"].to_numpy()
    regime = bars["regime"].to_numpy()
    regime_strength = bars["regime_strength"].to_numpy()
    is_bull_bar = bars["bar_is_bull"].to_numpy()
    sig_score = bars["signal_bar_score"].to_numpy()
    dates = bars["date"].to_numpy()
    ticker = bars["ticker"].iloc[0] if "ticker" in bars.columns else ""

    swings = swing_pivot(bars["high"], bars["low"], n=int(p["swing_n"]))
    is_sh = swings["is_swing_high"].to_numpy()
    is_sl = swings["is_swing_low"].to_numpy()

    max_push_ratio = float(p["max_push_ratio"])
    min_pushes_total_atr = float(p["min_pushes_total_atr"])
    min_score = float(p["min_signal_score"])
    min_strength = float(p["regime_strength_min"])
    buf = float(p["stop_atr_buffer"])
    r_mult = float(p["target_r_multiple"])

    for i in range(lookback, n):
        atr_i = float(atr[i]) if not pd.isna(atr[i]) else 0.0
        if atr_i == 0 or sig_score[i] < min_score:
            continue
        if regime_strength[i] < min_strength:
            continue

        # Long-only: bottom wedge in bear regime, bull signal bar.
        if regime[i] != Regime.BEAR_TREND.value or not is_bull_bar[i]:
            continue

        window = slice(i - lookback, i)
        sl_indices = np.where(is_sl[window])[0] + (i - lookback)
        sh_indices = np.where(is_sh[window])[0] + (i - lookback)
        if len(sl_indices) < 3:
            continue

        b3_idx, b2_idx, b1_idx = sl_indices[-1], sl_indices[-2], sl_indices[-3]
        b1 = float(lo[b1_idx])
        b2 = float(lo[b2_idx])
        b3 = float(lo[b3_idx])
        if not (b1 > b2 > b3):
            continue

        sh_between_12 = sh_indices[(sh_indices > b1_idx) & (sh_indices < b2_idx)]
        sh_between_23 = sh_indices[(sh_indices > b2_idx) & (sh_indices < b3_idx)]
        if len(sh_between_12) == 0 or len(sh_between_23) == 0:
            continue

        h1_idx = sh_between_12[-1]
        h2_idx = sh_between_23[-1]
        h1 = float(h[h1_idx])
        h2 = float(h[h2_idx])
        push1 = h1 - b2
        push2 = h2 - b3
        if push1 <= 0 or push2 <= 0:
            continue
        if push2 / push1 > max_push_ratio:
            continue
        if push1 + push2 < min_pushes_total_atr * atr_i:
            continue

        # Signal bar must be within 1-3 bars of B3 and close above B3
        if i - b3_idx > 3 or i - b3_idx < 1:
            continue
        if closes[i] <= b3:
            continue

        entry = float(h[i]) + 0.01
        stop = b3 - buf * atr_i
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
