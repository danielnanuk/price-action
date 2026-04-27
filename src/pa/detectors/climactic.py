"""Climactic reversal detector (long-only, bottom climax).

Brooks's "climactic reversal" pattern: a single bar of panic/euphoria —
oversized, full-bodied, in trend direction, making a new local extreme —
followed by a clear reversal bar. Captures the moment of trend exhaustion
in a single decisive K-line, distinct from Wedge's 3-push accumulation.

Long-only by default (bottom climax in bear regime). Top climax (short
side) had IS edge that broke OOS in 2025-2026 mostly-bull regime — same
pattern as Bear Flag and Top Wedge. Shorts retired here; revive when
bear cycles return to the dataset.

Algorithm (bottom climax, the live path):
  1. Bar i-1 (the climax): in mature bear_trend, AND
       a. range = high - low > min_climax_atr_mult x ATR (oversized)
       b. body = |close - open| / range > min_body_pct (full-bodied)
       c. close in lower (1 - min_close_pos) of range (close near low)
       d. low[i-1] = min(lows[i-1-N : i]) (new local low)
       e. close < open (bear bar in panic)
  2. Bar i (the reversal): bull bar AND close > close[i-1] AND
     signal_bar_score >= min_signal_score (real rejection of climax low).
  3. Entry: high[i] + 1 tick. Stop: low[i-1] - buffer x ATR.
     Target: 2R above entry (default; per_setup.yaml uses scale_trail).

Walk-forward results on daily 5y x S&P 500, baseline exit:

  tier      IS                       OOS
  strict    -3R PF 0.59 N=11         -1R PF 0.69 N=12
  standard  +22R PF 1.26 N=184       +13R PF 1.23 N=146
  loose     +48R PF 1.08 N=1253      +74R PF 1.23 N=696

Standard and loose tiers IS+OOS validated — long climactic is robust.
Strict has too few candidates (22 total). Standard contributes ~+12R
to OOS portfolio.
"""

from __future__ import annotations

import pandas as pd

from pa.detectors.base import CANDIDATE_COLS, SetupParams, empty_candidates
from pa.types import Regime, Side


def detect_climactic(bars: pd.DataFrame, params: SetupParams) -> pd.DataFrame:
    out: list[dict[str, object]] = []
    p = params.thresholds
    n = len(bars)
    lookback = int(p["new_extreme_lookback"])
    if n < lookback + 5:
        return empty_candidates()

    h = bars["high"].to_numpy()
    lo = bars["low"].to_numpy()
    opens = bars["open"].to_numpy()
    closes = bars["close"].to_numpy()
    atr = bars["atr14"].to_numpy()
    regime = bars["regime"].to_numpy()
    regime_strength = bars["regime_strength"].to_numpy()
    is_bull_bar = bars["bar_is_bull"].to_numpy()
    sig_score = bars["signal_bar_score"].to_numpy()
    dates = bars["date"].to_numpy()
    ticker = bars["ticker"].iloc[0] if "ticker" in bars.columns else ""

    min_climax_atr = float(p["min_climax_atr_mult"])
    min_body_pct = float(p["min_body_pct"])
    min_close_pos = float(p["min_close_pos"])
    min_score = float(p["min_signal_score"])
    min_strength = float(p["regime_strength_min"])
    buf = float(p["stop_atr_buffer"])
    r_mult = float(p["target_r_multiple"])

    for i in range(lookback + 1, n):
        atr_i = float(atr[i]) if not pd.isna(atr[i]) else 0.0
        if atr_i == 0 or sig_score[i] < min_score:
            continue
        if regime_strength[i] < min_strength:
            continue

        ci = i - 1  # climax bar index (the previous one)

        # Long-only: bottom climax in bear regime, climax is bear bar, reversal is bull bar
        if regime[ci] != Regime.BEAR_TREND.value:
            continue
        if is_bull_bar[ci]:  # climax must be bear in panic
            continue
        if not is_bull_bar[i]:  # reversal must be bull
            continue

        c_range = h[ci] - lo[ci]
        if c_range <= 0:
            continue
        c_body_pct = abs(closes[ci] - opens[ci]) / c_range
        c_close_pos = (closes[ci] - lo[ci]) / c_range  # 0 = at low, 1 = at high

        if c_range < min_climax_atr * atr_i:
            continue
        if c_body_pct < min_body_pct:
            continue
        # Bear panic: close near low → close_pos < 1 - min_close_pos
        if c_close_pos > 1 - min_close_pos:
            continue
        # New local low
        window_low = lo[ci - lookback : ci].min()
        if lo[ci] >= window_low:
            continue
        # Reversal bar must clearly reject the climax low
        if closes[i] <= closes[ci]:
            continue

        entry = float(h[i]) + 0.01
        stop = float(lo[ci]) - buf * atr_i
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
