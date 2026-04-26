"""Failed Breakout reversal: range breakout that closes back inside within K bars.

Algorithm (per spec section 6.4):
  - Look back range_lookback_bars to compute [range_low, range_high].
  - Range must be in trading_range regime AND at least min_range_atr_mult ATRs wide.
  - At bar i: detect breakout (close > range_high OR close < range_low).
  - Within next max_failure_bars, if a bar closes back inside the range:
    that bar is the failure confirmation.
  - Entry on next bar's open in opposite direction.
  - Stop on the breakout extreme + buffer*ATR.
  - Target = closer-of (2R, opposite range edge):
      short: target = max(entry - 2R, range_low)
      long:  target = min(entry + 2R, range_high)
"""

from __future__ import annotations

import pandas as pd

from pa.detectors.base import CANDIDATE_COLS, SetupParams, empty_candidates
from pa.types import Regime, Side


def detect_failed_breakout(bars: pd.DataFrame, params: SetupParams) -> pd.DataFrame:
    out: list[dict[str, object]] = []
    p = params.thresholds
    n = len(bars)
    if n < 30:
        return empty_candidates()

    h = bars["high"].to_numpy()
    lo = bars["low"].to_numpy()
    closes = bars["close"].to_numpy()
    opens = bars["open"].to_numpy()
    atr = bars["atr14"].to_numpy()
    regime = bars["regime"].to_numpy()
    regime_strength = bars["regime_strength"].to_numpy()
    dates = bars["date"].to_numpy()
    ticker = bars["ticker"].iloc[0] if "ticker" in bars.columns else ""

    rng_lookback = int(p["range_lookback_bars"])
    min_rng_atr = float(p["min_range_atr_mult"])
    max_fail_bars = int(p["max_failure_bars"])
    min_strength = float(p["regime_strength_min"])
    buf = float(p["stop_atr_buffer"])
    r_mult = float(p["target_r_multiple"])

    i = rng_lookback
    while i < n - max_fail_bars - 1:
        if regime[i] != Regime.TRADING_RANGE.value or regime_strength[i] < min_strength:
            i += 1
            continue
        atr_i = float(atr[i]) if not pd.isna(atr[i]) else 0.0
        if atr_i == 0:
            i += 1
            continue

        rng_lo = float(lo[i - rng_lookback : i].min())
        rng_hi = float(h[i - rng_lookback : i].max())
        if (rng_hi - rng_lo) < min_rng_atr * atr_i:
            i += 1
            continue

        emitted = False
        if closes[i] > rng_hi:  # upward breakout
            breakout_high = float(h[i])
            for k in range(1, max_fail_bars + 1):
                if i + k >= n - 1:
                    break
                if closes[i + k] < rng_hi:
                    confirm_idx = i + k
                    entry = float(opens[confirm_idx + 1])
                    stop = breakout_high + buf * atr_i
                    risk = stop - entry
                    if risk <= 0:
                        break
                    # SHORT: target is closer of (entry - 2R, range_low).
                    # max() picks the closer one (larger price = closer to entry).
                    target = max(entry - r_mult * risk, rng_lo)
                    out.append(
                        {
                            "ticker": ticker,
                            "signal_date": dates[confirm_idx + 1],
                            "side": Side.SHORT.value,
                            "entry_price": entry,
                            "stop_price": stop,
                            "target_price": target,
                            "setup_score": float(regime_strength[i]),
                            "regime_at_signal": regime[i],
                            "params_tier": params.tier.value,
                        }
                    )
                    i = confirm_idx + 2
                    emitted = True
                    break
        elif closes[i] < rng_lo:  # downward breakout
            breakout_low = float(lo[i])
            for k in range(1, max_fail_bars + 1):
                if i + k >= n - 1:
                    break
                if closes[i + k] > rng_lo:
                    confirm_idx = i + k
                    entry = float(opens[confirm_idx + 1])
                    stop = breakout_low - buf * atr_i
                    risk = entry - stop
                    if risk <= 0:
                        break
                    # LONG: target is closer of (entry + 2R, range_high).
                    # min() picks the closer one (smaller price = closer to entry).
                    target = min(entry + r_mult * risk, rng_hi)
                    out.append(
                        {
                            "ticker": ticker,
                            "signal_date": dates[confirm_idx + 1],
                            "side": Side.LONG.value,
                            "entry_price": entry,
                            "stop_price": stop,
                            "target_price": target,
                            "setup_score": float(regime_strength[i]),
                            "regime_at_signal": regime[i],
                            "params_tier": params.tier.value,
                        }
                    )
                    i = confirm_idx + 2
                    emitted = True
                    break

        if not emitted:
            i += 1

    if not out:
        return empty_candidates()
    return pd.DataFrame(out, columns=CANDIDATE_COLS)
