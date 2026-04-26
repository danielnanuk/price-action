"""Bull/Bear Flag detector: impulse leg + tight consolidation + breakout.

Algorithm (per spec section 6.3, bull side; bear is symmetric):
  - Find impulse leg ending at impulse_end: >= min_impulse_bars consecutive
    bull bars OR a stretch where (high.max() - low.min()) > min_impulse_atr_mult * ATR.
  - Identify consolidation: next >= min_consolidation_bars whose total range
    < impulse_size * max_consolidation_range_ratio.
  - Breakout: a subsequent bar's close > consolidation high (bull) or < low (bear).
  - Entry on next bar's open; stop = consolidation low - buffer*ATR (bull);
    target = entry + min(impulse_size, target_r_cap * R).
"""

from __future__ import annotations

import pandas as pd

from pa.detectors.base import CANDIDATE_COLS, SetupParams, empty_candidates
from pa.types import Regime, Side


def detect_flag(bars: pd.DataFrame, params: SetupParams) -> pd.DataFrame:
    out: list[dict[str, object]] = []
    p = params.thresholds
    n = len(bars)
    if n < 20:
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

    min_imp = int(p["min_impulse_bars"])
    min_imp_atr = float(p["min_impulse_atr_mult"])
    min_cons = int(p["min_consolidation_bars"])
    max_cons_ratio = float(p["max_consolidation_range_ratio"])
    min_strength = float(p["regime_strength_min"])
    buf = float(p["stop_atr_buffer"])
    r_cap = float(p["target_r_cap"])

    for i in range(min_imp + min_cons + 1, n - 1):
        cur_regime = regime[i]
        if cur_regime not in (Regime.BULL_TREND.value, Regime.BEAR_TREND.value):
            continue
        if regime_strength[i] < min_strength:
            continue

        side = Side.LONG if cur_regime == Regime.BULL_TREND.value else Side.SHORT
        atr_i = float(atr[i]) if not pd.isna(atr[i]) else 0.0
        if atr_i == 0:
            continue

        # Consolidation block: the min_cons bars BEFORE the breakout bar i,
        # i.e., bars [i - min_cons, i - 1]. Bar i itself is the breakout bar.
        cons_lo = float(lo[i - min_cons : i].min())
        cons_hi = float(h[i - min_cons : i].max())
        cons_range = cons_hi - cons_lo

        # Impulse leg: bars [i - min_cons - min_imp, i - min_cons - 1] (precedes cons).
        imp_start = i - min_cons - min_imp
        imp_end = i - min_cons - 1
        imp_lo = float(lo[imp_start : imp_end + 1].min())
        imp_hi = float(h[imp_start : imp_end + 1].max())
        imp_size = imp_hi - imp_lo

        if imp_size < min_imp_atr * atr_i:
            continue
        if cons_range > imp_size * max_cons_ratio:
            continue

        # Breakout check at bar i: close beyond consolidation in trend direction
        if side == Side.LONG:
            if not (closes[i] > cons_hi):
                continue
            entry = float(opens[i + 1])
            stop = cons_lo - buf * atr_i
            risk = entry - stop
            if risk <= 0:
                continue
            target = entry + min(imp_size, r_cap * risk)
        else:
            if not (closes[i] < cons_lo):
                continue
            entry = float(opens[i + 1])
            stop = cons_hi + buf * atr_i
            risk = stop - entry
            if risk <= 0:
                continue
            target = entry - min(imp_size, r_cap * risk)

        out.append(
            {
                "ticker": ticker,
                "signal_date": dates[i + 1],  # next bar = entry day
                "side": side.value,
                "entry_price": entry,
                "stop_price": stop,
                "target_price": target,
                "setup_score": float(regime_strength[i]),
                "regime_at_signal": cur_regime,
                "params_tier": params.tier.value,
            }
        )

    if not out:
        return empty_candidates()
    return pd.DataFrame(out, columns=CANDIDATE_COLS)
