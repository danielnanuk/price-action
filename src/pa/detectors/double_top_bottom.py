"""Double Top / Double Bottom reversal detector.

Algorithm (per spec section 6.5, double top; bottom symmetric):
  - At bar i, look back lookback_bars to find P1 (highest swing high).
  - Confirm at least min_pullback_bars after P1 with a low M where
    P1 - M >= min_pullback_atr_mult * ATR.
  - Confirm P2: a later swing high with |P2 - P1|/P1 < max_peak_diff_pct.
  - Bar i must be the first bear bar after P2 with signal_bar_score >= min.
  - Entry: short at lo(i) - 1 tick. Stop: max(P1, P2) + buffer*ATR.
  - Target: entry - r_mult * (stop - entry).
"""

from __future__ import annotations

import pandas as pd

from pa.detectors.base import CANDIDATE_COLS, SetupParams, empty_candidates
from pa.types import Side


def detect_double_top_bottom(bars: pd.DataFrame, params: SetupParams) -> pd.DataFrame:
    out: list[dict[str, object]] = []
    p = params.thresholds
    n = len(bars)
    if n < 40:
        return empty_candidates()

    h = bars["high"].to_numpy()
    lo = bars["low"].to_numpy()
    atr = bars["atr14"].to_numpy()
    is_bull_bar = bars["bar_is_bull"].to_numpy()
    sig_score = bars["signal_bar_score"].to_numpy()
    dates = bars["date"].to_numpy()
    ticker = bars["ticker"].iloc[0] if "ticker" in bars.columns else ""

    lookback = int(p["lookback_bars"])
    min_pb_bars = int(p["min_pullback_bars"])
    min_pb_atr = float(p["min_pullback_atr_mult"])
    max_diff = float(p["max_peak_diff_pct"])
    min_score = float(p["min_signal_score"])
    buf = float(p["stop_atr_buffer"])
    r_mult = float(p["target_r_multiple"])

    for i in range(lookback, n):
        atr_i = float(atr[i]) if not pd.isna(atr[i]) else 0.0
        if atr_i == 0 or sig_score[i] < min_score:
            continue
        window = slice(i - lookback, i)

        # ---- Double Top detection ----
        if not is_bull_bar[i]:
            window_h = h[window]
            p1_idx_local = int(window_h.argmax())
            p1_idx = i - lookback + p1_idx_local
            p1 = float(h[p1_idx])

            # Look for pullback after p1
            after_p1 = slice(p1_idx + 1, i)
            if i - p1_idx > min_pb_bars:
                m_idx_local = int(lo[after_p1].argmin())
                m_idx = p1_idx + 1 + m_idx_local
                m = float(lo[m_idx])
                if (p1 - m) >= min_pb_atr * atr_i:
                    # Look for p2 between m_idx and i
                    later = slice(m_idx + 1, i)
                    if i - m_idx > 2:
                        p2_idx_local = int(h[later].argmax())
                        p2_idx = m_idx + 1 + p2_idx_local
                        p2 = float(h[p2_idx])
                        if abs(p2 - p1) / p1 < max_diff and i == p2_idx + 1:
                            entry = float(lo[i]) - 0.01
                            stop = max(p1, p2) + buf * atr_i
                            risk = stop - entry
                            if risk > 0:
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
                                        "regime_at_signal": str(bars["regime"].iloc[i]),
                                        "params_tier": params.tier.value,
                                    }
                                )
                                continue

        # ---- Double Bottom detection ----
        if is_bull_bar[i]:
            window_l = lo[window]
            b1_idx_local = int(window_l.argmin())
            b1_idx = i - lookback + b1_idx_local
            b1 = float(lo[b1_idx])
            after_b1 = slice(b1_idx + 1, i)
            if i - b1_idx > min_pb_bars:
                m_idx_local = int(h[after_b1].argmax())
                m_idx = b1_idx + 1 + m_idx_local
                m = float(h[m_idx])
                if (m - b1) >= min_pb_atr * atr_i:
                    later = slice(m_idx + 1, i)
                    if i - m_idx > 2:
                        b2_idx_local = int(lo[later].argmin())
                        b2_idx = m_idx + 1 + b2_idx_local
                        b2 = float(lo[b2_idx])
                        if abs(b2 - b1) / b1 < max_diff and i == b2_idx + 1:
                            entry = float(h[i]) + 0.01
                            stop = min(b1, b2) - buf * atr_i
                            risk = entry - stop
                            if risk > 0:
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
                                        "regime_at_signal": str(bars["regime"].iloc[i]),
                                        "params_tier": params.tier.value,
                                    }
                                )

    if not out:
        return empty_candidates()
    return pd.DataFrame(out, columns=CANDIDATE_COLS)
