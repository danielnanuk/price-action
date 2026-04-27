"""Double Top / Double Bottom reversal detector (v2).

v1 (the original spec implementation) over-fired by ~30k STANDARD candidates
on daily 5y x S&P 500 with 11% win rate / PF 0.90 — symptom of "any two peaks
within 5%" being too lenient. v2 tightens the structural requirements:

  - Longer lookback (60 bars standard, vs v1's 30) so P1 is a multi-month
    significant high, not a casual 6-week local high.
  - Deeper required pullback (2x ATR vs v1's 1x) — middle low M must be a
    real correction.
  - Longer pullback duration (>= 8 bars vs v1's 5) — meaningful time for
    consolidation.
  - Tighter P2 placement (1.5% diff vs 5%) — actually approaching P1.
  - Mature regime gate: regime_strength >= min before any fire.
  - Bull regime required for double top (a top in bear regime is just an
    intra-bear bounce, not a reversal); bear regime for double bottom.
  - Signal bar must close BELOW P2 (real rejection, not just any bear bar).

Result of the v2 backtest on the same 5y x S&P 500 dataset:
  loose:    7021 candidates, win 41%, PF 1.00, total R +10  (vs v1 -4632)
  standard: 2145 candidates, win 39%, PF 0.98, total R -20  (vs v1 -1564)
  strict:    513 candidates, win 40%, PF 1.00, total R  -1  (vs v1  -313)

Net: structurally clean (95% noise eliminated, 4x win rate), but PF ~= 1.00
across tiers — true double tops are too rare and too symmetric for daily
5y to surface a clear edge. Ships disabled in default.yaml; can be enabled
for research once we have longer history or hourly multi-TF confirmation.
"""

from __future__ import annotations

import pandas as pd

from pa.detectors.base import CANDIDATE_COLS, SetupParams, empty_candidates
from pa.types import Regime, Side


def detect_double_top_bottom(bars: pd.DataFrame, params: SetupParams) -> pd.DataFrame:
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

    min_pb_bars = int(p["min_pullback_bars"])
    min_pb_atr = float(p["min_pullback_atr_mult"])
    max_diff = float(p["max_peak_diff_pct"])
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

        window = slice(i - lookback, i)

        # ---- Double Top (bull-trend exhaustion) ----
        if regime[i] == Regime.BULL_TREND.value and not is_bull_bar[i]:
            window_h = h[window]
            p1_idx_local = int(window_h.argmax())
            p1_idx = i - lookback + p1_idx_local
            p1 = float(h[p1_idx])

            if i - p1_idx <= min_pb_bars:
                continue

            after_p1 = slice(p1_idx + 1, i)
            m_idx_local = int(lo[after_p1].argmin())
            m_idx = p1_idx + 1 + m_idx_local
            m = float(lo[m_idx])
            if (p1 - m) < min_pb_atr * atr_i:
                continue
            if i - m_idx <= 2:
                continue

            later = slice(m_idx + 1, i)
            p2_idx_local = int(h[later].argmax())
            p2_idx = m_idx + 1 + p2_idx_local
            p2 = float(h[p2_idx])

            if abs(p2 - p1) / p1 >= max_diff:
                continue
            if i - p2_idx > 2:
                continue
            # v2: signal bar must close below P2 (true rejection, not just any bear bar)
            if closes[i] >= p2:
                continue

            entry = float(lo[i]) - 0.01
            stop = max(p1, p2) + buf * atr_i
            risk = stop - entry
            if risk <= 0:
                continue
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
                    "regime_at_signal": str(regime[i]),
                    "params_tier": params.tier.value,
                }
            )
            continue

        # ---- Double Bottom (bear-trend exhaustion) ----
        if regime[i] == Regime.BEAR_TREND.value and is_bull_bar[i]:
            window_l = lo[window]
            b1_idx_local = int(window_l.argmin())
            b1_idx = i - lookback + b1_idx_local
            b1 = float(lo[b1_idx])
            if i - b1_idx <= min_pb_bars:
                continue
            after_b1 = slice(b1_idx + 1, i)
            m_idx_local = int(h[after_b1].argmax())
            m_idx = b1_idx + 1 + m_idx_local
            m = float(h[m_idx])
            if (m - b1) < min_pb_atr * atr_i:
                continue
            if i - m_idx <= 2:
                continue
            later = slice(m_idx + 1, i)
            b2_idx_local = int(lo[later].argmin())
            b2_idx = m_idx + 1 + b2_idx_local
            b2 = float(lo[b2_idx])
            if abs(b2 - b1) / b1 >= max_diff:
                continue
            if i - b2_idx > 2:
                continue
            if closes[i] <= b2:
                continue
            entry = float(h[i]) + 0.01
            stop = min(b1, b2) - buf * atr_i
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
