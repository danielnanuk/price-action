"""Generic execution simulator: walks OHLCV after each candidate's signal_date,
applies stop/target/time-stop and emits a trade ledger.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

from pa.types import ExitReason, Side

TRADE_COLS = [
    "ticker",
    "signal_date",
    "entry_date",
    "entry_price",
    "exit_date",
    "exit_price",
    "exit_reason",
    "pnl_r",
    "pnl_pct",
    "mae_r",
    "mfe_r",
    "days_held",
    "regime_at_signal",
    "same_bar_ambiguous",
    "params_tier",
    "side",
]


def simulate(
    candidates: pd.DataFrame,
    ohlcv: pd.DataFrame,
    *,
    time_stop_bars: int,
    same_bar_priority: Literal["stop_first", "target_first"],
) -> pd.DataFrame:
    """Simulate trades for each row in `candidates` using `ohlcv` history.

    Assumes candidates already contain entry_price, stop_price, target_price.
    The engine enters on the bar AFTER signal_date (open price = entry_price
    if explicitly recorded, otherwise the open of the next bar — caller's
    responsibility).
    """
    if candidates.empty:
        return pd.DataFrame(columns=TRADE_COLS)

    ohlcv_idx = ohlcv.set_index("date").sort_index()
    rows: list[dict[str, object]] = []
    for cand in candidates.itertuples():
        rows.append(_simulate_one(cand, ohlcv_idx, time_stop_bars, same_bar_priority))
    return pd.DataFrame(rows, columns=TRADE_COLS)


def _simulate_one(
    cand: object,
    ohlcv_idx: pd.DataFrame,
    time_stop_bars: int,
    same_bar_priority: str,
) -> dict[str, object]:
    side = Side(cand.side)  # type: ignore[attr-defined]
    entry = float(cand.entry_price)  # type: ignore[attr-defined]
    stop = float(cand.stop_price)  # type: ignore[attr-defined]
    target = float(cand.target_price)  # type: ignore[attr-defined]
    risk = abs(entry - stop)
    if risk == 0:
        risk = 1e-9  # guard, should not happen in practice

    sig_date = pd.Timestamp(cand.signal_date)  # type: ignore[attr-defined]
    future = ohlcv_idx.loc[ohlcv_idx.index > sig_date].head(time_stop_bars)
    entry_date = future.index[0] if len(future) > 0 else sig_date

    exit_reason = ExitReason.END_OF_DATA
    exit_date = future.index[-1] if len(future) > 0 else sig_date
    exit_price = float(future["close"].iloc[-1]) if len(future) > 0 else entry
    same_bar = False
    mae_r = 0.0
    mfe_r = 0.0

    for i, (bar_date, row) in enumerate(future.iterrows()):
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])

        if side == Side.LONG:
            mae_r = min(mae_r, (low - entry) / risk)
            mfe_r = max(mfe_r, (high - entry) / risk)
            stop_hit = low <= stop
            target_hit = high >= target
        else:
            mae_r = min(mae_r, (entry - high) / risk)
            mfe_r = max(mfe_r, (entry - low) / risk)
            stop_hit = high >= stop
            target_hit = low <= target

        if stop_hit and target_hit:
            same_bar = True
            if same_bar_priority == "stop_first":
                exit_reason, exit_price = ExitReason.STOP_HIT, stop
            else:
                exit_reason, exit_price = ExitReason.TARGET_HIT, target
            exit_date = bar_date
            break
        if stop_hit:
            exit_reason, exit_price, exit_date = ExitReason.STOP_HIT, stop, bar_date
            break
        if target_hit:
            exit_reason, exit_price, exit_date = ExitReason.TARGET_HIT, target, bar_date
            break
        if i == len(future) - 1:
            exit_reason = ExitReason.TIME_STOP
            exit_price = close
            exit_date = bar_date

    direction = 1.0 if side == Side.LONG else -1.0
    pnl_pct = direction * (exit_price - entry) / entry
    pnl_r = direction * (exit_price - entry) / risk

    return {
        "ticker": cand.ticker,  # type: ignore[attr-defined]
        "signal_date": sig_date,
        "entry_date": entry_date,
        "entry_price": entry,
        "exit_date": exit_date,
        "exit_price": exit_price,
        "exit_reason": exit_reason.value,
        "pnl_r": pnl_r,
        "pnl_pct": pnl_pct,
        "mae_r": mae_r,
        "mfe_r": mfe_r,
        "days_held": ((exit_date - entry_date).days if isinstance(exit_date, pd.Timestamp) else 0),
        "regime_at_signal": cand.regime_at_signal,  # type: ignore[attr-defined]
        "same_bar_ambiguous": same_bar,
        "params_tier": cand.params_tier,  # type: ignore[attr-defined]
        "side": side.value,
    }
