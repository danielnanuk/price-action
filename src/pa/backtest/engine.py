"""Generic execution simulator: walks OHLCV after each candidate's signal_date,
applies stop/target/time-stop and emits a trade ledger.
"""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True, slots=True)
class ExitStrategy:
    """How the engine should manage open positions.

    Defaults match the original MVP behavior: 2R fixed target (read from the
    candidate's `target_price`), no scale-out, no trailing stop, 20-bar time
    stop, stop-first on same-bar collisions. Override individual fields to
    promote to scale-half + chandelier-trail or any other policy.

    Field semantics:
      use_fixed_target:    consult candidate.target_price as TP. If False, the
                           position can only exit via stop, time-stop, or trail.
      scale_at_1r:         when MFE >= 1R, lock in +1R on half the position and
                           run the other half to exit (trail or time-stop). The
                           reported pnl_r is the average of the two halves.
      trailing_atr_mult:   chandelier multiplier. After each bar, raise the
                           effective stop to high - K * ATR (long) or lower it
                           to low + K * ATR (short). 0 disables trailing.
                           When > 0, `bars` must contain an `atr14` column.
      time_stop_bars:      max bars to hold before exiting at last close.
      same_bar_priority:   when a single bar's range hits both the active stop
                           and target, which wins.
    """

    use_fixed_target: bool = True
    scale_at_1r: bool = False
    trailing_atr_mult: float = 0.0
    time_stop_bars: int = 20
    same_bar_priority: Literal["stop_first", "target_first"] = "stop_first"


def simulate(
    candidates: pd.DataFrame,
    bars: pd.DataFrame,
    *,
    strategy: ExitStrategy,
) -> pd.DataFrame:
    """Simulate trades for each row in `candidates` using `bars` history.

    `bars` must contain at least the OHLCV columns. If `strategy.trailing_atr_mult > 0`,
    `bars` must also contain an `atr14` column (merge indicators in before calling).

    Each candidate must already carry entry_price, stop_price, target_price set
    by the detector. The engine simulates from the bar after signal_date forward.
    """
    if candidates.empty:
        return pd.DataFrame(columns=TRADE_COLS)

    bars_idx = bars.set_index("date").sort_index()
    if strategy.trailing_atr_mult > 0 and "atr14" not in bars_idx.columns:
        raise ValueError("strategy.trailing_atr_mult > 0 requires 'atr14' column in bars")

    rows: list[dict[str, object]] = []
    for cand in candidates.itertuples():
        rows.append(_simulate_one(cand, bars_idx, strategy))
    return pd.DataFrame(rows, columns=TRADE_COLS)


def _simulate_one(
    cand: object,
    bars_idx: pd.DataFrame,
    strategy: ExitStrategy,
) -> dict[str, object]:
    side = Side(cand.side)  # type: ignore[attr-defined]
    entry = float(cand.entry_price)  # type: ignore[attr-defined]
    initial_stop = float(cand.stop_price)  # type: ignore[attr-defined]
    target = float(cand.target_price)  # type: ignore[attr-defined]
    risk = abs(entry - initial_stop)
    if risk == 0:
        risk = 1e-9

    sig_date = pd.Timestamp(cand.signal_date)  # type: ignore[attr-defined]
    future = bars_idx.loc[bars_idx.index > sig_date].head(strategy.time_stop_bars)
    entry_date = future.index[0] if len(future) > 0 else sig_date

    exit_reason = ExitReason.END_OF_DATA
    exit_date = future.index[-1] if len(future) > 0 else sig_date
    exit_price = float(future["close"].iloc[-1]) if len(future) > 0 else entry
    same_bar = False
    mae_r = 0.0
    mfe_r = 0.0

    effective_stop = initial_stop
    half_taken = False
    locked_pnl_r = 0.0  # locked when scaling half at +1R

    for i, (bar_date, row) in enumerate(future.iterrows()):
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])

        if side == Side.LONG:
            mae_r = min(mae_r, (low - entry) / risk)
            mfe_r = max(mfe_r, (high - entry) / risk)
            stop_hit = low <= effective_stop
            target_hit = strategy.use_fixed_target and high >= target

            if strategy.scale_at_1r and not half_taken and mfe_r >= 1.0:
                half_taken = True
                locked_pnl_r = 1.0

            if strategy.trailing_atr_mult > 0:
                atr_i = float(row["atr14"]) if not pd.isna(row["atr14"]) else risk
                trail = high - strategy.trailing_atr_mult * atr_i
                effective_stop = max(effective_stop, trail)
        else:
            mae_r = min(mae_r, (entry - high) / risk)
            mfe_r = max(mfe_r, (entry - low) / risk)
            stop_hit = high >= effective_stop
            target_hit = strategy.use_fixed_target and low <= target

            if strategy.scale_at_1r and not half_taken and mfe_r >= 1.0:
                half_taken = True
                locked_pnl_r = 1.0

            if strategy.trailing_atr_mult > 0:
                atr_i = float(row["atr14"]) if not pd.isna(row["atr14"]) else risk
                trail = low + strategy.trailing_atr_mult * atr_i
                effective_stop = min(effective_stop, trail)

        if stop_hit and target_hit:
            same_bar = True
            if strategy.same_bar_priority == "stop_first":
                exit_reason, exit_price = ExitReason.STOP_HIT, effective_stop
            else:
                exit_reason, exit_price = ExitReason.TARGET_HIT, target
            exit_date = bar_date
            break
        if stop_hit:
            exit_reason, exit_price, exit_date = (
                ExitReason.STOP_HIT,
                effective_stop,
                bar_date,
            )
            break
        if target_hit:
            exit_reason, exit_price, exit_date = ExitReason.TARGET_HIT, target, bar_date
            break
        if i == len(future) - 1:
            exit_reason = ExitReason.TIME_STOP
            exit_price = close
            exit_date = bar_date

    direction = 1.0 if side == Side.LONG else -1.0
    runner_pnl_r = direction * (exit_price - entry) / risk
    runner_pnl_pct = direction * (exit_price - entry) / entry

    if strategy.scale_at_1r and half_taken:
        # Average the two halves: first half locked at +1R, second is the runner.
        pnl_r = 0.5 * locked_pnl_r + 0.5 * runner_pnl_r
        # pnl_pct is reported as the runner's only — first half's pct is +risk/entry
        # which is approximately +1R in pct; we use runner pct for fidelity to actual
        # exit price recorded.
        pnl_pct = runner_pnl_pct
    else:
        pnl_r = runner_pnl_r
        pnl_pct = runner_pnl_pct

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
