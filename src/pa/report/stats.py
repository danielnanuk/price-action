"""Aggregate statistics from a trades ledger."""

from __future__ import annotations

import pandas as pd

from pa.types import ExitReason

EMPTY_STATS: dict[str, float] = {
    "n": 0,
    "win_rate": 0.0,
    "mean_r": 0.0,
    "profit_factor": 0.0,
    "mean_mae_r": 0.0,
    "mean_mfe_r": 0.0,
}


def compute_setup_stats(trades: pd.DataFrame) -> dict[str, float]:
    if trades.empty:
        return dict(EMPTY_STATS)
    n = len(trades)
    wins = (trades["exit_reason"] == ExitReason.TARGET_HIT.value).sum()
    win_rate = wins / n
    mean_r = float(trades["pnl_r"].mean())
    gross_win = trades.loc[trades["pnl_r"] > 0, "pnl_r"].sum()
    gross_loss = -trades.loc[trades["pnl_r"] < 0, "pnl_r"].sum()
    pf = float(gross_win / gross_loss) if gross_loss > 0 else float("inf")
    return {
        "n": int(n),
        "win_rate": float(win_rate),
        "mean_r": mean_r,
        "profit_factor": pf,
        "mean_mae_r": float(trades["mae_r"].mean()),
        "mean_mfe_r": float(trades["mfe_r"].mean()),
    }


def group_by_year(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    df = trades.copy()
    df["year"] = pd.to_datetime(df["signal_date"]).dt.year
    return (
        df.groupby("year")
        .apply(compute_setup_stats, include_groups=False)  # type: ignore[arg-type]
        .apply(pd.Series)
    )


def group_by_regime(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    return (
        trades.groupby("regime_at_signal")
        .apply(compute_setup_stats, include_groups=False)  # type: ignore[arg-type]
        .apply(pd.Series)
    )
