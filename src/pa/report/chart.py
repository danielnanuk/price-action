"""Render annotated K-line chart for a single trade."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle


def render_trade_chart(
    *,
    trade: dict[str, Any] | pd.Series,
    ohlcv: pd.DataFrame,
    out_path: Path,
    lookback: int = 30,
    lookforward: int = 15,
) -> None:
    sig_date = pd.Timestamp(trade["signal_date"])
    sorted_oh = ohlcv.sort_values("date").reset_index(drop=True)
    sig_idx = int(sorted_oh["date"].searchsorted(sig_date))

    start = max(0, sig_idx - lookback)
    end = min(len(sorted_oh), sig_idx + lookforward + 1)
    chunk = sorted_oh.iloc[start:end]

    fig, ax = plt.subplots(figsize=(10, 5))
    for _, bar in chunk.iterrows():
        color = "g" if bar["close"] >= bar["open"] else "r"
        ax.plot(
            [bar["date"], bar["date"]],
            [bar["low"], bar["high"]],
            color=color,
            linewidth=0.7,
        )
        body_low = min(bar["open"], bar["close"])
        body_height = abs(bar["close"] - bar["open"]) or 0.01
        ax.add_patch(
            Rectangle(
                (mdates.date2num(bar["date"]) - 0.3, body_low),
                0.6,
                body_height,
                color=color,
                alpha=0.7,
            )
        )

    ax.axhline(trade["entry_price"], color="blue", linestyle="--", lw=1, label="Entry")
    ax.axhline(trade["stop_price"], color="red", linestyle=":", lw=1, label="Stop")
    ax.axhline(trade["target_price"], color="green", linestyle=":", lw=1, label="Target")
    ax.axvline(sig_date, color="black", linestyle="-", lw=0.5, alpha=0.4)

    title = (
        f"{trade['ticker']}  {trade['side'].upper()}  "
        f"signal={sig_date.date()}  "
        f"exit={trade['exit_reason']}  R={trade['pnl_r']:.2f}"
    )
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper left", fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=80)
    plt.close(fig)
