"""Build the HTML report from a dict of trade ledgers."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from pa.report.chart import render_trade_chart
from pa.report.stats import compute_setup_stats, group_by_regime, group_by_year

TEMPLATES_DIR = Path(__file__).parent / "templates"


def build_report(
    *,
    run_id: str,
    trades_by_setup_tier: dict[tuple[str, str], pd.DataFrame],
    ohlcv_by_ticker: dict[str, pd.DataFrame],
    out_dir: Path,
    universe: str,
    date_start: str,
    date_end: str,
    samples_per_setup: int = 50,
    failures: list[dict[str, Any]] | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    env = Environment(
        loader=FileSystemLoader(TEMPLATES_DIR),
        autoescape=select_autoescape(),
    )
    # Expose Python's float() so templates can compare against float('inf').
    env.globals["float"] = float

    summary_rows: list[dict[str, Any]] = []
    all_trades: list[pd.DataFrame] = []
    samples_root = out_dir / "samples"
    samples_root.mkdir(exist_ok=True)

    for (setup, tier), trades in trades_by_setup_tier.items():
        stats = compute_setup_stats(trades)
        page_name = f"{setup}_{tier}.html"
        summary_rows.append(
            {
                "setup": setup,
                "tier": tier,
                "link": page_name,
                **stats,
            }
        )

        # Render up to N sample charts
        sample_dir = samples_root / f"{setup}_{tier}"
        sample_dir.mkdir(exist_ok=True)
        sample_meta: list[dict[str, Any]] = []
        for _, trade in trades.head(samples_per_setup).iterrows():
            ohlcv = ohlcv_by_ticker.get(trade["ticker"])
            if ohlcv is None or ohlcv.empty:
                continue
            img_name = f"{trade['ticker']}_{pd.Timestamp(trade['signal_date']).date()}.png"
            img_path = sample_dir / img_name
            try:
                render_trade_chart(
                    trade=trade,
                    ohlcv=ohlcv,
                    out_path=img_path,
                    lookback=30,
                    lookforward=15,
                )
            except Exception:  # render failure should not abort report
                continue
            sample_meta.append(
                {
                    "ticker": trade["ticker"],
                    "signal_date": str(pd.Timestamp(trade["signal_date"]).date()),
                    "exit_reason": trade["exit_reason"],
                    "pnl_r": float(trade["pnl_r"]),
                    "image": f"samples/{setup}_{tier}/{img_name}",
                }
            )

        page = env.get_template("setup.html.j2").render(
            run_id=run_id,
            setup=setup,
            tier=tier,
            stats=stats,
            by_regime=group_by_regime(trades),
            by_year=group_by_year(trades),
            samples=sample_meta,
        )
        (out_dir / page_name).write_text(page)
        all_trades.append(trades.assign(setup=setup, tier=tier))

    index = env.get_template("index.html.j2").render(
        run_id=run_id,
        run_date=datetime.now().date().isoformat(),
        universe=universe,
        date_start=date_start,
        date_end=date_end,
        summary=summary_rows,
        failures=failures or [],
    )
    (out_dir / "index.html").write_text(index)

    if all_trades:
        pd.concat(all_trades, ignore_index=True).to_parquet(out_dir / "data.parquet", index=False)
    else:
        pd.DataFrame().to_parquet(out_dir / "data.parquet", index=False)

    return out_dir
