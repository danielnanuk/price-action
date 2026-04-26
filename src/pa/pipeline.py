"""Pipeline orchestration: each function corresponds to a CLI subcommand."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pandas as pd

from pa.backtest import simulate
from pa.config import Config
from pa.data.cache import OhlcvCache
from pa.data.client import MassiveClient
from pa.detectors.double_top_bottom import detect_double_top_bottom
from pa.detectors.failed_breakout import detect_failed_breakout
from pa.detectors.flag import detect_flag
from pa.detectors.h2 import detect_h2
from pa.detectors.l2 import detect_l2
from pa.detectors.params import (
    double_tb_params,
    failed_breakout_params,
    flag_params,
    h2_params,
    l2_params,
)
from pa.indicators import compute_indicators
from pa.regime.classifier import classify_regime
from pa.regime.signal_bar import signal_bar_score
from pa.report.html import build_report
from pa.types import ParamTier

DETECTOR_REGISTRY = {
    "h2": (detect_h2, h2_params),
    "l2": (detect_l2, l2_params),
    "flag": (detect_flag, flag_params),
    "failed_breakout": (detect_failed_breakout, failed_breakout_params),
    "double_top_bottom": (detect_double_top_bottom, double_tb_params),
}


def _read_universe(cfg: Config) -> list[str]:
    return [
        line.strip()
        for line in cfg.universe.members_file.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]


def stage_fetch(cfg: Config) -> None:
    api_key = os.environ.get(cfg.data.api_key_env, "")
    if not api_key:
        raise RuntimeError(f"Missing {cfg.data.api_key_env} env var")
    client = MassiveClient(api_key=api_key, base_url=cfg.data.api_base_url)
    cache = OhlcvCache(cache_dir=cfg.data.cache_dir / "ohlcv", client=client)
    for ticker in _read_universe(cfg):
        try:
            cache.fetch_ohlcv(ticker, cfg.date_range.start, cfg.date_range.end)
        except Exception as exc:  # log and continue
            (cfg.data.cache_dir / "_failures").mkdir(parents=True, exist_ok=True)
            (cfg.data.cache_dir / "_failures" / "fetch.jsonl").open("a").write(
                f'{{"ticker":"{ticker}","reason":"{exc}"}}\n'
            )
    client.close()


def stage_indicate(cfg: Config) -> None:
    src = cfg.data.cache_dir / "ohlcv"
    dst = cfg.data.cache_dir / "indicators"
    dst.mkdir(parents=True, exist_ok=True)
    for f in sorted(src.glob("*.parquet")):
        ohlcv = pd.read_parquet(f)
        if ohlcv.empty:
            continue
        compute_indicators(ohlcv).to_parquet(dst / f.name, index=False)


def stage_regime(cfg: Config) -> None:
    ohlcv_dir = cfg.data.cache_dir / "ohlcv"
    ind_dir = cfg.data.cache_dir / "indicators"
    dst = cfg.data.cache_dir / "regime"
    dst.mkdir(parents=True, exist_ok=True)
    for f in sorted(ohlcv_dir.glob("*.parquet")):
        ohlcv = pd.read_parquet(f)
        if ohlcv.empty:
            continue
        ind = pd.read_parquet(ind_dir / f.name)
        regimes = classify_regime(ohlcv, ind)
        regimes["signal_bar_score"] = signal_bar_score(ohlcv, ind).values
        regimes.to_parquet(dst / f.name, index=False)


def stage_detect(cfg: Config) -> None:
    ohlcv_dir = cfg.data.cache_dir / "ohlcv"
    ind_dir = cfg.data.cache_dir / "indicators"
    rg_dir = cfg.data.cache_dir / "regime"
    dst = cfg.data.cache_dir / "candidates"
    dst.mkdir(parents=True, exist_ok=True)

    for setup in cfg.detectors.enabled:
        detect_fn, params_fn = DETECTOR_REGISTRY[setup]
        for tier_str in cfg.detectors.param_tiers:
            tier = ParamTier(tier_str)
            params = params_fn(tier)
            all_cands: list[pd.DataFrame] = []
            for f in sorted(ohlcv_dir.glob("*.parquet")):
                ticker = f.name.split("_")[0]
                ohlcv = pd.read_parquet(f)
                if ohlcv.empty:
                    continue
                bars = ohlcv.merge(pd.read_parquet(ind_dir / f.name), on="date").merge(
                    pd.read_parquet(rg_dir / f.name), on="date"
                )
                bars["ticker"] = ticker
                cands = detect_fn(bars, params)
                if not cands.empty:
                    all_cands.append(cands)
            combined = pd.concat(all_cands, ignore_index=True) if all_cands else pd.DataFrame()
            combined.to_parquet(dst / f"{setup}_{tier_str}.parquet", index=False)


def stage_backtest(cfg: Config) -> None:
    cand_dir = cfg.data.cache_dir / "candidates"
    ohlcv_dir = cfg.data.cache_dir / "ohlcv"
    dst = cfg.data.cache_dir / "trades"
    dst.mkdir(parents=True, exist_ok=True)

    ohlcv_by_ticker = {
        f.name.split("_")[0]: pd.read_parquet(f) for f in ohlcv_dir.glob("*.parquet")
    }

    for cand_file in sorted(cand_dir.glob("*.parquet")):
        cands = pd.read_parquet(cand_file)
        if cands.empty:
            cands.to_parquet(dst / cand_file.name, index=False)
            continue
        all_trades: list[pd.DataFrame] = []
        for ticker, group in cands.groupby("ticker"):
            ohlcv = ohlcv_by_ticker.get(str(ticker))
            if ohlcv is None or ohlcv.empty:
                continue
            trades = simulate(
                group,
                ohlcv,
                time_stop_bars=cfg.backtest.time_stop_bars,
                same_bar_priority=cfg.backtest.same_bar_priority,
            )
            all_trades.append(trades)
        combined = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
        combined.to_parquet(dst / cand_file.name, index=False)


def stage_report(cfg: Config) -> Path:
    trades_dir = cfg.data.cache_dir / "trades"
    ohlcv_dir = cfg.data.cache_dir / "ohlcv"
    run_id = uuid.uuid4().hex[:8]
    out_dir = cfg.report.output_dir / run_id

    trades_by_st: dict[tuple[str, str], pd.DataFrame] = {}
    for f in sorted(trades_dir.glob("*.parquet")):
        stem = f.stem  # e.g. "h2_standard"
        setup, _, tier = stem.rpartition("_")
        df = pd.read_parquet(f)
        if not df.empty:
            trades_by_st[(setup, tier)] = df

    ohlcv_by_ticker = {
        f.name.split("_")[0]: pd.read_parquet(f) for f in ohlcv_dir.glob("*.parquet")
    }

    return build_report(
        run_id=run_id,
        trades_by_setup_tier=trades_by_st,
        ohlcv_by_ticker=ohlcv_by_ticker,
        out_dir=out_dir,
        universe=cfg.universe.name,
        date_start=cfg.date_range.start.isoformat(),
        date_end=cfg.date_range.end.isoformat(),
        samples_per_setup=cfg.report.samples_per_setup,
    )


def run_all(cfg: Config) -> Path:
    stage_fetch(cfg)
    stage_indicate(cfg)
    stage_regime(cfg)
    stage_detect(cfg)
    stage_backtest(cfg)
    return stage_report(cfg)
