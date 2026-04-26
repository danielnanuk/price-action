"""End-to-end pipeline test on a synthetic 3-ticker, 1-year universe."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pa.config import Config, load_config
from pa.pipeline import (
    stage_backtest,
    stage_detect,
    stage_indicate,
    stage_regime,
    stage_report,
)
from pa.types import OHLCV_COLS


def _gen_synthetic_ohlcv(seed: int, n: int = 252) -> pd.DataFrame:
    """Generate a synthetic 1-year OHLCV frame (~252 trading days)."""
    rng = np.random.default_rng(seed)
    closes = 100 + np.cumsum(rng.normal(0.05, 0.8, n))
    highs = closes + rng.uniform(0.2, 0.8, n)
    lows = closes - rng.uniform(0.2, 0.8, n)
    return pd.DataFrame(
        {
            "date": pd.date_range("2021-04-26", periods=n, freq="B"),
            "open": closes - 0.1,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": rng.integers(1_000_000, 5_000_000, n),
            "vwap": closes,
        }
    )[OHLCV_COLS]


@pytest.fixture
def synthetic_universe(tmp_path: Path) -> Path:
    """Set up a tmp_path-based mini cache with 3 tickers' OHLCV."""
    ohlcv_dir = tmp_path / "data" / "ohlcv"
    ohlcv_dir.mkdir(parents=True)
    for i, t in enumerate(["AAPL", "MSFT", "GOOGL"], start=1):
        df = _gen_synthetic_ohlcv(seed=i)
        df.to_parquet(ohlcv_dir / f"{t}_seed{i:04d}.parquet", index=False)
    return tmp_path


def _mini_config(root: Path) -> Config:
    cfg_yaml = root / "config.yaml"
    members = root / "members.txt"
    members.write_text("AAPL\nMSFT\nGOOGL\n")
    cfg_yaml.write_text(
        f"""
universe: {{name: mini, members_file: {members}}}
date_range: {{start: "2021-04-26", end: "2022-04-26"}}
data:
  cache_dir: {root}/data
  api_base_url: "x"
  api_key_env: NOPE
  rate_limit_per_min: 1
detectors:
  enabled: [h2, l2, flag, failed_breakout, double_top_bottom]
  param_tiers: [strict, standard, loose]
backtest: {{time_stop_bars: 20, same_bar_priority: stop_first}}
report: {{output_dir: {root}/reports, samples_per_setup: 5}}
"""
    )
    return load_config(cfg_yaml)


def test_e2e_pipeline_runs(synthetic_universe: Path) -> None:
    cfg = _mini_config(synthetic_universe)
    stage_indicate(cfg)
    stage_regime(cfg)
    stage_detect(cfg)
    stage_backtest(cfg)
    out = stage_report(cfg)

    # Report artifacts exist
    assert (out / "index.html").exists()
    assert (out / "data.parquet").exists()

    # All 5 setups x 3 tiers = 15 candidate parquet files exist
    cands = list((cfg.data.cache_dir / "candidates").glob("*.parquet"))
    assert len(cands) == 15

    # Trades dir mirrors candidates: 15 trade parquet files
    trades = list((cfg.data.cache_dir / "trades").glob("*.parquet"))
    assert len(trades) == 15

    # Per-stage intermediates exist for each ticker
    for stage in ("indicators", "regime"):
        files = list((cfg.data.cache_dir / stage).glob("*.parquet"))
        assert len(files) == 3, f"expected 3 {stage} files, got {len(files)}"


def test_e2e_under_30s(synthetic_universe: Path) -> None:
    import time

    cfg = _mini_config(synthetic_universe)
    t0 = time.time()
    stage_indicate(cfg)
    stage_regime(cfg)
    stage_detect(cfg)
    stage_backtest(cfg)
    stage_report(cfg)
    elapsed = time.time() - t0
    assert elapsed < 30.0, f"pipeline too slow: {elapsed:.1f}s"
