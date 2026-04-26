"""Tests for pa.config — YAML loader + schema validation."""

from __future__ import annotations

from pathlib import Path

import pytest
from pa.config import Config, ConfigError, load_config


def test_load_default_config(tmp_path: Path) -> None:
    cfg_path = tmp_path / "test.yaml"
    cfg_path.write_text(
        """
universe:
  name: sp500
  members_file: members.txt
date_range:
  start: "2021-04-26"
  end: "2026-04-26"
data:
  cache_dir: data
  api_base_url: https://api.massive.com
  api_key_env: POLYGON_API_KEY
  rate_limit_per_min: 100
detectors:
  enabled: [h2]
  param_tiers: [standard]
backtest:
  time_stop_bars: 20
  same_bar_priority: stop_first
report:
  output_dir: reports
  samples_per_setup: 50
"""
    )
    cfg = load_config(cfg_path)
    assert isinstance(cfg, Config)
    assert cfg.detectors.enabled == ["h2"]
    assert cfg.backtest.time_stop_bars == 20


def test_invalid_same_bar_priority_raises(tmp_path: Path) -> None:
    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text(
        """
universe: {name: sp500, members_file: m.txt}
date_range: {start: "2021-01-01", end: "2026-01-01"}
data: {cache_dir: data, api_base_url: x, api_key_env: K, rate_limit_per_min: 1}
detectors: {enabled: [h2], param_tiers: [standard]}
backtest: {time_stop_bars: 20, same_bar_priority: WRONG}
report: {output_dir: r, samples_per_setup: 1}
"""
    )
    with pytest.raises(ConfigError):
        load_config(cfg_path)
