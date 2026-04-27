"""Smoke test that CLI parser dispatches correctly."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from pa.cli import main


def test_cli_dispatches_fetch(tmp_path: Path) -> None:
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(
        """
universe: {name: sp500, members_file: m.txt}
date_range: {start: "2021-01-01", end: "2021-02-01"}
data: {cache_dir: data, api_base_url: x, api_key_env: K, rate_limit_per_min: 1}
detectors: {enabled: [h2], param_tiers: [standard]}
backtest: {time_stop_bars: 20, same_bar_priority: stop_first}
report: {output_dir: r, samples_per_setup: 1}
"""
    )
    with patch("pa.cli.stage_fetch") as mock_fetch:
        rc = main(["fetch", "--config", str(cfg_path)])
    assert rc == 0
    mock_fetch.assert_called_once()
