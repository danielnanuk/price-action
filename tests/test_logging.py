"""Tests for pa.logging — JSON-line structured logger."""

from __future__ import annotations

import json
from pathlib import Path

from pa.logging import StageLogger


def test_logger_writes_jsonlines(tmp_path: Path) -> None:
    log_file = tmp_path / "fetch.jsonl"
    logger = StageLogger(stage="fetch", run_id="abc123", log_path=log_file)
    logger.info("started", ticker="AAPL", duration_ms=12)
    logger.warn("retry", ticker="AAPL", attempt=2)
    logger.close()

    lines = log_file.read_text().strip().split("\n")
    assert len(lines) == 2
    rec1 = json.loads(lines[0])
    assert rec1["stage"] == "fetch"
    assert rec1["run_id"] == "abc123"
    assert rec1["level"] == "info"
    assert rec1["msg"] == "started"
    assert rec1["ticker"] == "AAPL"
    assert "ts" in rec1


def test_logger_context_manager(tmp_path: Path) -> None:
    log_file = tmp_path / "fetch.jsonl"
    with StageLogger(stage="fetch", run_id="x", log_path=log_file) as log:
        log.error("boom", ticker="ZZZ")
    assert log_file.exists()
    rec = json.loads(log_file.read_text().strip())
    assert rec["level"] == "error"
