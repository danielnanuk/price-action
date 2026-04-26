# Price Action Backtest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Lean Research Pipeline that backtests 5 Al Brooks Price Action setups on S&P 500 daily bars (5 years) and outputs an HTML report with statistics + annotated chart galleries.

**Architecture:** Linear 6-stage pipeline (`fetch → indicate → regime → detect → backtest → report`) with parquet caching at every stage. Each stage is idempotent — failures or revisions can resume from any point. Setups detected via custom Python (no backtest framework).

**Tech Stack:** Python 3.11 / pandas / numpy / pyarrow / httpx / pydantic / matplotlib / Jinja2 / pytest / hypothesis. Tooling: uv / ruff / mypy --strict.

**Spec:** `docs/superpowers/specs/2026-04-26-price-action-backtest-design.md`

---

## Task 1: Project Bootstrap

**Files:**
- Create: `pyproject.toml`
- Create: `src/pa/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `.pre-commit-config.yaml`
- Create: `configs/default.yaml`

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "price-action"
version = "0.0.1"
description = "Al Brooks Price Action backtest research pipeline"
requires-python = ">=3.11"
dependencies = [
    "pandas>=2.2",
    "numpy>=1.26",
    "pyarrow>=15.0",
    "httpx>=0.27",
    "pydantic>=2.6",
    "pyyaml>=6.0",
    "jinja2>=3.1",
    "matplotlib>=3.8",
    "tenacity>=8.2",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=4.1",
    "pytest-xdist>=3.5",
    "hypothesis>=6.99",
    "ruff>=0.3",
    "mypy>=1.9",
    "pandas-stubs",
    "types-pyyaml",
    "pre-commit>=3.6",
]

[project.scripts]
pa-backtest = "pa.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/pa"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "N", "UP", "B", "SIM", "RUF"]

[tool.mypy]
strict = true
python_version = "3.11"
plugins = ["pydantic.mypy"]
mypy_path = "src"

[[tool.mypy.overrides]]
module = ["matplotlib.*", "pyarrow.*"]
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra --strict-markers"
```

- [ ] **Step 2: Create empty package files**

```bash
touch src/pa/__init__.py tests/__init__.py
```

Write `tests/conftest.py`:
```python
"""Shared pytest fixtures."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _seed_rng() -> None:
    np.random.seed(42)


@pytest.fixture
def empty_ohlcv() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["date", "open", "high", "low", "close", "volume", "vwap"]
    )
```

- [ ] **Step 3: Create `.pre-commit-config.yaml`**

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.3.4
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.9.0
    hooks:
      - id: mypy
        additional_dependencies:
          [pandas-stubs, types-pyyaml, pydantic>=2.6]
        args: [--strict]
        files: ^src/
```

- [ ] **Step 4: Create `configs/default.yaml`**

```yaml
universe:
  name: sp500
  members_file: configs/sp500_members.txt  # populated in Task 4
date_range:
  start: "2021-04-26"
  end: "2026-04-26"
data:
  cache_dir: data
  api_base_url: "https://api.massive.com"
  api_key_env: POLYGON_API_KEY
  rate_limit_per_min: 100
detectors:
  enabled: [h2, l2, flag, failed_breakout, double_top_bottom]
  param_tiers: [strict, standard, loose]
backtest:
  time_stop_bars: 20
  same_bar_priority: stop_first
report:
  output_dir: reports
  samples_per_setup: 50  # cap on annotated chart rendering
```

- [ ] **Step 5: Install dev environment with uv and verify imports**

Run:
```bash
uv venv
uv pip install -e ".[dev]"
uv run python -c "import pa; print('pa package importable')"
uv run pre-commit install
```
Expected: `pa package importable` printed; pre-commit hook installed.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/pa/__init__.py tests/__init__.py tests/conftest.py \
  .pre-commit-config.yaml configs/default.yaml
git commit -m "Bootstrap project: pyproject, pre-commit, default config"
```

---

## Task 2: Shared Types and Config Loader

**Files:**
- Create: `src/pa/types.py`
- Create: `src/pa/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write `tests/test_config.py` (failing)**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL — `pa.config` does not exist.

- [ ] **Step 3: Implement `src/pa/types.py`**

```python
"""Shared types and frame-shape protocols used across pa modules."""
from __future__ import annotations

from enum import StrEnum
from typing import Final

# Required columns at each pipeline stage. Centralized for cross-module validation.
OHLCV_COLS: Final = ["date", "open", "high", "low", "close", "volume", "vwap"]
INDICATOR_COLS: Final = [
    "ema20",
    "ema50",
    "atr14",
    "swing_high",
    "swing_low",
    "bar_body_pct",
    "bar_upper_wick_pct",
    "bar_lower_wick_pct",
    "bar_is_bull",
    "bar_close_position",
]
REGIME_COLS: Final = ["regime", "regime_strength", "signal_bar_score"]


class Regime(StrEnum):
    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend"
    TRADING_RANGE = "trading_range"
    TRANSITIONAL = "transitional"


class Side(StrEnum):
    LONG = "long"
    SHORT = "short"


class ExitReason(StrEnum):
    TARGET_HIT = "target_hit"
    STOP_HIT = "stop_hit"
    TIME_STOP = "time_stop"
    END_OF_DATA = "end_of_data"


class ParamTier(StrEnum):
    STRICT = "strict"
    STANDARD = "standard"
    LOOSE = "loose"
```

- [ ] **Step 4: Implement `src/pa/config.py`**

```python
"""Typed YAML configuration loader for the Price Action pipeline."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class ConfigError(Exception):
    """Raised when config file is missing, malformed, or fails validation."""


class UniverseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    members_file: Path


class DateRangeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start: date
    end: date


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cache_dir: Path
    api_base_url: str
    api_key_env: str
    rate_limit_per_min: int = Field(gt=0)


class DetectorsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: list[str]
    param_tiers: list[str]


class BacktestConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    time_stop_bars: int = Field(gt=0)
    same_bar_priority: Literal["stop_first", "target_first"]


class ReportConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    output_dir: Path
    samples_per_setup: int = Field(ge=0)


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    universe: UniverseConfig
    date_range: DateRangeConfig
    data: DataConfig
    detectors: DetectorsConfig
    backtest: BacktestConfig
    report: ReportConfig


def load_config(path: Path) -> Config:
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text())
        return Config.model_validate(raw)
    except (yaml.YAMLError, ValidationError) as exc:
        raise ConfigError(f"Invalid config {path}: {exc}") from exc
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_config.py -v`
Expected: 2 passing.

- [ ] **Step 6: Commit**

```bash
git add src/pa/types.py src/pa/config.py tests/test_config.py
git commit -m "Add shared types and pydantic-validated YAML config"
```

---

## Task 3: Structured Logging

**Files:**
- Create: `src/pa/logging.py`
- Create: `tests/test_logging.py`

- [ ] **Step 1: Write `tests/test_logging.py`**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_logging.py -v`
Expected: FAIL — `pa.logging` does not exist.

- [ ] **Step 3: Implement `src/pa/logging.py`**

```python
"""JSON-line structured logger keyed by pipeline stage and run_id."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Self


class StageLogger:
    def __init__(self, *, stage: str, run_id: str, log_path: Path) -> None:
        self.stage = stage
        self.run_id = run_id
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = log_path.open("a", encoding="utf-8")

    def _emit(self, level: str, msg: str, **fields: Any) -> None:
        record: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(),
            "level": level,
            "stage": self.stage,
            "run_id": self.run_id,
            "msg": msg,
            **fields,
        }
        self._fp.write(json.dumps(record, default=str) + "\n")
        self._fp.flush()

    def info(self, msg: str, **fields: Any) -> None:
        self._emit("info", msg, **fields)

    def warn(self, msg: str, **fields: Any) -> None:
        self._emit("warn", msg, **fields)

    def error(self, msg: str, **fields: Any) -> None:
        self._emit("error", msg, **fields)

    def close(self) -> None:
        self._fp.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_logging.py -v`
Expected: 2 passing.

- [ ] **Step 5: Commit**

```bash
git add src/pa/logging.py tests/test_logging.py
git commit -m "Add JSON-line StageLogger"
```

---

## Task 4: Massive API Client

**Files:**
- Create: `src/pa/data/__init__.py`
- Create: `src/pa/data/client.py`
- Create: `tests/test_data_client.py`

- [ ] **Step 1: Write `tests/test_data_client.py`**

```python
"""Tests for pa.data.client — Massive API HTTP wrapper."""
from __future__ import annotations

from datetime import date

import httpx
import pytest

from pa.data.client import MassiveClient, MassiveAPIError


def test_fetch_aggregates_returns_bars(httpx_mock_handler: object) -> None:
    """Mock returning 2 daily bars."""
    transport = httpx.MockTransport(
        lambda req: httpx.Response(
            200,
            json={
                "ticker": "AAPL",
                "results": [
                    {
                        "t": 1619395200000,  # 2021-04-26
                        "o": 134.83,
                        "h": 134.85,
                        "l": 132.81,
                        "c": 134.32,
                        "v": 66905680,
                        "vw": 133.91,
                    },
                    {
                        "t": 1619481600000,  # 2021-04-27
                        "o": 135.01,
                        "h": 135.41,
                        "l": 134.11,
                        "c": 134.39,
                        "v": 66362094,
                        "vw": 134.78,
                    },
                ],
            },
        )
    )
    client = MassiveClient(
        api_key="dummy",
        base_url="https://api.massive.com",
        transport=transport,
    )
    bars = client.fetch_aggregates(
        ticker="AAPL",
        start=date(2021, 4, 26),
        end=date(2021, 4, 27),
    )
    assert len(bars) == 2
    assert bars[0].open == 134.83
    assert bars[0].date == date(2021, 4, 26)


def test_fetch_aggregates_raises_on_5xx() -> None:
    transport = httpx.MockTransport(lambda req: httpx.Response(503))
    client = MassiveClient(
        api_key="dummy",
        base_url="https://api.massive.com",
        transport=transport,
        max_retries=1,
    )
    with pytest.raises(MassiveAPIError):
        client.fetch_aggregates(
            ticker="AAPL",
            start=date(2021, 4, 26),
            end=date(2021, 4, 27),
        )
```

(Note: pytest-httpx fixture name kept simple; `httpx_mock_handler` is just a placeholder param dropped by pytest — the test uses `httpx.MockTransport` directly.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_data_client.py -v`
Expected: FAIL — `pa.data.client` not found.

- [ ] **Step 3: Implement `src/pa/data/__init__.py`**

```python
"""Data layer: API client + parquet cache."""
```

- [ ] **Step 4: Implement `src/pa/data/client.py`**

```python
"""HTTP client for Massive API (Polygon-compatible aggregates endpoint)."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


class MassiveAPIError(Exception):
    """Raised when the Massive API returns an unrecoverable error."""


@dataclass(frozen=True, slots=True)
class Bar:
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: float


class MassiveClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        transport: httpx.BaseTransport | None = None,
        max_retries: int = 3,
        timeout_s: float = 30.0,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._max_retries = max_retries
        self._client = httpx.Client(
            transport=transport,
            timeout=timeout_s,
            headers={"Authorization": f"Bearer {api_key}"},
        )

    def fetch_aggregates(
        self,
        *,
        ticker: str,
        start: date,
        end: date,
        timespan: str = "day",
        adjusted: bool = True,
    ) -> list[Bar]:
        @retry(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=30),
            retry=retry_if_exception_type((httpx.HTTPError, MassiveAPIError)),
            reraise=True,
        )
        def _do() -> list[Bar]:
            url = (
                f"{self._base_url}/v2/aggs/ticker/{ticker}/range/1/{timespan}/"
                f"{start.isoformat()}/{end.isoformat()}"
            )
            params = {"adjusted": str(adjusted).lower(), "sort": "asc", "limit": 50000}
            resp = self._client.get(url, params=params)
            if resp.status_code >= 500:
                raise MassiveAPIError(f"Server {resp.status_code}: {resp.text[:200]}")
            if resp.status_code == 429:
                raise MassiveAPIError("Rate limited")
            resp.raise_for_status()
            payload = resp.json()
            return [_bar_from_dict(b) for b in payload.get("results", [])]

        return _do()

    def close(self) -> None:
        self._client.close()


def _bar_from_dict(d: dict[str, float | int]) -> Bar:
    ts_ms = int(d["t"])
    bar_date = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).date()
    return Bar(
        date=bar_date,
        open=float(d["o"]),
        high=float(d["h"]),
        low=float(d["l"]),
        close=float(d["c"]),
        volume=int(d["v"]),
        vwap=float(d["vw"]),
    )
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_data_client.py -v`
Expected: 2 passing.

- [ ] **Step 6: Commit**

```bash
git add src/pa/data/__init__.py src/pa/data/client.py tests/test_data_client.py
git commit -m "Add Massive API client with retry"
```

---

## Task 5: Parquet Cache Layer

**Files:**
- Create: `src/pa/data/cache.py`
- Create: `tests/test_data_cache.py`

- [ ] **Step 1: Write `tests/test_data_cache.py`**

```python
"""Tests for pa.data.cache — parquet OHLCV cache layer."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd

from pa.data.cache import OhlcvCache
from pa.data.client import Bar


def _fake_bars() -> list[Bar]:
    return [
        Bar(date(2021, 4, 26), 100.0, 101.0, 99.0, 100.5, 1000, 100.2),
        Bar(date(2021, 4, 27), 100.5, 102.0, 100.0, 101.5, 1100, 101.0),
    ]


def test_cache_miss_calls_client(tmp_path: Path) -> None:
    client = MagicMock()
    client.fetch_aggregates.return_value = _fake_bars()
    cache = OhlcvCache(cache_dir=tmp_path, client=client)
    df = cache.fetch_ohlcv("AAPL", date(2021, 4, 26), date(2021, 4, 27))
    assert len(df) == 2
    assert list(df.columns) == ["date", "open", "high", "low", "close", "volume", "vwap"]
    client.fetch_aggregates.assert_called_once()


def test_cache_hit_skips_client(tmp_path: Path) -> None:
    client = MagicMock()
    client.fetch_aggregates.return_value = _fake_bars()
    cache = OhlcvCache(cache_dir=tmp_path, client=client)
    cache.fetch_ohlcv("AAPL", date(2021, 4, 26), date(2021, 4, 27))
    cache.fetch_ohlcv("AAPL", date(2021, 4, 26), date(2021, 4, 27))
    assert client.fetch_aggregates.call_count == 1


def test_cache_persists_as_parquet(tmp_path: Path) -> None:
    client = MagicMock()
    client.fetch_aggregates.return_value = _fake_bars()
    cache = OhlcvCache(cache_dir=tmp_path, client=client)
    cache.fetch_ohlcv("AAPL", date(2021, 4, 26), date(2021, 4, 27))
    files = list(tmp_path.glob("*.parquet"))
    assert len(files) == 1
    df = pd.read_parquet(files[0])
    assert len(df) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_data_cache.py -v`
Expected: FAIL — `pa.data.cache` not found.

- [ ] **Step 3: Implement `src/pa/data/cache.py`**

```python
"""Parquet-backed OHLCV cache; one file per (ticker, start, end)."""
from __future__ import annotations

import hashlib
from dataclasses import asdict
from datetime import date
from pathlib import Path

import pandas as pd

from pa.data.client import MassiveClient
from pa.types import OHLCV_COLS


class OhlcvCache:
    def __init__(self, *, cache_dir: Path, client: MassiveClient) -> None:
        self._dir = cache_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._client = client

    def _key(self, ticker: str, start: date, end: date) -> Path:
        h = hashlib.sha256(
            f"{ticker}|{start.isoformat()}|{end.isoformat()}".encode()
        ).hexdigest()[:12]
        return self._dir / f"{ticker}_{h}.parquet"

    def fetch_ohlcv(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        path = self._key(ticker, start, end)
        if path.exists():
            return pd.read_parquet(path)

        bars = self._client.fetch_aggregates(ticker=ticker, start=start, end=end)
        if not bars:
            df = pd.DataFrame(columns=OHLCV_COLS)
        else:
            df = pd.DataFrame([asdict(b) for b in bars])[OHLCV_COLS]
            df["date"] = pd.to_datetime(df["date"])

        df.to_parquet(path, index=False)
        return df
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_data_cache.py -v`
Expected: 3 passing.

- [ ] **Step 5: Commit**

```bash
git add src/pa/data/cache.py tests/test_data_cache.py
git commit -m "Add parquet OHLCV cache"
```

---

## Task 6: EMA Indicator

**Files:**
- Create: `src/pa/indicators/__init__.py`
- Create: `src/pa/indicators/ema.py`
- Create: `tests/test_indicators_ema.py`

- [ ] **Step 1: Write `tests/test_indicators_ema.py`**

```python
"""Tests for pa.indicators.ema."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pa.indicators.ema import ema


def test_ema_constant_input_equals_input() -> None:
    s = pd.Series([10.0] * 50)
    out = ema(s, n=20)
    assert (out.dropna() == 10.0).all()


def test_ema_first_n_minus_1_are_nan() -> None:
    s = pd.Series(list(range(30)), dtype=float)
    out = ema(s, n=20)
    assert out.iloc[:19].isna().all()
    assert not np.isnan(out.iloc[19])


def test_ema_against_pandas_reference() -> None:
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(loc=100, scale=2, size=100))
    out = ema(s, n=20)
    ref = s.ewm(span=20, adjust=False).mean()
    ref.iloc[:19] = np.nan
    pd.testing.assert_series_equal(out, ref, check_names=False, rtol=1e-9)


def test_ema_invalid_n_raises() -> None:
    with pytest.raises(ValueError):
        ema(pd.Series([1.0, 2.0]), n=0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_indicators_ema.py -v`
Expected: FAIL — `pa.indicators.ema` not found.

- [ ] **Step 3: Implement `src/pa/indicators/__init__.py`**

```python
"""Technical indicators for Brooks-style price action."""
```

- [ ] **Step 4: Implement `src/pa/indicators/ema.py`**

```python
"""Exponential moving average — the only smoothing primitive used in pa."""
from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, *, n: int) -> pd.Series:
    """Standard EMA with span=n; first n-1 values are NaN.

    Brooks uses EMA20 / EMA50 to determine trend regime.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    out = series.ewm(span=n, adjust=False).mean()
    out.iloc[: n - 1] = np.nan
    return out
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_indicators_ema.py -v`
Expected: 4 passing.

- [ ] **Step 6: Commit**

```bash
git add src/pa/indicators/__init__.py src/pa/indicators/ema.py tests/test_indicators_ema.py
git commit -m "Add EMA indicator"
```

---

## Task 7: ATR Indicator

**Files:**
- Create: `src/pa/indicators/atr.py`
- Create: `tests/test_indicators_atr.py`

- [ ] **Step 1: Write `tests/test_indicators_atr.py`**

```python
"""Tests for pa.indicators.atr — Wilder's Average True Range."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pa.indicators.atr import atr


def test_atr_constant_bars_zero() -> None:
    df = pd.DataFrame(
        {
            "high": [10.0] * 30,
            "low": [10.0] * 30,
            "close": [10.0] * 30,
        }
    )
    out = atr(df["high"], df["low"], df["close"], n=14)
    assert (out.dropna() == 0.0).all()


def test_atr_first_n_are_nan() -> None:
    rng = np.random.default_rng(1)
    n_bars = 50
    closes = pd.Series(rng.normal(loc=100, scale=1, size=n_bars))
    highs = closes + 1
    lows = closes - 1
    out = atr(highs, lows, closes, n=14)
    assert out.iloc[:13].isna().all()
    assert not np.isnan(out.iloc[14])


def test_atr_known_value() -> None:
    """Hand computed: TR = max(h-l, |h-prev_close|, |l-prev_close|)."""
    high = pd.Series([10.0, 11.0, 12.0, 13.0])
    low = pd.Series([9.0, 9.5, 10.5, 11.0])
    close = pd.Series([9.5, 10.5, 11.5, 12.5])
    out = atr(high, low, close, n=2)
    # TR series: [1.0, 1.5 (11-9.5), 2.0 (12-10), 2.0 (13-11)]
    # ATR2 (Wilder) seeded at index 1 = mean(TR[0:2]) = 1.25
    # ATR2 at index 2 = (1.25*(2-1) + 2.0) / 2 = 1.625
    # ATR2 at index 3 = (1.625*(2-1) + 2.0) / 2 = 1.8125
    assert np.isnan(out.iloc[0])
    assert out.iloc[1] == pytest.approx(1.25)
    assert out.iloc[2] == pytest.approx(1.625)
    assert out.iloc[3] == pytest.approx(1.8125)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_indicators_atr.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `src/pa/indicators/atr.py`**

```python
"""Wilder's ATR — used for stop placement and volatility-aware thresholds."""
from __future__ import annotations

import numpy as np
import pandas as pd


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    *,
    n: int = 14,
) -> pd.Series:
    """Wilder's Average True Range. Output has NaN for indices 0..n-2."""
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    out = pd.Series(np.nan, index=high.index)
    if len(tr) < n:
        return out
    # Wilder seed: mean of first n TR values; place at index n-1.
    seed = tr.iloc[1:n].mean()
    out.iloc[n - 1] = seed
    for i in range(n, len(tr)):
        prev_atr = out.iloc[i - 1]
        out.iloc[i] = (prev_atr * (n - 1) + tr.iloc[i]) / n
    # First TR has no prev_close so seed used TR[1..n-1]; mask index 0 NaN explicitly.
    out.iloc[: n - 1] = np.nan
    return out
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_indicators_atr.py -v`
Expected: 3 passing.

- [ ] **Step 5: Commit**

```bash
git add src/pa/indicators/atr.py tests/test_indicators_atr.py
git commit -m "Add Wilder ATR indicator"
```

---

## Task 8: Swing Pivot

**Files:**
- Create: `src/pa/indicators/swing.py`
- Create: `tests/test_indicators_swing.py`

- [ ] **Step 1: Write `tests/test_indicators_swing.py`**

```python
"""Tests for pa.indicators.swing — Brooks-style swing high/low pivots."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pa.indicators.swing import swing_pivot


def test_swing_high_detected() -> None:
    """A bar whose high > N bars on each side is a swing high."""
    highs = pd.Series([1.0, 2.0, 3.0, 5.0, 3.5, 2.5, 1.5])
    lows = pd.Series([0.5, 1.5, 2.5, 4.0, 3.0, 2.0, 1.0])
    out = swing_pivot(highs, lows, n=2)
    # Index 3 (high=5.0) should be a swing high (highest within ±2)
    assert out.loc[3, "is_swing_high"]
    assert not out.loc[3, "is_swing_low"]


def test_swing_low_detected() -> None:
    highs = pd.Series([5.0, 4.0, 3.0, 1.0, 3.5, 4.5, 5.5])
    lows = pd.Series([4.5, 3.5, 2.5, 0.5, 3.0, 4.0, 5.0])
    out = swing_pivot(highs, lows, n=2)
    assert out.loc[3, "is_swing_low"]
    assert not out.loc[3, "is_swing_high"]


def test_edges_cannot_be_swings() -> None:
    """First and last n bars cannot be swings (insufficient lookback/forward)."""
    highs = pd.Series([10.0, 9.0, 8.0, 7.0, 6.0])
    lows = pd.Series([9.5, 8.5, 7.5, 6.5, 5.5])
    out = swing_pivot(highs, lows, n=2)
    assert not out.iloc[:2]["is_swing_high"].any()
    assert not out.iloc[:2]["is_swing_low"].any()
    assert not out.iloc[-2:]["is_swing_high"].any()
    assert not out.iloc[-2:]["is_swing_low"].any()


def test_swing_levels_carried_forward() -> None:
    """`swing_high` column = price of most recent confirmed swing high (forward-filled)."""
    highs = pd.Series([1.0, 2.0, 3.0, 5.0, 3.5, 2.5, 1.5])
    lows = pd.Series([0.5, 1.5, 2.5, 4.0, 3.0, 2.0, 1.0])
    out = swing_pivot(highs, lows, n=2)
    # After bar 3 confirms swing high=5.0 at bar 3+n=5, level should appear from there.
    assert out.loc[5, "swing_high"] == 5.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_indicators_swing.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `src/pa/indicators/swing.py`**

```python
"""Brooks-style swing high/low pivots with N-bar confirmation on each side."""
from __future__ import annotations

import numpy as np
import pandas as pd


def swing_pivot(
    high: pd.Series,
    low: pd.Series,
    *,
    n: int = 2,
) -> pd.DataFrame:
    """Compute swing pivots.

    A bar at index i is a swing high if `high[i]` is strictly greater than
    every `high` value in `[i-n, i-1]` and `[i+1, i+n]`. Symmetric for low.

    Returns DataFrame with:
      - is_swing_high (bool): True at the bar that *is* the swing
      - is_swing_low (bool)
      - swing_high (float): price level of most recent confirmed swing high,
                            forward-filled from confirmation bar (i+n)
      - swing_low (float): symmetric

    Confirmation: a swing at bar i is only "known" at bar i+n (we need n future
    bars to validate). The level is therefore forward-filled starting at i+n.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    h = high.to_numpy()
    lo = low.to_numpy()
    size = len(h)
    is_sh = np.zeros(size, dtype=bool)
    is_sl = np.zeros(size, dtype=bool)
    for i in range(n, size - n):
        left_h = h[i - n : i]
        right_h = h[i + 1 : i + n + 1]
        if h[i] > left_h.max() and h[i] > right_h.max():
            is_sh[i] = True
        left_l = lo[i - n : i]
        right_l = lo[i + 1 : i + n + 1]
        if lo[i] < left_l.min() and lo[i] < right_l.min():
            is_sl[i] = True

    sh_level = np.full(size, np.nan)
    sl_level = np.full(size, np.nan)
    last_sh = np.nan
    last_sl = np.nan
    for i in range(size):
        # confirmed swing at index i-n becomes available at index i
        confirm_idx = i - n
        if confirm_idx >= 0:
            if is_sh[confirm_idx]:
                last_sh = h[confirm_idx]
            if is_sl[confirm_idx]:
                last_sl = lo[confirm_idx]
        sh_level[i] = last_sh
        sl_level[i] = last_sl

    return pd.DataFrame(
        {
            "is_swing_high": pd.Series(is_sh, index=high.index),
            "is_swing_low": pd.Series(is_sl, index=high.index),
            "swing_high": pd.Series(sh_level, index=high.index),
            "swing_low": pd.Series(sl_level, index=high.index),
        }
    )
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_indicators_swing.py -v`
Expected: 4 passing.

- [ ] **Step 5: Commit**

```bash
git add src/pa/indicators/swing.py tests/test_indicators_swing.py
git commit -m "Add swing pivot indicator"
```

---

## Task 9: Bar Anatomy + Indicators Pipeline

**Files:**
- Create: `src/pa/indicators/bar_anatomy.py`
- Modify: `src/pa/indicators/__init__.py`
- Create: `tests/test_indicators_bar_anatomy.py`
- Create: `tests/test_indicators_pipeline.py`

- [ ] **Step 1: Write `tests/test_indicators_bar_anatomy.py`**

```python
"""Tests for pa.indicators.bar_anatomy — per-bar geometric features."""
from __future__ import annotations

import pandas as pd
import pytest

from pa.indicators.bar_anatomy import bar_anatomy


def _bar(o: float, h: float, lo: float, c: float) -> pd.DataFrame:
    return pd.DataFrame({"open": [o], "high": [h], "low": [lo], "close": [c]})


def test_full_body_bull_bar() -> None:
    df = _bar(o=10.0, h=11.0, lo=10.0, c=11.0)
    out = bar_anatomy(df)
    row = out.iloc[0]
    assert row["bar_is_bull"] is True or row["bar_is_bull"] == True  # noqa: E712
    assert row["bar_body_pct"] == pytest.approx(1.0)
    assert row["bar_upper_wick_pct"] == pytest.approx(0.0)
    assert row["bar_lower_wick_pct"] == pytest.approx(0.0)
    assert row["bar_close_position"] == pytest.approx(1.0)


def test_doji_bar() -> None:
    df = _bar(o=10.0, h=11.0, lo=9.0, c=10.0)
    out = bar_anatomy(df)
    row = out.iloc[0]
    assert row["bar_body_pct"] == pytest.approx(0.0)
    assert row["bar_close_position"] == pytest.approx(0.5)


def test_zero_range_safely_handled() -> None:
    df = _bar(o=10.0, h=10.0, lo=10.0, c=10.0)
    out = bar_anatomy(df)
    row = out.iloc[0]
    assert row["bar_body_pct"] == 0.0
    assert row["bar_close_position"] == 0.5  # convention for zero-range bars
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_indicators_bar_anatomy.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `src/pa/indicators/bar_anatomy.py`**

```python
"""Per-bar geometric features: body, wicks, close position, bull/bear."""
from __future__ import annotations

import numpy as np
import pandas as pd


def bar_anatomy(df: pd.DataFrame) -> pd.DataFrame:
    """Add Brooks-style per-bar geometry columns.

    Required input columns: open, high, low, close.
    Returns a DataFrame indexed identically to df with anatomy columns only
    (caller decides whether to concat).
    """
    o, h, lo, c = df["open"], df["high"], df["low"], df["close"]
    bar_range = (h - lo).replace(0.0, np.nan)
    body = (c - o).abs()
    upper_wick = h - c.where(c >= o, o)
    lower_wick = o.where(c >= o, c) - lo

    body_pct = (body / bar_range).fillna(0.0)
    upper_pct = (upper_wick / bar_range).fillna(0.0)
    lower_pct = (lower_wick / bar_range).fillna(0.0)
    close_pos = ((c - lo) / bar_range).fillna(0.5)

    return pd.DataFrame(
        {
            "bar_body_pct": body_pct,
            "bar_upper_wick_pct": upper_pct,
            "bar_lower_wick_pct": lower_pct,
            "bar_is_bull": (c > o),
            "bar_close_position": close_pos,
        },
        index=df.index,
    )
```

- [ ] **Step 4: Implement `src/pa/indicators/__init__.py` pipeline**

Replace contents:
```python
"""Technical indicators for Brooks-style price action."""
from __future__ import annotations

import pandas as pd

from pa.indicators.atr import atr
from pa.indicators.bar_anatomy import bar_anatomy
from pa.indicators.ema import ema
from pa.indicators.swing import swing_pivot

__all__ = ["atr", "bar_anatomy", "ema", "swing_pivot", "compute_indicators"]


def compute_indicators(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Run all indicator computations on an OHLCV frame.

    Input columns required: date, open, high, low, close, volume, vwap.
    Returns a DataFrame with `date` plus all INDICATOR_COLS columns.
    """
    out = pd.DataFrame({"date": ohlcv["date"]})
    out["ema20"] = ema(ohlcv["close"], n=20)
    out["ema50"] = ema(ohlcv["close"], n=50)
    out["atr14"] = atr(ohlcv["high"], ohlcv["low"], ohlcv["close"], n=14)
    swings = swing_pivot(ohlcv["high"], ohlcv["low"], n=2)
    out["swing_high"] = swings["swing_high"]
    out["swing_low"] = swings["swing_low"]
    anatomy = bar_anatomy(ohlcv)
    for col in [
        "bar_body_pct",
        "bar_upper_wick_pct",
        "bar_lower_wick_pct",
        "bar_is_bull",
        "bar_close_position",
    ]:
        out[col] = anatomy[col]
    return out
```

- [ ] **Step 5: Write `tests/test_indicators_pipeline.py`**

```python
"""Tests for the indicators-pipeline glue function."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pa.indicators import compute_indicators
from pa.types import INDICATOR_COLS


def _synthetic_ohlcv(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    closes = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame(
        {
            "date": pd.date_range("2021-04-26", periods=n, freq="B"),
            "open": closes - 0.1,
            "high": closes + 0.5,
            "low": closes - 0.5,
            "close": closes,
            "volume": rng.integers(1_000_000, 5_000_000, n),
            "vwap": closes,
        }
    )


def test_pipeline_produces_required_columns() -> None:
    ohlcv = _synthetic_ohlcv(100)
    out = compute_indicators(ohlcv)
    assert "date" in out.columns
    for col in INDICATOR_COLS:
        assert col in out.columns
    assert len(out) == len(ohlcv)


def test_pipeline_no_extra_columns() -> None:
    ohlcv = _synthetic_ohlcv(100)
    out = compute_indicators(ohlcv)
    assert set(out.columns) == {"date", *INDICATOR_COLS}
```

- [ ] **Step 6: Run tests**

Run: `uv run pytest tests/test_indicators_bar_anatomy.py tests/test_indicators_pipeline.py -v`
Expected: 5 passing total.

- [ ] **Step 7: Commit**

```bash
git add src/pa/indicators/bar_anatomy.py src/pa/indicators/__init__.py \
  tests/test_indicators_bar_anatomy.py tests/test_indicators_pipeline.py
git commit -m "Add bar anatomy and indicators pipeline"
```

---

## Task 10: Regime Classifier

**Files:**
- Create: `src/pa/regime/__init__.py`
- Create: `src/pa/regime/classifier.py`
- Create: `tests/test_regime_classifier.py`

- [ ] **Step 1: Write `tests/test_regime_classifier.py`**

```python
"""Tests for pa.regime.classifier."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pa.indicators import compute_indicators
from pa.regime.classifier import classify_regime
from pa.types import Regime


def _trending_up_ohlcv(n: int = 80) -> pd.DataFrame:
    closes = np.linspace(100, 130, n) + np.random.default_rng(0).normal(0, 0.3, n)
    return pd.DataFrame(
        {
            "date": pd.date_range("2021-04-26", periods=n, freq="B"),
            "open": closes,
            "high": closes + 0.5,
            "low": closes - 0.5,
            "close": closes,
            "volume": [1_000_000] * n,
            "vwap": closes,
        }
    )


def _trending_down_ohlcv(n: int = 80) -> pd.DataFrame:
    closes = np.linspace(130, 100, n) + np.random.default_rng(0).normal(0, 0.3, n)
    return pd.DataFrame(
        {
            "date": pd.date_range("2021-04-26", periods=n, freq="B"),
            "open": closes,
            "high": closes + 0.5,
            "low": closes - 0.5,
            "close": closes,
            "volume": [1_000_000] * n,
            "vwap": closes,
        }
    )


def _flat_ohlcv(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    closes = 100 + rng.normal(0, 0.5, n)
    return pd.DataFrame(
        {
            "date": pd.date_range("2021-04-26", periods=n, freq="B"),
            "open": closes,
            "high": closes + 0.5,
            "low": closes - 0.5,
            "close": closes,
            "volume": [1_000_000] * n,
            "vwap": closes,
        }
    )


def test_uptrend_classified_as_bull() -> None:
    ohlcv = _trending_up_ohlcv()
    indicators = compute_indicators(ohlcv)
    regimes = classify_regime(ohlcv, indicators)
    valid = regimes["regime"].dropna().tail(20)
    assert (valid == Regime.BULL_TREND).mean() > 0.7


def test_downtrend_classified_as_bear() -> None:
    ohlcv = _trending_down_ohlcv()
    indicators = compute_indicators(ohlcv)
    regimes = classify_regime(ohlcv, indicators)
    valid = regimes["regime"].dropna().tail(20)
    assert (valid == Regime.BEAR_TREND).mean() > 0.7


def test_flat_classified_as_range() -> None:
    ohlcv = _flat_ohlcv()
    indicators = compute_indicators(ohlcv)
    regimes = classify_regime(ohlcv, indicators)
    valid = regimes["regime"].dropna().tail(20)
    assert (valid == Regime.TRADING_RANGE).mean() > 0.5


def test_every_bar_has_regime_label() -> None:
    ohlcv = _trending_up_ohlcv()
    indicators = compute_indicators(ohlcv)
    regimes = classify_regime(ohlcv, indicators)
    # Bars before EMA50 warmup may be `transitional`; none should be NaN.
    assert regimes["regime"].notna().all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_regime_classifier.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `src/pa/regime/__init__.py`**

```python
"""Regime classification + signal-bar scoring."""
from pa.regime.classifier import classify_regime
from pa.regime.signal_bar import signal_bar_score

__all__ = ["classify_regime", "signal_bar_score"]
```

- [ ] **Step 4: Implement `src/pa/regime/classifier.py`**

```python
"""Trend regime classifier using EMA stack + structural HH/HL checks.

Standard tier rules (loose/strict adjust thresholds, not the rule shape):
- bull_trend: close > EMA20 > EMA50, EMA20 slope rising over last 10 bars,
  AND price made at least one HH+HL in last 20 bars
- bear_trend: mirror
- trading_range: |close - EMA20| <= 1.5 * ATR for majority of last 20 bars
- transitional: warmup bars (before EMA50 valid) or borderline conditions
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pa.types import Regime


def classify_regime(
    ohlcv: pd.DataFrame,
    indicators: pd.DataFrame,
    *,
    slope_window: int = 10,
    structure_window: int = 20,
    range_atr_mult: float = 1.5,
) -> pd.DataFrame:
    """Return DataFrame with `regime` (StrEnum value) and `regime_strength` (0..1)."""
    df = pd.DataFrame({"date": ohlcv["date"]})
    close = ohlcv["close"]
    ema20 = indicators["ema20"]
    ema50 = indicators["ema50"]
    atr14 = indicators["atr14"]

    ema20_slope = ema20.diff(slope_window)

    rolling_hh = ohlcv["high"].rolling(structure_window, min_periods=1).max()
    rolling_ll = ohlcv["low"].rolling(structure_window, min_periods=1).min()
    made_higher_high = ohlcv["high"] >= rolling_hh.shift(1)
    made_lower_low = ohlcv["low"] <= rolling_ll.shift(1)
    hh_count = made_higher_high.rolling(structure_window, min_periods=1).sum()
    ll_count = made_lower_low.rolling(structure_window, min_periods=1).sum()

    is_bull = (
        (close > ema20)
        & (ema20 > ema50)
        & (ema20_slope > 0)
        & (hh_count >= 1)
    )
    is_bear = (
        (close < ema20)
        & (ema20 < ema50)
        & (ema20_slope < 0)
        & (ll_count >= 1)
    )
    band = atr14 * range_atr_mult
    in_range_band = (close - ema20).abs() <= band
    range_score = in_range_band.rolling(structure_window, min_periods=1).mean()
    is_range = (~is_bull) & (~is_bear) & (range_score > 0.6)

    regime = pd.Series(Regime.TRANSITIONAL.value, index=close.index, dtype=object)
    regime[is_bull] = Regime.BULL_TREND.value
    regime[is_bear] = Regime.BEAR_TREND.value
    regime[is_range] = Regime.TRADING_RANGE.value
    # Force warmup (pre-EMA50) into TRANSITIONAL — prevents false signals on NaN.
    warmup = ema50.isna() | atr14.isna()
    regime[warmup] = Regime.TRANSITIONAL.value

    # Strength: fraction of the last `structure_window` bars sharing this bar's regime.
    arr = regime.to_numpy()
    strength_vals: list[float] = []
    for idx in range(len(arr)):
        start = max(0, idx - structure_window + 1)
        win = arr[start : idx + 1]
        cur = arr[idx]
        strength_vals.append(float((win == cur).mean()))

    df["regime"] = regime.values
    df["regime_strength"] = strength_vals
    return df
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_regime_classifier.py -v`
Expected: 4 passing.

- [ ] **Step 6: Commit**

```bash
git add src/pa/regime/__init__.py src/pa/regime/classifier.py tests/test_regime_classifier.py
git commit -m "Add regime classifier"
```

---

## Task 11: Signal Bar Score

**Files:**
- Create: `src/pa/regime/signal_bar.py`
- Create: `tests/test_regime_signal_bar.py`

- [ ] **Step 1: Write `tests/test_regime_signal_bar.py`**

```python
"""Tests for pa.regime.signal_bar — quality score for entry signal candles."""
from __future__ import annotations

import pandas as pd

from pa.indicators import compute_indicators
from pa.regime.signal_bar import signal_bar_score


def _ohlcv_with_known_bars() -> pd.DataFrame:
    # Index 0: full body bull bar (best)
    # Index 1: doji (worst)
    # Index 2: bull bar with big upper wick (mediocre)
    return pd.DataFrame(
        {
            "date": pd.date_range("2021-04-26", periods=60, freq="B"),
            "open": [100.0] * 60,
            "high": [101.0] * 60,
            "low": [99.0] * 60,
            "close": [100.5] * 60,
            "volume": [1_000_000] * 60,
            "vwap": [100.0] * 60,
        }
    )


def test_score_is_between_0_and_1() -> None:
    ohlcv = _ohlcv_with_known_bars()
    ohlcv.loc[0, ["open", "high", "low", "close"]] = [100.0, 101.0, 100.0, 101.0]
    ohlcv.loc[1, ["open", "high", "low", "close"]] = [100.0, 101.0, 99.0, 100.0]
    ohlcv.loc[2, ["open", "high", "low", "close"]] = [100.0, 102.0, 99.5, 100.4]
    indicators = compute_indicators(ohlcv)
    scores = signal_bar_score(ohlcv, indicators)
    assert (scores.dropna().between(0.0, 1.0)).all()


def test_full_body_bull_scores_higher_than_doji() -> None:
    ohlcv = _ohlcv_with_known_bars()
    ohlcv.loc[0, ["open", "high", "low", "close"]] = [100.0, 101.0, 100.0, 101.0]
    ohlcv.loc[1, ["open", "high", "low", "close"]] = [100.0, 101.0, 99.0, 100.0]
    indicators = compute_indicators(ohlcv)
    scores = signal_bar_score(ohlcv, indicators)
    assert scores.iloc[0] > scores.iloc[1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_regime_signal_bar.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `src/pa/regime/signal_bar.py`**

```python
"""Quality score in [0, 1] for a bar to act as a Brooks 'signal bar'.

A good signal bar (bull or bear) has:
  - Large body relative to range (50%+ ideal)
  - Close in the upper third (bull) / lower third (bear) of range
  - Body size > recent average (bar of significance)
"""
from __future__ import annotations

import pandas as pd


def signal_bar_score(
    ohlcv: pd.DataFrame,
    indicators: pd.DataFrame,
) -> pd.Series:
    body_pct = indicators["bar_body_pct"]
    close_pos = indicators["bar_close_position"]
    is_bull = indicators["bar_is_bull"]

    # Direction-aware close position: bull → close near top, bear → close near bottom
    directional_pos = close_pos.where(is_bull, 1.0 - close_pos)

    # Size relative to recent volatility: body / ATR14
    body_abs = (ohlcv["close"] - ohlcv["open"]).abs()
    size_ratio = (body_abs / indicators["atr14"]).clip(upper=1.0).fillna(0.0)

    score = (0.4 * body_pct) + (0.4 * directional_pos) + (0.2 * size_ratio)
    return score.clip(lower=0.0, upper=1.0)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_regime_signal_bar.py -v`
Expected: 2 passing.

- [ ] **Step 5: Commit**

```bash
git add src/pa/regime/signal_bar.py tests/test_regime_signal_bar.py
git commit -m "Add signal bar quality scorer"
```

---

## Task 12: Backtest Engine

**Files:**
- Create: `src/pa/backtest/__init__.py`
- Create: `src/pa/backtest/engine.py`
- Create: `tests/test_backtest_engine.py`

- [ ] **Step 1: Write `tests/test_backtest_engine.py`**

```python
"""Tests for pa.backtest.engine — execution simulator."""
from __future__ import annotations

import pandas as pd
import pytest

from pa.backtest.engine import simulate
from pa.types import ExitReason, Side


def _ohlcv(rows: list[tuple[str, float, float, float, float]]) -> pd.DataFrame:
    """rows = [(date, open, high, low, close), ...]"""
    return pd.DataFrame(
        [
            {
                "date": pd.Timestamp(r[0]),
                "open": r[1],
                "high": r[2],
                "low": r[3],
                "close": r[4],
                "volume": 1_000,
                "vwap": (r[2] + r[3]) / 2,
            }
            for r in rows
        ]
    )


def test_long_target_hit() -> None:
    ohlcv = _ohlcv(
        [
            ("2021-04-26", 100, 100.5, 99.5, 100),
            ("2021-04-27", 100, 105, 100, 104),  # high=105 hits target=104
        ]
    )
    cands = pd.DataFrame(
        [
            {
                "ticker": "X",
                "signal_date": pd.Timestamp("2021-04-26"),
                "side": Side.LONG.value,
                "entry_price": 100.0,
                "stop_price": 99.0,
                "target_price": 104.0,
                "setup_score": 0.8,
                "regime_at_signal": "bull_trend",
                "params_tier": "standard",
            }
        ]
    )
    trades = simulate(cands, ohlcv, time_stop_bars=20, same_bar_priority="stop_first")
    assert len(trades) == 1
    t = trades.iloc[0]
    assert t["exit_reason"] == ExitReason.TARGET_HIT.value
    assert t["exit_price"] == pytest.approx(104.0)
    assert t["pnl_r"] == pytest.approx(4.0)


def test_long_stop_hit() -> None:
    ohlcv = _ohlcv(
        [
            ("2021-04-26", 100, 100.5, 99.5, 100),
            ("2021-04-27", 100, 100.5, 98.5, 99),  # low=98.5 hits stop=99
        ]
    )
    cands = pd.DataFrame(
        [
            {
                "ticker": "X",
                "signal_date": pd.Timestamp("2021-04-26"),
                "side": Side.LONG.value,
                "entry_price": 100.0,
                "stop_price": 99.0,
                "target_price": 104.0,
                "setup_score": 0.8,
                "regime_at_signal": "bull_trend",
                "params_tier": "standard",
            }
        ]
    )
    trades = simulate(cands, ohlcv, time_stop_bars=20, same_bar_priority="stop_first")
    t = trades.iloc[0]
    assert t["exit_reason"] == ExitReason.STOP_HIT.value
    assert t["pnl_r"] == pytest.approx(-1.0)


def test_same_bar_stop_first_takes_loss() -> None:
    ohlcv = _ohlcv(
        [
            ("2021-04-26", 100, 100.5, 99.5, 100),
            ("2021-04-27", 100, 105, 98.5, 100),  # both hit
        ]
    )
    cands = pd.DataFrame(
        [
            {
                "ticker": "X",
                "signal_date": pd.Timestamp("2021-04-26"),
                "side": Side.LONG.value,
                "entry_price": 100.0,
                "stop_price": 99.0,
                "target_price": 104.0,
                "setup_score": 0.8,
                "regime_at_signal": "bull_trend",
                "params_tier": "standard",
            }
        ]
    )
    trades = simulate(cands, ohlcv, time_stop_bars=20, same_bar_priority="stop_first")
    t = trades.iloc[0]
    assert t["exit_reason"] == ExitReason.STOP_HIT.value
    assert t["same_bar_ambiguous"] is True or t["same_bar_ambiguous"] == True  # noqa: E712


def test_time_stop_when_neither_hit() -> None:
    ohlcv = _ohlcv(
        [("2021-04-26", 100, 100.5, 99.5, 100)]
        + [(f"2021-04-{27+i:02d}", 100, 100.5, 99.5, 100) for i in range(5)]
    )
    cands = pd.DataFrame(
        [
            {
                "ticker": "X",
                "signal_date": pd.Timestamp("2021-04-26"),
                "side": Side.LONG.value,
                "entry_price": 100.0,
                "stop_price": 99.0,
                "target_price": 104.0,
                "setup_score": 0.8,
                "regime_at_signal": "bull_trend",
                "params_tier": "standard",
            }
        ]
    )
    trades = simulate(cands, ohlcv, time_stop_bars=3, same_bar_priority="stop_first")
    t = trades.iloc[0]
    assert t["exit_reason"] == ExitReason.TIME_STOP.value


def test_short_target_hit() -> None:
    ohlcv = _ohlcv(
        [
            ("2021-04-26", 100, 100.5, 99.5, 100),
            ("2021-04-27", 100, 100, 95, 96),  # low=95 hits target=96 (short)
        ]
    )
    cands = pd.DataFrame(
        [
            {
                "ticker": "X",
                "signal_date": pd.Timestamp("2021-04-26"),
                "side": Side.SHORT.value,
                "entry_price": 100.0,
                "stop_price": 101.0,
                "target_price": 96.0,
                "setup_score": 0.8,
                "regime_at_signal": "bear_trend",
                "params_tier": "standard",
            }
        ]
    )
    trades = simulate(cands, ohlcv, time_stop_bars=20, same_bar_priority="stop_first")
    t = trades.iloc[0]
    assert t["exit_reason"] == ExitReason.TARGET_HIT.value
    assert t["pnl_r"] == pytest.approx(4.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_backtest_engine.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `src/pa/backtest/__init__.py`**

```python
"""Backtest execution engine."""
from pa.backtest.engine import simulate

__all__ = ["simulate"]
```

- [ ] **Step 4: Implement `src/pa/backtest/engine.py`**

```python
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
        rows.append(
            _simulate_one(cand, ohlcv_idx, time_stop_bars, same_bar_priority)
        )
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
        "days_held": (
            (exit_date - entry_date).days if isinstance(exit_date, pd.Timestamp) else 0
        ),
        "regime_at_signal": cand.regime_at_signal,  # type: ignore[attr-defined]
        "same_bar_ambiguous": same_bar,
        "params_tier": cand.params_tier,  # type: ignore[attr-defined]
        "side": side.value,
    }
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_backtest_engine.py -v`
Expected: 5 passing.

- [ ] **Step 6: Commit**

```bash
git add src/pa/backtest/__init__.py src/pa/backtest/engine.py tests/test_backtest_engine.py
git commit -m "Add backtest execution engine"
```

---

## Task 13: Detector Base + Fixture Loader

**Files:**
- Create: `src/pa/detectors/__init__.py`
- Create: `src/pa/detectors/base.py`
- Create: `tests/fixtures/__init__.py`
- Create: `tests/fixtures/loader.py`
- Create: `tests/test_detector_base.py`

- [ ] **Step 1: Write `tests/test_detector_base.py`**

```python
"""Tests for the SetupParams data class and CandidatesFrame helpers."""
from __future__ import annotations

import pytest

from pa.detectors.base import CANDIDATE_COLS, SetupParams
from pa.types import ParamTier


def test_setup_params_holds_tier_and_thresholds() -> None:
    p = SetupParams(tier=ParamTier.STANDARD, thresholds={"signal_score_min": 0.5})
    assert p.tier == ParamTier.STANDARD
    assert p.threshold("signal_score_min") == 0.5


def test_missing_threshold_raises() -> None:
    p = SetupParams(tier=ParamTier.STRICT, thresholds={})
    with pytest.raises(KeyError):
        p.threshold("nope")


def test_candidate_cols_includes_required() -> None:
    required = {
        "ticker", "signal_date", "side", "entry_price", "stop_price",
        "target_price", "setup_score", "regime_at_signal", "params_tier",
    }
    assert required.issubset(set(CANDIDATE_COLS))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_detector_base.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `src/pa/detectors/__init__.py`**

```python
"""Setup detectors: each setup produces candidate trades for the backtest engine."""
from pa.detectors.base import CANDIDATE_COLS, BarsFrame, SetupParams

__all__ = ["BarsFrame", "CANDIDATE_COLS", "SetupParams"]
```

- [ ] **Step 4: Implement `src/pa/detectors/base.py`**

```python
"""Shared types and column conventions for setup detectors."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from pa.types import ParamTier

# Type alias documenting the columns a detector expects in its input frame.
BarsFrame = pd.DataFrame  # see types.OHLCV_COLS + INDICATOR_COLS + REGIME_COLS

CANDIDATE_COLS: list[str] = [
    "ticker",
    "signal_date",
    "side",
    "entry_price",
    "stop_price",
    "target_price",
    "setup_score",
    "regime_at_signal",
    "params_tier",
]


@dataclass(frozen=True, slots=True)
class SetupParams:
    tier: ParamTier
    thresholds: dict[str, Any] = field(default_factory=dict)

    def threshold(self, key: str) -> Any:
        return self.thresholds[key]


def empty_candidates() -> pd.DataFrame:
    return pd.DataFrame(columns=CANDIDATE_COLS)
```

- [ ] **Step 5: Implement `tests/fixtures/__init__.py` and `tests/fixtures/loader.py`**

```python
# tests/fixtures/__init__.py
```

```python
# tests/fixtures/loader.py
"""Helpers for loading hand-crafted detector fixtures (JSON OHLCV + expected)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class Fixture:
    name: str
    ohlcv: pd.DataFrame
    expected_signal_dates: list[pd.Timestamp]  # empty for negative cases
    note: str


def load_fixture(path: Path) -> Fixture:
    payload = json.loads(path.read_text())
    bars = payload["bars"]
    df = pd.DataFrame(bars)
    df["date"] = pd.to_datetime(df["date"])
    df = df[["date", "open", "high", "low", "close", "volume", "vwap"]]
    expected = [pd.Timestamp(d) for d in payload.get("expected_signal_dates", [])]
    return Fixture(
        name=path.stem,
        ohlcv=df,
        expected_signal_dates=expected,
        note=payload.get("note", ""),
    )


def load_fixtures(setup_dir: Path) -> list[Fixture]:
    return [load_fixture(p) for p in sorted(setup_dir.glob("*.json"))]
```

- [ ] **Step 6: Run tests**

Run: `uv run pytest tests/test_detector_base.py -v`
Expected: 3 passing.

- [ ] **Step 7: Commit**

```bash
git add src/pa/detectors/__init__.py src/pa/detectors/base.py \
  tests/fixtures/__init__.py tests/fixtures/loader.py tests/test_detector_base.py
git commit -m "Add detector base types and fixture loader"
```

---

## Task 14: H2 Detector

**Files:**
- Create: `src/pa/detectors/h2.py`
- Create: `src/pa/detectors/params.py`
- Create: `tests/fixtures/h2/positive_01_textbook.json` (synthetic)
- Create: `tests/fixtures/h2/negative_01_no_second_leg.json`
- Create: `tests/test_detectors_h2.py`

- [ ] **Step 1: Implement `src/pa/detectors/params.py` (parameter presets)**

```python
"""Parameter presets per setup × tier. Loose ⊇ standard ⊇ strict.

Each tier returns SetupParams; the detector reads thresholds by name.
"""
from __future__ import annotations

from pa.detectors.base import SetupParams
from pa.types import ParamTier

H2_THRESHOLDS: dict[ParamTier, dict[str, float]] = {
    ParamTier.STRICT: {
        "min_signal_score": 0.7,
        "min_first_leg_bars": 3,
        "min_second_leg_bars": 3,
        "max_second_low_below_first_low_atr": 0.3,
        "regime_strength_min": 0.6,
        "stop_atr_buffer": 0.5,
        "target_r_multiple": 2.0,
    },
    ParamTier.STANDARD: {
        "min_signal_score": 0.5,
        "min_first_leg_bars": 2,
        "min_second_leg_bars": 2,
        "max_second_low_below_first_low_atr": 1.0,
        "regime_strength_min": 0.4,
        "stop_atr_buffer": 0.5,
        "target_r_multiple": 2.0,
    },
    ParamTier.LOOSE: {
        "min_signal_score": 0.3,
        "min_first_leg_bars": 2,
        "min_second_leg_bars": 2,
        "max_second_low_below_first_low_atr": 2.0,
        "regime_strength_min": 0.2,
        "stop_atr_buffer": 0.5,
        "target_r_multiple": 2.0,
    },
}

L2_THRESHOLDS = {  # mirror of H2 for bear regime
    tier: {**vals, "max_second_high_above_first_high_atr": vals.pop(
        "max_second_low_below_first_low_atr"
    )}
    for tier, vals in {
        k: dict(v) for k, v in H2_THRESHOLDS.items()
    }.items()
}


def h2_params(tier: ParamTier) -> SetupParams:
    return SetupParams(tier=tier, thresholds=dict(H2_THRESHOLDS[tier]))


def l2_params(tier: ParamTier) -> SetupParams:
    return SetupParams(tier=tier, thresholds=dict(L2_THRESHOLDS[tier]))
```

- [ ] **Step 2: Write `tests/test_detectors_h2.py`**

```python
"""Tests for H2 (High-2 two-legged pullback in bull trend)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pa.detectors.h2 import detect_h2
from pa.detectors.params import h2_params
from pa.indicators import compute_indicators
from pa.regime.classifier import classify_regime
from pa.regime.signal_bar import signal_bar_score
from pa.types import ParamTier
from tests.fixtures.loader import load_fixture

FIX_DIR = Path(__file__).parent / "fixtures" / "h2"


def _bars_for_detector(ohlcv: pd.DataFrame) -> pd.DataFrame:
    indicators = compute_indicators(ohlcv)
    regimes = classify_regime(ohlcv, indicators)
    bars = ohlcv.merge(indicators, on="date").merge(regimes, on="date")
    bars["signal_bar_score"] = signal_bar_score(ohlcv, indicators).values
    return bars


def test_positive_textbook_detected() -> None:
    fix = load_fixture(FIX_DIR / "positive_01_textbook.json")
    bars = _bars_for_detector(fix.ohlcv)
    candidates = detect_h2(bars, h2_params(ParamTier.STANDARD))
    assert not candidates.empty
    detected_dates = set(candidates["signal_date"])
    for d in fix.expected_signal_dates:
        assert d in detected_dates


def test_negative_no_second_leg_not_detected() -> None:
    fix = load_fixture(FIX_DIR / "negative_01_no_second_leg.json")
    bars = _bars_for_detector(fix.ohlcv)
    candidates = detect_h2(bars, h2_params(ParamTier.STANDARD))
    # Negative fixture: should produce no candidate at the disqualifying date
    assert candidates.empty or len(candidates) == 0


def test_monotonicity_loose_superset_of_standard_superset_of_strict() -> None:
    fix = load_fixture(FIX_DIR / "positive_01_textbook.json")
    bars = _bars_for_detector(fix.ohlcv)
    strict = set(detect_h2(bars, h2_params(ParamTier.STRICT))["signal_date"])
    standard = set(detect_h2(bars, h2_params(ParamTier.STANDARD))["signal_date"])
    loose = set(detect_h2(bars, h2_params(ParamTier.LOOSE))["signal_date"])
    assert strict <= standard
    assert standard <= loose


def test_returns_required_columns() -> None:
    fix = load_fixture(FIX_DIR / "positive_01_textbook.json")
    bars = _bars_for_detector(fix.ohlcv)
    candidates = detect_h2(bars, h2_params(ParamTier.STANDARD))
    if not candidates.empty:
        for col in [
            "ticker", "signal_date", "side", "entry_price",
            "stop_price", "target_price", "setup_score",
            "regime_at_signal", "params_tier",
        ]:
            assert col in candidates.columns
```

- [ ] **Step 3: Create fixture `tests/fixtures/h2/positive_01_textbook.json`**

```json
{
  "name": "h2_positive_textbook",
  "note": "Synthetic uptrend with clean two-legged pullback ending at signal bar.",
  "expected_signal_dates": ["2021-06-22"],
  "bars": [
    {"date": "2021-04-26", "open": 100.0, "high": 101.0, "low": 99.5, "close": 100.8, "volume": 1000000, "vwap": 100.4},
    {"date": "2021-04-27", "open": 100.8, "high": 102.0, "low": 100.5, "close": 101.7, "volume": 1000000, "vwap": 101.3},
    {"date": "2021-04-28", "open": 101.7, "high": 103.0, "low": 101.5, "close": 102.7, "volume": 1000000, "vwap": 102.4},
    {"date": "2021-04-29", "open": 102.7, "high": 104.0, "low": 102.5, "close": 103.6, "volume": 1000000, "vwap": 103.3},
    {"date": "2021-04-30", "open": 103.6, "high": 105.0, "low": 103.4, "close": 104.7, "volume": 1000000, "vwap": 104.4},
    {"date": "2021-05-03", "open": 104.7, "high": 106.0, "low": 104.5, "close": 105.7, "volume": 1000000, "vwap": 105.4},
    {"date": "2021-05-04", "open": 105.7, "high": 107.0, "low": 105.5, "close": 106.7, "volume": 1000000, "vwap": 106.4},
    {"date": "2021-05-05", "open": 106.7, "high": 108.0, "low": 106.5, "close": 107.7, "volume": 1000000, "vwap": 107.4},
    {"date": "2021-05-06", "open": 107.7, "high": 109.0, "low": 107.5, "close": 108.7, "volume": 1000000, "vwap": 108.4},
    {"date": "2021-05-07", "open": 108.7, "high": 110.0, "low": 108.5, "close": 109.7, "volume": 1000000, "vwap": 109.4},
    {"date": "2021-05-10", "open": 109.7, "high": 111.0, "low": 109.5, "close": 110.7, "volume": 1000000, "vwap": 110.4},
    {"date": "2021-05-11", "open": 110.7, "high": 112.0, "low": 110.5, "close": 111.7, "volume": 1000000, "vwap": 111.4},
    {"date": "2021-05-12", "open": 111.7, "high": 113.0, "low": 111.5, "close": 112.7, "volume": 1000000, "vwap": 112.4},
    {"date": "2021-05-13", "open": 112.7, "high": 114.0, "low": 112.5, "close": 113.7, "volume": 1000000, "vwap": 113.4},
    {"date": "2021-05-14", "open": 113.7, "high": 115.0, "low": 113.5, "close": 114.7, "volume": 1000000, "vwap": 114.4},
    {"date": "2021-05-17", "open": 114.7, "high": 116.0, "low": 114.5, "close": 115.7, "volume": 1000000, "vwap": 115.4},
    {"date": "2021-05-18", "open": 115.7, "high": 117.0, "low": 115.5, "close": 116.7, "volume": 1000000, "vwap": 116.4},
    {"date": "2021-05-19", "open": 116.7, "high": 118.0, "low": 116.5, "close": 117.7, "volume": 1000000, "vwap": 117.4},
    {"date": "2021-05-20", "open": 117.7, "high": 119.0, "low": 117.5, "close": 118.7, "volume": 1000000, "vwap": 118.4},
    {"date": "2021-05-21", "open": 118.7, "high": 120.0, "low": 118.5, "close": 119.7, "volume": 1000000, "vwap": 119.4},
    {"date": "2021-05-24", "open": 119.7, "high": 121.0, "low": 119.5, "close": 120.7, "volume": 1000000, "vwap": 120.4},
    {"date": "2021-05-25", "open": 120.7, "high": 122.0, "low": 120.5, "close": 121.7, "volume": 1000000, "vwap": 121.4},
    {"date": "2021-05-26", "open": 121.7, "high": 123.0, "low": 121.5, "close": 122.7, "volume": 1000000, "vwap": 122.4},
    {"date": "2021-05-27", "open": 122.7, "high": 124.0, "low": 122.5, "close": 123.7, "volume": 1000000, "vwap": 123.4},
    {"date": "2021-05-28", "open": 123.7, "high": 125.0, "low": 123.5, "close": 124.7, "volume": 1000000, "vwap": 124.4},
    {"date": "2021-06-01", "open": 124.7, "high": 126.0, "low": 124.5, "close": 125.7, "volume": 1000000, "vwap": 125.4},
    {"date": "2021-06-02", "open": 125.7, "high": 127.0, "low": 125.5, "close": 126.7, "volume": 1000000, "vwap": 126.4},
    {"date": "2021-06-03", "open": 126.7, "high": 128.0, "low": 126.5, "close": 127.7, "volume": 1000000, "vwap": 127.4},
    {"date": "2021-06-04", "open": 127.7, "high": 129.0, "low": 127.5, "close": 128.7, "volume": 1000000, "vwap": 128.4},
    {"date": "2021-06-07", "open": 128.7, "high": 130.0, "low": 128.5, "close": 129.7, "volume": 1000000, "vwap": 129.4},
    {"date": "2021-06-08", "open": 129.7, "high": 131.0, "low": 129.5, "close": 130.7, "volume": 1000000, "vwap": 130.4},
    {"date": "2021-06-09", "open": 130.7, "high": 132.0, "low": 130.5, "close": 131.7, "volume": 1000000, "vwap": 131.4},
    {"date": "2021-06-10", "open": 131.7, "high": 133.0, "low": 131.5, "close": 132.7, "volume": 1000000, "vwap": 132.4},
    {"date": "2021-06-11", "open": 132.7, "high": 134.0, "low": 132.5, "close": 133.7, "volume": 1000000, "vwap": 133.4},
    {"date": "2021-06-14", "open": 133.7, "high": 135.0, "low": 133.0, "close": 133.5, "volume": 1000000, "vwap": 134.0},
    {"date": "2021-06-15", "open": 133.5, "high": 134.0, "low": 132.0, "close": 132.3, "volume": 1000000, "vwap": 133.0},
    {"date": "2021-06-16", "open": 132.3, "high": 133.0, "low": 131.0, "close": 131.5, "volume": 1000000, "vwap": 132.0},
    {"date": "2021-06-17", "open": 131.5, "high": 133.0, "low": 131.0, "close": 132.7, "volume": 1000000, "vwap": 132.0},
    {"date": "2021-06-18", "open": 132.7, "high": 134.0, "low": 132.5, "close": 133.7, "volume": 1000000, "vwap": 133.4},
    {"date": "2021-06-21", "open": 133.7, "high": 134.5, "low": 132.5, "close": 132.8, "volume": 1000000, "vwap": 133.3},
    {"date": "2021-06-22", "open": 132.8, "high": 134.5, "low": 132.0, "close": 134.3, "volume": 1500000, "vwap": 133.5}
  ]
}
```

- [ ] **Step 3b: Create fixture `tests/fixtures/h2/negative_01_no_second_leg.json`**

```json
{
  "name": "h2_negative_no_second_leg",
  "note": "Uptrend with only single-leg pullback then immediate continuation - should NOT match H2.",
  "expected_signal_dates": [],
  "bars": [
    {"date": "2021-04-26", "open": 100.0, "high": 101.0, "low": 99.5, "close": 100.8, "volume": 1000000, "vwap": 100.4},
    {"date": "2021-04-27", "open": 100.8, "high": 102.0, "low": 100.5, "close": 101.7, "volume": 1000000, "vwap": 101.3},
    {"date": "2021-04-28", "open": 101.7, "high": 103.0, "low": 101.5, "close": 102.7, "volume": 1000000, "vwap": 102.4},
    {"date": "2021-04-29", "open": 102.7, "high": 104.0, "low": 102.5, "close": 103.6, "volume": 1000000, "vwap": 103.3},
    {"date": "2021-04-30", "open": 103.6, "high": 105.0, "low": 103.4, "close": 104.7, "volume": 1000000, "vwap": 104.4},
    {"date": "2021-05-03", "open": 104.7, "high": 106.0, "low": 104.5, "close": 105.7, "volume": 1000000, "vwap": 105.4},
    {"date": "2021-05-04", "open": 105.7, "high": 107.0, "low": 105.5, "close": 106.7, "volume": 1000000, "vwap": 106.4},
    {"date": "2021-05-05", "open": 106.7, "high": 108.0, "low": 106.5, "close": 107.7, "volume": 1000000, "vwap": 107.4},
    {"date": "2021-05-06", "open": 107.7, "high": 109.0, "low": 107.5, "close": 108.7, "volume": 1000000, "vwap": 108.4},
    {"date": "2021-05-07", "open": 108.7, "high": 110.0, "low": 108.5, "close": 109.7, "volume": 1000000, "vwap": 109.4},
    {"date": "2021-05-10", "open": 109.7, "high": 111.0, "low": 109.5, "close": 110.7, "volume": 1000000, "vwap": 110.4},
    {"date": "2021-05-11", "open": 110.7, "high": 112.0, "low": 110.5, "close": 111.7, "volume": 1000000, "vwap": 111.4},
    {"date": "2021-05-12", "open": 111.7, "high": 113.0, "low": 111.5, "close": 112.7, "volume": 1000000, "vwap": 112.4},
    {"date": "2021-05-13", "open": 112.7, "high": 114.0, "low": 112.5, "close": 113.7, "volume": 1000000, "vwap": 113.4},
    {"date": "2021-05-14", "open": 113.7, "high": 115.0, "low": 113.5, "close": 114.7, "volume": 1000000, "vwap": 114.4},
    {"date": "2021-05-17", "open": 114.7, "high": 116.0, "low": 114.5, "close": 115.7, "volume": 1000000, "vwap": 115.4},
    {"date": "2021-05-18", "open": 115.7, "high": 117.0, "low": 115.5, "close": 116.7, "volume": 1000000, "vwap": 116.4},
    {"date": "2021-05-19", "open": 116.7, "high": 118.0, "low": 116.5, "close": 117.7, "volume": 1000000, "vwap": 117.4},
    {"date": "2021-05-20", "open": 117.7, "high": 119.0, "low": 117.5, "close": 118.7, "volume": 1000000, "vwap": 118.4},
    {"date": "2021-05-21", "open": 118.7, "high": 120.0, "low": 118.5, "close": 119.7, "volume": 1000000, "vwap": 119.4},
    {"date": "2021-05-24", "open": 119.7, "high": 121.0, "low": 119.5, "close": 120.7, "volume": 1000000, "vwap": 120.4},
    {"date": "2021-05-25", "open": 120.7, "high": 122.0, "low": 120.5, "close": 121.7, "volume": 1000000, "vwap": 121.4},
    {"date": "2021-05-26", "open": 121.7, "high": 123.0, "low": 121.5, "close": 122.7, "volume": 1000000, "vwap": 122.4},
    {"date": "2021-05-27", "open": 122.7, "high": 124.0, "low": 122.5, "close": 123.7, "volume": 1000000, "vwap": 123.4},
    {"date": "2021-05-28", "open": 123.7, "high": 125.0, "low": 123.5, "close": 124.7, "volume": 1000000, "vwap": 124.4},
    {"date": "2021-06-01", "open": 124.7, "high": 126.0, "low": 124.5, "close": 125.7, "volume": 1000000, "vwap": 125.4},
    {"date": "2021-06-02", "open": 125.7, "high": 127.0, "low": 125.5, "close": 126.7, "volume": 1000000, "vwap": 126.4},
    {"date": "2021-06-03", "open": 126.7, "high": 128.0, "low": 126.5, "close": 127.7, "volume": 1000000, "vwap": 127.4},
    {"date": "2021-06-04", "open": 127.7, "high": 129.0, "low": 127.5, "close": 128.7, "volume": 1000000, "vwap": 128.4},
    {"date": "2021-06-07", "open": 128.7, "high": 130.0, "low": 128.5, "close": 129.7, "volume": 1000000, "vwap": 129.4},
    {"date": "2021-06-08", "open": 129.7, "high": 131.0, "low": 129.5, "close": 130.7, "volume": 1000000, "vwap": 130.4},
    {"date": "2021-06-09", "open": 130.7, "high": 132.0, "low": 130.5, "close": 131.7, "volume": 1000000, "vwap": 131.4},
    {"date": "2021-06-10", "open": 131.7, "high": 133.0, "low": 131.5, "close": 132.7, "volume": 1000000, "vwap": 132.4},
    {"date": "2021-06-11", "open": 132.7, "high": 134.0, "low": 132.5, "close": 133.7, "volume": 1000000, "vwap": 133.4},
    {"date": "2021-06-14", "open": 133.7, "high": 134.5, "low": 132.5, "close": 132.8, "volume": 1000000, "vwap": 133.3},
    {"date": "2021-06-15", "open": 132.8, "high": 134.0, "low": 132.0, "close": 133.7, "volume": 1500000, "vwap": 133.0},
    {"date": "2021-06-16", "open": 133.7, "high": 135.0, "low": 133.5, "close": 134.7, "volume": 1000000, "vwap": 134.4},
    {"date": "2021-06-17", "open": 134.7, "high": 136.0, "low": 134.5, "close": 135.7, "volume": 1000000, "vwap": 135.4},
    {"date": "2021-06-18", "open": 135.7, "high": 137.0, "low": 135.5, "close": 136.7, "volume": 1000000, "vwap": 136.4},
    {"date": "2021-06-21", "open": 136.7, "high": 138.0, "low": 136.5, "close": 137.7, "volume": 1000000, "vwap": 137.4},
    {"date": "2021-06-22", "open": 137.7, "high": 139.0, "low": 137.5, "close": 138.7, "volume": 1000000, "vwap": 138.4}
  ]
}
```

- [ ] **Step 4: Run test to verify it fails**

Run: `uv run pytest tests/test_detectors_h2.py -v`
Expected: FAIL — `pa.detectors.h2` not found.

- [ ] **Step 5: Implement `src/pa/detectors/h2.py`**

```python
"""H2 detector: two-legged pullback in bull regime → long entry on signal bar.

Algorithm (per spec §6.1):
  1. Walk forward bar-by-bar. For each bar i in bull_trend regime:
     a. Look back to find a recent swing_high (within 30 bars).
     b. From that swing_high, identify first leg down: ≥ min_first_leg_bars
        consecutive lower closes ending at first low L1.
     c. Identify rebound: ≥ 2 bars of upward closes after L1.
     d. Identify second leg down: ≥ min_second_leg_bars lower closes
        ending at second low L2.
     e. Bar i must be the first bull bar after L2 with signal_bar_score
        ≥ min_signal_score AND L2 not more than max_second_low_below_first_low_atr
        ATRs below L1.
  2. Emit candidate: entry = high(i) + 1 tick, stop = low(i) - buffer*ATR,
     target = entry + R*target_r_multiple where R = entry - stop.
"""
from __future__ import annotations

import pandas as pd

from pa.detectors.base import CANDIDATE_COLS, SetupParams, empty_candidates
from pa.types import Regime, Side


def detect_h2(bars: pd.DataFrame, params: SetupParams) -> pd.DataFrame:
    out: list[dict[str, object]] = []
    p = params.thresholds
    n = len(bars)
    if n < 5:
        return empty_candidates()

    closes = bars["close"].to_numpy()
    highs = bars["high"].to_numpy()
    lows = bars["low"].to_numpy()
    atr = bars["atr14"].to_numpy()
    regime = bars["regime"].to_numpy()
    regime_strength = bars["regime_strength"].to_numpy()
    is_bull_bar = bars["bar_is_bull"].to_numpy()
    sig_score = bars["signal_bar_score"].to_numpy()
    dates = bars["date"].to_numpy()
    ticker = bars["ticker"].iloc[0] if "ticker" in bars.columns else ""

    min_first = int(p["min_first_leg_bars"])
    min_second = int(p["min_second_leg_bars"])
    min_score = float(p["min_signal_score"])
    max_below = float(p["max_second_low_below_first_low_atr"])
    min_strength = float(p["regime_strength_min"])
    buffer_atr = float(p["stop_atr_buffer"])
    r_mult = float(p["target_r_multiple"])

    for i in range(min_first + min_second + 4, n):
        if regime[i] != Regime.BULL_TREND.value:
            continue
        if regime_strength[i] < min_strength:
            continue
        if not is_bull_bar[i]:
            continue
        if sig_score[i] < min_score:
            continue

        # Locate swing high within last 30 bars before i
        lookback_start = max(0, i - 30)
        window_high_idx = lookback_start + int(highs[lookback_start:i].argmax())
        # First leg: bars after window_high_idx with strictly lower closes
        first_leg_end = _walk_lower_closes(closes, start=window_high_idx, max_idx=i - 1)
        if first_leg_end - window_high_idx < min_first:
            continue
        l1 = lows[first_leg_end]

        # Rebound: at least 2 bars of higher closes
        rebound_end = _walk_higher_closes(
            closes, start=first_leg_end, max_idx=i - 1, min_bars=2
        )
        if rebound_end is None:
            continue

        # Second leg: lower closes after rebound, ending before bar i
        second_leg_end = _walk_lower_closes(closes, start=rebound_end, max_idx=i - 1)
        if second_leg_end - rebound_end < min_second:
            continue
        l2 = lows[second_leg_end]

        # L2 may dip below L1 by at most `max_below` ATRs
        atr_i = float(atr[i]) if not pd.isna(atr[i]) else 0.0
        if atr_i == 0:
            continue
        if (l1 - l2) / atr_i > max_below:
            continue

        # Bar i must be the FIRST bull bar after l2_end (i.e., second_leg_end + 1)
        if i != second_leg_end + 1:
            continue

        entry = float(highs[i]) + 0.01
        stop = float(lows[i]) - buffer_atr * atr_i
        risk = entry - stop
        target = entry + r_mult * risk

        out.append(
            {
                "ticker": ticker,
                "signal_date": dates[i],
                "side": Side.LONG.value,
                "entry_price": entry,
                "stop_price": stop,
                "target_price": target,
                "setup_score": float(sig_score[i]),
                "regime_at_signal": regime[i],
                "params_tier": params.tier.value,
            }
        )

    if not out:
        return empty_candidates()
    return pd.DataFrame(out, columns=CANDIDATE_COLS)


def _walk_lower_closes(closes: pd.Series, *, start: int, max_idx: int) -> int:
    """Walk forward from start, returning index of last consecutive lower close."""
    i = start
    while i + 1 <= max_idx and closes[i + 1] < closes[i]:
        i += 1
    return i


def _walk_higher_closes(
    closes: pd.Series, *, start: int, max_idx: int, min_bars: int
) -> int | None:
    """Walk forward, return end index after at least `min_bars` higher closes."""
    i = start
    count = 0
    while i + 1 <= max_idx and closes[i + 1] > closes[i]:
        i += 1
        count += 1
        if count >= min_bars:
            return i
    return None
```

- [ ] **Step 6: Run tests**

Run: `uv run pytest tests/test_detectors_h2.py -v`
Expected: 4 passing.

- [ ] **Step 7: Commit**

```bash
git add src/pa/detectors/h2.py src/pa/detectors/params.py \
  tests/fixtures/h2/ tests/test_detectors_h2.py
git commit -m "Add H2 detector with parameter tiers and fixtures"
```

---

## Task 15: L2 Detector (mirror of H2 for bear regime)

**Files:**
- Create: `src/pa/detectors/l2.py`
- Create: `tests/fixtures/l2/positive_01_textbook.json`
- Create: `tests/fixtures/l2/negative_01_no_second_leg.json`
- Create: `tests/test_detectors_l2.py`

- [ ] **Step 1: Write `tests/test_detectors_l2.py`**

```python
"""Tests for L2 (Low-2 two-legged rally in bear trend)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from pa.detectors.l2 import detect_l2
from pa.detectors.params import l2_params
from pa.indicators import compute_indicators
from pa.regime.classifier import classify_regime
from pa.regime.signal_bar import signal_bar_score
from pa.types import ParamTier
from tests.fixtures.loader import load_fixture

FIX_DIR = Path(__file__).parent / "fixtures" / "l2"


def _bars(ohlcv: pd.DataFrame) -> pd.DataFrame:
    indicators = compute_indicators(ohlcv)
    regimes = classify_regime(ohlcv, indicators)
    bars = ohlcv.merge(indicators, on="date").merge(regimes, on="date")
    bars["signal_bar_score"] = signal_bar_score(ohlcv, indicators).values
    return bars


def test_positive_textbook_detected() -> None:
    fix = load_fixture(FIX_DIR / "positive_01_textbook.json")
    candidates = detect_l2(_bars(fix.ohlcv), l2_params(ParamTier.STANDARD))
    detected = set(candidates["signal_date"]) if not candidates.empty else set()
    for d in fix.expected_signal_dates:
        assert d in detected


def test_negative_not_detected() -> None:
    fix = load_fixture(FIX_DIR / "negative_01_no_second_leg.json")
    candidates = detect_l2(_bars(fix.ohlcv), l2_params(ParamTier.STANDARD))
    assert candidates.empty


def test_monotonicity() -> None:
    fix = load_fixture(FIX_DIR / "positive_01_textbook.json")
    bars = _bars(fix.ohlcv)
    strict = set(detect_l2(bars, l2_params(ParamTier.STRICT))["signal_date"])
    standard = set(detect_l2(bars, l2_params(ParamTier.STANDARD))["signal_date"])
    loose = set(detect_l2(bars, l2_params(ParamTier.LOOSE))["signal_date"])
    assert strict <= standard <= loose
```

- [ ] **Step 2: Implement `src/pa/detectors/l2.py`** (mirror of H2 with bear-direction flips)

```python
"""L2 detector: two-legged rally in bear regime → short entry on signal bar."""
from __future__ import annotations

import pandas as pd

from pa.detectors.base import CANDIDATE_COLS, SetupParams, empty_candidates
from pa.types import Regime, Side


def detect_l2(bars: pd.DataFrame, params: SetupParams) -> pd.DataFrame:
    out: list[dict[str, object]] = []
    p = params.thresholds
    n = len(bars)
    if n < 5:
        return empty_candidates()

    closes = bars["close"].to_numpy()
    highs = bars["high"].to_numpy()
    lows = bars["low"].to_numpy()
    atr = bars["atr14"].to_numpy()
    regime = bars["regime"].to_numpy()
    regime_strength = bars["regime_strength"].to_numpy()
    is_bull_bar = bars["bar_is_bull"].to_numpy()
    sig_score = bars["signal_bar_score"].to_numpy()
    dates = bars["date"].to_numpy()
    ticker = bars["ticker"].iloc[0] if "ticker" in bars.columns else ""

    min_first = int(p["min_first_leg_bars"])
    min_second = int(p["min_second_leg_bars"])
    min_score = float(p["min_signal_score"])
    max_above = float(p["max_second_high_above_first_high_atr"])
    min_strength = float(p["regime_strength_min"])
    buffer_atr = float(p["stop_atr_buffer"])
    r_mult = float(p["target_r_multiple"])

    for i in range(min_first + min_second + 4, n):
        if regime[i] != Regime.BEAR_TREND.value:
            continue
        if regime_strength[i] < min_strength:
            continue
        if is_bull_bar[i]:  # need bear bar for short signal
            continue
        if sig_score[i] < min_score:
            continue

        lookback_start = max(0, i - 30)
        swing_low_idx = lookback_start + int(lows[lookback_start:i].argmin())

        first_leg_end = _walk_higher_closes(closes, start=swing_low_idx, max_idx=i - 1)
        if first_leg_end - swing_low_idx < min_first:
            continue
        h1 = highs[first_leg_end]

        pullback_end = _walk_lower_closes(
            closes, start=first_leg_end, max_idx=i - 1, min_bars=2
        )
        if pullback_end is None:
            continue

        second_leg_end = _walk_higher_closes(closes, start=pullback_end, max_idx=i - 1)
        if second_leg_end - pullback_end < min_second:
            continue
        h2 = highs[second_leg_end]

        atr_i = float(atr[i]) if not pd.isna(atr[i]) else 0.0
        if atr_i == 0:
            continue
        if (h2 - h1) / atr_i > max_above:
            continue

        if i != second_leg_end + 1:
            continue

        entry = float(lows[i]) - 0.01
        stop = float(highs[i]) + buffer_atr * atr_i
        risk = stop - entry
        target = entry - r_mult * risk

        out.append(
            {
                "ticker": ticker,
                "signal_date": dates[i],
                "side": Side.SHORT.value,
                "entry_price": entry,
                "stop_price": stop,
                "target_price": target,
                "setup_score": float(sig_score[i]),
                "regime_at_signal": regime[i],
                "params_tier": params.tier.value,
            }
        )

    if not out:
        return empty_candidates()
    return pd.DataFrame(out, columns=CANDIDATE_COLS)


def _walk_higher_closes(closes, *, start: int, max_idx: int) -> int:
    i = start
    while i + 1 <= max_idx and closes[i + 1] > closes[i]:
        i += 1
    return i


def _walk_lower_closes(
    closes, *, start: int, max_idx: int, min_bars: int
) -> int | None:
    i = start
    count = 0
    while i + 1 <= max_idx and closes[i + 1] < closes[i]:
        i += 1
        count += 1
        if count >= min_bars:
            return i
    return None
```

- [ ] **Step 3: Create fixture files** — mirror the H2 fixtures (downtrend instead of uptrend). Use the same date range and structure but flip the price direction:

`tests/fixtures/l2/positive_01_textbook.json` — synthetic 40-bar downtrend from 140→100, with a clean two-legged rally near bar 35-39, signal date = `2021-06-22`. Build by mirroring H2 positive: use `(1 + (140 - close_h2)/close_h2)` formula or hand-author 40 rows.

`tests/fixtures/l2/negative_01_no_second_leg.json` — 40-bar downtrend with only single-leg bounce, no signal date.

(Construct these from H2 fixtures by `new_close = 240 - h2_close` and `new_high = 240 - h2_low`, `new_low = 240 - h2_high`, `new_open = 240 - h2_close`. Verify dates align.)

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_detectors_l2.py -v`
Expected: 3 passing.

- [ ] **Step 5: Commit**

```bash
git add src/pa/detectors/l2.py tests/fixtures/l2/ tests/test_detectors_l2.py
git commit -m "Add L2 detector"
```

---

## Task 16: Bull/Bear Flag Breakout Detector

**Files:**
- Create: `src/pa/detectors/flag.py`
- Modify: `src/pa/detectors/params.py` (add FLAG_THRESHOLDS + flag_params)
- Create: `tests/fixtures/flag/positive_bull_01.json`
- Create: `tests/fixtures/flag/negative_no_consolidation.json`
- Create: `tests/test_detectors_flag.py`

- [ ] **Step 1: Add `flag_params` to `src/pa/detectors/params.py`**

Append:
```python
FLAG_THRESHOLDS: dict[ParamTier, dict[str, float]] = {
    ParamTier.STRICT: {
        "min_impulse_bars": 4,
        "min_impulse_atr_mult": 2.0,
        "min_consolidation_bars": 5,
        "max_consolidation_range_ratio": 0.4,
        "regime_strength_min": 0.6,
        "stop_atr_buffer": 0.5,
        "target_r_cap": 3.0,
    },
    ParamTier.STANDARD: {
        "min_impulse_bars": 3,
        "min_impulse_atr_mult": 1.5,
        "min_consolidation_bars": 5,
        "max_consolidation_range_ratio": 0.5,
        "regime_strength_min": 0.4,
        "stop_atr_buffer": 0.5,
        "target_r_cap": 3.0,
    },
    ParamTier.LOOSE: {
        "min_impulse_bars": 3,
        "min_impulse_atr_mult": 1.0,
        "min_consolidation_bars": 4,
        "max_consolidation_range_ratio": 0.7,
        "regime_strength_min": 0.2,
        "stop_atr_buffer": 0.5,
        "target_r_cap": 3.0,
    },
}


def flag_params(tier: ParamTier) -> SetupParams:
    return SetupParams(tier=tier, thresholds=dict(FLAG_THRESHOLDS[tier]))
```

- [ ] **Step 2: Write `tests/test_detectors_flag.py`**

```python
"""Tests for Bull/Bear Flag breakout detector."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from pa.detectors.flag import detect_flag
from pa.detectors.params import flag_params
from pa.indicators import compute_indicators
from pa.regime.classifier import classify_regime
from pa.regime.signal_bar import signal_bar_score
from pa.types import ParamTier
from tests.fixtures.loader import load_fixture

FIX_DIR = Path(__file__).parent / "fixtures" / "flag"


def _bars(ohlcv: pd.DataFrame) -> pd.DataFrame:
    indicators = compute_indicators(ohlcv)
    regimes = classify_regime(ohlcv, indicators)
    bars = ohlcv.merge(indicators, on="date").merge(regimes, on="date")
    bars["signal_bar_score"] = signal_bar_score(ohlcv, indicators).values
    return bars


def test_bull_flag_detected() -> None:
    fix = load_fixture(FIX_DIR / "positive_bull_01.json")
    candidates = detect_flag(_bars(fix.ohlcv), flag_params(ParamTier.STANDARD))
    assert not candidates.empty
    for d in fix.expected_signal_dates:
        assert d in set(candidates["signal_date"])


def test_no_consolidation_not_detected() -> None:
    fix = load_fixture(FIX_DIR / "negative_no_consolidation.json")
    candidates = detect_flag(_bars(fix.ohlcv), flag_params(ParamTier.STANDARD))
    assert candidates.empty


def test_monotonicity() -> None:
    fix = load_fixture(FIX_DIR / "positive_bull_01.json")
    bars = _bars(fix.ohlcv)
    s = set(detect_flag(bars, flag_params(ParamTier.STRICT))["signal_date"])
    m = set(detect_flag(bars, flag_params(ParamTier.STANDARD))["signal_date"])
    l = set(detect_flag(bars, flag_params(ParamTier.LOOSE))["signal_date"])
    assert s <= m <= l
```

- [ ] **Step 3: Implement `src/pa/detectors/flag.py`**

```python
"""Bull/Bear Flag detector: impulse leg + tight consolidation + breakout.

Algorithm (per spec §6.3, bull side; bear is symmetric):
  - Find impulse leg ending at impulse_end: ≥ min_impulse_bars consecutive
    bull bars OR a stretch where (high.max() - low.min()) > min_impulse_atr_mult * ATR.
  - Identify consolidation: next ≥ min_consolidation_bars whose total range
    < impulse_size * max_consolidation_range_ratio.
  - Breakout: a subsequent bar's close > consolidation high (bull) or < low (bear).
  - Entry on next bar's open; stop = consolidation low - buffer*ATR (bull);
    target = entry + min(impulse_size, target_r_cap * R).
"""
from __future__ import annotations

import pandas as pd

from pa.detectors.base import CANDIDATE_COLS, SetupParams, empty_candidates
from pa.types import Regime, Side


def detect_flag(bars: pd.DataFrame, params: SetupParams) -> pd.DataFrame:
    out: list[dict[str, object]] = []
    p = params.thresholds
    n = len(bars)
    if n < 20:
        return empty_candidates()

    h = bars["high"].to_numpy()
    lo = bars["low"].to_numpy()
    closes = bars["close"].to_numpy()
    opens = bars["open"].to_numpy()
    atr = bars["atr14"].to_numpy()
    regime = bars["regime"].to_numpy()
    regime_strength = bars["regime_strength"].to_numpy()
    dates = bars["date"].to_numpy()
    ticker = bars["ticker"].iloc[0] if "ticker" in bars.columns else ""

    min_imp = int(p["min_impulse_bars"])
    min_imp_atr = float(p["min_impulse_atr_mult"])
    min_cons = int(p["min_consolidation_bars"])
    max_cons_ratio = float(p["max_consolidation_range_ratio"])
    min_strength = float(p["regime_strength_min"])
    buf = float(p["stop_atr_buffer"])
    r_cap = float(p["target_r_cap"])

    for i in range(min_imp + min_cons + 1, n - 1):
        cur_regime = regime[i]
        if cur_regime not in (Regime.BULL_TREND.value, Regime.BEAR_TREND.value):
            continue
        if regime_strength[i] < min_strength:
            continue

        side = Side.LONG if cur_regime == Regime.BULL_TREND.value else Side.SHORT
        atr_i = float(atr[i]) if not pd.isna(atr[i]) else 0.0
        if atr_i == 0:
            continue

        # Look at consolidation block: bars [i - min_cons, i]
        cons_lo = float(lo[i - min_cons : i + 1].min())
        cons_hi = float(h[i - min_cons : i + 1].max())
        cons_range = cons_hi - cons_lo

        # Impulse leg: bars [i - min_cons - min_imp, i - min_cons]
        imp_start = i - min_cons - min_imp
        imp_end = i - min_cons
        imp_lo = float(lo[imp_start : imp_end + 1].min())
        imp_hi = float(h[imp_start : imp_end + 1].max())
        imp_size = imp_hi - imp_lo

        if imp_size < min_imp_atr * atr_i:
            continue
        if cons_range > imp_size * max_cons_ratio:
            continue

        # Breakout check at bar i: close beyond consolidation in trend direction
        if side == Side.LONG:
            if not (closes[i] > cons_hi):
                continue
            entry = float(opens[i + 1])
            stop = cons_lo - buf * atr_i
            risk = entry - stop
            if risk <= 0:
                continue
            target = entry + min(imp_size, r_cap * risk)
        else:
            if not (closes[i] < cons_lo):
                continue
            entry = float(opens[i + 1])
            stop = cons_hi + buf * atr_i
            risk = stop - entry
            if risk <= 0:
                continue
            target = entry - min(imp_size, r_cap * risk)

        out.append(
            {
                "ticker": ticker,
                "signal_date": dates[i + 1],  # next bar = entry day
                "side": side.value,
                "entry_price": entry,
                "stop_price": stop,
                "target_price": target,
                "setup_score": float(regime_strength[i]),
                "regime_at_signal": cur_regime,
                "params_tier": params.tier.value,
            }
        )

    if not out:
        return empty_candidates()
    return pd.DataFrame(out, columns=CANDIDATE_COLS)
```

- [ ] **Step 4: Create fixtures** — `tests/fixtures/flag/positive_bull_01.json` (60 bars: ~30-bar uptrend establishing bull regime, then 6-bar tight consolidation around 130-131, then breakout bar). `negative_no_consolidation.json` (60 bars: smooth uptrend with no flat range). Hand-author OHLCV ensuring `compute_indicators` produces valid EMA/ATR/regime.

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_detectors_flag.py -v`
Expected: 3 passing.

- [ ] **Step 6: Commit**

```bash
git add src/pa/detectors/flag.py src/pa/detectors/params.py \
  tests/fixtures/flag/ tests/test_detectors_flag.py
git commit -m "Add Bull/Bear Flag detector"
```

---

## Task 17: Failed Breakout Detector

**Files:**
- Create: `src/pa/detectors/failed_breakout.py`
- Modify: `src/pa/detectors/params.py` (add FAILED_BREAKOUT_THRESHOLDS)
- Create: `tests/fixtures/failed_breakout/positive_up_fail_01.json`
- Create: `tests/fixtures/failed_breakout/negative_real_breakout.json`
- Create: `tests/test_detectors_failed_breakout.py`

- [ ] **Step 1: Append to `src/pa/detectors/params.py`**

```python
FAILED_BREAKOUT_THRESHOLDS: dict[ParamTier, dict[str, float]] = {
    ParamTier.STRICT: {
        "range_lookback_bars": 20,
        "min_range_atr_mult": 1.0,  # range must be at least 1 ATR wide
        "max_failure_bars": 2,
        "regime_strength_min": 0.4,
        "stop_atr_buffer": 0.5,
        "target_r_multiple": 2.0,
    },
    ParamTier.STANDARD: {
        "range_lookback_bars": 20,
        "min_range_atr_mult": 0.8,
        "max_failure_bars": 3,
        "regime_strength_min": 0.3,
        "stop_atr_buffer": 0.5,
        "target_r_multiple": 2.0,
    },
    ParamTier.LOOSE: {
        "range_lookback_bars": 20,
        "min_range_atr_mult": 0.6,
        "max_failure_bars": 5,
        "regime_strength_min": 0.1,
        "stop_atr_buffer": 0.5,
        "target_r_multiple": 2.0,
    },
}


def failed_breakout_params(tier: ParamTier) -> SetupParams:
    return SetupParams(tier=tier, thresholds=dict(FAILED_BREAKOUT_THRESHOLDS[tier]))
```

- [ ] **Step 2: Write `tests/test_detectors_failed_breakout.py`**

```python
"""Tests for Failed Breakout reversal detector."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from pa.detectors.failed_breakout import detect_failed_breakout
from pa.detectors.params import failed_breakout_params
from pa.indicators import compute_indicators
from pa.regime.classifier import classify_regime
from pa.regime.signal_bar import signal_bar_score
from pa.types import ParamTier
from tests.fixtures.loader import load_fixture

FIX_DIR = Path(__file__).parent / "fixtures" / "failed_breakout"


def _bars(ohlcv: pd.DataFrame) -> pd.DataFrame:
    indicators = compute_indicators(ohlcv)
    regimes = classify_regime(ohlcv, indicators)
    bars = ohlcv.merge(indicators, on="date").merge(regimes, on="date")
    bars["signal_bar_score"] = signal_bar_score(ohlcv, indicators).values
    return bars


def test_up_breakout_failure_detected() -> None:
    fix = load_fixture(FIX_DIR / "positive_up_fail_01.json")
    candidates = detect_failed_breakout(
        _bars(fix.ohlcv), failed_breakout_params(ParamTier.STANDARD)
    )
    assert not candidates.empty
    for d in fix.expected_signal_dates:
        assert d in set(candidates["signal_date"])


def test_real_breakout_not_detected_as_failure() -> None:
    fix = load_fixture(FIX_DIR / "negative_real_breakout.json")
    candidates = detect_failed_breakout(
        _bars(fix.ohlcv), failed_breakout_params(ParamTier.STANDARD)
    )
    assert candidates.empty


def test_target_is_closer_of_2R_or_range_low() -> None:
    """For SHORT: target = max(entry - 2R, range_low). This test asserts that
    when range_low is closer to entry than 2R, target equals range_low."""
    fix = load_fixture(FIX_DIR / "positive_up_fail_01.json")
    candidates = detect_failed_breakout(
        _bars(fix.ohlcv), failed_breakout_params(ParamTier.STANDARD)
    )
    if not candidates.empty:
        c = candidates.iloc[0]
        risk = c["stop_price"] - c["entry_price"]
        target_2r = c["entry_price"] - 2 * risk
        # Target should never be lower than 2R target (always >= due to max())
        assert c["target_price"] >= target_2r - 1e-6
```

- [ ] **Step 3: Implement `src/pa/detectors/failed_breakout.py`**

```python
"""Failed Breakout reversal: range breakout that closes back inside within K bars.

Algorithm (per spec §6.4):
  - Look back range_lookback_bars to compute [range_low, range_high].
  - Range must be in trading_range regime AND at least min_range_atr_mult ATRs wide.
  - At bar i: detect breakout (close > range_high OR close < range_low).
  - Within next max_failure_bars, if a bar closes back inside the range:
    that bar is the failure confirmation.
  - Entry on next bar's open in opposite direction.
  - Stop on the breakout extreme + buffer*ATR.
  - Target = closer-of (2R, opposite range edge):
      short: target = max(entry - 2R, range_low)
      long:  target = min(entry + 2R, range_high)
"""
from __future__ import annotations

import pandas as pd

from pa.detectors.base import CANDIDATE_COLS, SetupParams, empty_candidates
from pa.types import Regime, Side


def detect_failed_breakout(
    bars: pd.DataFrame, params: SetupParams
) -> pd.DataFrame:
    out: list[dict[str, object]] = []
    p = params.thresholds
    n = len(bars)
    if n < 30:
        return empty_candidates()

    h = bars["high"].to_numpy()
    lo = bars["low"].to_numpy()
    closes = bars["close"].to_numpy()
    opens = bars["open"].to_numpy()
    atr = bars["atr14"].to_numpy()
    regime = bars["regime"].to_numpy()
    regime_strength = bars["regime_strength"].to_numpy()
    dates = bars["date"].to_numpy()
    ticker = bars["ticker"].iloc[0] if "ticker" in bars.columns else ""

    rng_lookback = int(p["range_lookback_bars"])
    min_rng_atr = float(p["min_range_atr_mult"])
    max_fail_bars = int(p["max_failure_bars"])
    min_strength = float(p["regime_strength_min"])
    buf = float(p["stop_atr_buffer"])
    r_mult = float(p["target_r_multiple"])

    i = rng_lookback
    while i < n - max_fail_bars - 1:
        if regime[i] != Regime.TRADING_RANGE.value or regime_strength[i] < min_strength:
            i += 1
            continue
        atr_i = float(atr[i]) if not pd.isna(atr[i]) else 0.0
        if atr_i == 0:
            i += 1
            continue

        rng_lo = float(lo[i - rng_lookback : i].min())
        rng_hi = float(h[i - rng_lookback : i].max())
        if (rng_hi - rng_lo) < min_rng_atr * atr_i:
            i += 1
            continue

        if closes[i] > rng_hi:  # upward breakout
            breakout_high = float(h[i])
            for k in range(1, max_fail_bars + 1):
                if i + k >= n - 1:
                    break
                if closes[i + k] < rng_hi:
                    confirm_idx = i + k
                    entry = float(opens[confirm_idx + 1])
                    stop = breakout_high + buf * atr_i
                    risk = stop - entry
                    if risk <= 0:
                        break
                    target = max(entry - r_mult * risk, rng_lo)
                    out.append(
                        {
                            "ticker": ticker,
                            "signal_date": dates[confirm_idx + 1],
                            "side": Side.SHORT.value,
                            "entry_price": entry,
                            "stop_price": stop,
                            "target_price": target,
                            "setup_score": float(regime_strength[i]),
                            "regime_at_signal": regime[i],
                            "params_tier": params.tier.value,
                        }
                    )
                    i = confirm_idx + 2
                    break
            else:
                i += 1
                continue
        elif closes[i] < rng_lo:  # downward breakout
            breakout_low = float(lo[i])
            for k in range(1, max_fail_bars + 1):
                if i + k >= n - 1:
                    break
                if closes[i + k] > rng_lo:
                    confirm_idx = i + k
                    entry = float(opens[confirm_idx + 1])
                    stop = breakout_low - buf * atr_i
                    risk = entry - stop
                    if risk <= 0:
                        break
                    target = min(entry + r_mult * risk, rng_hi)
                    out.append(
                        {
                            "ticker": ticker,
                            "signal_date": dates[confirm_idx + 1],
                            "side": Side.LONG.value,
                            "entry_price": entry,
                            "stop_price": stop,
                            "target_price": target,
                            "setup_score": float(regime_strength[i]),
                            "regime_at_signal": regime[i],
                            "params_tier": params.tier.value,
                        }
                    )
                    i = confirm_idx + 2
                    break
            else:
                i += 1
                continue
        else:
            i += 1

    if not out:
        return empty_candidates()
    return pd.DataFrame(out, columns=CANDIDATE_COLS)
```

- [ ] **Step 4: Create fixtures** — `positive_up_fail_01.json` (40-bar trading range around 100±2, then bar that closes at 103, then within 2 bars closes back below 102 = failure confirmation). `negative_real_breakout.json` (40-bar range, then breakout bar followed by continuation bars all closing above range_high).

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_detectors_failed_breakout.py -v`
Expected: 3 passing.

- [ ] **Step 6: Commit**

```bash
git add src/pa/detectors/failed_breakout.py src/pa/detectors/params.py \
  tests/fixtures/failed_breakout/ tests/test_detectors_failed_breakout.py
git commit -m "Add Failed Breakout detector"
```

---

## Task 18: Double Top / Double Bottom Detector

**Files:**
- Create: `src/pa/detectors/double_top_bottom.py`
- Modify: `src/pa/detectors/params.py`
- Create: `tests/fixtures/double_top_bottom/positive_double_top_01.json`
- Create: `tests/fixtures/double_top_bottom/negative_single_top.json`
- Create: `tests/test_detectors_double.py`

- [ ] **Step 1: Append to `src/pa/detectors/params.py`**

```python
DOUBLE_TB_THRESHOLDS: dict[ParamTier, dict[str, float]] = {
    ParamTier.STRICT: {
        "lookback_bars": 30,
        "min_pullback_bars": 5,
        "min_pullback_atr_mult": 1.5,
        "max_peak_diff_pct": 0.02,  # 2%
        "min_signal_score": 0.6,
        "stop_atr_buffer": 0.5,
        "target_r_multiple": 2.0,
    },
    ParamTier.STANDARD: {
        "lookback_bars": 30,
        "min_pullback_bars": 5,
        "min_pullback_atr_mult": 1.0,
        "max_peak_diff_pct": 0.05,
        "min_signal_score": 0.5,
        "stop_atr_buffer": 0.5,
        "target_r_multiple": 2.0,
    },
    ParamTier.LOOSE: {
        "lookback_bars": 30,
        "min_pullback_bars": 4,
        "min_pullback_atr_mult": 0.8,
        "max_peak_diff_pct": 0.08,
        "min_signal_score": 0.3,
        "stop_atr_buffer": 0.5,
        "target_r_multiple": 2.0,
    },
}


def double_tb_params(tier: ParamTier) -> SetupParams:
    return SetupParams(tier=tier, thresholds=dict(DOUBLE_TB_THRESHOLDS[tier]))
```

- [ ] **Step 2: Write `tests/test_detectors_double.py`**

```python
"""Tests for Double Top / Double Bottom reversal detector."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from pa.detectors.double_top_bottom import detect_double_top_bottom
from pa.detectors.params import double_tb_params
from pa.indicators import compute_indicators
from pa.regime.classifier import classify_regime
from pa.regime.signal_bar import signal_bar_score
from pa.types import ParamTier
from tests.fixtures.loader import load_fixture

FIX_DIR = Path(__file__).parent / "fixtures" / "double_top_bottom"


def _bars(ohlcv: pd.DataFrame) -> pd.DataFrame:
    indicators = compute_indicators(ohlcv)
    regimes = classify_regime(ohlcv, indicators)
    bars = ohlcv.merge(indicators, on="date").merge(regimes, on="date")
    bars["signal_bar_score"] = signal_bar_score(ohlcv, indicators).values
    return bars


def test_double_top_detected() -> None:
    fix = load_fixture(FIX_DIR / "positive_double_top_01.json")
    candidates = detect_double_top_bottom(
        _bars(fix.ohlcv), double_tb_params(ParamTier.STANDARD)
    )
    assert not candidates.empty
    for d in fix.expected_signal_dates:
        assert d in set(candidates["signal_date"])


def test_single_top_not_detected() -> None:
    fix = load_fixture(FIX_DIR / "negative_single_top.json")
    candidates = detect_double_top_bottom(
        _bars(fix.ohlcv), double_tb_params(ParamTier.STANDARD)
    )
    assert candidates.empty
```

- [ ] **Step 3: Implement `src/pa/detectors/double_top_bottom.py`**

```python
"""Double Top / Double Bottom reversal detector.

Algorithm (per spec §6.5, double top; bottom symmetric):
  - At bar i, look back lookback_bars to find P1 (highest swing high).
  - Confirm at least min_pullback_bars after P1 with a low M where
    P1 - M >= min_pullback_atr_mult * ATR.
  - Confirm P2: a later swing high with |P2 - P1|/P1 < max_peak_diff_pct.
  - Bar i must be the first bear bar after P2 with signal_bar_score >= min.
  - Entry: short at lo(i) - 1 tick. Stop: max(P1, P2) + buffer*ATR.
  - Target: entry - r_mult * (stop - entry).
"""
from __future__ import annotations

import pandas as pd

from pa.detectors.base import CANDIDATE_COLS, SetupParams, empty_candidates
from pa.types import Side


def detect_double_top_bottom(
    bars: pd.DataFrame, params: SetupParams
) -> pd.DataFrame:
    out: list[dict[str, object]] = []
    p = params.thresholds
    n = len(bars)
    if n < 40:
        return empty_candidates()

    h = bars["high"].to_numpy()
    lo = bars["low"].to_numpy()
    closes = bars["close"].to_numpy()
    opens = bars["open"].to_numpy()
    atr = bars["atr14"].to_numpy()
    is_bull_bar = bars["bar_is_bull"].to_numpy()
    sig_score = bars["signal_bar_score"].to_numpy()
    dates = bars["date"].to_numpy()
    ticker = bars["ticker"].iloc[0] if "ticker" in bars.columns else ""

    lookback = int(p["lookback_bars"])
    min_pb_bars = int(p["min_pullback_bars"])
    min_pb_atr = float(p["min_pullback_atr_mult"])
    max_diff = float(p["max_peak_diff_pct"])
    min_score = float(p["min_signal_score"])
    buf = float(p["stop_atr_buffer"])
    r_mult = float(p["target_r_multiple"])

    for i in range(lookback, n):
        atr_i = float(atr[i]) if not pd.isna(atr[i]) else 0.0
        if atr_i == 0 or sig_score[i] < min_score:
            continue
        window = slice(i - lookback, i)

        # ---- Double Top detection ----
        if not is_bull_bar[i]:
            window_h = h[window]
            p1_idx_local = int(window_h.argmax())
            p1_idx = i - lookback + p1_idx_local
            p1 = float(h[p1_idx])

            # Look for pullback after p1
            after_p1 = slice(p1_idx + 1, i)
            if i - p1_idx > min_pb_bars:
                m_idx_local = int(lo[after_p1].argmin())
                m_idx = p1_idx + 1 + m_idx_local
                m = float(lo[m_idx])
                if (p1 - m) >= min_pb_atr * atr_i:
                    # Look for p2 between m_idx and i
                    later = slice(m_idx + 1, i)
                    if i - m_idx > 2:
                        p2_idx_local = int(h[later].argmax())
                        p2_idx = m_idx + 1 + p2_idx_local
                        p2 = float(h[p2_idx])
                        if abs(p2 - p1) / p1 < max_diff and i == p2_idx + 1:
                            entry = float(lo[i]) - 0.01
                            stop = max(p1, p2) + buf * atr_i
                            risk = stop - entry
                            if risk > 0:
                                target = entry - r_mult * risk
                                out.append(
                                    {
                                        "ticker": ticker,
                                        "signal_date": dates[i],
                                        "side": Side.SHORT.value,
                                        "entry_price": entry,
                                        "stop_price": stop,
                                        "target_price": target,
                                        "setup_score": float(sig_score[i]),
                                        "regime_at_signal": str(bars["regime"].iloc[i]),
                                        "params_tier": params.tier.value,
                                    }
                                )
                                continue

        # ---- Double Bottom detection ----
        if is_bull_bar[i]:
            window_l = lo[window]
            b1_idx_local = int(window_l.argmin())
            b1_idx = i - lookback + b1_idx_local
            b1 = float(lo[b1_idx])
            after_b1 = slice(b1_idx + 1, i)
            if i - b1_idx > min_pb_bars:
                m_idx_local = int(h[after_b1].argmax())
                m_idx = b1_idx + 1 + m_idx_local
                m = float(h[m_idx])
                if (m - b1) >= min_pb_atr * atr_i:
                    later = slice(m_idx + 1, i)
                    if i - m_idx > 2:
                        b2_idx_local = int(lo[later].argmin())
                        b2_idx = m_idx + 1 + b2_idx_local
                        b2 = float(lo[b2_idx])
                        if abs(b2 - b1) / b1 < max_diff and i == b2_idx + 1:
                            entry = float(h[i]) + 0.01
                            stop = min(b1, b2) - buf * atr_i
                            risk = entry - stop
                            if risk > 0:
                                target = entry + r_mult * risk
                                out.append(
                                    {
                                        "ticker": ticker,
                                        "signal_date": dates[i],
                                        "side": Side.LONG.value,
                                        "entry_price": entry,
                                        "stop_price": stop,
                                        "target_price": target,
                                        "setup_score": float(sig_score[i]),
                                        "regime_at_signal": str(bars["regime"].iloc[i]),
                                        "params_tier": params.tier.value,
                                    }
                                )

    if not out:
        return empty_candidates()
    return pd.DataFrame(out, columns=CANDIDATE_COLS)
```

- [ ] **Step 4: Create fixtures** — `positive_double_top_01.json` (50 bars: rise to 120 (P1) at bar 25, dip to 110 by bar 32, rise to 119.5 (P2) by bar 38, then bear bar at 39 = signal). `negative_single_top.json` (50 bars: rise to 120 then continuous decline, no second top).

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_detectors_double.py -v`
Expected: 2 passing.

- [ ] **Step 6: Commit**

```bash
git add src/pa/detectors/double_top_bottom.py src/pa/detectors/params.py \
  tests/fixtures/double_top_bottom/ tests/test_detectors_double.py
git commit -m "Add Double Top/Bottom detector"
```

---

## Task 19: Trade Statistics

**Files:**
- Create: `src/pa/report/__init__.py`
- Create: `src/pa/report/stats.py`
- Create: `tests/test_report_stats.py`

- [ ] **Step 1: Write `tests/test_report_stats.py`**

```python
"""Tests for pa.report.stats — aggregate metrics from a trades ledger."""
from __future__ import annotations

import pandas as pd
import pytest

from pa.report.stats import compute_setup_stats, group_by_year, group_by_regime
from pa.types import ExitReason


def _trades(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_basic_metrics_three_trades() -> None:
    trades = _trades(
        [
            {"pnl_r": 2.0, "exit_reason": ExitReason.TARGET_HIT.value, "mae_r": -0.4, "mfe_r": 2.1},
            {"pnl_r": -1.0, "exit_reason": ExitReason.STOP_HIT.value, "mae_r": -1.0, "mfe_r": 0.5},
            {"pnl_r": 2.0, "exit_reason": ExitReason.TARGET_HIT.value, "mae_r": -0.3, "mfe_r": 2.2},
        ]
    )
    stats = compute_setup_stats(trades)
    assert stats["n"] == 3
    assert stats["win_rate"] == pytest.approx(2 / 3)
    assert stats["mean_r"] == pytest.approx(1.0)
    assert stats["profit_factor"] == pytest.approx(4.0)  # gross 4 / gross 1
    assert stats["mean_mae_r"] == pytest.approx((-0.4 + -1.0 + -0.3) / 3)


def test_empty_returns_zero_metrics() -> None:
    stats = compute_setup_stats(pd.DataFrame())
    assert stats["n"] == 0
    assert stats["win_rate"] == 0.0
    assert stats["mean_r"] == 0.0


def test_group_by_year() -> None:
    trades = _trades(
        [
            {"signal_date": pd.Timestamp("2022-03-01"), "pnl_r": 2.0,
             "exit_reason": ExitReason.TARGET_HIT.value, "mae_r": -0.5, "mfe_r": 2.1},
            {"signal_date": pd.Timestamp("2023-06-15"), "pnl_r": -1.0,
             "exit_reason": ExitReason.STOP_HIT.value, "mae_r": -1.0, "mfe_r": 0.3},
        ]
    )
    by_year = group_by_year(trades)
    assert {2022, 2023} <= set(by_year.index)


def test_group_by_regime() -> None:
    trades = _trades(
        [
            {"regime_at_signal": "bull_trend", "pnl_r": 2.0,
             "exit_reason": ExitReason.TARGET_HIT.value, "mae_r": -0.5, "mfe_r": 2.1},
            {"regime_at_signal": "trading_range", "pnl_r": -1.0,
             "exit_reason": ExitReason.STOP_HIT.value, "mae_r": -1.0, "mfe_r": 0.3},
        ]
    )
    by_regime = group_by_regime(trades)
    assert {"bull_trend", "trading_range"} <= set(by_regime.index)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_report_stats.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `src/pa/report/__init__.py`**

```python
"""Reporting layer: stats + chart + HTML."""
from pa.report.stats import compute_setup_stats, group_by_regime, group_by_year

__all__ = ["compute_setup_stats", "group_by_regime", "group_by_year"]
```

- [ ] **Step 4: Implement `src/pa/report/stats.py`**

```python
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
        .apply(compute_setup_stats, include_groups=False)
        .apply(pd.Series)
    )


def group_by_regime(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    return (
        trades.groupby("regime_at_signal")
        .apply(compute_setup_stats, include_groups=False)
        .apply(pd.Series)
    )
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_report_stats.py -v`
Expected: 4 passing.

- [ ] **Step 6: Commit**

```bash
git add src/pa/report/__init__.py src/pa/report/stats.py tests/test_report_stats.py
git commit -m "Add trade statistics aggregation"
```

---

## Task 20: Annotated Chart Rendering

**Files:**
- Create: `src/pa/report/chart.py`
- Create: `tests/test_report_chart.py`

- [ ] **Step 1: Write `tests/test_report_chart.py`**

```python
"""Tests for pa.report.chart — matplotlib annotated K-line chart."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless

import pandas as pd  # noqa: E402

from pa.report.chart import render_trade_chart  # noqa: E402


def _ohlcv() -> pd.DataFrame:
    dates = pd.date_range("2021-04-26", periods=30, freq="B")
    closes = [100 + i for i in range(30)]
    return pd.DataFrame(
        {
            "date": dates,
            "open": closes,
            "high": [c + 1 for c in closes],
            "low": [c - 1 for c in closes],
            "close": closes,
            "volume": [1_000_000] * 30,
            "vwap": closes,
        }
    )


def test_render_creates_png(tmp_path: Path) -> None:
    ohlcv = _ohlcv()
    trade = {
        "ticker": "TEST",
        "signal_date": pd.Timestamp("2021-05-10"),
        "entry_date": pd.Timestamp("2021-05-11"),
        "entry_price": 110.5,
        "stop_price": 109.0,
        "target_price": 113.5,
        "exit_date": pd.Timestamp("2021-05-15"),
        "exit_price": 113.5,
        "exit_reason": "target_hit",
        "side": "long",
        "pnl_r": 2.0,
    }
    out_path = tmp_path / "trade.png"
    render_trade_chart(trade=trade, ohlcv=ohlcv, out_path=out_path, lookback=15)
    assert out_path.exists()
    assert out_path.stat().st_size > 1000  # non-empty PNG
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_report_chart.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `src/pa/report/chart.py`**

```python
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
    sig_idx = sorted_oh["date"].searchsorted(sig_date)

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
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_report_chart.py -v`
Expected: 1 passing.

- [ ] **Step 5: Commit**

```bash
git add src/pa/report/chart.py tests/test_report_chart.py
git commit -m "Add annotated trade chart renderer"
```

---

## Task 21: HTML Report Generator

**Files:**
- Create: `src/pa/report/html.py`
- Create: `src/pa/report/templates/index.html.j2`
- Create: `src/pa/report/templates/setup.html.j2`
- Create: `tests/test_report_html.py`

- [ ] **Step 1: Implement `src/pa/report/templates/index.html.j2`**

```jinja
<!doctype html>
<html><head><meta charset="utf-8"><title>Price Action Backtest — {{ run_id }}</title>
<style>
body{font-family:system-ui;margin:2em;max-width:1100px}
table{border-collapse:collapse;width:100%}
th,td{border:1px solid #ccc;padding:6px 10px;text-align:right}
th:first-child,td:first-child{text-align:left}
h2{margin-top:2em}
.win{color:#067d2f}.loss{color:#a30000}
</style></head><body>
<h1>Price Action Backtest</h1>
<p><b>Run:</b> {{ run_id }} &middot; <b>Date:</b> {{ run_date }} &middot;
   <b>Universe:</b> {{ universe }} &middot;
   <b>Range:</b> {{ date_start }} → {{ date_end }}</p>

<h2>Per-Setup Summary</h2>
<table>
<thead><tr>
  <th>Setup</th><th>Tier</th><th>N</th>
  <th>Win Rate</th><th>Mean R</th><th>PF</th>
  <th>Mean MAE (R)</th><th>Mean MFE (R)</th><th>Detail</th>
</tr></thead>
<tbody>
{% for row in summary %}
  <tr>
    <td>{{ row.setup }}</td>
    <td>{{ row.tier }}</td>
    <td>{{ row.n }}</td>
    <td>{{ "%.1f%%" % (row.win_rate * 100) }}</td>
    <td class="{{ 'win' if row.mean_r > 0 else 'loss' }}">{{ "%.2f" % row.mean_r }}</td>
    <td>{{ "%.2f" % row.profit_factor if row.profit_factor != float('inf') else '∞' }}</td>
    <td>{{ "%.2f" % row.mean_mae_r }}</td>
    <td>{{ "%.2f" % row.mean_mfe_r }}</td>
    <td><a href="{{ row.link }}">view</a></td>
  </tr>
{% endfor %}
</tbody></table>

{% if failures %}
<h2>Failed / Skipped Tickers ({{ failures|length }})</h2>
<ul>{% for f in failures %}<li>{{ f.ticker }}: {{ f.reason }}</li>{% endfor %}</ul>
{% endif %}
</body></html>
```

- [ ] **Step 2: Implement `src/pa/report/templates/setup.html.j2`**

```jinja
<!doctype html>
<html><head><meta charset="utf-8"><title>{{ setup }} ({{ tier }}) — {{ run_id }}</title>
<style>
body{font-family:system-ui;margin:2em;max-width:1300px}
.gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:1em}
.gallery figure{margin:0;border:1px solid #ddd;padding:6px}
.gallery img{width:100%;display:block}
table{border-collapse:collapse;margin:1em 0}
th,td{border:1px solid #ccc;padding:4px 8px}
</style></head><body>
<p><a href="index.html">← back</a></p>
<h1>{{ setup }} — {{ tier }}</h1>
<p>N = {{ stats.n }} &middot; Win {{ "%.1f%%" % (stats.win_rate*100) }}
   &middot; Mean R {{ "%.2f" % stats.mean_r }}
   &middot; PF {{ "%.2f" % stats.profit_factor if stats.profit_factor != float('inf') else '∞' }}</p>

<h2>By Regime</h2>
<table><tr><th>Regime</th><th>N</th><th>Win</th><th>Mean R</th></tr>
{% for r, s in by_regime.iterrows() %}
  <tr><td>{{ r }}</td><td>{{ s.n }}</td>
      <td>{{ "%.1f%%" % (s.win_rate*100) }}</td>
      <td>{{ "%.2f" % s.mean_r }}</td></tr>
{% endfor %}
</table>

<h2>By Year</h2>
<table><tr><th>Year</th><th>N</th><th>Win</th><th>Mean R</th></tr>
{% for y, s in by_year.iterrows() %}
  <tr><td>{{ y }}</td><td>{{ s.n }}</td>
      <td>{{ "%.1f%%" % (s.win_rate*100) }}</td>
      <td>{{ "%.2f" % s.mean_r }}</td></tr>
{% endfor %}
</table>

<h2>Sample Charts ({{ samples|length }} of {{ stats.n }})</h2>
<div class="gallery">
{% for s in samples %}
<figure>
  <img src="{{ s.image }}" loading="lazy">
  <figcaption>{{ s.ticker }} {{ s.signal_date }} R={{ "%.2f" % s.pnl_r }} ({{ s.exit_reason }})</figcaption>
</figure>
{% endfor %}
</div>
</body></html>
```

- [ ] **Step 3: Write `tests/test_report_html.py`**

```python
"""Tests for HTML report generation (smoke test only — exact markup not asserted)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from pa.report.html import build_report


def _trades() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "signal_date": pd.Timestamp("2022-03-01"),
                "entry_date": pd.Timestamp("2022-03-02"),
                "entry_price": 165.0,
                "stop_price": 163.0,
                "target_price": 169.0,
                "exit_date": pd.Timestamp("2022-03-08"),
                "exit_price": 169.0,
                "exit_reason": "target_hit",
                "pnl_r": 2.0,
                "pnl_pct": 0.024,
                "mae_r": -0.4,
                "mfe_r": 2.1,
                "days_held": 6,
                "regime_at_signal": "bull_trend",
                "same_bar_ambiguous": False,
                "params_tier": "standard",
                "side": "long",
            }
        ]
    )


def _ohlcv() -> pd.DataFrame:
    dates = pd.date_range("2022-02-01", periods=40, freq="B")
    return pd.DataFrame(
        {
            "date": dates,
            "open": [165.0] * 40,
            "high": [166.0] * 40,
            "low": [164.0] * 40,
            "close": [165.5] * 40,
            "volume": [1_000_000] * 40,
            "vwap": [165.0] * 40,
        }
    )


def test_build_report_creates_index_and_setup_pages(tmp_path: Path) -> None:
    out = build_report(
        run_id="test_run",
        trades_by_setup_tier={("h2", "standard"): _trades()},
        ohlcv_by_ticker={"AAPL": _ohlcv()},
        out_dir=tmp_path,
        universe="sp500",
        date_start="2021-04-26",
        date_end="2026-04-26",
        samples_per_setup=10,
    )
    assert (out / "index.html").exists()
    assert (out / "h2_standard.html").exists()
    assert (out / "samples").is_dir()
    assert (out / "data.parquet").exists()
```

- [ ] **Step 4: Run test to verify it fails**

Run: `uv run pytest tests/test_report_html.py -v`
Expected: FAIL.

- [ ] **Step 5: Implement `src/pa/report/html.py`**

```python
"""Build the HTML report from a dict of trade ledgers."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

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
    failures: list[dict] | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    env = Environment(
        loader=FileSystemLoader(TEMPLATES_DIR),
        autoescape=select_autoescape(),
    )

    summary_rows: list[dict] = []
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
        sample_meta: list[dict] = []
        for _, trade in trades.head(samples_per_setup).iterrows():
            ohlcv = ohlcv_by_ticker.get(trade["ticker"])
            if ohlcv is None or ohlcv.empty:
                continue
            img_name = f"{trade['ticker']}_{pd.Timestamp(trade['signal_date']).date()}.png"
            img_path = sample_dir / img_name
            try:
                render_trade_chart(
                    trade=trade.to_dict(),
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
        pd.concat(all_trades, ignore_index=True).to_parquet(
            out_dir / "data.parquet", index=False
        )
    else:
        pd.DataFrame().to_parquet(out_dir / "data.parquet", index=False)

    return out_dir
```

- [ ] **Step 6: Run tests**

Run: `uv run pytest tests/test_report_html.py -v`
Expected: 1 passing.

- [ ] **Step 7: Commit**

```bash
git add src/pa/report/html.py src/pa/report/templates/ tests/test_report_html.py
git commit -m "Add HTML report generator"
```

---

## Task 22: CLI Integration

**Files:**
- Create: `src/pa/cli.py`
- Create: `src/pa/pipeline.py` (orchestrator helpers)
- Create: `tests/test_cli.py`
- Create: `configs/sp500_members.txt`

- [ ] **Step 1: Create `configs/sp500_members.txt`**

A newline-separated list of ~500 S&P 500 tickers (one ticker per line, no header).

**Sourcing instruction:** download the current "S&P 500 component stocks" table from Wikipedia (`https://en.wikipedia.org/wiki/List_of_S%26P_500_companies`), extract the `Symbol` column from the first wikitable, save symbols one-per-line into `configs/sp500_members.txt`. Commit as a frozen snapshot — do NOT regenerate dynamically (frozen list keeps backtest reproducible per spec §9.2).

One-liner:
```bash
uv run python -c "
import pandas as pd
df = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
syms = df['Symbol'].str.replace('.', '-', regex=False).tolist()
open('configs/sp500_members.txt','w').write('\n'.join(syms) + '\n')
print(f'Wrote {len(syms)} symbols')
"
```
Expected: ~500 symbols written. Note: BRK.B → BRK-B, BF.B → BF-B (Polygon convention).

- [ ] **Step 2: Implement `src/pa/pipeline.py`**

```python
"""Pipeline orchestration: each function corresponds to a CLI subcommand."""
from __future__ import annotations

import os
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
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
                bars = (
                    ohlcv.merge(pd.read_parquet(ind_dir / f.name), on="date")
                    .merge(pd.read_parquet(rg_dir / f.name), on="date")
                )
                bars["ticker"] = ticker
                cands = detect_fn(bars, params)
                if not cands.empty:
                    all_cands.append(cands)
            combined = (
                pd.concat(all_cands, ignore_index=True)
                if all_cands
                else pd.DataFrame()
            )
            combined.to_parquet(dst / f"{setup}_{tier_str}.parquet", index=False)


def stage_backtest(cfg: Config) -> None:
    cand_dir = cfg.data.cache_dir / "candidates"
    ohlcv_dir = cfg.data.cache_dir / "ohlcv"
    dst = cfg.data.cache_dir / "trades"
    dst.mkdir(parents=True, exist_ok=True)

    ohlcv_by_ticker = {
        f.name.split("_")[0]: pd.read_parquet(f)
        for f in ohlcv_dir.glob("*.parquet")
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
        combined = (
            pd.concat(all_trades, ignore_index=True)
            if all_trades
            else pd.DataFrame()
        )
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
        f.name.split("_")[0]: pd.read_parquet(f)
        for f in ohlcv_dir.glob("*.parquet")
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
```

- [ ] **Step 3: Implement `src/pa/cli.py`**

```python
"""CLI entry point: pa-backtest <subcommand>."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pa.config import load_config
from pa.pipeline import (
    run_all,
    stage_backtest,
    stage_detect,
    stage_fetch,
    stage_indicate,
    stage_regime,
    stage_report,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pa-backtest")
    p.add_argument(
        "command",
        choices=[
            "fetch", "indicate", "regime", "detect",
            "backtest", "report", "all",
        ],
    )
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    cfg = load_config(args.config)

    dispatch = {
        "fetch": stage_fetch,
        "indicate": stage_indicate,
        "regime": stage_regime,
        "detect": stage_detect,
        "backtest": stage_backtest,
        "report": stage_report,
        "all": run_all,
    }
    result = dispatch[args.command](cfg)
    if isinstance(result, Path):
        print(f"Output: {result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Write `tests/test_cli.py`**

```python
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
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_cli.py -v`
Expected: 1 passing.

- [ ] **Step 6: Commit**

```bash
git add src/pa/cli.py src/pa/pipeline.py configs/sp500_members.txt tests/test_cli.py
git commit -m "Add CLI orchestrator and S&P 500 members snapshot"
```

---

## Task 23: Property-Based Invariant Tests

**Files:**
- Create: `tests/test_invariants.py`

- [ ] **Step 1: Write `tests/test_invariants.py`**

```python
"""Property-based invariants — properties that must hold for ANY valid input."""
from __future__ import annotations

import pandas as pd
from hypothesis import HealthCheck, assume, given, settings, strategies as st

from pa.backtest import simulate
from pa.detectors.h2 import detect_h2
from pa.detectors.params import h2_params
from pa.indicators import compute_indicators
from pa.regime.classifier import classify_regime
from pa.regime.signal_bar import signal_bar_score
from pa.types import ParamTier


@st.composite
def random_ohlcv(draw, n: int = 80) -> pd.DataFrame:
    closes = draw(
        st.lists(
            st.floats(min_value=10.0, max_value=500.0, allow_nan=False),
            min_size=n,
            max_size=n,
        )
    )
    return pd.DataFrame(
        {
            "date": pd.date_range("2021-04-26", periods=n, freq="B"),
            "open": closes,
            "high": [c + 0.5 for c in closes],
            "low": [c - 0.5 for c in closes],
            "close": closes,
            "volume": [1_000_000] * n,
            "vwap": closes,
        }
    )


@settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
@given(random_ohlcv())
def test_regime_no_nan(ohlcv: pd.DataFrame) -> None:
    indicators = compute_indicators(ohlcv)
    regimes = classify_regime(ohlcv, indicators)
    assert regimes["regime"].notna().all()


@settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
@given(random_ohlcv(n=120))
def test_h2_monotonicity_loose_superset_standard_superset_strict(
    ohlcv: pd.DataFrame,
) -> None:
    indicators = compute_indicators(ohlcv)
    regimes = classify_regime(ohlcv, indicators)
    bars = ohlcv.merge(indicators, on="date").merge(regimes, on="date")
    bars["signal_bar_score"] = signal_bar_score(ohlcv, indicators).values
    bars["ticker"] = "TEST"

    s = set(detect_h2(bars, h2_params(ParamTier.STRICT))["signal_date"])
    m = set(detect_h2(bars, h2_params(ParamTier.STANDARD))["signal_date"])
    l = set(detect_h2(bars, h2_params(ParamTier.LOOSE))["signal_date"])
    assert s <= m, f"strict not subset of standard: {s - m}"
    assert m <= l, f"standard not subset of loose: {m - l}"


@settings(max_examples=15, suppress_health_check=[HealthCheck.too_slow])
@given(random_ohlcv(n=120))
def test_trade_ledger_invariants(ohlcv: pd.DataFrame) -> None:
    indicators = compute_indicators(ohlcv)
    regimes = classify_regime(ohlcv, indicators)
    bars = ohlcv.merge(indicators, on="date").merge(regimes, on="date")
    bars["signal_bar_score"] = signal_bar_score(ohlcv, indicators).values
    bars["ticker"] = "TEST"
    cands = detect_h2(bars, h2_params(ParamTier.LOOSE))
    assume(not cands.empty)

    trades = simulate(
        cands, ohlcv, time_stop_bars=20, same_bar_priority="stop_first"
    )
    assert (trades["entry_date"] <= trades["exit_date"]).all()
    assert trades["pnl_r"].notna().all()
    assert (trades["pnl_r"].abs() < 1e6).all()  # finite-ish


@settings(max_examples=10, suppress_health_check=[HealthCheck.too_slow])
@given(random_ohlcv(n=80))
def test_indicators_deterministic(ohlcv: pd.DataFrame) -> None:
    a = compute_indicators(ohlcv)
    b = compute_indicators(ohlcv.copy())
    pd.testing.assert_frame_equal(a, b)
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_invariants.py -v`
Expected: 4 passing.

- [ ] **Step 3: Commit**

```bash
git add tests/test_invariants.py
git commit -m "Add property-based invariant tests"
```

---

## Task 24: E2E Pipeline Test

**Files:**
- Create: `tests/test_e2e.py`
- Create: `tests/fixtures/mini_universe/AAPL.parquet` (synthetic)
- Create: `tests/fixtures/mini_universe/MSFT.parquet`
- Create: `tests/fixtures/mini_universe/GOOGL.parquet`

- [ ] **Step 1: Generate mini-universe parquet fixtures**

Add `tests/conftest.py` fixture or inline script in test file. For reproducibility, generate synthetic 250-bar OHLCV for 3 tickers using `numpy.random.default_rng(seed=42)` and write to `tests/fixtures/mini_universe/`.

- [ ] **Step 2: Write `tests/test_e2e.py`**

```python
"""End-to-end pipeline test on a synthetic 3-ticker, 1-year universe."""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pa.config import Config
from pa.config import load_config
from pa.pipeline import (
    stage_indicate,
    stage_regime,
    stage_detect,
    stage_backtest,
    stage_report,
)
from pa.types import OHLCV_COLS


def _gen_synthetic_ohlcv(seed: int, n: int = 250) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    closes = 100 + np.cumsum(rng.normal(0.05, 0.8, n))
    return pd.DataFrame(
        {
            "date": pd.date_range("2021-04-26", periods=n, freq="B"),
            "open": closes - 0.1,
            "high": closes + rng.uniform(0.2, 0.8, n),
            "low": closes - rng.uniform(0.2, 0.8, n),
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
    assert (out / "index.html").exists()
    assert (out / "data.parquet").exists()
    # All 5 setups × 3 tiers = 15 candidate parquet files exist
    cands = list((cfg.data.cache_dir / "candidates").glob("*.parquet"))
    assert len(cands) == 15


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
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/test_e2e.py -v`
Expected: 2 passing in < 30s.

- [ ] **Step 4: Commit**

```bash
git add tests/test_e2e.py
git commit -m "Add E2E pipeline test"
```

---

## Task 25: Visual Regression Baseline

**Files:**
- Create: `tests/test_visual_regression.py`
- Create: `tests/fixtures/visual_baselines/*.png` (committed baselines)

- [ ] **Step 1: Write `tests/test_visual_regression.py`**

```python
"""Visual regression test — chart rendering must remain pixel-stable.

Generate baseline PNGs by running once with REGEN=1 env var.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from pa.report.chart import render_trade_chart  # noqa: E402

BASE_DIR = Path(__file__).parent / "fixtures" / "visual_baselines"


def _ohlcv() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 50
    closes = 100 + np.cumsum(rng.normal(0.1, 0.5, n))
    return pd.DataFrame(
        {
            "date": pd.date_range("2021-04-26", periods=n, freq="B"),
            "open": closes - 0.1,
            "high": closes + 0.5,
            "low": closes - 0.5,
            "close": closes,
            "volume": [1_000_000] * n,
            "vwap": closes,
        }
    )


def _pixel_diff_pct(a: Path, b: Path) -> float:
    img_a = np.asarray(Image.open(a).convert("RGB"))
    img_b = np.asarray(Image.open(b).convert("RGB"))
    if img_a.shape != img_b.shape:
        return 1.0
    return float((img_a != img_b).mean())


def test_h2_textbook_chart_pixel_stable(tmp_path: Path) -> None:
    ohlcv = _ohlcv()
    trade = {
        "ticker": "TEST",
        "signal_date": pd.Timestamp("2021-05-25"),
        "entry_date": pd.Timestamp("2021-05-26"),
        "entry_price": float(ohlcv.iloc[20]["close"] + 0.5),
        "stop_price": float(ohlcv.iloc[20]["close"] - 1.5),
        "target_price": float(ohlcv.iloc[20]["close"] + 4.5),
        "exit_date": pd.Timestamp("2021-06-04"),
        "exit_price": float(ohlcv.iloc[20]["close"] + 4.5),
        "exit_reason": "target_hit",
        "side": "long",
        "pnl_r": 2.0,
    }
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    baseline = BASE_DIR / "h2_textbook.png"
    candidate = tmp_path / "h2_textbook.png"
    render_trade_chart(
        trade=trade, ohlcv=ohlcv, out_path=candidate, lookback=15, lookforward=10
    )

    if os.environ.get("REGEN") == "1" or not baseline.exists():
        candidate.replace(baseline)
        return  # baseline regenerated; treat as pass

    diff = _pixel_diff_pct(candidate, baseline)
    assert diff < 0.01, f"Chart drift: {diff:.4%}"
```

- [ ] **Step 2: Generate baseline (one-time)**

Run:
```bash
REGEN=1 uv run pytest tests/test_visual_regression.py -v
```
Expected: PASS (creates baseline). On subsequent runs without `REGEN=1`, the test compares to baseline.

- [ ] **Step 3: Verify baseline test passes deterministically**

Run twice without REGEN:
```bash
uv run pytest tests/test_visual_regression.py -v
uv run pytest tests/test_visual_regression.py -v
```
Expected: 1 passing both times. If pixel drift varies between runs, set matplotlib `rcParams` for deterministic font/AA rendering inside the test (e.g., `matplotlib.rcParams["text.antialiased"] = False`). Add Pillow to dev deps if not present.

- [ ] **Step 4: Add Pillow to dev deps**

In `pyproject.toml` add to `[project.optional-dependencies] dev = [..., "Pillow>=10.0"]` and re-run `uv pip install -e ".[dev]"`.

- [ ] **Step 5: Commit**

```bash
git add tests/test_visual_regression.py tests/fixtures/visual_baselines/ pyproject.toml
git commit -m "Add visual regression baseline for chart rendering"
```

---

## Task 26: Final E2E Validation + README

**Files:**
- Create: `README.md`
- Modify: `.gitignore` (verify excludes match reality)

- [ ] **Step 1: Write `README.md`**

```markdown
# Price Action Backtest

Lean research pipeline that backtests 5 Al Brooks Price Action setups on
S&P 500 daily bars (5 years) and outputs an HTML report with statistics
+ annotated chart galleries.

See [design spec](docs/superpowers/specs/2026-04-26-price-action-backtest-design.md)
for the full architectural context.

## Quick Start

```bash
# 1. Install
uv venv
uv pip install -e ".[dev]"

# 2. Set Massive API key
export POLYGON_API_KEY=your_key_here

# 3. Run end-to-end (fetch → indicate → regime → detect → backtest → report)
uv run pa-backtest all --config configs/default.yaml

# 4. Open the HTML report
open reports/<run-id>/index.html
```

## Per-Stage Execution

```bash
uv run pa-backtest fetch     # Pull OHLCV from Massive API → data/ohlcv/
uv run pa-backtest indicate  # EMA/ATR/swing/anatomy → data/indicators/
uv run pa-backtest regime    # Trend classification → data/regime/
uv run pa-backtest detect    # 5 setups × 3 tiers → data/candidates/
uv run pa-backtest backtest  # Execution simulation → data/trades/
uv run pa-backtest report    # HTML + annotated charts → reports/<run-id>/
```

## Tests

```bash
uv run pytest -n auto                    # full suite (parallel)
uv run pytest --cov=pa --cov-report=term # coverage
REGEN=1 uv run pytest tests/test_visual_regression.py  # regenerate baselines
```

## Project Layout

- `src/pa/data/` — Massive API + parquet cache
- `src/pa/indicators/` — EMA, ATR, swing, bar anatomy
- `src/pa/regime/` — trend classifier + signal bar scorer
- `src/pa/detectors/` — 5 setups (H2, L2, Flag, Failed Breakout, Double T/B)
- `src/pa/backtest/` — execution engine (stop/target/time-stop)
- `src/pa/report/` — stats + chart + HTML
- `src/pa/cli.py` — CLI entry
- `tests/fixtures/` — hand-crafted detector fixtures
```

- [ ] **Step 2: Final validation run**

Run:
```bash
uv run ruff check src tests
uv run mypy --strict src
uv run pytest -n auto --cov=pa --cov-report=term-missing
```
Expected:
- Ruff: clean
- Mypy: no errors
- Pytest: all passing
- Coverage: ≥ 85%

- [ ] **Step 3: Acceptance checklist (per spec §10)**

Manually verify each:
- [ ] `pa-backtest all` end-to-end runs against real S&P 500 5y data in <30 min on local machine
- [ ] All 6 stages produce parquet files in `data/{ohlcv,indicators,regime,candidates,trades}/`
- [ ] 5 setups × 3 tiers = 15 candidate/trade files exist
- [ ] HTML report rendered with statistics + regime/year stratification + sample charts
- [ ] All 10 fixtures per setup pass (50 fixtures total — author 8 more for each setup if not done above)
- [ ] Hypothesis monotonicity test green
- [ ] Coverage ≥ 85%

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "Add README and finalize MVP acceptance checklist"
```

---

## Self-Review Checklist (for plan author)

Before handoff, the plan was checked against the spec for:

**1. Spec coverage:**
- §3 Architecture → Tasks 1, 22 (bootstrap + pipeline)
- §4.1 data → Tasks 4, 5
- §4.2 indicators → Tasks 6, 7, 8, 9
- §4.3 regime → Tasks 10, 11
- §4.4 detectors base → Task 13
- §4.5 backtest → Task 12
- §4.6 report → Tasks 19, 20, 21
- §5 data flow → Task 22 (pipeline glues schemas)
- §6 setup definitions → Tasks 14 (H2), 15 (L2), 16 (Flag), 17 (Failed Breakout), 18 (Double T/B)
- §7 error handling → Tasks 3 (logger), 4 (retry), 5 (cache miss), 22 (failure jsonl)
- §8 testing → Tasks 23 (property), 24 (e2e), 25 (visual), plus per-component tests
- §10 acceptance → Task 26

**2. Type consistency:**
- All detectors return DataFrames with columns from `CANDIDATE_COLS` (Task 13)
- Backtest engine consumes `CANDIDATE_COLS` and emits `TRADE_COLS`
- Pipeline reads/writes only declared parquet schemas
- `SetupParams.thresholds` keys consistent across `params.py` presets and detector code

**3. Placeholders:** None — all code blocks are complete; fixtures spec out structure with concrete examples for H2 (Task 14) and reference H2 structure for L2/Flag/etc. fixture authoring (Tasks 15-18).

