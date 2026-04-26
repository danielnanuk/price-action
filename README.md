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

# 3. Run end-to-end (fetch -> indicate -> regime -> detect -> backtest -> report)
uv run pa-backtest all --config configs/default.yaml

# 4. Open the HTML report
open reports/<run-id>/index.html
```

## Per-Stage Execution

```bash
uv run pa-backtest fetch     # Pull OHLCV from Massive API -> data/ohlcv/
uv run pa-backtest indicate  # EMA/ATR/swing/anatomy -> data/indicators/
uv run pa-backtest regime    # Trend classification -> data/regime/
uv run pa-backtest detect    # 5 setups x 3 tiers -> data/candidates/
uv run pa-backtest backtest  # Execution simulation -> data/trades/
uv run pa-backtest report    # HTML + annotated charts -> reports/<run-id>/
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
