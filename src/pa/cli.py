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
            "fetch",
            "indicate",
            "regime",
            "detect",
            "backtest",
            "report",
            "all",
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
