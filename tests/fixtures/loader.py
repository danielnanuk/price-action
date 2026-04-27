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
