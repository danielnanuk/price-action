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
