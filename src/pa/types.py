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
