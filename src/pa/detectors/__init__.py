"""Setup detectors: each setup produces candidate trades for the backtest engine."""

from pa.detectors.base import CANDIDATE_COLS, BarsFrame, SetupParams

__all__ = ["BarsFrame", "CANDIDATE_COLS", "SetupParams"]
