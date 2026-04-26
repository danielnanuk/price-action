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
        "ticker",
        "signal_date",
        "side",
        "entry_price",
        "stop_price",
        "target_price",
        "setup_score",
        "regime_at_signal",
        "params_tier",
    }
    assert required.issubset(set(CANDIDATE_COLS))
