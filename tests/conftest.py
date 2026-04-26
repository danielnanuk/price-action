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
    return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "vwap"])
