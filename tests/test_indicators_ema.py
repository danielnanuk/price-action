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
