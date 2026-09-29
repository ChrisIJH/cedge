import pandas as pd
import pytest
from cedge_core.regime.signals import credit_ratio, trailing_zscore


def test_trailing_zscore_known_value():
    assert trailing_zscore(pd.Series([1, 2, 3]), window=3) == pytest.approx(1.0)


def test_trailing_zscore_insufficient_history_is_none():
    assert trailing_zscore(pd.Series([1, 2]), window=3) is None


def test_trailing_zscore_zero_variance_is_none():
    assert trailing_zscore(pd.Series([5, 5, 5]), window=3) is None


def test_credit_ratio():
    hyg = pd.Series([80.0, 82.0])
    ief = pd.Series([100.0, 100.0])
    result = credit_ratio(hyg, ief)
    assert result.tolist() == pytest.approx([0.80, 0.82])