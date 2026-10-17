from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from cedge_core.portfolio.stats import (
    build_cumulative,
    build_yearly_breakdown,
    calc_return_stats,
    calc_stats,
    compute_performance_stats,
)


def _make_ret_series(seed: int, n: int = 252) -> pd.Series:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.Series(rng.normal(0.0005, 0.01, n), index=dates)


def _reference_calc_stats(ret_series: pd.Series) -> dict:
    """Copy of the ORIGINAL calc_stats formula, kept here as an independent oracle."""
    ann_ret = (1 + ret_series).prod() ** (252 / len(ret_series)) - 1
    ann_vol = ret_series.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    cum = (1 + ret_series).cumprod()
    max_dd = (cum / cum.cummax() - 1).min()
    total = cum.iloc[-1] - 1
    return {
        'Total Return': total,
        'Annual Return': ann_ret,
        'Annual Vol': ann_vol,
        'Sharpe': sharpe,
        'Max DD': max_dd,
    }


def _reference_calc_return_stats(ret_series: pd.Series, name: str) -> dict:
    """Copy of the ORIGINAL calc_return_stats formula, kept here as an independent oracle."""
    total_ret = (1 + ret_series).prod() - 1
    annual_ret = (1 + total_ret) ** (252 / len(ret_series)) - 1
    annual_vol = ret_series.std() * np.sqrt(252)
    sharpe = annual_ret / annual_vol if annual_vol > 0 else 0
    cum = (1 + ret_series).cumprod()
    max_dd = (cum / cum.cummax() - 1).min()
    return {
        'Strategy': name,
        'Total Return': total_ret,
        'Annual Return': annual_ret,
        'Annual Vol': annual_vol,
        'Sharpe': sharpe,
        'Max Drawdown': max_dd,
    }


class TestComputePerformanceStats:
    def test_matches_reference_calc_stats_on_random_data(self):
        ret = _make_ret_series(seed=1)
        expected = _reference_calc_stats(ret)
        stats = compute_performance_stats(ret)
        actual = stats.to_policy_dict()
        for key in expected:
            assert actual[key] == pytest.approx(expected[key], abs=1e-9, nan_ok=True)

    def test_matches_reference_calc_return_stats_on_random_data(self):
        ret = _make_ret_series(seed=2)
        expected = _reference_calc_return_stats(ret, "MyStrategy")
        actual = calc_return_stats(ret, "MyStrategy")
        for key in expected:
            assert actual[key] == pytest.approx(expected[key], abs=1e-9)

    def test_zero_vol_sharpe_is_nan_everywhere(self):
        """A flat series has no volatility, so Sharpe is undefined — NaN, not 0.
        (calc_return_stats used to coerce it to 0, which read as "no risk-adjusted
        return" rather than "undefined".)"""
        ret = pd.Series([0.0] * 10, index=pd.date_range("2024-01-01", periods=10, freq="B"))
        assert np.isnan(compute_performance_stats(ret).to_policy_dict()["Sharpe"])
        assert np.isnan(calc_return_stats(ret, "Flat")["Sharpe"])

    def test_empty_series_raises_everywhere(self):
        """No silent fallbacks: an empty series is an error, never a row of zeros."""
        empty = pd.Series([], dtype=float)
        with pytest.raises(ValueError, match="empty"):
            calc_stats(empty)
        with pytest.raises(ValueError, match="Empty"):
            calc_return_stats(empty, "Empty")

class TestBuildYearlyBreakdown:
    def test_two_policies_two_years(self):
        idx_2023 = pd.date_range("2023-01-01", periods=3, freq="B")
        idx_2024 = pd.date_range("2024-01-01", periods=3, freq="B")
        idx = idx_2023.append(idx_2024)
        a = pd.Series([0.01, 0.01, 0.01, 0.02, 0.02, 0.02], index=idx)
        b = pd.Series([0.0, 0.0, 0.0, 0.01, 0.01, 0.01], index=idx)

        result = build_yearly_breakdown({'A': a, 'B': b})

        assert list(result['year']) == [2023, 2024]
        assert result[result['year'] == 2023]['A'].iloc[0] == pytest.approx((1.01 ** 3) - 1, abs=1e-9)
        assert result[result['year'] == 2024]['B'].iloc[0] == pytest.approx((1.01 ** 3) - 1, abs=1e-9)


class TestBuildCumulative:
    def test_cumulative_returns_two_policies(self):
        dates = pd.Series(pd.date_range("2024-01-01", periods=3, freq="B"))
        a = pd.Series([0.01, 0.01, 0.01])
        b = pd.Series([0.02, -0.01, 0.0])

        result = build_cumulative({'Policy A': a, 'Policy B': b}, dates)

        assert result['Policy A'].iloc[-1] == pytest.approx((1.01 ** 3) - 1, abs=1e-9)
        assert result['Policy B'].iloc[-1] == pytest.approx((1.02 * 0.99 * 1.0) - 1, abs=1e-9)