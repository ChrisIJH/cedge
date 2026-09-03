from __future__ import annotations

import numpy as np
import pandas as pd

import pytest


from cedge_core.portfolio.stats import (
    PerformanceStats,
    compute_performance_stats,
    calc_stats,
    calc_return_stats,
    build_yearly_breakdown,
    build_cumulative,
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
    if len(ret_series) == 0:
        return {'Strategy': name, 'Total Return': 0, 'Annual Return': 0,
                'Annual Vol': 0, 'Sharpe': 0, 'Max Drawdown': 0}
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

    def test_zero_vol_sharpe_differs_between_policy_and_strategy_dict(self):
        """
        Edge case: when annual_vol == 0 (e.g. a series of all-zero returns),
        the two ORIGINAL functions disagreed: calc_stats returned NaN,
        calc_return_stats returned 0. Both behaviors must be preserved exactly.
        """
        ret = pd.Series([0.0] * 10, index=pd.date_range("2024-01-01", periods=10, freq="B"))
        stats = compute_performance_stats(ret)
        assert np.isnan(stats.to_policy_dict()["Sharpe"])
        assert calc_return_stats(ret, "Flat")["Sharpe"] == 0

    def test_calc_stats_raises_on_empty_series_like_original(self):
        """
        The ORIGINAL calc_stats had no length guard and raised ZeroDivisionError
        on an empty series (252 / len(ret_series) with len == 0). Preserved as-is.
        """
        with pytest.raises(ZeroDivisionError):
            calc_stats(pd.Series([], dtype=float))

    def test_calc_return_stats_returns_zeros_on_empty_series(self):
        result = calc_return_stats(pd.Series([], dtype=float), "Empty")
        assert result == {
            'Strategy': "Empty", 'Total Return': 0, 'Annual Return': 0,
            'Annual Vol': 0, 'Sharpe': 0, 'Max Drawdown': 0,
        }


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