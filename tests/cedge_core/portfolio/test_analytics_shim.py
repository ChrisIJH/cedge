from __future__ import annotations


class TestBackwardCompatImports:
    def test_policy_comparison_import(self):
        from cedge_core.portfolio.analytics import (
            build_cumulative,
            build_yearly_breakdown,
            calc_stats,
        )
        assert callable(calc_stats)
        assert callable(build_yearly_breakdown)
        assert callable(build_cumulative)

    def test_factor_portfolio_import(self):
        from cedge_core.portfolio.analytics import (
            build_performance,
            calc_return_stats,
            get_daily_prices,
        )
        assert callable(get_daily_prices)
        assert callable(calc_return_stats)
        assert callable(build_performance)

    def test_bloomberg_compare_import(self):
        from cedge_core.portfolio.analytics import calc_stats, get_daily_prices
        assert callable(calc_stats)
        assert callable(get_daily_prices)

    def test_build_optimizer_snapshot_import(self):
        from cedge_core.portfolio.turnover import compute_turnover
        assert callable(compute_turnover)

    def test_policy_b_backtest_import(self):
        from cedge_core.portfolio.analytics import calc_stats
        assert callable(calc_stats)