"""
cedge_core/portfolio/analytics.py

Backward-compatible re-export shim. The real implementation lives in:
  - ch_api/portfolio/stats.py       (functional core — pure calculation)
  - ch_api/portfolio/prices.py      (imperative shell — DB access)
  - ch_api/portfolio/performance.py (orchestration)

Existing consumers (core/policy_comparison.py, core/factor_portfolio.py,
core/bloomberg_compare.py, clients/policy_b_backtest.py) import from this
module and do not need to change.
"""
from cedge_core.portfolio.performance import build_performance
from cedge_core.portfolio.prices import get_daily_prices
from cedge_core.portfolio.stats import (
    build_cumulative,
    build_yearly_breakdown,
    calc_return_stats,
    calc_stats,
)

__all__ = [
    "build_cumulative",
    "build_performance",
    "build_yearly_breakdown",
    "calc_return_stats",
    "calc_stats",
    "get_daily_prices",
]