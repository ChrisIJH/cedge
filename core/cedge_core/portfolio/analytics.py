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
from cedge_core.portfolio.stats import (
    calc_stats,
    calc_return_stats,
    build_yearly_breakdown,
    build_cumulative,
)
from cedge_core.portfolio.prices import get_daily_prices
from cedge_core.portfolio.performance import build_performance

__all__ = [
    "calc_stats",
    "calc_return_stats",
    "build_yearly_breakdown",
    "build_cumulative",
    "get_daily_prices",
    "build_performance",
]