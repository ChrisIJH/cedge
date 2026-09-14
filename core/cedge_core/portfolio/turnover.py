"""
cedge_core/portfolio/turnover.py

Portfolio turnover and gross exposure calculations.

Sources:
  - compute_turnover: batch/build_optimizer_snapshot.py
"""
from typing import Optional

import pandas as pd


def compute_gross(w: pd.Series) -> float:
    """Sum of absolute weights — gross exposure. Used as turnover fallback on first rebalance."""
    return float(w.abs().sum())


def compute_turnover(w: pd.Series, w_prev: Optional[pd.Series]) -> float:
    """
    Two-way turnover: sum of absolute weight changes between current and previous portfolio.
    Falls back to gross exposure when no previous weights (first rebalancing).
    """
    if w_prev is None or w_prev.empty:
        return compute_gross(w)
    idx = w.index.union(w_prev.index)
    return float(
        (w.reindex(idx, fill_value=0.0)
         - w_prev.reindex(idx, fill_value=0.0))
        .abs()
        .sum()
    )
