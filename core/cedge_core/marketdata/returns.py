"""
cedge_core/marketdata/returns.py

Functional Programming - no I/O
"""
from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd


def to_returns(prices: Union[pd.DataFrame, pd.Series],
               method: str = "simple",
               dropna: str = "any") -> Union[pd.DataFrame, pd.Series]:
    """prices: wide (date x ticker) -> returns (첫 행 제거).
    method: 'simple'(pct_change) | 'log'.
    dropna: 'any' — drop a date if any ticker is NaN (one aligned panel, the
                    default for covariance and portfolio returns)
            'all' — drop a date only if every ticker is NaN (keeps tickers with
                    different histories; per-ticker consumers skip NaN themselves)
    """
    if dropna not in ("any", "all"):
        raise ValueError(f"Unknown dropna:{dropna} (use 'any' or 'all')")
    if method == "simple":
        rets = prices.pct_change()
    elif method == "log":
        rets = np.log(prices).diff()
    else:
        raise ValueError(f"Unknown method:{method} (use 'simple' or 'log')")

    if isinstance(rets, pd.Series):
        return rets.dropna()
    return rets.dropna(how=dropna)

SPLIT_FACTORS = (2, 3, 4, 5, 6, 8, 10, 20, 50, 100)


def flag_suspect_moves(returns: pd.DataFrame, tolerance: float = 0.10,
                       extreme: float = 0.75) -> list[dict]:
    """Daily moves that look like data errors rather than market moves.

    Flags a day when the price ratio (1 + return) is within `tolerance` of a
    split factor k or 1/k — an unadjusted split — or when the move is at least
    `extreme`. A heuristic: a genuine crash can be flagged too, so callers
    should report these as warnings, not drop the data.
    """
    factors = np.array(SPLIT_FACTORS, dtype=float)
    factors = np.concatenate([factors, 1 / factors])
    flagged = []
    for ticker in returns.columns:
        r = returns[ticker].dropna()
        values = r.to_numpy(dtype=float)
        ratio = 1 + values
        split_like = (np.abs(ratio[:, None] / factors[None, :] - 1) <= tolerance).any(axis=1)
        hit = split_like | (np.abs(values) >= extreme)
        for date, ret, is_split in zip(r.index[hit], values[hit], split_like[hit]):
            flagged.append({
                "ticker": ticker,
                "date": str(pd.Timestamp(date).date()),
                "return": float(ret),
                "reason": "price ratio close to a split factor" if is_split
                          else f"move of {extreme:.0%} or more",
            })
    return flagged