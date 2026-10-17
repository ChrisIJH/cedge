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