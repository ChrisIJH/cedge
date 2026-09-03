"""
cedge_core/marketdata/returns.py

Functional Programming - no I/O
"""
from __future__ import annotations
from typing import Union
import numpy as np
import pandas as pd

def to_returns(prices: Union[pd.DataFrame, pd.Series],
            method: str="simple",) -> Union[pd.DataFrame, pd.Series]:
    """prices: wide (date × ticker) -> returns (첫 행 제거).
    method: 'simple'(pct_change) | 'log'."""
    if method=="simple":
        rets= prices.pct_change().dropna()
    elif method=="log":
        rets  = np.log(prices).diff().dropna() # dataframe 이다.
    else:
        raise ValueError(f"Unknown method:{method} (use 'simple' or 'log')")

    return rets