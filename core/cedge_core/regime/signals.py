"""
cedge_core/regime/signals.py

Pure functions — no DB, no I/O. Rolling z-scores and derived ratios used
as inputs to the regime composite score.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def trailing_zscore(series: pd.Series, window: int = 252) -> Optional[float]:
    """
    Z-score of the latest observation against the trailing `window`
    values (ddof=1). Returns None if fewer than `window` non-null
    observations are available, or the trailing window has zero variance.
    """
    s = series.dropna()
    if len(s) < window:
        return None

    tail = s.iloc[-window:]
    x_t = float(tail.iloc[-1])
    mu = float(tail.mean())
    sd = float(tail.std(ddof=1))

    if sd == 0 or np.isnan(sd):
        return None

    z = (x_t - mu) / sd
    if np.isnan(z) or np.isinf(z):
        return None

    return float(z)


def rolling_zscore_series(series: pd.Series, window: int = 252) -> pd.Series:
    """Full rolling z-score series (not just the latest value) — feeds the
    historical panel `pc1_score` needs to estimate its covariance."""
    return (series - series.rolling(window).mean()) / series.rolling(window).std()


def credit_ratio(hyg: pd.Series, ief: pd.Series) -> pd.Series:
    """HYG/IEF price ratio — proxy for credit spread direction (rising
    ratio = credit tightening / risk-on)."""
    return hyg / ief


   