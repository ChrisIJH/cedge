"""
cedge_core/marketdata/weights.py

Portfolio weight transforms.

Turn the weights into exposures and weighted returns.

"""
from __future__ import annotations

import numpy as np
import pandas as pd


def exposures(weights: dict) -> dict:
    w = np.array(list(weights.values()), float)
    return {"gross": float(np.abs(w).sum()),
            "net": float(w.sum())}

def portfolio_returns(returns_df: pd.DataFrame,
                      weights: dict):
    """Fixed-weight, daily-rebalanced 포트폴리오 수익률 (1-D).
      r_p,t = sum_i w_i * r_i,t   (합=1 강제 안 함; 음수=short)
    반환: (rp ndarray[T], dates Index[T], exposures dict{gross,net})
    단일 종목 = 가중치 1 특수케이스."""
    tickers = list(weights)
    w = np.array([weights[t] for t in tickers])

    sub = returns_df[tickers].dropna()
    rp = sub @ w

    return rp, sub.index, exposures(weights)












