"""
cedge_core/regime/backtest.py

Pure functions — no DB. Validates regime_scores.score/posterior against
subsequent SPY returns: forward-return computation and label bucketing.
Consumes RegimeScoreRepository's history (scores.py) and SPY prices
from the existing marketdata/portfolio repositories; fetches neither
itself.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd

_LABELS = ("risk_off", "soft_patch", "risk_on")



def forward_returns(ret: pd.Series, horizons: Sequence[int] = (1, 5, 20)) -> pd.DataFrame:
    """
    Ported from CHResearch's regime_backtest.py::compute_forward_metrics.
    Compounded forward return: fwd_h[s] covers ret[s .. s+h-1] (h days
    starting at s, inclusive of s itself — NOT "tomorrow through h days
    from now"). Trailing (h-1) rows are NaN (no future data yet).

    fwd_h[s] = prod(1 + ret[s : s+h]) - 1
    """
    res = pd.DataFrame(index=ret.index)
    for h in horizons:
        gross = (1.0 + ret).rolling(h).apply(lambda x: np.prod(x), raw=True)
        fwd = gross.shift(-h + 1) - 1.0
        res[f"fwd_{h}"] = fwd
    return res


def label_from_posterior(posterior: Dict[str, float]) -> str:
    """Argmax label from a {risk_off, soft_patch, risk_on} posterior dict
    (the same shape posterior.mixture_posterior returns)."""
    return max(_LABELS, key=lambda label: posterior.get(label, 0.0))


def bucket_stats_by_label(regime_history: pd.DataFrame,
                          fwd_returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    regime_history: must have columns
      posterior_risk_on, posterior_soft_patch, posterior_risk_off
      (as returned by scores.get_regime_history), same index as
      fwd_returns_df.
    fwd_returns_df: output of forward_returns(...), same index.

    Returns one row per label (risk_off/soft_patch/risk_on) x fwd column,
    with mean forward return and sample count.
    """
    labels = regime_history.apply(
        lambda row: label_from_posterior({
            "risk_off": row["posterior_risk_off"],
            "soft_patch": row["posterior_soft_patch"],
            "risk_on": row["posterior_risk_on"],
        }),
        axis=1,
    )
    joined = fwd_returns_df.copy()
    joined["regime_label"] = labels

    stats = joined.groupby("regime_label").agg(["mean", "count"])
    stats.columns = ["_".join(c) for c in stats.columns]
    return stats.reindex(_LABELS).dropna(how="all")