"""
core/cedge_core/model_portfolio/model_change.py

Model-level tools

- compare models: what a proposed model change does before it ships: turnover,
    flows by fund, change by sleevs, tracking error between versions and 
    to the benchmark
- make_invariant: the closest variant of a base model under a client's rules
    (excluded funds, a fund universe, caps), found by minimizing tracking error to
    the base and rounded to a weight grid that still sums to one.

Covariance-convention: annualized, a DataFrame with index - columns - tickers.
Pure function: the caller loads prices.
"""
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from cedge_core.model_portfolio.attribution import SLEEVE_OF
from cedge_core.model_portfolio.models import ModelPortfolio
from cedge_core.optimization.covariance import ensure_psd, sample_covariance

CHANGE_TOL = 0.0005

def annualized_covariance( adj_close: pd.DataFrame, periods: int = 252) -> pd.DataFrame:
    """Wide adjusted closes (index= = dates, columns=ticker)
    """
    px = adj_close.sort_index()
    returns = px.pct_change(fill_method=None).iloc[1:]
    cov = ensure_psd(sample_covariance(returns)*periods)
    return pd.DataFrame(cov, index=px.columns, columns=px.columns)


def _vector(w: Mapping[str, float], ticker: Sequence[str]) -> np.ndarray:
    return np.asarray([float(w.get(t, 0.0)) for t in ticker])


def tracking_error(w: Mapping[str, float], ref: Mapping[str, float], cov: pd.DataFrame) -> float:
    """sqsrt( (w - ref)' cov (w - ref)). A ticker missing from w or ref counts as 0"""
    tickers = sorted(set(w) | set(ref))
    missing = [t for t in tickers if t not in cov.index]
    if missing:
        raise ValueError(f"tickers not in the covariance: {missing}")
    d = _vector(w, tickers) - _vector(ref, tickers)
    s = cov.loc[tickers, tickers].to_numpy()
    return float(np.sqrt(d.T @ s @ d))


@dataclass(frozen=True)
class ModelChange:
    diff: pd.Series                  # new - old by ticker (union of both, sorted)
    diff_by_sleeve: pd.Series        # diff summed by sleeve (sorted)
    turnover: float                  # one-way: 1/2 sum |diff|
    n_changed: int                   # funds with |diff| > CHANGE_TOL
    te_old_new: float                # new vs old
    te_old_bench: Optional[float]    # old vs benchmark (None without a benchmark)
    te_new_bench: Optional[float]
    flows: Optional[pd.Series]       # aum x diff, dollars (None without aum)


def compare_models(old: ModelPortfolio, new: ModelPortfolio, cov: pd.DataFrame,
                   benchmark: Optional[ModelPortfolio] = None, aum: Optional[float] = None,
                   sleeve_of: Mapping[str, str] = SLEEVE_OF) -> ModelChange:
    """What changing old -> new does, before it ships. A ticker with no sleeve
    is a ValueError naming it."""
    tickers = sorted(set(old.weights) | set(new.weights))
    no_sleeve = [t for t in tickers if t not in sleeve_of]
    if no_sleeve:
        raise ValueError(f"tickers with no sleeve: {no_sleeve}")

    diff = pd.Series(_vector(new.weights, tickers) - _vector(old.weights, tickers), index=tickers)
    by_sleeve = diff.groupby(diff.index.map(dict(sleeve_of))).sum().sort_index()
    te_bench = (None, None) if benchmark is None else (
        tracking_error(old.weights, benchmark.weights, cov),
        tracking_error(new.weights, benchmark.weights, cov),
    )
    return ModelChange(
        diff=diff,
        diff_by_sleeve=by_sleeve,
        turnover=float(diff.abs().sum() / 2),
        n_changed=int((diff.abs() > CHANGE_TOL).sum()),
        te_old_new=tracking_error(new.weights, old.weights, cov),
        te_old_bench=te_bench[0],
        te_new_bench=te_bench[1],
        flows=None if aum is None else diff * aum,
    )