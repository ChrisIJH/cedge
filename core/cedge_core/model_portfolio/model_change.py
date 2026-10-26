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
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import cvxpy as cp
import numpy as np
import pandas as pd

from cedge_core.model_portfolio.attribution import SLEEVE_OF
from cedge_core.model_portfolio.models import ModelPortfolio
from cedge_core.optimization.covariance import ensure_psd, sample_covariance

CHANGE_TOL = 0.0005     # |weight change| above this counts as a changed fund
SUM_TOL = 1e-6          # weights summing within this of 1 count as fully invested
GRID_TOL = 1e-9         # float slack when flooring weights onto the grid
DUAL_TOL = 1e-6         # a rule whose dual exceeds this held the solution back

_SOLVERS = (cp.OSQP, cp.ECOS, cp.SCS)      # same order as optimization.mean_variance._SOLVERS
_SOLVER_OPTS: Dict[str, Dict[str, Any]] = {
    cp.OSQP: {"eps_abs": 1e-10, "eps_rel": 1e-10, "polish": True, "max_iter": 100_000},
}

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
    """sqrt( (w - ref)' cov (w - ref)). A ticker missing from w or ref counts as 0"""
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



def round_to_grid(w: pd.Series, step: float = 0.005) -> pd.Series:
    """Weights on a grid of `step` that sum to exactly one (largest remainder):
    floor every weight to the grid, then give the leftover steps to the largest
    remainders, ties broken by ticker order. 1 / step must be a whole number."""
    n_steps = round(1 / step)
    if abs(n_steps * step - 1) > GRID_TOL:
        raise ValueError(f"1 / step must be a whole number, got step={step}")
    if abs(float(w.sum()) - 1) > SUM_TOL:
        raise ValueError(f"weights sum to {float(w.sum()):.8f}, not 1")
    w = w.sort_index().clip(lower=0.0)
    units = w.to_numpy() * n_steps
    floors = np.floor(units + GRID_TOL)
    left = n_steps - int(floors.sum())
    remainders = units - floors
    order = sorted(range(len(w)), key=lambda i: (-remainders[i], i))
    floors[order[:left]] += 1
    return pd.Series(floors / n_steps, index=w.index)


@dataclass(frozen=True)
class Variant:
    status: str                         # "optimal", or the solver's word for why not
    weights: Optional[pd.Series]        # continuous solution
    rounded: Optional[pd.Series]        # on the grid, sums to one
    te: Optional[float]                 # continuous vs base
    te_rounded: Optional[float]
    te_share: Optional[pd.Series]       # each fund's share of te_rounded^2 (sums to 1)
    binding: Tuple[str, ...]            # rules that held the solution back
    violations: Tuple[str, ...]         # rules broken by rounding

    def to_model(self, model_id: str, version: str) -> ModelPortfolio:
        """The rounded variant as a model the account engine can run."""
        if self.rounded is None:
            raise ValueError(f"no variant to convert: status {self.status}")
        return ModelPortfolio(model_id, version,
                              {t: float(x) for t, x in self.rounded.items() if x > 0})


def _solve(problem: cp.Problem) -> str:
    """Try each solver in turn; return the final status (never raises)."""
    status = "solver_failed"
    for solver in _SOLVERS:
        try:
            problem.solve(solver=solver, **_SOLVER_OPTS.get(solver, {}))
        except cp.error.SolverError:
            continue
        status = problem.status
        if status in ("optimal", "infeasible"):
            return status
    return status


def make_variant(base: ModelPortfolio, cov: pd.DataFrame, exclude: Sequence[str] = (),
                 universe: Optional[Sequence[str]] = None, max_weight: Optional[float] = None,
                 max_sleeve: Optional[Mapping[str, float]] = None, keep_sleeves: bool = False,
                 grid: float = 0.005, sleeve_of: Mapping[str, str] = SLEEVE_OF) -> Variant:
    """Closest variant of `base` under the rules, by tracking error:

        minimize (w - w_base)' cov (w - w_base)
        s.t. sum w = 1, w >= 0, w = 0 for excluded funds and funds outside the universe,
             w <= max_weight, sleeve sums <= max_sleeve (or == base's if keep_sleeves).

    The universe defaults to the base model's own funds. Rules that cannot all hold
    come back as status "infeasible", never as an exception."""
    allowed = set(base.weights if universe is None else universe) - set(exclude)
    tickers = sorted(set(base.weights) | allowed)
    missing = [t for t in tickers if t not in cov.index or t not in sleeve_of]
    if missing:
        raise ValueError(f"tickers missing from the covariance or sleeve map: {missing}")

    wb = _vector(base.weights, tickers)
    s = ensure_psd(cov.loc[tickers, tickers].to_numpy())
    sleeve = np.array([sleeve_of[t] for t in tickers])
    w = cp.Variable(len(tickers))

    rules: Dict[str, Any] = {"budget": cp.sum(w) == 1, "long_only": w >= 0}
    for i, t in enumerate(tickers):
        if t not in allowed:
            rules[f"exclude:{t}"] = w[i] == 0
        elif max_weight is not None:
            rules[f"max_weight:{t}"] = w[i] <= max_weight
    for name in sorted(set(sleeve)):
        in_sleeve = (sleeve == name).astype(float)
        if keep_sleeves:
            rules[f"keep_sleeve:{name}"] = in_sleeve @ w == in_sleeve @ wb
        if max_sleeve is not None and name in max_sleeve:
            rules[f"max_sleeve:{name}"] = in_sleeve @ w <= max_sleeve[name]

    problem = cp.Problem(cp.Minimize(cp.quad_form(w - wb, cp.psd_wrap(s))), list(rules.values()))
    status = _solve(problem)
    if status != "optimal":
        return Variant(status, None, None, None, None, None, (), ())

    cont = pd.Series(np.clip(w.value, 0.0, None), index=tickers)
    cont = cont / cont.sum()
    rounded = round_to_grid(cont, grid)
    binding = tuple(name for name, c in rules.items()
                    if name.split(":")[0] in ("max_weight", "max_sleeve", "keep_sleeve")
                    and c.dual_value is not None and float(np.max(np.abs(c.dual_value))) > DUAL_TOL)

    violations = [f"exclude:{t}" for t in tickers if t not in allowed and rounded[t] > 0]
    if max_weight is not None:
        violations += [f"max_weight:{t}" for t in tickers if rounded[t] > max_weight + SUM_TOL]
    by_sleeve = rounded.groupby(sleeve).sum()
    base_by_sleeve = pd.Series(wb, index=tickers).groupby(sleeve).sum()
    for name in sorted(set(sleeve)):
        if keep_sleeves and abs(by_sleeve[name] - base_by_sleeve[name]) > SUM_TOL:
            violations.append(f"keep_sleeve:{name}")
        if max_sleeve is not None and name in max_sleeve and by_sleeve[name] > max_sleeve[name] + SUM_TOL:
            violations.append(f"max_sleeve:{name}")

    d = rounded.to_numpy() - wb
    te2 = float(d @ s @ d)
    share = pd.Series(d * (s @ d) / te2 if te2 > 0 else np.zeros(len(d)), index=tickers)
    return Variant(status, cont, rounded,
                   tracking_error(cont.to_dict(), base.weights, cov),
                   tracking_error(rounded.to_dict(), base.weights, cov),
                   share, binding, tuple(violations))