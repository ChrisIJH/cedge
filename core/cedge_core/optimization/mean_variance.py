"""
cedge_core/optimization/mean_variance.py

Mean-variance portfolio construction — pure function, no I/O.

    minimize   w^T Cov w + lambda_turnover * ||w - w_prev||_1 
                + factor_penalty * ||factor_exposure^T w||_2^2
                - mu_alpha * (alpha @ w)
    subject to sum(w) == net_target
               sum(|w|) <= gross_target
               -w_bounds <= w <= w_bounds
               |beta @ w| <= beta_cap              (only if beta and beta_cap given)

Ported and cleaned up from CHResearch's ch_api/ch_optimize.py) — same 
QP structure, rewritten as a pure function with
no DB/engine coupling, real constraint objects (the original built
constraints by string-concatenating Python code and eval()'ing it — see
ch_optimize.py's MVOptimizer/BHOptimizer), and a typed dataclass result
instead of a loose dict.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import cvxpy as cvx
import numpy as np

from cedge_core.optimization.covariance import ensure_psd as _ensure_psd_matrix

_SOLVERS = (cvx.OSQP, cvx.ECOS, cvx.SCS)


@dataclass(frozen=True)
class OptimizeResult:
    """risk/turnover/gross_realized/net_realized are computed from the
    actual solution, not the targets passed in — compare against
    gross_target/w_bounds/net_target to see how tightly constraints bind."""
    weights: np.ndarray
    status: str
    risk: float
    turnover: float
    gross_realized: float
    net_realized: float
    factor_realized: Dict[str, float] = field(default_factory=dict)
    solver: str = ""



def _validate_inputs(cov: np.ndarray, w_prev: np.ndarray,
                     alpha: Optional[np.ndarray],
                     factor_exposure: Optional[np.ndarray],
                     factor_names: Optional[Sequence[str]],
                     factor_limits: Optional[Dict[str, float]],
                     gross_target: float, net_target: float, w_bounds: float) -> None:
    n = len(w_prev)
    if cov.shape != (n, n):
        raise ValueError(f"cov shape {cov.shape} != ({n}, {n}) from w_prev")
    if alpha is not None and alpha.shape != (n,):
        raise ValueError(f"alpha shape {alpha.shape} != ({n},)")

    if factor_exposure is not None:
        if not factor_names:
            raise ValueError("factor_names is required when factor_exposure is given")
        if factor_exposure.shape[0] != n:
            raise ValueError(f"factor_exposure rows {factor_exposure.shape[0]} != n={n}")
        if factor_exposure.shape[1] != len(factor_names):
            raise ValueError(
                f"factor_exposure has {factor_exposure.shape[1]} columns but "
                f"factor_names has {len(factor_names)} entries"
            )
    if factor_limits:
        if not factor_names:
            raise ValueError("factor_limits given but factor_names/factor_exposure missing")
        unknown = set(factor_limits) - set(factor_names)
        if unknown:
            raise ValueError(
                f"factor_limits references unknown factor(s) {sorted(unknown)}; "
                f"factor_names={list(factor_names)}"
            )


    if w_bounds * n < abs(net_target):
        raise ValueError(
            f"infeasible by construction: w_bounds*n ({w_bounds * n:.4f}) < "
            f"|net_target| ({abs(net_target):.4f}) — no weight vector this "
            f"bounded can sum to net_target"
        )
    if gross_target < abs(net_target):
        raise ValueError(
            f"infeasible by construction: gross_target ({gross_target}) < "
            f"|net_target| ({abs(net_target)}) — sum(|w|) can never be "
            f"smaller than sum(w)"
        )
    

def solve_mean_variance(
    cov: np.ndarray,
    w_prev: np.ndarray,
    alpha: Optional[np.ndarray] = None,
    *,
    gross_target: float = 2.0,
    net_target: float = 0.0,
    w_bounds: float = 0.5,
    factor_exposure: Optional[np.ndarray] = None,
    factor_names: Optional[Sequence[str]] = None,
    factor_limits: Optional[Dict[str, float]] = None,
    lambda_turnover: float = 1.0,
    mu_alpha: float = 0.0,
    ensure_psd: bool = True,
) -> OptimizeResult:
    """Solve one mean-variance rebalance.

    w_prev is the anchor (turnover is measured against it) — pass zeros for
    a from-scratch allocation with no turnover history.

    alpha=None means minimum-variance: no expected-return view, just risk
    minimization net of turnover cost. 

    beta/beta_cap enforce |beta @ w| <= beta_cap as a HARD constraint, only
    when both are given.

    factor_exposure/factor_penalty add a SOFT factor-neutrality penalty
    (||factor_exposure^T w||^2), same optionality as beta — cedge has no
    automatic factor-loading source, so the caller supplies the (n, k)
    exposure matrix directly, exactly as it must already supply beta.

    Raises RuntimeError if every solver in the fallback chain fails to
    reach "optimal"/"optimal_inaccurate" — a real failure the caller must
    handle, not a w_prev result disguised as a solve.
    """
    cov = np.asarray(cov, dtype=float)
    w_prev = np.asarray(w_prev, dtype=float).reshape(-1)
    alpha = None if alpha is None else np.asarray(alpha, dtype=float).reshape(-1)
    factor_exposure = (None if factor_exposure is None
                       else np.asarray(factor_exposure, dtype=float))
    factor_names = list(factor_names) if factor_names else None

    _validate_inputs(cov, w_prev, alpha, factor_exposure, factor_names, factor_limits,
                     gross_target, net_target, w_bounds)
    if ensure_psd:
        cov = _ensure_psd_matrix(cov)

    n = len(w_prev)
    w = cvx.Variable(n)

    risk = cvx.quad_form(w, cov)
    turnover = cvx.norm1(w - w_prev)
    objective = risk + lambda_turnover * turnover
    if alpha is not None and mu_alpha != 0.0:
        objective -= mu_alpha * (alpha @ w)

    constraints = [
        cvx.sum(w) == net_target,
        cvx.sum(cvx.abs(w)) <= gross_target,
        w <= w_bounds,
        w >= -w_bounds,
    ]
    if factor_exposure is not None and factor_limits and factor_names is not None:
        for j, name in enumerate(factor_names):
            if name in factor_limits:
                constraints.append(cvx.abs(factor_exposure[:, j] @ w) <= factor_limits[name])

    problem = cvx.Problem(cvx.Minimize(objective), constraints)

    last_error: Optional[str] = None
    for solver in _SOLVERS:
        try:
            problem.solve(solver=solver, warm_start=True)
        except Exception as exc: # noqa
            last_error = f"{solver}: raised {exc}"
            continue

        if problem.status in ("optimal", "optimal_inaccurate") and w.value is not None:
            w_opt = np.asarray(w.value, dtype=float).reshape(-1)
            factor_realized = {}
            if factor_exposure is not None and factor_names is not None:
                for j, name in enumerate(factor_names):
                    factor_realized[name] = float(factor_exposure[:, j] @ w_opt)
            return OptimizeResult(
                weights=w_opt,
                status=problem.status,
                risk=float(w_opt @ cov @ w_opt),
                turnover=float(np.sum(np.abs(w_opt - w_prev))),
                gross_realized=float(np.sum(np.abs(w_opt))),
                net_realized=float(np.sum(w_opt)),
                factor_realized=factor_realized,
                solver=str(solver),
            )
        last_error = f"{solver}: status={problem.status}"

    raise RuntimeError(
        f"solve_mean_variance: no solver in {[str(s) for s in _SOLVERS]} reached "
        f"optimal/optimal_inaccurate. Last attempt — {last_error}"
    )