"""
cedge_core/estimators.py

Statistical estimators shared across the platform. Pure functions — no I/O.

"""
import numpy as np


def realized_beta(asset_ret, market_ret):
    """OLS realized residual market beta:  asset = alpha + beta*market + eps.

    Raw daily simple returns, FULL-SAMPLE, rf ignored at daily horizon.
    NOT rolling (rolling/dynamic hedge is handled elsewhere).

    Returns: {beta, alpha, r2, n}.   beta = Cov(asset, market)/Var(market).
    """
    a = np.asarray(asset_ret, float)
    m = np.asarray(market_ret, float)
    mask = ~(np.isnan(a) | np.isnan(m))
    a, m = a[mask], m[mask]
    n = len(a)
    mbar, abar = m.mean(), a.mean()
    var_m = np.mean((m - mbar) ** 2)
    cov = np.mean((m - mbar) * (a - abar))
    beta = cov / var_m
    alpha = abar - beta * mbar
    resid = a - (alpha + beta * m)
    ss_tot = np.sum((a - abar) ** 2)
    r2 = 1 - np.sum(resid ** 2) / ss_tot if ss_tot > 0 else float("nan")
    return {"beta": float(beta), "alpha": float(alpha), "r2": float(r2), "n": int(n)}
