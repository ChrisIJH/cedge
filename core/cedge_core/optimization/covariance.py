"""
cedge_core/optimization/covariance.py

Turning a returns panel into a covariance matrix an optimizer can use.
Pure functions — no I/O. The DataFrame is expected to already be loaded
(by services/marketdata), not fetched here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sample_covariance(returns: pd.DataFrame) -> np.ndarray:
    """Sample covariance of a wide (date x ticker) returns DataFrame.
    Column order is preserved — the caller is responsible for remembering
    which row/column of the result corresponds to which ticker."""
    if returns.empty:
        raise ValueError("returns is empty")
    if returns.isna().any().any():
        raise ValueError("returns contains NaN — align/drop before calling")
    return returns.cov().to_numpy()


def ensure_psd(cov: np.ndarray, eig_floor: float = 1e-10) -> np.ndarray:
    """Clip negative eigenvalues to eig_floor so cov is safe to use in a
    convex QP. A covariance estimated from a short window can end up
    not-quite-PSD from floating point error alone, which a solver will
    reject as non-convex."""
    sym = 0.5 * (cov + cov.T)
    eigvals, eigvecs = np.linalg.eigh(sym)
    if eigvals.min() < eig_floor:
        eigvals = np.maximum(eigvals, eig_floor)
        sym = (eigvecs * eigvals) @ eigvecs.T
        sym = 0.5 * (sym + sym.T)
    return sym

def factor_model_covariance(
        factor_betas: pd.DataFrame, # index=ticker, columns=factor_name, values=beta_mean(B)
        factor_cov: pd.DataFrame,     # index=factor_name, columns=factor_name, values=cov_ij (Sigma_f)
        resid_var: pd.Series,         # index=ticker, values=resid_var (diag of D)
) -> np.ndarray:
    """Factor risk model: Sigma_asset = B @ Sigma_f @ B.T + D.

    Decomposes asset covariance into a low-rank factor structure plus
    per-asset idiosyncratic variance — the rank is bounded by the factor
    count, not the asset count, which is more stable than
    sample_covariance for small windows or wide universes.

    Callers must pre-align all three inputs (same ticker index, same
    factor index/columns) — this function trusts that alignment rather
    than silently intersecting or filling gaps with zero. Ported from
    CHResearch's ch_api/ch_factor/factor_risk.py::assemble_sigma_asset,
    minus its DB loading and universe-intersection logic, which belongs
    to a repository layer, not here.
    """
    if factor_betas.empty:
        raise ValueError("factor_betas is empty")
    if factor_betas.isna().any().any():
        raise ValueError("factor_betas contains NaN — align/fill before calling")
    if resid_var.isna().any():
        raise ValueError("resid_var contains NaN — align/fill before calling")

    if factor_cov.isna().any().any():
        raise ValueError(
            "factor_cov contains NaN - factor_covariance stores only one "
            "triangle, so the loader must mirror it, not leave the other "
            "side unpopulated"
        )

    factors = factor_betas.columns
    tickers = factor_betas.index

    if not factor_cov.index.equals(factors) or not factor_cov.columns.equals(factors):
        raise ValueError(
            "factor_cov index/columns must exactly match factor_betas columns "
            f"(factor_betas factors={list(factors)}, "
            f"factor_cov index={list(factor_cov.index)}, "
            f"factor_cov columns={list(factor_cov.columns)})"
        )
    if not resid_var.index.equals(tickers):
        raise ValueError(
            "resid_var index must exactly match factor_betas index (tickers) "
            f"(factor_betas tickers={list(tickers)}, resid_var index={list(resid_var.index)})"
        )
    
    B = factor_betas.to_numpy(dtype=float)
    sigma_f = factor_cov.to_numpy(dtype=float)

    if not np.allclose(sigma_f, sigma_f.T, rtol=1e-9, atol=1e-15):
        raise ValueError(
            "factor_cov is not symmetric - check the loader's triangle mirroring"
        )


    d = np.diag(resid_var.to_numpy(dtype=float))

    return B @ sigma_f @ B.T + d