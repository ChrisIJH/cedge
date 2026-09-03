"""Parametric (Normal linear) VaR / ES — pure compute (no DB).

⚠ ALPHA-CONVENTION WARNING (read before using anything in this file):
    This module's `confidence` parameter is the CONFIDENCE LEVEL itself
    (e.g. confidence=0.99 for 99% VaR, z = Phi^-1(0.99) > 0).

    var_es_model.py in this SAME package uses the OPPOSITE convention:
    `parametric_var(w, alpha=0.05)` — alpha is the LEFT-TAIL PROBABILITY
    (0.05 = 95% VaR, z = Phi^-1(0.05) < 0, then negated).

    These are deliberately NOT unified (changing either would break its own
    already-verified known-answer tests). The parameter is named `confidence`
    here — not `alpha` — specifically so a call site can never silently pass
    the wrong convention to the wrong function. Function names in this file
    also carry the `_confidence` suffix for the same reason.

Simple returns throughout. VaR/ES = positive loss.
Look-ahead 차단: rolling estimates use info through day t-1 only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def normal_linear_var_confidence(sigma: float, confidence: float = 0.99,
                                 horizon: int = 1, mu: float = 0.0) -> float:
    """Normal linear VaR (parametric, loss-positive convention).

        VaR_{h,c} = z_c * sigma * sqrt(h) - mu * h,   z_c = Phi^-1(c)

    `sigma` is the 1-period (daily) standard deviation of simple returns.
    `confidence` is the confidence level itself (0.99 -> 99% VaR), NOT a
    tail probability — see the module-level warning above.
    Returns the loss as a positive number (VaR > 0 for a typical portfolio).
    """
    z = stats.norm.ppf(confidence)
    return z * sigma * np.sqrt(horizon) - mu * horizon


def normal_linear_es_confidence(sigma: float, confidence: float = 0.99,
                                horizon: int = 1, mu: float = 0.0) -> float:
    """Normal linear Expected Shortfall (parametric, loss-positive convention).

        ES_{h,c} = [ phi(z_c) / (1 - c) ] * sigma * sqrt(h) - mu * h

    (a) the denominator is the TAIL PROBABILITY (1 - confidence), not confidence itself.
    (b) the numerator is the density phi(z_c), NOT the CDF Phi(z_c).
    (c) ES must be >= VaR at the same confidence (ES is a deeper, averaged loss).
    """
    z = stats.norm.ppf(confidence)
    return (stats.norm.pdf(z) / (1 - confidence)) * sigma * np.sqrt(horizon) - mu * horizon


def normal_linear_es_confidence_WRONG(sigma: float, confidence: float = 0.99,
                                      horizon: int = 1, mu: float = 0.0) -> float:
    """Intentionally WRONG ES — omits the 1/(1-confidence) tail normalization.

    Kept here (and tested) to demonstrate: without the tail normalization,
    ES can fall BELOW VaR, violating coherence — a tail-average loss can
    never be smaller than the threshold it's averaging beyond.
    """
    z = stats.norm.ppf(confidence)
    return float(stats.norm.pdf(z) * sigma * np.sqrt(horizon))


def portfolio_sigma(weights: np.ndarray, cov: np.ndarray) -> float:
    """Portfolio return standard deviation: sigma_p = sqrt(w' Sigma w).

    `weights` are used exactly as given — this function does NOT normalize
    them (e.g. to sum to 1). Normalization is the caller's responsibility.
    """
    return float(np.sqrt(weights.T @ cov @ weights))


def rolling_parametric_var_confidence(returns: pd.Series, confidence: float = 0.99,
                                      window: int = 252, vol_method: str = "sample",
                                      lam: float = 0.94, horizon: int = 1) -> pd.DataFrame:
    """Rolling parametric VaR/ES over a return series.

    vol_method: "sample" (rolling sample std) or "ewma" (RiskMetrics, lam=0.94).
    Returns a DataFrame indexed like `returns`, columns ["mu", "sigma", "var", "es"].

    LOOK-AHEAD DISCIPLINE: sigma/mu are computed INCLUSIVE of day t (both the
    rolling-window and EWMA estimators naturally include day t's own return),
    then shifted once — a single shift point shared by both vol_method
    branches, so day t's VaR/ES depends only on information through day t-1.

    mu always uses a plain trailing-window mean regardless of vol_method —
    only the volatility estimator differs between "sample" and "ewma", which
    isolates that comparison to sigma alone.
    """
    if vol_method == "sample":
        sigma_incl = returns.rolling(window).std(ddof=1)
    elif vol_method == "ewma":
        sigma_incl = np.sqrt((returns ** 2).ewm(alpha=1 - lam, adjust=False).mean())
    else:
        raise ValueError(f"unknown vol_method: {vol_method!r}")

    mu_incl = returns.rolling(window).mean()

    sigma = sigma_incl.shift(1)
    mu = mu_incl.shift(1)

    z = stats.norm.ppf(confidence)
    var = z * sigma * np.sqrt(horizon) - mu * horizon
    es = (stats.norm.pdf(z) / (1 - confidence)) * sigma * np.sqrt(horizon) - mu * horizon

    return pd.DataFrame({"mu": mu, "sigma": sigma, "var": var, "es": es},
                        index=returns.index)


def rolling_parametric_var_confidence_WRONG(returns: pd.Series, confidence: float = 0.99,
                                            window: int = 252) -> pd.DataFrame:
    """Intentionally WRONG rolling VaR — NO shift (look-ahead bug demo).

    Estimates sigma on a window that INCLUDES day t's own return, then
    reports "day t's VaR" using that same-day-inclusive sigma — a look-ahead
    bug (on day t you would not yet know day t's return).
    """
    sigma = returns.rolling(window).std()
    z = stats.norm.ppf(confidence)
    var = z * sigma
    es = (stats.norm.pdf(z) / (1 - confidence)) * sigma
    return pd.DataFrame({"sigma": sigma, "var": var, "es": es}, index=returns.index)
