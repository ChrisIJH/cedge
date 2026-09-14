"""Known-answer gates for cedge_core.risk.param_var (Normal linear VaR/ES).

Values independently verified two ways:
1. closed-form phi(z)/(1-confidence) vs a numerical tail-mean integral (10-digit agreement)
2. Monte Carlo, 4,000,000 draws, seed=42, Cholesky (4-digit agreement)

⚠ `confidence` here is the confidence level itself (0.99 -> 99% VaR) — the
OPPOSITE convention from var_es_model.parametric_var's `alpha` (tail
probability). See param_var.py's module docstring.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from cedge_core.risk.param_var import (
    normal_linear_es_confidence,
    normal_linear_es_confidence_WRONG,
    normal_linear_var_confidence,
    portfolio_sigma,
    rolling_parametric_var_confidence,
    rolling_parametric_var_confidence_WRONG,
)
from scipy import stats


def test_z_and_multiplier():
    assert stats.norm.ppf(0.95) == pytest.approx(1.6448536270, abs=1e-9)
    assert stats.norm.ppf(0.99) == pytest.approx(2.3263478740, abs=1e-9)

    z95, z99 = stats.norm.ppf(0.95), stats.norm.ppf(0.99)
    assert stats.norm.pdf(z95) / (1 - 0.95) == pytest.approx(2.0627128075, abs=1e-9)
    assert stats.norm.pdf(z99) / (1 - 0.99) == pytest.approx(2.6652142203, abs=1e-9)


def test_es_ge_var():
    sigma = 0.01
    ratios = {0.95: 1.2540403436, 0.99: 1.1456645199}
    for confidence in (0.90, 0.95, 0.975, 0.99):
        var = normal_linear_var_confidence(sigma, confidence=confidence, horizon=1, mu=0.0)
        es = normal_linear_es_confidence(sigma, confidence=confidence, horizon=1, mu=0.0)
        assert es >= var, f"ES < VaR at confidence={confidence} — coherence violated"
    for confidence, ratio in ratios.items():
        var = normal_linear_var_confidence(sigma, confidence=confidence, horizon=1, mu=0.0)
        es = normal_linear_es_confidence(sigma, confidence=confidence, horizon=1, mu=0.0)
        assert es / var == pytest.approx(ratio, abs=1e-9)


def test_es_wrong_breaks_coherence():
    sigma, confidence = 0.01, 0.99
    var = normal_linear_var_confidence(sigma, confidence=confidence, horizon=1, mu=0.0)
    es_wrong = normal_linear_es_confidence_WRONG(sigma, confidence=confidence, horizon=1, mu=0.0)
    assert es_wrong < var, "expected the WRONG (unnormalized) ES to fall BELOW VaR"


def test_sqrt_h_scaling():
    sigma, confidence = 0.01, 0.99
    var1 = normal_linear_var_confidence(sigma, confidence=confidence, horizon=1, mu=0.0)
    var10 = normal_linear_var_confidence(sigma, confidence=confidence, horizon=10, mu=0.0)
    assert var10 / var1 == pytest.approx(np.sqrt(10), abs=1e-12)

    mu = 0.0005
    var1_mu = normal_linear_var_confidence(sigma, confidence=confidence, horizon=1, mu=mu)
    var10_mu = normal_linear_var_confidence(sigma, confidence=confidence, horizon=10, mu=mu)
    assert var10_mu / var1_mu != pytest.approx(np.sqrt(10), abs=1e-6)


def test_portfolio_sigma():
    w = np.array([0.6, 0.4])
    cov = np.array([[0.040, 0.018],
                    [0.018, 0.090]])
    sigma_p = portfolio_sigma(w, cov)
    assert sigma_p ** 2 == pytest.approx(0.037440000000, abs=1e-12)
    assert sigma_p == pytest.approx(0.193494185959, abs=1e-9)
    assert sigma_p / np.sqrt(252) == pytest.approx(0.012188988004, abs=1e-9)


def test_portfolio_var_es_known_answer():
    w = np.array([0.6, 0.4])
    cov = np.array([[0.040, 0.018],
                    [0.018, 0.090]])
    sigma_daily = portfolio_sigma(w, cov) / np.sqrt(252)

    var_1d = normal_linear_var_confidence(sigma_daily, confidence=0.99, horizon=1, mu=0.0)
    es_1d = normal_linear_es_confidence(sigma_daily, confidence=0.99, horizon=1, mu=0.0)
    var_10d = normal_linear_var_confidence(sigma_daily, confidence=0.99, horizon=10, mu=0.0)

    assert var_1d == pytest.approx(0.028355826331, abs=1e-9)
    assert es_1d == pytest.approx(0.032486264161, abs=1e-9)
    assert var_10d == pytest.approx(0.089668996141, abs=1e-9)


def test_no_lookahead():
    rng = np.random.default_rng(42)
    returns = pd.Series(rng.normal(0, 0.01, 400))

    result = rolling_parametric_var_confidence(returns, confidence=0.99, window=252, vol_method="sample")
    result_wrong = rolling_parametric_var_confidence_WRONG(returns, confidence=0.99, window=252)

    fv_correct = result["var"].first_valid_index()
    fv_wrong = result_wrong["var"].first_valid_index()
    assert fv_correct is not None and fv_wrong is not None
    assert fv_correct > fv_wrong

    diff = (result["var"] - result_wrong["var"]).abs().dropna()
    assert diff.max() > 0
