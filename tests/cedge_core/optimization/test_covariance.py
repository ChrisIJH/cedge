import numpy as np
import pandas as pd
import pytest
from cedge_core.optimization.covariance import factor_model_covariance


def test_factor_model_covariance_known_answer():
    """Sigma = B @ Sigma_f @ B.T + D, hand-computed on a tiny example."""
    factor_betas = pd.DataFrame(
        {"mkt": [1.0, 0.5], "mom": [0.0, 1.0]},
        index=["AAPL", "MSFT"],
    )
    factor_cov = pd.DataFrame(
        {"mkt": [0.04, 0.01], "mom": [0.01, 0.02]},
        index=["mkt", "mom"],
    )
    resid_var = pd.Series([0.001, 0.002], index=["AAPL", "MSFT"])

    result = factor_model_covariance(factor_betas, factor_cov, resid_var)

    B = factor_betas.to_numpy()
    Sf = factor_cov.to_numpy()
    D = np.diag(resid_var.to_numpy())
    expected = B @ Sf @ B.T + D

    np.testing.assert_allclose(result, expected)


def test_factor_model_covariance_misaligned_factors_raises():
    factor_betas = pd.DataFrame({"mkt": [1.0]}, index=["AAPL"])
    factor_cov = pd.DataFrame({"mom": [0.02]}, index=["mom"])  # wrong factor name
    resid_var = pd.Series([0.001], index=["AAPL"])

    with pytest.raises(ValueError, match="factor_cov index/columns must exactly match"):
        factor_model_covariance(factor_betas, factor_cov, resid_var)


def test_factor_model_covariance_misaligned_tickers_raises():
    factor_betas = pd.DataFrame({"mkt": [1.0, 0.5]}, index=["AAPL", "MSFT"])
    factor_cov = pd.DataFrame({"mkt": [0.04]}, index=["mkt"], columns=["mkt"])
    resid_var = pd.Series([0.001], index=["AAPL"])  # missing MSFT

    with pytest.raises(ValueError, match="resid_var index must exactly match"):
        factor_model_covariance(factor_betas, factor_cov, resid_var)


def test_factor_model_covariance_nan_betas_raises():
    factor_betas = pd.DataFrame({"mkt": [1.0, np.nan]}, index=["AAPL", "MSFT"])
    factor_cov = pd.DataFrame({"mkt": [0.04]}, index=["mkt"], columns=["mkt"])
    resid_var = pd.Series([0.001, 0.002], index=["AAPL", "MSFT"])

    with pytest.raises(ValueError, match="factor_betas contains NaN"):
        factor_model_covariance(factor_betas, factor_cov, resid_var)


def test_factor_model_covariance_asymmetric_factor_cov_raises():
    """Guard for the CHResearch bug: triangle-only factor_covariance left
    unmirrored silently zeroes every cross-factor term."""
    factor_betas = pd.DataFrame({"mkt": [1.0], "mom": [0.5]}, index=["AAPL"])
    factor_cov = pd.DataFrame(
        {"mkt": [0.04, 0.0], "mom": [0.01, 0.02]},   # (mkt,mom)=0.01, (mom,mkt)=0.0
        index=["mkt", "mom"],
    )
    resid_var = pd.Series([0.001], index=["AAPL"])

    with pytest.raises(ValueError, match="not symmetric"):
        factor_model_covariance(factor_betas, factor_cov, resid_var)


def test_factor_model_covariance_nan_factor_cov_raises():
    factor_betas = pd.DataFrame({"mkt": [1.0], "mom": [0.5]}, index=["AAPL"])
    factor_cov = pd.DataFrame(
        {"mkt": [0.04, np.nan], "mom": [np.nan, 0.02]},   # unmirrored, as pivoted
        index=["mkt", "mom"],
    )
    resid_var = pd.Series([0.001], index=["AAPL"])

    with pytest.raises(ValueError, match="factor_cov contains NaN"):
        factor_model_covariance(factor_betas, factor_cov, resid_var)