"""model_change: known answers worked by hand from small covariances built from vols and correlations."""
import numpy as np
import pandas as pd
import pytest
from cedge_core.model_portfolio.model_change import (
    annualized_covariance,
    compare_models,
    tracking_error,
)
from cedge_core.model_portfolio.models import ModelPortfolio


def cov_from(vols, corr):
    """Covariance DataFrame from {ticker: vol} and a correlation matrix in the same order."""
    t = list(vols)
    v = np.array([vols[k] for k in t])
    return pd.DataFrame(np.outer(v, v) * np.asarray(corr, dtype=float), index=t, columns=t)


# --- annualized_covariance -------------------------------------------------

R = pd.DataFrame({"A": [0.01, -0.01, 0.02, 0.0], "B": [0.0, 0.01, -0.01, 0.005]})
PX = pd.concat([pd.DataFrame({"A": [100.0], "B": [100.0]}), 100 * (1 + R).cumprod()],
               ignore_index=True).set_axis(pd.bdate_range("2026-01-05", periods=5))


def test_annualized_covariance_is_daily_cov_times_252():
    got = annualized_covariance(PX)
    assert list(got.index) == list(got.columns) == ["A", "B"]
    np.testing.assert_allclose(got.to_numpy(), (R.cov() * 252).to_numpy(), atol=1e-12)


def test_annualized_covariance_sorts_dates_first():
    pd.testing.assert_frame_equal(annualized_covariance(PX.iloc[::-1]), annualized_covariance(PX))


def test_annualized_covariance_rejects_gaps():
    with pytest.raises(ValueError):
        annualized_covariance(PX.assign(B=[100.0, np.nan, 101.0, 100.0, 100.5]))


# --- tracking_error ---------------------------------------------------------

def test_tracking_error_two_assets():
    # sigma 20% / 10%, rho 0.5, w (0.6, 0.4) vs (0.5, 0.5), d = (+0.1, -0.1):
    # TE^2 = 0.01 * 0.04 + 0.01 * 0.01 - 2 * 0.01 * 0.5 * 0.2 * 0.1 = 0.0003
    cov = cov_from({"A": 0.2, "B": 0.1}, [[1, 0.5], [0.5, 1]])
    te = tracking_error({"A": 0.6, "B": 0.4}, {"A": 0.5, "B": 0.5}, cov)
    assert te == pytest.approx(np.sqrt(0.0003), abs=1e-12)


def test_tracking_error_counts_a_missing_ticker_as_zero():
    # all in A vs all in B, uncorrelated: d = (1, -1), TE^2 = 0.04 + 0.01
    cov = cov_from({"A": 0.2, "B": 0.1}, np.eye(2))
    assert tracking_error({"A": 1.0}, {"B": 1.0}, cov) == pytest.approx(np.sqrt(0.05), abs=1e-12)


def test_tracking_error_names_a_ticker_missing_from_cov():
    cov = cov_from({"A": 0.2, "B": 0.1}, np.eye(2))
    with pytest.raises(ValueError, match="Z"):
        tracking_error({"Z": 1.0}, {"A": 1.0}, cov)


# --- compare_models ---------------------------------------------------------
# old {A .5, B .5} -> new {A .4, B .5, C .1}; A equity, B and C bonds; uncorrelated,
# vols 20% / 10% / 10%; benchmark {A .6, B .4}; aum $1bn.
#   diff (A -0.1, B 0, C +0.1): turnover 0.10, flows A -$100m, C +$100m, 2 funds change
#   new vs old    d = (-0.1, 0, +0.1):    0.01 * 0.04 + 0.01 * 0.01               = 0.0005
#   old vs bench  d = (-0.1, +0.1, 0):    same                                     = 0.0005
#   new vs bench  d = (-0.2, +0.1, +0.1): 0.04 * 0.04 + 0.01 * 0.01 + 0.01 * 0.01 = 0.0018

SLEEVES = {"A": "equity", "B": "bonds", "C": "bonds"}
COV3 = cov_from({"A": 0.2, "B": 0.1, "C": 0.1}, np.eye(3))
OLD = ModelPortfolio("M", "1.0", {"A": 0.5, "B": 0.5})
NEW = ModelPortfolio("M", "2.0", {"A": 0.4, "B": 0.5, "C": 0.1})
BENCH = ModelPortfolio("BM", "1.0", {"A": 0.6, "B": 0.4})


def test_compare_models_turnover_and_flows():
    ch = compare_models(OLD, NEW, COV3, aum=1e9, sleeve_of=SLEEVES)
    assert ch.diff.to_dict() == pytest.approx({"A": -0.1, "B": 0.0, "C": 0.1}, abs=1e-12)
    assert ch.turnover == pytest.approx(0.10, abs=1e-12)
    assert ch.n_changed == 2
    assert ch.flows is not None
    assert ch.flows["A"] == pytest.approx(-1e8) and ch.flows["C"] == pytest.approx(1e8)
    assert ch.te_old_new == pytest.approx(np.sqrt(0.0005), abs=1e-12)
    assert ch.te_old_bench is None and ch.te_new_bench is None


def test_compare_models_sleeves_and_benchmark():
    ch = compare_models(OLD, NEW, COV3, benchmark=BENCH, sleeve_of=SLEEVES)
    assert ch.diff_by_sleeve.to_dict() == pytest.approx({"bonds": 0.1, "equity": -0.1}, abs=1e-12)
    assert ch.te_old_bench == pytest.approx(np.sqrt(0.0005), abs=1e-12)
    assert ch.te_new_bench == pytest.approx(np.sqrt(0.0018), abs=1e-12)
    assert ch.flows is None


def test_compare_models_names_a_ticker_with_no_sleeve():
    with pytest.raises(ValueError, match="C"):
        compare_models(OLD, NEW, COV3, sleeve_of={"A": "equity", "B": "bonds"})


def test_no_change_is_all_zero():
    ch = compare_models(OLD, OLD, COV3, sleeve_of=SLEEVES)
    assert ch.turnover == 0.0 and ch.n_changed == 0 and ch.te_old_new == 0.0