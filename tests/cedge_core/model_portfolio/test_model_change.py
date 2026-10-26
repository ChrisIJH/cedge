"""model_change: known answers worked by hand from small covariances built from vols and correlations."""
import numpy as np
import pandas as pd
import pytest
from cedge_core.model_portfolio.accounts import Account
from cedge_core.model_portfolio.model_change import (
    annualized_covariance,
    compare_models,
    make_variant,
    round_to_grid,
    tracking_error,
)
from cedge_core.model_portfolio.models import ModelPortfolio
from cedge_core.model_portfolio.run import RunInputs, run
from cedge_core.model_portfolio.trades import Rules


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


# --- round_to_grid ----------------------------------------------------------

def test_round_to_grid_sums_to_exactly_one():
    # plain rounding to 0.5% gives .655 + .15 + .145 + .045 = 0.995; the leftover
    # step goes to the largest remainder (IVV, .6572 -> .66)
    w = pd.Series({"IVV": 0.6572, "IJR": 0.0, "EFA": 0.15, "AGG": 0.1471, "TIP": 0.0457})
    got = round_to_grid(w, 0.005)
    assert got.to_dict() == pytest.approx({"AGG": 0.145, "EFA": 0.15, "IJR": 0.0, "IVV": 0.66, "TIP": 0.045},
                                          abs=1e-12)
    assert got.sum() == pytest.approx(1.0, abs=1e-12)


def test_round_to_grid_breaks_ties_by_ticker():
    got = round_to_grid(pd.Series({"B": 0.5025, "A": 0.4975}), 0.005)    # both remainders 0.5
    assert got.to_dict() == pytest.approx({"A": 0.5, "B": 0.5}, abs=1e-12)


def test_round_to_grid_rejects_a_step_that_does_not_divide_one():
    with pytest.raises(ValueError, match="step"):
        round_to_grid(pd.Series({"A": 1.0}), 0.003)


# --- make_variant -----------------------------------------------------------

def test_variant_with_one_fund_left_has_one_answer():
    # universe {IVV, IJR}, base (.8, .2), IJR excluded: the only portfolio left is all IVV.
    # sigma 16% / 20%, rho 0.9: TE = 0.2 * sqrt(0.0256 + 0.04 - 2 * 0.9 * 0.16 * 0.2) = 0.2 * sqrt(0.008)
    cov = cov_from({"IVV": 0.16, "IJR": 0.20}, [[1, 0.9], [0.9, 1]])
    v = make_variant(ModelPortfolio("M", "1.0", {"IVV": 0.8, "IJR": 0.2}), cov, exclude=["IJR"],
                     sleeve_of={"IVV": "eq", "IJR": "eq"})
    assert v.status == "optimal"
    assert v.weights.to_dict() == pytest.approx({"IJR": 0.0, "IVV": 1.0}, abs=1e-6)
    assert v.te == pytest.approx(0.2 * np.sqrt(0.008), abs=1e-6)
    # shares d_i * (cov d)_i / TE^2, d = (IJR -0.2, IVV +0.2), cov(IVV, IJR) = 0.0288:
    #   IJR: -0.2 * (0.2 * 0.0288 - 0.2 * 0.04) / 0.00032 = +1.4   IVV: -0.4  (they sum to 1;
    #   the missing fund carries more than all of it, IVV takes some back)
    assert v.te_share.to_dict() == pytest.approx({"IJR": 1.4, "IVV": -0.4}, abs=1e-6)


def test_variant_moves_weight_to_the_correlated_fund():
    # all vols 20%, rho(A, C) = 0.9, others 0; base (.5, .3, .2), C excluded, dC = -0.2,
    # dB = 0.2 - dA:  TE^2 / 0.04 = dA^2 + (0.2 - dA)^2 + 0.04 - 0.36 dA,
    # derivative 4 dA - 0.76 = 0 -> dA = 0.19 -> (0.69, 0.31, 0); TE^2 = 0.04 * 0.0078 = 0.000312
    cov = cov_from({"A": 0.2, "B": 0.2, "C": 0.2}, [[1, 0, 0.9], [0, 1, 0], [0.9, 0, 1]])
    v = make_variant(ModelPortfolio("M", "1.0", {"A": 0.5, "B": 0.3, "C": 0.2}), cov, exclude=["C"],
                     sleeve_of={"A": "eq", "B": "eq", "C": "eq"})
    assert v.weights.to_dict() == pytest.approx({"A": 0.69, "B": 0.31, "C": 0.0}, abs=1e-6)
    assert v.te == pytest.approx(np.sqrt(0.000312), abs=1e-6)


def test_variant_reports_the_binding_cap():
    # uncorrelated, equal vols; base (.6, .2, .2), every fund capped at 45%: A gives up 0.15,
    # split evenly -> (.45, .275, .275), TE = 0.2 * sqrt(0.15^2 + 2 * 0.075^2)
    cov = cov_from({"A": 0.2, "B": 0.2, "C": 0.2}, np.eye(3))
    v = make_variant(ModelPortfolio("M", "1.0", {"A": 0.6, "B": 0.2, "C": 0.2}), cov, max_weight=0.45,
                     sleeve_of={"A": "eq", "B": "eq", "C": "eq"})
    assert v.rounded.to_dict() == pytest.approx({"A": 0.45, "B": 0.275, "C": 0.275}, abs=1e-12)
    assert v.binding == ("max_weight:A",)
    assert v.te == pytest.approx(0.2 * np.sqrt(0.03375), abs=1e-6)
    assert v.violations == ()


def test_rules_that_cannot_hold_come_back_as_a_status():
    # keep the bond sleeve at 40% while excluding every bond fund
    sleeves = {"E": "equity", "B1": "bonds", "B2": "bonds"}
    v = make_variant(ModelPortfolio("M", "1.0", {"E": 0.6, "B1": 0.2, "B2": 0.2}),
                     cov_from({"E": 0.16, "B1": 0.05, "B2": 0.05}, np.eye(3)),
                     exclude=["B1", "B2"], keep_sleeves=True, sleeve_of=sleeves)
    assert v.status == "infeasible"
    assert v.weights is None and v.rounded is None and v.binding == ()


def test_variant_is_deterministic():
    cov = cov_from({"A": 0.2, "B": 0.2, "C": 0.2}, [[1, 0, 0.9], [0, 1, 0], [0.9, 0, 1]])
    base = ModelPortfolio("M", "1.0", {"A": 0.5, "B": 0.3, "C": 0.2})
    one = make_variant(base, cov, exclude=["C"], sleeve_of={"A": "eq", "B": "eq", "C": "eq"})
    two = make_variant(base, cov, exclude=["C"], sleeve_of={"A": "eq", "B": "eq", "C": "eq"})
    pd.testing.assert_series_equal(one.rounded, two.rounded)
    assert one.te_rounded == two.te_rounded


def test_variant_feeds_the_account_engine():
    # the rounded variant is a valid model: run() trades an account onto it and reconciles
    sleeves = {t: "eq" for t in ("A", "B", "C")}
    cov = cov_from({"A": 0.2, "B": 0.2, "C": 0.2}, [[1, 0, 0.9], [0, 1, 0], [0.9, 0, 1]])
    v = make_variant(ModelPortfolio("M", "1.0", {"A": 0.5, "B": 0.3, "C": 0.2}), cov,
                     exclude=["C"], sleeve_of=sleeves)
    variant = v.to_model("M_NO_C", "1.0")
    assert set(variant.weights) == {"A", "B"}
    acct = Account("X", "M_NO_C", {"A": 50.0, "B": 30.0, "C": 20.0}, 0.0)
    res = run(RunInputs({"M_NO_C": variant}, "2026-10-16", Rules(), "test"), [acct],
              {"A": 100.0, "B": 100.0, "C": 100.0})
    c = res.counts
    assert c["n_target"] == c["n_processed"] + c["n_exception"] + c["n_skipped"] == 1
    assert res.post_accounts[0].shares.get("C", 0.0) == 0.0          # the excluded fund is sold