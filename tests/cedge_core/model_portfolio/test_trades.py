"""Known answers for generate_trades (hand-checked against the BLK_T1 reference)."""
import pytest
from cedge_core.model_portfolio.trades import (
    CASH_SHORT,
    EXC_TOO_SMALL,
    SKIPPED_IN_BAND,
    STILL_OUT_OF_BAND,
    TRADED,
    Rules,
    generate_trades,
)

MODEL = {"EQ": 0.60, "BOND": 0.40}
PRICES = {"EQ": 187.37, "BOND": 96.42}


def _w(res, t):
    return res.post_shares[t] * PRICES[t] / res.total_value


def test_in_band_is_skipped():
    res = generate_trades({"EQ": 305, "BOND": 392}, 3_400.0, MODEL, PRICES,
                          Rules(band=0.05, cash_buffer=0.05))
    assert res.status == SKIPPED_IN_BAND
    assert res.trades == {}
    assert res.total_value == pytest.approx(98_344.49, abs=0.01)
    assert res.max_drift_before == pytest.approx(0.011099, abs=1e-5)


def test_case1_full_rebalance():
    rules = Rules(band=0.05, cash_buffer=0.05, min_trade=250)
    res = generate_trades({"EQ": 352, "BOND": 318}, 3_400.0, MODEL, PRICES, rules)
    assert res.status == TRADED
    assert res.trades == {"EQ": -47, "BOND": 74}
    # cash starts at 3.4%, below the 5% buffer: refilling it trims BOND from 76 to 74
    assert res.flags == (CASH_SHORT,)
    assert _w(res, "EQ") == pytest.approx(0.5714, abs=1e-4)
    assert _w(res, "BOND") == pytest.approx(0.3779, abs=1e-4)
    assert res.post_cash / res.total_value == pytest.approx(0.0507, abs=1e-4)
    assert res.turnover == pytest.approx(0.0797, abs=1e-4)


def test_case1_rerun_is_idempotent():
    rules = Rules(band=0.05, cash_buffer=0.05, min_trade=250)
    first = generate_trades({"EQ": 352, "BOND": 318}, 3_400.0, MODEL, PRICES, rules)
    again = generate_trades(first.post_shares, first.post_cash, MODEL, PRICES, rules)
    assert again.status == SKIPPED_IN_BAND
    assert again.trades == {}


def test_case3_small_account_whole_shares():
    res = generate_trades({"EQ": 12, "BOND": 6}, 150.0, MODEL, PRICES,
                          Rules(band=0.05, cash_buffer=0.02))
    assert res.trades == {"EQ": -2, "BOND": 4}
    assert set(res.flags) == {CASH_SHORT, STILL_OUT_OF_BAND}
    assert _w(res, "EQ") == pytest.approx(0.6294, abs=1e-4)
    assert _w(res, "BOND") == pytest.approx(0.3239, abs=1e-4)
    assert res.post_cash / res.total_value == pytest.approx(0.0467, abs=1e-4)


def test_case3_fractional_hits_target():
    res = generate_trades({"EQ": 12, "BOND": 6}, 150.0, MODEL, PRICES,
                          Rules(band=0.05, cash_buffer=0.02, fractional=True))
    assert _w(res, "EQ") == pytest.approx(0.588, abs=1e-6)
    assert _w(res, "BOND") == pytest.approx(0.392, abs=1e-6)
    assert res.max_drift_after == pytest.approx(0.0, abs=1e-9)


def test_legacy_ticker_sold_in_full():
    # 352 * 841.89 / 841.89 evaluates to 351.99999999999994; naive truncation keeps 1 share
    res = generate_trades({"EQ": 300, "BOND": 300, "OLD": 352}, 1_000.0, MODEL,
                          {**PRICES, "OLD": 841.89}, Rules(band=0.05, cash_buffer=0.02))
    assert res.trades["OLD"] == -352
    assert res.post_shares["OLD"] == 0


def test_tiny_account_is_exception():
    # $200 cannot buy one EQ share at the 58.8% target
    res = generate_trades({}, 200.0, MODEL, PRICES, Rules(band=0.05, cash_buffer=0.02))
    assert res.status == EXC_TOO_SMALL
    assert res.trades == {}
    assert STILL_OUT_OF_BAND in res.flags


def test_invariants_hold():
    res = generate_trades({"EQ": 352, "BOND": 318}, 3_400.0, MODEL, PRICES,
                          Rules(band=0.01, cash_buffer=0.05))
    assert res.violations == ()
    assert res.post_cash >= 0
    assert all(float(q).is_integer() and q >= 0 for q in res.post_shares.values())


def test_leftover_cash_is_spent_so_rerun_is_a_no_op():
    # real 2026-10-14 closes. Proportional scaling + floor bought only 1 AGG and left
    # deployable cash, so every rerun bought 1 more AGG. The top-up spends it now.
    px = {"IVV": 773.09, "IJR": 135.41, "EFA": 102.775, "AGG": 94.82, "TIP": 104.82}
    model = {"IVV": 0.42, "IJR": 0.08, "EFA": 0.10, "AGG": 0.30, "TIP": 0.10}
    rules = Rules(band=0.03, cash_buffer=0.02)
    first = generate_trades({"IVV": 11, "IJR": 11, "EFA": 17, "AGG": 54, "TIP": 16},
                            655.09, model, px, rules)
    again = generate_trades(first.post_shares, first.post_cash, model, px, rules)
    assert first.status == TRADED
    assert first.trades == {"AGG": 2}           # was {"AGG": 1} before the top-up
    assert again.trades == {}